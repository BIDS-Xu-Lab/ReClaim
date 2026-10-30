#!/usr/bin/env python3
"""
Build the cost post-training pairs and the annual-cost bin artifact.

Consumes the supervision split produced by build_train_cohort
(`<baseline_dir>/{train,val}/preprocessed_input.parquet`), so post-training
uses the same patients, prediction points and token-sum labels as the
LightGBM baselines (matched supervision).

Outputs (under --out_dir):
  bin_artifact_mantissa.json      frozen bin definition: the existing <COST-me>
                                  tokens as annual-cost bins (bin id =
                                  quantized dollars), plus per-bin mean dollars
                                  and counts of the training split (descriptive;
                                  decoding uses the bins' own dollar values)
  mantissa/train_pairs.parquet    n_train_pairs rows
  mantissa/val_pairs.parquet      n_val_pairs rows
      columns: enrollee_id, context_seq, response_seq,
               the total dollar label, the total bin token

Response format (loss is masked to everything after context_seq):
  <INSTRUCT-COST> <tok_total> <eos>

Bins: labels are right-censored at --label_cap ($1M, the task definition),
re-rounded to one significant digit with the encode_cost operator and
expressed with the EXISTING <COST-me> tokens (no new vocabulary).
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset, disable_caching

disable_caching()

from utils import setup_logging, ID_COL, hash_salt, duck_connect, select_smallest_ids
from sequence_parsing import GOLD_COST_COLS
from posttrain_bins import encode_cost


def parse_args():
    p = argparse.ArgumentParser(description="Build cost post-training pairs + bin artifact")
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="supervision-split dir containing train/ and val/ preprocessed_input.parquet")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--label_cap", type=float, default=1_000_000,
                   help="Right-censor dollar labels before binning (task definition)")
    p.add_argument("--n_train_pairs", type=int, default=100_000)
    p.add_argument("--n_val_pairs", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=66)
    return p.parse_args()


# =============================================================================
# Bins
# =============================================================================

class MantissaBins:
    """Existing <COST-me> grid: labels re-rounded via the pre-training rounding operator."""

    def fit(self, total_costs: np.ndarray):
        pass

    @staticmethod
    def _code(q: float) -> str:
        if q <= 0:
            return "00"
        e = int(np.floor(np.log10(q) + 1e-9))  # epsilon guards exact powers of 10
        m = int(round(q / 10 ** e))
        return f"{m}{e}"

    def assign(self, values: np.ndarray) -> np.ndarray:
        # bin id = quantized dollar value itself (used only as a grouping key)
        return encode_cost(np.asarray(values, dtype=float)).astype(np.int64)

    def token_for(self, bin_ids: np.ndarray) -> list[str]:
        return [f"<COST-{self._code(q)}>" for q in bin_ids]

    def artifact(self) -> dict:
        return {"note": "existing <COST-me> tokens; bin id = quantized dollars"}


# =============================================================================
# Helpers
# =============================================================================

def load_labels(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["enrollee_id"] + GOLD_COST_COLS)


def conditional_means(bin_ids: np.ndarray, values: np.ndarray) -> dict:
    """Per-bin mean dollars (for EV decoding), as {bin_id: mean}."""
    df = pd.DataFrame({"bin": bin_ids, "v": values})
    return df.groupby("bin")["v"].mean().to_dict()


def build_pairs(split_path: str, n_pairs: int, seed: int, binner,
                split_name: str) -> pd.DataFrame:
    ds = load_dataset("parquet", data_files=split_path, split="train")
    n = len(ds)
    if n_pairs < n:
        # Content-keyed subsample (see utils.py): the n_pairs patients with the
        # smallest "pairs"-salted id-hashes — independent of row order/layout.
        eids = ds[ID_COL]
        keep = select_smallest_ids(duck_connect(), eids, n_pairs, hash_salt(seed, "pairs"))
        ds = ds.select(np.flatnonzero(np.fromiter((e in keep for e in eids), dtype=bool, count=len(eids))))   # keeps canonical id order
    logging.info("  [%s] Sampled %d/%d pairs (seed=%d, id-hash salt=%r)",
                 split_name, len(ds), n, seed, hash_salt(seed, "pairs"))

    cols = ["enrollee_id", "context_seq"] + GOLD_COST_COLS
    df = ds.remove_columns([c for c in ds.column_names if c not in cols]).to_pandas()

    resp_tokens = []
    for tc in GOLD_COST_COLS:
        vals = df[tc].clip(upper=binner.label_cap) if getattr(binner, "label_cap", None) else df[tc]
        ids = binner.assign(vals.values)
        toks = binner.token_for(ids)
        df[f"bin_token_{tc}"] = toks
        resp_tokens.append(toks)

    df["response_seq"] = [
        "<INSTRUCT-COST> " + " ".join(ts) + " <eos>"
        for ts in zip(*resp_tokens)
    ]
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    args = parse_args()
    logging.info("POST-TRAIN PAIR BUILD STARTING (mantissa bins)...")
    logging.info("  Baseline dir: %s", args.baseline_dir)
    logging.info("  Out dir:      %s", args.out_dir)

    train_path = os.path.join(args.baseline_dir, "train", "preprocessed_input.parquet")
    val_path = os.path.join(args.baseline_dir, "val", "preprocessed_input.parquet")

    # ------------------------------------------------------------------
    # 1. Fit bins on the FULL train split's total-cost labels
    # ------------------------------------------------------------------
    labels_train = load_labels(train_path)
    logging.info("  Full train split: %d rows for bin fitting", len(labels_train))

    binner = MantissaBins()
    binner.label_cap = args.label_cap if args.label_cap and args.label_cap > 0 else None
    capped = (labels_train["next_annual_cost"].clip(upper=binner.label_cap)
              if binner.label_cap else labels_train["next_annual_cost"])
    binner.fit(capped.values)

    # ------------------------------------------------------------------
    # 2. Conditional means per target (on the full train split)
    # ------------------------------------------------------------------
    artifact = binner.artifact()
    artifact["seed"] = args.seed
    artifact["fitted_on_rows"] = len(labels_train)
    artifact["conditional_mean_dollars"] = {}
    artifact["bin_counts"] = {}
    for tc in GOLD_COST_COLS:
        ids = binner.assign(labels_train[tc].values)
        artifact["conditional_mean_dollars"][tc] = {
            str(k): float(v) for k, v in conditional_means(ids, labels_train[tc].values).items()}
        counts = pd.Series(ids).value_counts().sort_index()
        artifact["bin_counts"][tc] = {str(k): int(v) for k, v in counts.items()}

    total_counts = pd.Series(binner.assign(labels_train["next_annual_cost"].values))
    logging.info("  Total-cost bins: min count=%d  median count=%d  (HNHC >=30k rows: %d)",
                 total_counts.value_counts().min(), total_counts.value_counts().median(),
                 int((labels_train["next_annual_cost"] >= 30000).sum()))

    os.makedirs(args.out_dir, exist_ok=True)
    artifact_path = os.path.join(args.out_dir, "bin_artifact_mantissa.json")
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=1)
    logging.info("  Saved bin artifact: %s", artifact_path)
    del labels_train

    # ------------------------------------------------------------------
    # 3. Build train/val pairs
    # ------------------------------------------------------------------
    pairs_dir = os.path.join(args.out_dir, "mantissa")
    os.makedirs(pairs_dir, exist_ok=True)

    for split_name, path, n_pairs in [("train", train_path, args.n_train_pairs),
                                      ("val", val_path, args.n_val_pairs)]:
        df = build_pairs(path, n_pairs, args.seed, binner, split_name)
        out_path = os.path.join(pairs_dir, f"{split_name}_pairs.parquet")
        df.to_parquet(out_path, index=False)
        logging.info("  [%s] Saved %d pairs to: %s", split_name, len(df), out_path)
        logging.info("  [%s] HNHC (>=30k) pairs: %d (%.2f%%)", split_name,
                     int((df['next_annual_cost'] >= 30000).sum()),
                     100 * (df['next_annual_cost'] >= 30000).mean())
        if split_name == "train":
            ex = df.iloc[0]
            logging.info("  Example response: %s", ex["response_seq"])
            logging.info("  Example label:    total=%d", ex["next_annual_cost"])

    logging.info("POST-TRAIN PAIR BUILD COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
