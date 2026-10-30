#!/usr/bin/env python3
"""
Feature engineering for the supervised cost-prediction baselines.

Reads preprocessed data (context_seq, future_seq, next_annual_cost*) and the
token vocabulary to construct:
  - Outcome: next_annual_cost (+ per-visit-type breakdowns), already computed
    by preprocess_find_prediction_point.py
  - Features:
      * Demographics: sex, age at the prediction point
      * prev_cost: total cost over the history window
      * Clinical: binary/count indicators of DX-MAJOR (3-char ICD-10) and RX
        (RxNorm ingredient) tokens over a configurable history window

Note: context_seq already carries <sos> and ends at a bare <NY>; age at
the prediction point = first <AGE-XX> value + number of <NY> tokens after it.

Outputs baseline_features_{history}_{type}.parquet next to each input.
"""

import os
import argparse
import logging
import re

import numpy as np
import pandas as pd
from datasets import load_dataset, disable_caching
from sklearn.feature_extraction.text import CountVectorizer

disable_caching()

from utils import setup_logging
from sequence_parsing import GOLD_COST_COLS, parse_cost_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature engineering for cost prediction baselines")
    parser.add_argument("--section_dirs", type=str, nargs='+', required=True,
                        help="Directories each containing a preprocessed_input.parquet")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Directory of vocab parquet files (category, token, token_freq)")
    parser.add_argument("--history_length", type=str, default="1year",
                        choices=["1year", "all"])
    parser.add_argument("--token_feature_type", type=str, default="binary",
                        choices=["binary", "count"])
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()


def sanitise_column_name(name: str) -> str:
    """Remove angle brackets from a column name for downstream model compatibility."""
    return name.replace("<", "").replace(">", "")


# =============================================================================
# Demographics
# =============================================================================

def extract_demographics(example: dict) -> dict:
    """Extract sex and age (at the prediction point) from context_seq."""
    tokens = example["context_seq"].split()

    sex = -1
    age_at_anchor = -1
    n_ny_after_age = 0
    seen_age = False

    for t in tokens:
        if t.startswith("<SEX-") and t.endswith(">"):
            val = t[5:-1]
            sex = 1 if val == "1" else (0 if val == "2" else -1)
        elif t.startswith("<AGE-") and t.endswith(">"):
            try:
                age_at_anchor = int(t[5:-1])
                seen_age = True
            except ValueError:
                pass
        elif seen_age and t.startswith("<NY"):
            n_ny_after_age += 1

    age = age_at_anchor + n_ny_after_age if age_at_anchor >= 0 else -1
    return {"sex": sex, "age": age}


# =============================================================================
# History windowing
# =============================================================================

def extract_relevant_tokens(example: dict, history_length: str) -> dict:
    """
    Extract the relevant history tokens from context_seq.

    "1year" : tokens between the second-to-last <NY> and the end of context_seq
              (context ends at the prediction-point <NY>, so this is the last
              full observation year).
    "all"   : tokens after <DOBYR-...> through the end of context_seq.

    Also aggregates COST tokens over the window into prev_cost.
    """
    tokens = example["context_seq"].strip().split()

    if history_length == "1year":
        ny_positions = [i for i, t in enumerate(tokens) if t.startswith("<NY")]
        if len(ny_positions) >= 2:
            start = ny_positions[-2] + 1
        else:
            return {"relevant_tokens": "", "prev_cost": 0}
        relevant_tokens = tokens[start:]
    elif history_length == "all":
        start_idx = 2
        for i, t in enumerate(tokens):
            if t.startswith("<DOBYR-"):
                start_idx = i + 1
                break
        relevant_tokens = tokens[start_idx:]
    else:
        raise ValueError(f"Unsupported history_length: {history_length}")

    prev_cost = 0
    for token in relevant_tokens:
        if token.startswith("<COST-"):
            prev_cost += parse_cost_value(token)

    return {"relevant_tokens": " ".join(relevant_tokens), "prev_cost": prev_cost}


# =============================================================================
# Vocab loading
# =============================================================================

def load_vocab_features(vocab_path: str) -> list[str]:
    """Return sorted list of DX-MAJOR (3-char ICD-10) and RX feature tokens."""
    vocab_ds = load_dataset("parquet",
                            data_files=os.path.join(vocab_path, "*.parquet"),
                            split="train")
    logging.info("  Vocab: loaded %d rows", len(vocab_ds))

    special_tokens = {
        "<DX-MISSING>", "<DX-NOMAP>", "<DX-PRINCIPAL>", "<DX-SECONDARY>",
        "<RX-MISSING>", "<RX-NOMAP>", "<RX-COMBSTART>", "<RX-COMBEND>",
    }

    vocab_relevant = vocab_ds.filter(
        lambda x: (
            x["token"] is not None
            and (
                (x["token"].startswith("<DX-MAJOR_") and re.match(r"<DX-MAJOR_[A-Z]", x["token"]))
                or (x["category"] == "RX" and x["token"] not in special_tokens)
            )
        )
    )

    relevant_tokens = sorted(set(vocab_relevant["token"]))
    logging.info("  Vocab: %d DX/RX feature tokens", len(relevant_tokens))
    return relevant_tokens


# =============================================================================
# Pipeline
# =============================================================================

def run_feature_pipeline(section_dir: str, args, feature_tokens: list[str],
                         token2idx: dict, num_proc: int):
    section = os.path.basename(os.path.normpath(section_dir))
    logging.info("=" * 60)
    logging.info("  [%s] Feature engineering", section.upper())
    logging.info("=" * 60)

    input_path = os.path.join(section_dir, "preprocessed_input.parquet")
    if not os.path.exists(input_path):
        logging.warning("  [%s] Input not found, skipping: %s", section, input_path)
        return
    ds = load_dataset("parquet", data_files=input_path, split="train")
    logging.info("  [%s] Loaded %d rows", section, len(ds))

    for col in ("context_seq", "future_seq", "enrollee_id", "next_annual_cost"):
        if col not in ds.column_names:
            raise ValueError(f"[{section}] Missing required column: {col}")

    available_targets = [c for c in GOLD_COST_COLS if c in ds.column_names]
    logging.info("  [%s] Available target columns: %s", section, available_targets)

    ds = ds.map(extract_demographics, num_proc=num_proc,
                desc=f"[{section}] Extracting demographics")
    ds = ds.map(lambda ex: extract_relevant_tokens(ex, args.history_length),
                num_proc=num_proc,
                desc=f"[{section}] Extracting relevant tokens ({args.history_length})")

    vectorizer = CountVectorizer(
        vocabulary=token2idx,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
    )

    logging.info("  [%s] Vectorizing clinical tokens...", section)
    clinical_sparse = vectorizer.transform(ds["relevant_tokens"])
    logging.info("  [%s] Clinical matrix shape: %s, nnz: %d",
                 section, clinical_sparse.shape, clinical_sparse.nnz)

    if args.token_feature_type == "binary":
        clinical_sparse = (clinical_sparse > 0).astype(np.int32)
        logging.info("  [%s] Binarised clinical features (count -> 0/1)", section)
    else:
        clinical_sparse = clinical_sparse.astype(np.int32)

    logging.info("  [%s] Assembling final DataFrame...", section)
    feature_names = [sanitise_column_name(t) for t in feature_tokens]

    result_df = pd.DataFrame({
        "enrollee_id": ds["enrollee_id"],
        "sex": ds["sex"],
        "age": ds["age"],
        "prev_cost": ds["prev_cost"],
    })

    clinical_df = pd.DataFrame.sparse.from_spmatrix(clinical_sparse, columns=feature_names)
    dtype = np.int32 if args.token_feature_type == "count" else np.int8
    clinical_df = clinical_df.sparse.to_dense().astype(dtype)

    result_df = pd.concat([result_df, clinical_df], axis=1)
    for tc in available_targets:
        result_df[tc] = ds[tc]

    logging.info("  [%s] Result shape: %s", section, result_df.shape)
    for tc in available_targets:
        logging.info("  [%s] %-40s mean=%.2f  median=%.2f  max=%d  zero_frac=%.3f",
                     section, tc, result_df[tc].mean(), result_df[tc].median(),
                     result_df[tc].max(), (result_df[tc] == 0).mean())

    clinical_nonzero = (clinical_sparse > 0).sum(axis=1).A1
    logging.info("  [%s] Clinical features per patient: mean=%.1f  median=%.1f  max=%d",
                 section, clinical_nonzero.mean(), np.median(clinical_nonzero),
                 clinical_nonzero.max())

    out_path = os.path.join(
        section_dir,
        f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet")
    result_df.to_parquet(out_path, index=False)
    logging.info("  [%s] Saved to: %s", section, out_path)


def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("FEATURE ENGINEERING STARTING...")

    args = parse_args()
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("  Config: history_length=%s, token_feature_type=%s, num_proc=%d",
                 args.history_length, args.token_feature_type, num_proc)
    logging.info("  Vocab:  %s", args.vocab_path)

    feature_tokens = load_vocab_features(args.vocab_path)
    token2idx = {t: i for i, t in enumerate(feature_tokens)}

    for section_dir in args.section_dirs:
        run_feature_pipeline(section_dir, args, feature_tokens, token2idx, num_proc)

    logging.info("FEATURE ENGINEERING COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
