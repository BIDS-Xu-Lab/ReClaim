#!/usr/bin/env python3
"""
Aggregate ReClaim MC generations into per-enrollee cost predictions.

Discovers all generated_data_{N}.parquet files in the model folder, parses
next-year cost from each generated trajectory (COST tokens up to the first
<NY> or <eos>), averages across the N simulations per enrollee, and writes
cost_predictions_{N}.csv.

Each trajectory's parsed cost is capped at $1M BEFORE averaging (the same
cap the evaluation applies to final predictions).  Without this, a single
sampled extreme token (e.g. <COST-98> = 8x10^8) poisons the Monte-Carlo mean and
collapses R2.

The CSV also carries prob_ge_<cutoff> = fraction of the N
trajectories whose (capped) total is at or above each --cutoffs value: the MC
anchor's HNHC score, identical to what e4_eval/eval_binary_auc.py derives from
the parquet, so AUCs at every N can be computed from the CSVs alone (the parquets
are too large to move between clusters).
"""

GEN_CAP = 1_000_000

import os
import re
import argparse
import logging

import numpy as np
import pandas as pd
from datasets import Dataset, disable_caching

disable_caching()

from utils import setup_logging
from sequence_parsing import PRED_COST_COLS, extract_outcome


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate cost predictions from ReClaim generations')
    parser.add_argument("--model_gen_dir", type=str, required=True,
                        help='Model generation directory (contains generated_data_*.parquet)')
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--cutoffs", type=float, nargs="*",
                        default=[float(os.environ.get("CUTOFF_HNHC", "30000"))],
                        help="Exceedance cutoffs written as prob_ge_<int> columns (default: $CUTOFF_HNHC)")
    return parser.parse_args()


def discover_repetitions(model_dir: str) -> list[int]:
    pattern = re.compile(r'^generated_data_(\d+)\.parquet$')
    reps = [int(m.group(1)) for f in os.listdir(model_dir)
            if (m := pattern.match(f))]
    return sorted(reps)


def process_repetition(data_file: str, num_proc: int, cutoffs=()) -> pd.DataFrame:
    """Parse costs from one generated_data parquet and average per enrollee; also the
    per-enrollee fraction of trajectories at/above each cutoff (prob_ge_<cutoff>)."""
    prob_cols = {c: f"prob_ge_{int(c)}" for c in cutoffs}
    total = PRED_COST_COLS[0]                      # predicted_next_annual_cost
    ds = Dataset.from_parquet(data_file)
    logging.info("    Loaded %d rows from %s", len(ds), os.path.basename(data_file))

    is_list = isinstance(ds[0]['generated_seq'], list)

    def _mean_outcomes(sequences):
        outcomes = [extract_outcome(s, prefix="predicted_") for s in sequences]
        out = {col: float(np.mean([min(o[col], GEN_CAP) for o in outcomes]))
               for col in PRED_COST_COLS}
        capped = [min(o[total], GEN_CAP) for o in outcomes]
        for c, name in prob_cols.items():
            out[name] = float(np.mean([v >= c for v in capped]))
        return out

    def _single_outcome(sequence):
        o = extract_outcome(sequence, prefix="predicted_")
        for c, name in prob_cols.items():
            o[name] = float(min(o[total], GEN_CAP) >= c)      # 0/1 per row; the groupby mean below = fraction
        return o

    if is_list:
        ds = ds.map(lambda ex: _mean_outcomes(ex["generated_seq"]),
                    num_proc=num_proc, desc="Extracting costs")
    else:
        ds = ds.map(lambda ex: _single_outcome(ex["generated_seq"]),
                    num_proc=num_proc, desc="Extracting costs")

    for col in PRED_COST_COLS:
        logging.info("    Per-row %-45s mean=%.2f  median=%.2f  std=%.2f",
                     col, np.mean(ds[col]), np.median(ds[col]), np.std(ds[col]))

    df = ds.to_pandas()
    result = df.groupby('enrollee_id')[PRED_COST_COLS + list(prob_cols.values())].mean().reset_index()
    for name in prob_cols.values():
        logging.info("    %-20s mean=%.4f (fraction of trajectories at/above the cutoff)", name, result[name].mean())
    logging.info("    Aggregated: %d enrollees", len(result))
    return result


def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("COST AGGREGATION STARTING...")

    args = parse_args()
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)
    logging.info("  Model gen dir: %s", args.model_gen_dir)

    repetitions = discover_repetitions(args.model_gen_dir)
    if not repetitions:
        logging.error("  No generated_data_*.parquet in %s", args.model_gen_dir)
        return
    logging.info("  Found %d repetition file(s): %s", len(repetitions), repetitions)

    for rep in repetitions:
        data_file = os.path.join(args.model_gen_dir, f"generated_data_{rep}.parquet")
        logging.info("  --- Repetition %d ---", rep)
        result = process_repetition(data_file, num_proc, args.cutoffs)
        output_path = os.path.join(args.model_gen_dir, f"cost_predictions_{rep}.csv")
        result.to_csv(output_path, index=False)
        logging.info("    Saved: %s", output_path)

    logging.info("COST AGGREGATION COMPLETED")
        # Quick-look preview (formal metrics incl. trajectory-based AUC: e4_eval)
    try:
        import numpy as _np
        _sec = os.path.dirname(os.path.dirname(os.path.normpath(args.model_gen_dir)))
        _gold = pd.read_parquet(os.path.join(_sec, "preprocessed_input.parquet"),
                                columns=["enrollee_id", "next_annual_cost"])
        _pred = pd.read_csv(os.path.join(args.model_gen_dir,
                                         f"cost_predictions_{max(repetitions)}.csv"))
        _m = _gold.merge(_pred[["enrollee_id", "predicted_next_annual_cost"]], on="enrollee_id")
        _y = _np.minimum(_m["next_annual_cost"].to_numpy(float), 1_000_000.0)
        _p = _m["predicted_next_annual_cost"].to_numpy(float)
        _r2 = 1 - ((_y - _p) ** 2).sum() / ((_y - _y.mean()) ** 2).sum()
        logging.info("PREVIEW [%s] R2=%.3f MAE=%.0f (censored gold; formal metrics via e4_eval)",
                     os.path.basename(_sec), _r2, _np.abs(_y - _p).mean())
    except Exception as e:
        logging.warning("preview skipped: %s", e)
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
