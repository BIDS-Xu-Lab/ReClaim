#!/usr/bin/env python3
"""
Cost aggregation for RECLAIM model predictions.

Discovers all generated_data_{N}.parquet files in the model folder,
parses next_annual_cost from generated_seq, aggregates (mean) across
simulations per enrollee_id, and writes cost_predictions_{N}.csv for
each repetition.

Cost parsing rule: Sum COST tokens up to the first <NY-...> token.
(Consistent with preprocess_find_prediction_point.py extract_outcome logic.)
"""

import os
import re
import argparse
import logging

import numpy as np
import pandas as pd
from datasets import Dataset

from utils import setup_logging


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate cost predictions from RECLAIM generations')
    parser.add_argument("--model", type=str, required=True,
                        help='Model name (e.g., v6-s)')
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(
                            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                            "evaluation",
                            "versions",
                            "v6",
                            "data",
                            "model_gen",
                        ),
                        help='Base directory containing the generated data')
    parser.add_argument("--num_proc", type=int, default=None,
                        help='Number of processes for dataset.map (default: CPU count - 1)')
    return parser.parse_args()


# =============================================================================
# Cost Parsing (from preprocess_find_prediction_point.py)
# =============================================================================

def parse_cost_value(token: str) -> int:
    """Parse a single COST token to an integer dollar amount.
    
    E.g., <COST-53> -> 5 * 10^3 = 5000
    """
    if token == "<COST-MISSING>":
        return 0
    if not (token.startswith("<COST-") and token.endswith(">")):
        return 0
    cost_code = token[6:-1]
    if len(cost_code) >= 2:
        mantissa = int(cost_code[0])
        exponent = int(cost_code[1:])
        return mantissa * (10 ** exponent)
    return 0


VT_TYPES = ("inpatient", "outpatient", "pharmacy")
COST_COLS = [
    "predicted_next_annual_cost",
    *(f"predicted_next_annual_cost_{vt}" for vt in VT_TYPES),
]


def extract_outcome(sequence: str) -> dict:
    """Sum COST tokens in sequence up to the first <NY-...> token.

    Costs are attributed to the most recent <VT-...> token (inpatient,
    outpatient, or pharmacy).  Returns total cost and per-VT breakdown.
    """
    total = 0
    costs_by_vt = {vt: 0 for vt in VT_TYPES}
    current_vt = None
    for token in sequence.strip().split():
        if token.startswith("<NY"):
            break
        if token.startswith("<VT-") and token.endswith(">"):
            vt_type = token[4:-1]
            if vt_type in costs_by_vt:
                current_vt = vt_type
        elif token.startswith("<COST-"):
            val = parse_cost_value(token)
            total += val
            if current_vt is not None:
                costs_by_vt[current_vt] += val
    return {
        "predicted_next_annual_cost": total,
        **{f"predicted_next_annual_cost_{vt}": costs_by_vt[vt] for vt in VT_TYPES},
    }


# =============================================================================
# Repetition Discovery
# =============================================================================

def discover_repetitions(model_dir: str) -> list[int]:
    """Find all generated_data_{N}.parquet files and return sorted list of N."""
    pattern = re.compile(r'^generated_data_(\d+)\.parquet$')
    repetitions = []
    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            repetitions.append(int(m.group(1)))
    repetitions.sort()
    return repetitions


# =============================================================================
# Per-Repetition Processing
# =============================================================================

def process_repetition(data_file: str, num_proc: int) -> pd.DataFrame:
    """Load one generated_data parquet, parse costs, aggregate by enrollee_id.

    The ``generated_seq`` column may be either:
      - a list of K strings (new format: one list per unique prompt), or
      - a single string (old format: one row per generation).

    Returns a DataFrame with columns:
        enrollee_id, predicted_next_annual_cost,
        predicted_next_annual_cost_inpatient,
        predicted_next_annual_cost_outpatient,
        predicted_next_annual_cost_pharmacy
    """
    ds = Dataset.from_parquet(data_file)
    logging.info("    Loaded %d rows from %s", len(ds), os.path.basename(data_file))

    first = ds[0]['generated_seq']
    is_list = isinstance(first, list)

    def _mean_outcomes(sequences):
        outcomes = [extract_outcome(s) for s in sequences]
        return {col: np.mean([o[col] for o in outcomes]) for col in COST_COLS}

    if is_list:
        ds = ds.map(
            lambda ex: _mean_outcomes(ex["generated_seq"]),
            num_proc=num_proc,
            desc="Extracting costs",
        )
    else:
        ds = ds.map(
            lambda ex: extract_outcome(ex["generated_seq"]),
            num_proc=num_proc,
            desc="Extracting costs",
        )

    for col in COST_COLS:
        logging.info("    Per-row %-45s mean=%.2f  median=%.2f  std=%.2f",
                     col, np.mean(ds[col]), np.median(ds[col]), np.std(ds[col]))

    df = ds.to_pandas()
    result = df.groupby('enrollee_id')[COST_COLS].mean().reset_index()

    logging.info("    Aggregated: %d enrollees", len(result))
    for col in COST_COLS:
        logging.info("    Agg %-49s mean=%.2f  median=%.2f  std=%.2f",
                     col, result[col].mean(), result[col].median(), result[col].std())

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("COST AGGREGATION STARTING...")

    args = parse_args()
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("  Model: %s", args.model)
    logging.info("  Output base dir: %s", args.output_dir)
    logging.info("  Num proc: %d", num_proc)

    # ------------------------------------------------------------------
    # 1. Discover all repetition files
    # ------------------------------------------------------------------
    repetitions = discover_repetitions(args.output_dir)
    if not repetitions:
        logging.error("  No generated_data_*.parquet files found in %s", args.output_dir)
        return
    logging.info("  Found %d repetition(s): %s", len(repetitions), repetitions)

    # ------------------------------------------------------------------
    # 2. Process each repetition
    # ------------------------------------------------------------------
    for rep in repetitions:
        data_file = os.path.join(args.output_dir, f"generated_data_{rep}.parquet")
        logging.info("  --- Repetition %d ---", rep)

        result = process_repetition(data_file, num_proc)

        output_path = os.path.join(args.output_dir, f"cost_predictions_{rep}.csv")
        result.to_csv(output_path, index=False)
        logging.info("    Saved: %s", output_path)

    logging.info("")
    logging.info("COST AGGREGATION COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
