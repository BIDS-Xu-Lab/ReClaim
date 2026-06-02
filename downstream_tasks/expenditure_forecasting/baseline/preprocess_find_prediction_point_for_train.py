#!/usr/bin/env python3
"""
Preprocess sequence data for cost prediction task (v6 only).

Data version: v6 (relative month-based)
- <NY-XXXX> marks year boundaries
- Split at a middle <NY> token: context includes everything up to and including <NY>,
  future is everything after.
"""

import os
import argparse
import logging
from typing import Optional, Dict
from glob import glob
import numpy as np
from datasets import load_dataset, Dataset, disable_caching

disable_caching()

from utils import setup_logging



def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess patient sequences for cost prediction task'
    )
    parser.add_argument("--reclaim_data_dir", type=str,
                        default=os.path.join(
                            os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"),
                            "v6", "marketscan_full_combined"))
    parser.add_argument("--sample_n", type=int, default=5_000_000,
                        help='Number of examples to sample')
    parser.add_argument("--processed_data_dir", type=str,
                        default=os.path.join(
                            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                            "baseline", "versions", "v6", "data"))
    parser.add_argument("--num_proc", type=int, default=None,
                        help='Number of processes (default: CPU count - 1)')
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--min_age", type=int, default=16,
                        help='Minimum age to include (default: 16)')
    return parser.parse_args()


# =============================================================================
# Utility Functions
# =============================================================================

def sample_subset(dataset: Dataset, n: int, seed: int) -> Dataset:
    """Deterministically sample n examples from a dataset.

    Args:
        dataset: HuggingFace Dataset to sample from.
        n: Number of examples to sample (must be < len(dataset)).
        seed: Random seed for reproducibility.

    Returns:
        Dataset with n randomly sampled examples (deterministic given seed).
    """
    initial_count = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.choice(initial_count, size=n, replace=False)
    indices = np.sort(indices)  # preserve original order for sequential I/O
    dataset = dataset.select(indices)
    logging.info("  Sampled %d/%d examples (seed=%d)", n, initial_count, seed)
    return dataset


def tokenize(sequence: str) -> np.ndarray:
    """Convert sequence string to numpy array of tokens."""
    return np.array(sequence.strip().split())


def parse_age_value(sequence: str) -> Optional[int]:
    """Extract first age value from sequence. Returns None if not found."""
    for token in sequence.strip().split():
        if token.startswith('<AGE-') and token.endswith('>'):
            try:
                return int(token[5:-1])
            except ValueError:
                return None
    return None


def parse_cost_value(token: str) -> int:
    """Parse a single COST token to an integer dollar amount."""
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
GOLD_COST_COLS = [
    "next_annual_cost",
    *(f"next_annual_cost_{vt}" for vt in VT_TYPES),
]


def extract_outcome(example: dict) -> dict:
    """Sum COST tokens in future_seq up to the first <NY-...> token.

    Costs are attributed to the most recent <VT-...> token (inpatient,
    outpatient, or pharmacy).  Returns total cost and per-VT breakdown.
    """
    total = 0
    costs_by_vt = {vt: 0 for vt in VT_TYPES}
    current_vt = None
    for token in example["future_seq"].strip().split():
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
        "next_annual_cost": total,
        **{f"next_annual_cost_{vt}": costs_by_vt[vt] for vt in VT_TYPES},
    }


def _ineligible_result(n_tokens: int) -> dict:
    """Return standardized result for ineligible sequences."""
    return {
        'context_seq': None,
        'future_seq': None,
        'n_valid_points': 0,
        'n_tokens': n_tokens,
        'is_eligible': False
    }


# =============================================================================
# V6 Processing (NY-Based Split)
# =============================================================================

def process_example(example, idx, seed):
    """
    Process v6 format by splitting at a middle <NY> token.

    Finds all <NY-...> year boundary tokens, excludes the first and last,
    and randomly picks one of the remaining "middle" years as the split point.
    The <NY> token itself is included in the context sequence.
    """
    rng = np.random.default_rng(seed + idx)

    tokens = tokenize(example['seq_thru_2022'])
    n = len(tokens)

    # Find all <NY-...> token positions
    is_ny = np.char.startswith(tokens, '<NY')
    ny_indices = np.where(is_ny)[0]

    # Need at least 3 NY tokens to have a middle year
    if len(ny_indices) < 3:
        return _ineligible_result(n)

    # Middle years: exclude first and last NY tokens
    middle_indices = ny_indices[1:-1]
    n_valid = len(middle_indices)

    # Randomly pick a middle NY token as the split point
    split_idx = int(rng.choice(middle_indices))

    # Context includes everything up to and including the <NY> token
    # Future is everything after
    return {
        'context_seq': ' '.join(tokens[:split_idx + 1]),
        'future_seq': ' '.join(tokens[split_idx + 1:]),
        'n_valid_points': n_valid,
        'n_tokens': n,
        'is_eligible': True
    }


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("PREPROCESS STARTING...")

    args = parse_args()

    logging.info("Config: version=v6, min_age=%d", args.min_age)
    logging.info("  Input: %s", args.reclaim_data_dir)
    logging.info("  Output: %s", args.processed_data_dir)

    # Load data (always from 'train' split)
    trajectory_dir = os.path.join(args.reclaim_data_dir, "final_data", "train")
    parquet_files = sorted(glob(os.path.join(trajectory_dir, "*.parquet")))

    dataset = load_dataset('parquet', data_files=parquet_files, split='train') # set as train to load all the data
    logging.info("  Loaded %d sequences", len(dataset))

    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    # Sample to target size upfront
    if args.sample_n is not None and len(dataset) > args.sample_n:
        dataset = sample_subset(dataset, args.sample_n, args.seed)

    # Filter by age (before adding <sos> to avoid unnecessary work)
    initial_count = len(dataset)
    dataset = dataset.filter(
        lambda x: (age := parse_age_value(x['seq_thru_2022'])) is not None and age >= args.min_age,
        num_proc=num_proc,
        desc="Age filter"
    )
    logging.info("  Age filter: %d -> %d (kept %.1f%%)",
                 initial_count, len(dataset), 100 * len(dataset) / initial_count)

    # Add <sos> token
    dataset = dataset.map(
        lambda x: {'seq_thru_2022': '<sos> ' + str(x.get('seq_thru_2022', ''))},
        num_proc=num_proc,
        desc="Adding <sos>"
    )

    # Process sequences
    processed = dataset.map(
        lambda x, i: process_example(x, i, args.seed),
        with_indices=True,
        num_proc=num_proc,
        desc="Processing"
    )

    # Filter eligible
    eligible = processed.filter(lambda x: x['is_eligible'], num_proc=num_proc, desc="Filtering")
    logging.info("  Eligible: %d/%d (%.1f%%)", len(eligible), len(dataset), 100 * len(eligible) / len(dataset))

    # Extract outcome (next_annual_cost by total and VT type)
    eligible = eligible.map(extract_outcome, num_proc=num_proc, desc="Extracting cost outcome")
    for col in GOLD_COST_COLS:
        vals = eligible[col]
        logging.info("  %-40s mean=%.2f  median=%.2f  max=%d  zero_frac=%.3f",
                     col, np.mean(vals), np.median(vals), max(vals),
                     np.mean([c == 0 for c in vals]))

    # Log example
    if len(eligible) > 0:
        ex = eligible[0]
        logging.info("  Example context (last 10): %s", ' '.join(ex['context_seq'].split()[-10:]))
        logging.info("  Example future (first 10): %s", ' '.join(ex['future_seq'].split()[:10]))

    # Split into train (90%) and val (10%)
    split = eligible.train_test_split(test_size=0.1, seed=args.seed, shuffle=False)
    train_ds = split['train']
    val_ds = split['test']
    logging.info("  Train/val split: %d train, %d val", len(train_ds), len(val_ds))

    # Save
    train_path = os.path.join(args.processed_data_dir, "train", "preprocessed_input.parquet")
    val_path = os.path.join(args.processed_data_dir, "val", "preprocessed_input.parquet")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    logging.info("  Saved train to: %s", train_path)
    logging.info("  Saved val to: %s", val_path)

    # Summary
    if len(eligible) > 0:
        for name, ds in [("train", train_ds), ("val", val_ds)]:
            df = ds.to_pandas()
            logging.info("  %s stats: n=%d, tokens=%.1f, valid_points=%.1f (mean)",
                         name, len(df), df['n_tokens'].mean(), df['n_valid_points'].mean())

    logging.info("PREPROCESS COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
