#!/usr/bin/env python3
"""
Prepare evaluation data from trajectory parquet files.

This script loads parquet files, optionally filters by enrollee IDs,
optionally samples a fixed number of rows, then builds disease points
using add_disease_points_ages_and_locations.
"""

import argparse
import glob
import os
from typing import Set

import pandas as pd

from utils.data_ops import add_disease_points_ages_and_locations


def _load_enrollee_ids(path: str) -> Set[str]:
    """Load enrollee IDs from csv/parquet/txt."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.empty:
            return set()
        col = "enrollee_id" if "enrollee_id" in df.columns else df.columns[0]
        return set(df[col].astype(str).dropna().tolist())

    if lower.endswith(".csv"):
        df = pd.read_csv(path)
        if df.empty:
            return set()
        col = "enrollee_id" if "enrollee_id" in df.columns else df.columns[0]
        return set(df[col].astype(str).dropna().tolist())

    if lower.endswith(".txt"):
        ids: Set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
        return ids

    raise ValueError("Unsupported enrollee_ids_file format. Use .csv, .parquet, or .txt")


def _resolve_sample_size(args: argparse.Namespace) -> int:
    """Resolve sample size from --sample_size / --data_size."""
    if args.sample_size is not None and args.data_size is not None:
        raise ValueError("Use only one of --sample_size or --data_size")
    sample_size = args.sample_size if args.sample_size is not None else args.data_size
    if sample_size is not None and sample_size <= 0:
        raise ValueError("--sample_size must be > 0")
    return sample_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare evaluation data from trajectory parquet files")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--output_file", type=str, default="traj_test_pd.parquet", help="Output parquet filename")
    parser.add_argument("--demo_end_index", type=int, default=9, help="Index where demographics end")
    parser.add_argument("--age_index", type=int, default=3, help="Index where age starts")
    parser.add_argument("--seq_len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument(
        "--first-occurrence",
        dest="first_occurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use first occurrence of disease",
    )
    parser.add_argument(
        "--att_type",
        type=str,
        default="day",
        choices=["day", "week", "month"],
        help="Type of ATT tokens",
    )
    parser.add_argument("--absolute", action="store_true", default=False, help="Use absolute disease points")
    parser.add_argument("--longitudinal", action="store_true", help="Use longitudinal data")

    # New sampling controls.
    parser.add_argument("--sample_size", type=int, default=None, help="Sample fixed number of rows")
    parser.add_argument("--sample_random_state", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--enrollee_ids_file",
        type=str,
        default=None,
        help="Optional .csv/.parquet/.txt file containing enrollee IDs to keep",
    )

    # Backward-compatible alias.
    parser.add_argument("--data_size", type=int, default=None, help="Deprecated alias of --sample_size")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(f"{args.data_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {args.data_dir}")

    print(f"[load] parquet files found: {len(parquet_files)}")
    eval_raw_data = pd.concat((pd.read_parquet(f, engine="pyarrow") for f in parquet_files), ignore_index=True)
    input_rows = len(eval_raw_data)
    print(f"[load] input rows: {input_rows}")

    if args.enrollee_ids_file:
        if "enrollee_id" not in eval_raw_data.columns:
            raise ValueError("Column 'enrollee_id' not found in input data; cannot apply --enrollee_ids_file")
        keep_ids = _load_enrollee_ids(args.enrollee_ids_file)
        print(f"[filter] enrollee_ids loaded: {len(keep_ids)} from {args.enrollee_ids_file}")
        eval_raw_data = eval_raw_data[eval_raw_data["enrollee_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
        print(f"[filter] rows after enrollee_id filter: {len(eval_raw_data)}")

    sample_size = _resolve_sample_size(args)
    if sample_size is not None:
        available = len(eval_raw_data)
        if sample_size > available:
            print(f"[sample] requested {sample_size} > available {available}; using all {available}")
        else:
            eval_raw_data = eval_raw_data.sample(
                n=sample_size,
                random_state=args.sample_random_state,
                replace=False,
            ).reset_index(drop=True)
        print(f"[sample] rows after sample_size: {len(eval_raw_data)}")

    if eval_raw_data.empty:
        raise ValueError("No rows left after filtering/sampling")

    print("[process] building disease points/ages/visit locations...")
    traj_test_pd = add_disease_points_ages_and_locations(
        eval_raw_data,
        demo_end_index=args.demo_end_index,
        age_index=args.age_index,
        seq_len=args.seq_len,
        att_type=args.att_type,
        absolute=args.absolute,
        longitudinal=args.longitudinal,
        first_occurrence=args.first_occurrence,
    )
    print(f"[process] output rows after feature extraction: {len(traj_test_pd)}")

    if traj_test_pd.empty:
        raise ValueError("Output is empty after add_disease_points_ages_and_locations")

    output_path = os.path.join(args.output_dir, args.output_file)
    traj_test_pd.to_parquet(output_path, index=False)
    print(f"[save] wrote: {output_path}")
    print(
        f"[summary] files={len(parquet_files)} input={input_rows} "
        f"post_filter_sample={len(eval_raw_data)} final={len(traj_test_pd)}"
    )


if __name__ == "__main__":
    main()
