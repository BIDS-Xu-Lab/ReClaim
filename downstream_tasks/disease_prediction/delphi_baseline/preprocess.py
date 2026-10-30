"""
Delphi Data Preprocessing

Converts tokenized patient trajectories (Parquet) into the Delphi binary
format (.bin) used for evaluation, plus a patient-id mapping.

Pipeline: Parquet → Spark filtering → per-patient parsing → Binary output
"""

import socket
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import os
import sys
import logging
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from tqdm import tqdm
from datetime import datetime


# === Logging Setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_dir, f"preprocessing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath)
    ]
)
logger = logging.getLogger("DelphiPreprocessing")
logger.info(f"Log file saved to: {log_filepath}")
logger.info(f"Current node: {socket.gethostname()}")

# PySpark needs a Java runtime: either `java` on PATH or JAVA_HOME set.
if "JAVA_HOME" in os.environ:
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ.get("PATH", "")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    """Parse command line arguments for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Delphi Data Preprocessing Pipeline")

    # --- Data paths ---
    parser.add_argument("--input_folder_path", type=str, required=True,
                        help="Path to folder containing input parquet files (required)")
    parser.add_argument("--output_folder_path", type=str, default='./data/',
                        help="Path to save output files (default: ./data/)")
    parser.add_argument("--delphi_disease_map", type=str, default="./nature_disease_map.csv",
                        help="Path to the Delphi disease map CSV file (nature_disease_map.csv)")

    # --- Split & output naming ---
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Data split type (default: test)")
    parser.add_argument("--output_prefix", type=str, default="ehr",
                        help="Prefix for output folder name (default: ehr)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output folder name, e.g., '_v2' (default: empty)")

    # --- Column names ---
    parser.add_argument("--pid_col", type=str, default="person_id",
                        help="Column name for Patient ID (default: person_id)")
    parser.add_argument("--seq_col", type=str, default="eval_seq",
                        help="Column name for the token sequence (default: eval_seq)")

    # --- Sequence parsing parameters ---
    parser.add_argument("--sex_index", type=int, default=0,
                        help="Index of sex token in sequence (default: 0)")
    parser.add_argument("--age_index", type=int, default=2,
                        help="Index of age token in sequence (default: 2)")
    parser.add_argument("--demo_end_index", type=int, default=3,
                        help="Index where demographic tokens end (default: 3)")

    # --- ATT token configuration ---
    parser.add_argument("--att_time_unit", type=str, default="months", choices=["days", "weeks", "months"],
                        help="Time unit for <ATT-X> tokens: 'days', 'weeks' (X*7), or 'months' (X*30) (default: months)")
    parser.add_argument("--att_time_mode", type=str, default="relative", choices=["relative", "absolute"],
                        help="Time mode: 'relative' (interval from previous) or 'absolute' (from initial age) (default: relative)")
    parser.add_argument(
        "--keep_sex_only_patients",
        action="store_true",
        help="Keep patients with no valid disease events as sex-only sequences (default: False)",
    )
    parser.add_argument("--spark_driver_memory", type=str, default="32g",
                        help="Spark driver memory (default: 32g)")

    return parser.parse_args()


# =============================================================================
# Spark & Data Loading
# =============================================================================

def init_spark_session(app_name='delphi_preprocessing', driver_memory="32g"):
    """Initialize and return a SparkSession."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "16g")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.pivotMaxValues", "1000000")
    spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
    return spark


def format_size(count):
    """Format a count into a human-readable string (e.g., 2M, 51K)."""
    if count >= 1_000_000:
        return f"{round(count / 1_000_000)}M"
    elif count >= 1_000:
        return f"{round(count / 1_000)}K"
    else:
        return str(count)


def filter_data(df, args):
    """Keep patients with complete demographic tokens."""
    total_count = df.count()
    logger.info(f"Total patients (before filtering): {total_count:,}")

    seq_c = col(args.seq_col)

    # Demographic filters
    filtered_df = (
        df
        .filter(seq_c.contains('<SEX-1>') | seq_c.contains('<SEX-2>'))
        .filter(seq_c.contains('<DOBYR'))
        .filter(seq_c.contains('<AGE'))
        .filter(~seq_c.contains('<ATT-LT>'))
    )

    return filtered_df


# =============================================================================
# Disease Mapping Loaders
# =============================================================================

def load_disease_mapping_csv(csv_file_path):
    """
    Load disease mapping from the Delphi CSV file (nature_disease_map.csv).

    The CSV has columns: disease, icd_code, disease_id.
    IDs are shifted by -1 (original 13 -> 12) to match Delphi's internal token space.

    Returns:
        dict: Mapping of disease codes (e.g., 'A00') to integer IDs (starting from 12).
    """
    logger.info(f"Loading disease mapping from CSV: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Disease mapping CSV file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    required_cols = ['disease', 'disease_id']
    for col_name in required_cols:
        if col_name not in df.columns:
            raise ValueError(f"Required column '{col_name}' not found in CSV. Available: {df.columns.tolist()}")

    unique_mappings = df[['disease', 'disease_id']].drop_duplicates()

    mapping = {}
    for _, row in unique_mappings.iterrows():
        mapping[row['disease']] = int(row['disease_id']) - 1  # Shift: 13 -> 12

    logger.info(f"Loaded {len(mapping)} diseases from CSV.")
    if mapping:
        logger.info(f"ID range: {min(mapping.values())} to {max(mapping.values())}")
        logger.info(f"Sample mappings: {list(mapping.items())[:5]}")
    
    return mapping


# =============================================================================
# Sequence Parsing
# =============================================================================

# Track unknown diseases for reporting
unknown_diseases = set()


def parse_disease_id(token_str, mapping):
    """
    Extract disease code from token and look up its ID.
    Token format: <DX-MAJOR_A00> -> 'A00' -> mapping lookup.
    """
    if not token_str.startswith('<DX-MAJOR_'):
        return -1
    major_code = token_str[10:-1]  # Remove '<DX-MAJOR_' and '>'
    result = mapping.get(major_code, -1)
    if result == -1:
        unknown_diseases.add(major_code)
    return result


def get_prediction_points_data(
        seq, disease_mapping, sex_mapping,
        demo_end_index=3, age_index=2, sex_index=0,
        max_length=4096, first_occurrence=True,
        att_time_unit="days", att_time_mode="relative"):
    """
    Parse a sequence string into model input format (prediction points).

    Args:
        seq: Space-separated token string
        disease_mapping: Dict mapping disease codes to IDs
        sex_mapping: Dict mapping sex tokens to IDs
        demo_end_index: Index after demographic tokens (start of events)
        age_index: Index of age token
        sex_index: Index of sex token
        max_length: Maximum sequence length to process
        first_occurrence: Only record the first occurrence of each disease
        att_time_unit: Time unit for ATT tokens - 'days', 'weeks', or 'months'
        att_time_mode: 'relative' (interval from previous) or 'absolute' (from initial age)

    Returns:
        tuple: (positions, ages_in_days, disease_ids, sex_id)
    """
    tokens = seq.split(" ")

    prediction_points = []
    prediction_points_ages = []
    prediction_points_disease_ids = []

    # Some evaluation sequences include a leading <sos> token, which shifts the
    # demographic tokens by one position relative to training-format sequences.
    seq_offset = 1 if tokens and tokens[0] == "<sos>" else 0
    sex_index += seq_offset
    age_index += seq_offset
    demo_end_index += seq_offset

    # Parse sex
    sex_id = 0
    if len(tokens) > sex_index and tokens[sex_index] in sex_mapping:
        sex_id = sex_mapping[tokens[sex_index]]

    # Parse age (format: <AGE-XX>)
    if len(tokens) <= age_index:
        return [], [], [], sex_id

    age_token = tokens[age_index]
    try:
        if age_token.startswith('<AGE-') and age_token.endswith('>'):
            day_age = int(age_token[5:-1]) * 365.25
        else:
            raise ValueError("Format mismatch")
    except (ValueError, IndexError):
        return [], [], [], sex_id

    days_passed = 0
    disease_tokens = set()

    # Time unit multiplier (convert ATT value to days)
    time_multiplier = {"days": 1, "weeks": 7, "months": 30}.get(att_time_unit, 1)

    length = min(len(tokens) - 2, max_length - 2)
    i = demo_end_index

    while i < length:
        current_token = tokens[i]

        if current_token.startswith('<ATT-'):
            try:
                att_value = int(current_token[5:-1]) if current_token != "<ATT-0>" else 0
                if att_time_mode == "relative":
                    # Relative: ATT value is interval from previous time point
                    days_passed += att_value * time_multiplier
                else:
                    # Absolute: ATT value is 1-indexed month from age reference
                    days_passed = max(att_value - 1, 0) * time_multiplier
            except ValueError:
                pass

        elif current_token.startswith('<AGE-') and current_token.endswith('>'):
            # Mid-sequence AGE token resets time reference in absolute mode
            if att_time_mode == "absolute":
                try:
                    day_age = int(current_token[5:-1]) * 365.25
                    days_passed = 0
                except (ValueError, IndexError):
                    pass

        elif current_token.startswith('<DX-MAJOR_'):
            disease_id = parse_disease_id(current_token, disease_mapping)

            should_record = (not first_occurrence) or (current_token not in disease_tokens)
            if first_occurrence:
                disease_tokens.add(current_token)

            if should_record:
                prediction_points.append(i)
                prediction_points_ages.append(day_age + days_passed)
                prediction_points_disease_ids.append(disease_id)

        i += 1

    return prediction_points, prediction_points_ages, prediction_points_disease_ids, sex_id


# =============================================================================
# Patient Processing
# =============================================================================

def process_all_patients(
    rows_iter,
    pid_col,
    seq_col,
    args,
    disease_mapping,
    sex_mapping,
    save_path,
    total_count,
    pid_mapping_tsv_path=None,
):
    """
    Stream patients from Spark, process them, and write directly to a binary file.

    Each patient produces: 1 sex token record + N disease token records.
    Patients with < 2 records (sex-only or only invalid diseases) are skipped.

    Args:
        rows_iter: Iterator of Spark Row objects.
        pid_col (str): Patient ID column.
        seq_col (str): Sequence column.
        args: Parsed argparse Namespace.
        disease_mapping (dict): Disease code -> token id mapping.
        sex_mapping (dict): Sex token -> token id mapping.
        save_path (str): Output .bin path.
        total_count (int): Number of rows (for progress bar).

    Returns:
        dict: Mapping from original patient IDs to new contiguous integer IDs.
    """
    # For large datasets, storing a full mapping from original patient IDs to
    # contiguous indices in memory is prohibitively heavy. We therefore
    # (a) always stream a mapping to disk (TSV), and
    # (b) only keep an in-memory dict for smaller datasets (<= 1M patients),
    #     where it can be convenient without causing OOM.
    save_pid_mapping = total_count <= 1_000_000
    pid_map = {} if save_pid_mapping else None
    next_pid_id = 0

    invalid_token_count = 0
    total_rows = 0
    kept_patients = 0
    kept_records = 0

    # Sequence length stats (per kept patient)
    seq_count = 0
    seq_sum = 0.0
    seq_sumsq = 0.0
    seq_min = None
    seq_max = 0

    token_min = None
    token_max = None

    logger.info("Processing patients and writing binary output (streaming)...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    map_f = open(pid_mapping_tsv_path, "w") if pid_mapping_tsv_path else None

    with open(save_path, "wb") as f:
        for row in tqdm(rows_iter, total=total_count, desc="Processing patients"):
            total_rows += 1

            original_pid = row[pid_col]
            seq = row[seq_col]

            if not seq:
                continue

            pid_id = pid_map.get(original_pid) if save_pid_mapping else None
            pid_is_new = False

            locs, ages, dids, sex_id = get_prediction_points_data(
                seq,
                disease_mapping=disease_mapping,
                sex_mapping=sex_mapping,
                demo_end_index=args.demo_end_index,
                age_index=args.age_index,
                sex_index=args.sex_index,
                first_occurrence=True,
                att_time_unit=args.att_time_unit,
                att_time_mode=args.att_time_mode,
            )

            disease_records = []
            for age, did in zip(ages, dids):
                if did == -1:
                    invalid_token_count += 1
                    continue
                if pid_id is None:
                    pid_id = next_pid_id
                    next_pid_id += 1
                    pid_is_new = True
                disease_records.append((pid_id, age, did))

            if not disease_records and not args.keep_sex_only_patients:
                continue

            if pid_id is None:
                pid_id = next_pid_id
                next_pid_id += 1
                pid_is_new = True

            if save_pid_mapping and pid_is_new:
                pid_map[original_pid] = pid_id
            if map_f is not None and pid_is_new:
                map_f.write(f"{original_pid}\t{pid_id}\n")

            # One demographic sex token (age=0) + disease tokens, sorted by age
            disease_records.sort(key=lambda x: x[1])
            records = [(pid_id, 0.0, sex_id)] + disease_records

            kept_patients += 1
            kept_records += len(records)

            # Sequence length stats
            L = len(records)
            seq_count += 1
            seq_sum += L
            seq_sumsq += float(L) * L
            seq_min = L if seq_min is None or L < seq_min else seq_min
            seq_max = max(seq_max, L)

            # Token ID stats
            for _, _, tid in records:
                if token_min is None or tid < token_min:
                    token_min = tid
                if token_max is None or tid > token_max:
                    token_max = tid

            np.asarray(records, dtype=np.uint32).tofile(f)

    if map_f is not None:
        map_f.close()

    logger.info("=== Processing Summary ===")
    logger.info(f"Total rows seen: {total_rows:,}")
    if args.keep_sex_only_patients:
        logger.info(f"Patients kept (including sex-only): {kept_patients:,}")
    else:
        logger.info(f"Patients kept (>= 1 disease event): {kept_patients:,}")
    logger.info(f"Total records written: {kept_records:,}")
    logger.info(f"Invalid tokens skipped: {invalid_token_count:,}")

    if seq_count > 0:
        mean_len = seq_sum / seq_count
        var = max(seq_sumsq / seq_count - mean_len ** 2, 0.0)
        std = var ** 0.5
        logger.info("=== Sequence Length Statistics ===")
        logger.info(f"Total patients: {seq_count:,}")
        logger.info(f"Avg: {mean_len:.2f}, Min: {seq_min}, Max: {seq_max}, Std: {std:.2f}")

    if token_min is not None and token_max is not None:
        logger.info(f"Token ID range: {token_min} to {token_max}")
        logger.info(f"Recommended vocab_size: {token_max + 2}")

    return pid_map or {}


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    args = parse_arguments()

    if not os.path.exists(args.input_folder_path):
        raise FileNotFoundError(f"Input folder not found: {args.input_folder_path}")

    os.makedirs(args.output_folder_path, exist_ok=True)

    # --- Log configuration ---
    logger.info("=== Configuration ===")
    logger.info(f"Split: {args.split}")
    logger.info(f"Input folder: {args.input_folder_path}")
    logger.info(f"Output folder: {args.output_folder_path}")
    logger.info(f"Columns: PID='{args.pid_col}', Seq='{args.seq_col}'")
    logger.info(f"ATT Time Unit: {args.att_time_unit}")
    logger.info(f"ATT Time Mode: {args.att_time_mode}")
    logger.info(f"Disease Map (CSV): {args.delphi_disease_map}")

    # --- Initialize Spark ---
    spark = init_spark_session(driver_memory=args.spark_driver_memory)
    logger.info(f"Spark UI: {spark.sparkContext.uiWebUrl}")

    # --- Load data ---
    try:
        traj_df = spark.read.parquet(args.input_folder_path)
        logger.info(f"Loaded trajectory data. Schema: {traj_df.columns}")
        logger.info(f"Total rows: {traj_df.count():,}")
    except Exception as e:
        logger.error(f"Failed to load data from {args.input_folder_path}: {e}")
        raise

    traj_df.show(5, truncate=100)

    # --- Filter data (Spark) ---
    traj_filtered_df = filter_data(traj_df, args)
    selected_df = traj_filtered_df.select(args.pid_col, args.seq_col).repartition(512)
    actual_size = selected_df.count()
    logger.info(f"Filtered to {actual_size:,} samples with complete demographic info")

    # --- Create output directory ---
    size_str = format_size(actual_size)
    save_dir = os.path.join(
        args.output_folder_path,
        f"{args.output_prefix}_data_{args.split}_{size_str}{args.output_suffix}"
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")

    # --- Load disease mapping (Delphi CSV) ---
    disease_mapping = load_disease_mapping_csv(args.delphi_disease_map)
    logger.info(f"Loaded {len(disease_mapping)} diseases.")

    sex_mapping = {'<SEX-2>': 1, '<SEX-1>': 2}
    unknown_diseases.clear()

    save_path = os.path.join(save_dir, f"{args.split}.bin")

    # --- Process all patients (streaming, memory-efficient) ---
    pid_mapping_tsv_path = os.path.join(save_dir, "pid_mapping.tsv")
    pid_mapping = process_all_patients(
        rows_iter=selected_df.toLocalIterator(),
        pid_col=args.pid_col,
        seq_col=args.seq_col,
        args=args,
        disease_mapping=disease_mapping,
        sex_mapping=sex_mapping,
        save_path=save_path,
        total_count=actual_size,
        pid_mapping_tsv_path=pid_mapping_tsv_path,
    )

    if os.path.exists(save_path):
        logger.info(f"=== {args.split.upper()} Data Saved ===")
        logger.info(f"File: {save_path}")
    else:
        logger.warning("No records were written; binary file was not created.")

    # Report unknown diseases
    if unknown_diseases:
        logger.info(f"Unknown diseases (not in mapping): {len(unknown_diseases)}")
        logger.info(f"Examples (first 20): {sorted(unknown_diseases)[:20]}")

    # Save disease mapping (for evaluation script to auto-detect)
    disease_mapping_path = os.path.join(save_dir, "disease_mapping.json")
    with open(disease_mapping_path, "w") as f:
        json.dump(disease_mapping, f, indent=2)
    logger.info(f"Saved disease mapping to: {disease_mapping_path}")

    # Save patient ID mapping (JSON only when reasonably sized; TSV always)
    if pid_mapping:
        pid_mapping_path = os.path.join(save_dir, "pid_mapping.json")
        with open(pid_mapping_path, "w") as f:
            json.dump({str(k): v for k, v in pid_mapping.items()}, f)
        logger.info(f"Saved patient ID mapping (JSON) to: {pid_mapping_path}")
    logger.info(f"Saved patient ID mapping (TSV) to: {pid_mapping_tsv_path}")

    logger.info("=== Preprocessing Pipeline Complete ===")

    try:
        spark.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
