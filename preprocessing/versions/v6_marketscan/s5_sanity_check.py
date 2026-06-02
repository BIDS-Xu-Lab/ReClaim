#!/usr/bin/env python3
"""
Reclaim Data Sanity Check
Performs various sanity checks on trajectory data from S2.

Checks performed:
1. Missing token check: <DOBYR>, <SEX>, <AGE>
2. Temporal information check: start_year, end_year, num_years_passed comparison
3. Split point checks (for rows with first_token_index_from_2023 available):
   - Token at split index is <NY>
   - First token of event_seq_from_2023 is <NY>
   - Last token of seq_thru_2022 is <ATT-*>
   - DOBYR + AGE + count(<NY>) = 2022 for seq_thru_2022
"""

import logging
import os
import argparse
import re
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, BooleanType, StringType

from utils import setup_logging, time_execution, spark_log_contingency_table

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "
INDENT4 = "        "

# Sample size for sanity checks
SAMPLE_SIZE = 100_000


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run sanity checks on trajectory data')
    parser.add_argument("--input_folder_path", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"), "v6", "marketscan_val"),
                       help='Base directory containing processed data')
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
                       help='Number of rows to sample for sanity checks (default: 100000)')
    parser.add_argument("--seed", type=int, default=42,
                       help='Random seed for sampling')
    return parser.parse_args()


# =============================================================================
# Helper UDFs for token extraction and checking
# =============================================================================

def check_token_exists(seq: str, token_pattern: str) -> bool:
    """Check if a token matching the pattern exists in the sequence"""
    if seq is None:
        return False
    return bool(re.search(token_pattern, seq))


def extract_token_value(seq: str, token_pattern: str) -> int:
    """Extract numeric value from a token matching the pattern"""
    if seq is None:
        return None
    match = re.search(token_pattern, seq)
    if match:
        return int(match.group(1))
    return None


def count_token_occurrences(seq: str, token_pattern: str) -> int:
    """Count occurrences of a token matching the pattern"""
    if seq is None:
        return 0
    return len(re.findall(token_pattern, seq))


def get_token_at_index(seq: str, index: int) -> str:
    """Get the token at a specific index (0-based)"""
    if seq is None or index is None:
        return None
    tokens = seq.split(" ")
    if 0 <= index < len(tokens):
        return tokens[index]
    return None


def get_first_token(seq: str) -> str:
    """Get the first token from a sequence"""
    if seq is None or seq.strip() == "":
        return None
    tokens = seq.strip().split(" ")
    return tokens[0] if tokens else None


def get_last_token(seq: str) -> str:
    """Get the last token from a sequence"""
    if seq is None or seq.strip() == "":
        return None
    tokens = seq.strip().split(" ")
    return tokens[-1] if tokens else None


def is_att_token(token: str) -> bool:
    """Check if a token is an ATT token (<ATT-N>)"""
    if token is None:
        return False
    return bool(re.match(r'^<ATT-\d+>$', token))


def is_ny_token(token: str) -> bool:
    """Check if a token is the NY token (<NY>)"""
    return token == "<NY>"


def get_tokens_around_index(seq: str, index: int, window: int = 5) -> str:
    """
    Get tokens around a specific index (±window tokens) for diagnostic purposes.
    Returns a string with the token at index marked with [*TOKEN*].
    """
    if seq is None or index is None:
        return None
    tokens = seq.split(" ")
    if index < 0 or index >= len(tokens):
        return None
    
    start_idx = max(0, index - window)
    end_idx = min(len(tokens), index + window + 1)
    
    # Build result with the target token marked
    result_tokens = []
    for i in range(start_idx, end_idx):
        if i == index:
            result_tokens.append(f"[*{tokens[i]}*]")
        else:
            result_tokens.append(tokens[i])
    
    return " ".join(result_tokens)


# =============================================================================
# Sanity Check Functions
# =============================================================================

@time_execution
def check_missing_tokens(_spark: SparkSession, df, sample_count: int) -> None:
    """
    Check 1: Missing token check
    For each seq, check if specified tokens exist: <DOBYR>, <SEX>, <AGE>
    Log the missing rate for each token type.
    """
    logging.info("********** CHECK 1: Missing Token Check **********")
    
    # Register UDFs
    check_dobyr_udf = F.udf(lambda seq: check_token_exists(seq, r'<DOBYR-\d+>'), BooleanType())
    check_sex_udf = F.udf(lambda seq: check_token_exists(seq, r'<SEX-[12MISSING]+>'), BooleanType())
    check_age_udf = F.udf(lambda seq: check_token_exists(seq, r'<AGE-\d+>'), BooleanType())
    
    # Add columns for token existence
    df_with_checks = df.withColumn("has_dobyr", check_dobyr_udf(F.col("seq"))) \
                       .withColumn("has_sex", check_sex_udf(F.col("seq"))) \
                       .withColumn("has_age", check_age_udf(F.col("seq")))
    
    # Calculate missing rates
    stats = df_with_checks.agg(
        F.sum(F.when(~F.col("has_dobyr"), 1).otherwise(0)).alias("missing_dobyr"),
        F.sum(F.when(~F.col("has_sex"), 1).otherwise(0)).alias("missing_sex"),
        F.sum(F.when(~F.col("has_age"), 1).otherwise(0)).alias("missing_age")
    ).collect()[0]
    
    missing_dobyr_rate = (stats["missing_dobyr"] / sample_count) * 100
    missing_sex_rate = (stats["missing_sex"] / sample_count) * 100
    missing_age_rate = (stats["missing_age"] / sample_count) * 100
    
    logging.info("%sSample size: %s", INDENT1, f"{sample_count:,}")
    logging.info("%s<DOBYR> missing count: %s (%.4f%%)", INDENT1, f"{stats['missing_dobyr']:,}", missing_dobyr_rate)
    logging.info("%s<SEX> missing count: %s (%.4f%%)", INDENT1, f"{stats['missing_sex']:,}", missing_sex_rate)
    logging.info("%s<AGE> missing count: %s (%.4f%%)", INDENT1, f"{stats['missing_age']:,}", missing_age_rate)
    
    # Alert if any missing rate is concerning
    for token_name, rate in [("<DOBYR>", missing_dobyr_rate), ("<SEX>", missing_sex_rate), ("<AGE>", missing_age_rate)]:
        if rate > 0:
            logging.warning("%sWARNING: %s has non-zero missing rate: %.4f%%", INDENT2, token_name, rate)


@time_execution
def check_temporal_information(_spark: SparkSession, df, _sample_count: int) -> None:
    """
    Check 2: Temporal information check
    - Extract DOBYR and AGE values, sum to get start_year_seq
    - Count <NY> tokens to get num_years_passed_seq
    - Calculate end_year_seq = start_year_seq + num_years_passed_seq
    - Use floor(total_duration/365) to get num_year_passed_dur
    - Compare num_years_passed_seq with num_year_passed_dur
    - Log distribution and rate of delta > 1, > 2
    """
    logging.info("********** CHECK 2: Temporal Information Check **********")
    
    # Register UDFs
    extract_dobyr_udf = F.udf(lambda seq: extract_token_value(seq, r'<DOBYR-(\d+)>'), IntegerType())
    extract_age_udf = F.udf(lambda seq: extract_token_value(seq, r'<AGE-(\d+)>'), IntegerType())
    count_ny_udf = F.udf(lambda seq: count_token_occurrences(seq, r'<NY>'), IntegerType())
    
    # Extract temporal values
    df_temporal = df.withColumn("dobyr", extract_dobyr_udf(F.col("seq"))) \
                    .withColumn("age", extract_age_udf(F.col("seq"))) \
                    .withColumn("num_years_passed_seq", count_ny_udf(F.col("seq"))) \
                    .withColumn("start_year_seq", F.col("dobyr") + F.col("age")) \
                    .withColumn("end_year_seq", F.col("start_year_seq") + F.col("num_years_passed_seq")) \
                    .withColumn("num_year_passed_dur", F.floor(F.col("total_duration") / 365).cast(IntegerType())) \
                    .withColumn("year_delta", F.abs(F.col("num_years_passed_seq") - F.col("num_year_passed_dur")))
    
    # Cache for multiple operations
    df_temporal.cache()
    
    # Log distributions using contingency table (discrete year values)
    logging.info("%sDistribution of start_year_seq:", INDENT1)
    spark_log_contingency_table(df_temporal.filter(F.col("start_year_seq").isNotNull()), "start_year_seq")
    
    logging.info("%sDistribution of end_year_seq:", INDENT1)
    spark_log_contingency_table(df_temporal.filter(F.col("end_year_seq").isNotNull()), "end_year_seq")
    
    logging.info("%sDistribution of num_years_passed_seq:", INDENT1)
    spark_log_contingency_table(df_temporal.filter(F.col("num_years_passed_seq").isNotNull()), "num_years_passed_seq")
    
    logging.info("%sDistribution of num_year_passed_dur (from total_duration):", INDENT1)
    spark_log_contingency_table(df_temporal.filter(F.col("num_year_passed_dur").isNotNull()), "num_year_passed_dur")
    
    # Calculate delta comparison stats
    valid_comparison_df = df_temporal.filter(
        F.col("num_years_passed_seq").isNotNull() & 
        F.col("num_year_passed_dur").isNotNull()
    )
    valid_count = valid_comparison_df.count()
    
    if valid_count > 0:
        delta_stats = valid_comparison_df.agg(
            F.sum(F.when(F.col("year_delta") > 1, 1).otherwise(0)).alias("delta_gt_1"),
            F.sum(F.when(F.col("year_delta") > 2, 1).otherwise(0)).alias("delta_gt_2"),
            F.avg("year_delta").alias("avg_delta"),
            F.max("year_delta").alias("max_delta")
        ).collect()[0]
        
        rate_delta_gt_1 = (delta_stats["delta_gt_1"] / valid_count) * 100
        rate_delta_gt_2 = (delta_stats["delta_gt_2"] / valid_count) * 100
        
        logging.info("%sTemporal consistency comparison (num_years_passed_seq vs num_year_passed_dur):", INDENT1)
        logging.info("%sValid comparisons: %s", INDENT2, f"{valid_count:,}")
        logging.info("%sAverage delta: %.2f years", INDENT2, delta_stats["avg_delta"])
        logging.info("%sMax delta: %d years", INDENT2, delta_stats["max_delta"])
        logging.info("%sCount with delta > 1 year: %s (%.4f%%)", INDENT2, f"{delta_stats['delta_gt_1']:,}", rate_delta_gt_1)
        logging.info("%sCount with delta > 2 years: %s (%.4f%%)", INDENT2, f"{delta_stats['delta_gt_2']:,}", rate_delta_gt_2)
        
        # Show some examples of high delta cases
        if delta_stats["delta_gt_2"] > 0:
            logging.info("%sExamples of cases with delta > 2:", INDENT2)
            high_delta_examples = valid_comparison_df.filter(F.col("year_delta") > 2) \
                .select("enrollee_id", "start_year_seq", "end_year_seq", "num_years_passed_seq", 
                        "num_year_passed_dur", "year_delta", "total_duration") \
                .limit(10).collect()
            for row in high_delta_examples:
                logging.info("%senrollee_id=%s, start=%s, end=%s, ny_count=%s, dur_years=%s, delta=%s, total_dur=%s", 
                            INDENT3, row["enrollee_id"], row["start_year_seq"], row["end_year_seq"],
                            row["num_years_passed_seq"], row["num_year_passed_dur"], row["year_delta"],
                            row["total_duration"])
    else:
        logging.warning("%sNo valid comparisons available (all values NULL)", INDENT1)
    
    df_temporal.unpersist()


@time_execution
def check_split_point(_spark: SparkSession, df, sample_count: int) -> None:
    """
    Check 3: Split point checks
    For rows where first_token_index_from_2023 is available (not null):
    - Check if token at split index in seq is <NY>
    - Check if first token of event_seq_from_2023 is <NY>
    - Check if last token of seq_thru_2022 is <ATT-*>
    - Check if DOBYR + AGE + count(<NY> in seq_thru_2022) = 2022
    """
    logging.info("********** CHECK 3: Split Point Checks **********")
    
    # Filter to rows with first_token_index_from_2023 available
    df_with_split = df.filter(F.col("first_token_index_from_2023").isNotNull())
    split_count = df_with_split.count()
    
    logging.info("%sRows with first_token_index_from_2023 available: %s (out of %s sampled)", 
                 INDENT1, f"{split_count:,}", f"{sample_count:,}")
    
    if split_count == 0:
        logging.warning("%sNo rows with first_token_index_from_2023 available, skipping split point checks", INDENT1)
        return
    
    # Register UDFs
    get_token_at_index_udf = F.udf(get_token_at_index, StringType())
    get_first_token_udf = F.udf(get_first_token, StringType())
    get_last_token_udf = F.udf(get_last_token, StringType())
    is_ny_token_udf = F.udf(is_ny_token, BooleanType())
    is_att_token_udf = F.udf(is_att_token, BooleanType())
    extract_dobyr_udf = F.udf(lambda seq: extract_token_value(seq, r'<DOBYR-(\d+)>'), IntegerType())
    extract_age_udf = F.udf(lambda seq: extract_token_value(seq, r'<AGE-(\d+)>'), IntegerType())
    count_ny_udf = F.udf(lambda seq: count_token_occurrences(seq, r'<NY>'), IntegerType())
    get_tokens_around_index_udf = F.udf(lambda seq, idx: get_tokens_around_index(seq, idx, window=5), StringType())
    
    # Add check columns
    df_split_checks = df_with_split \
        .withColumn("token_at_split_index", get_token_at_index_udf(F.col("seq"), F.col("first_token_index_from_2023"))) \
        .withColumn("tokens_around_split", get_tokens_around_index_udf(F.col("seq"), F.col("first_token_index_from_2023"))) \
        .withColumn("first_token_from_2023", get_first_token_udf(F.col("event_seq_from_2023"))) \
        .withColumn("last_token_thru_2022", get_last_token_udf(F.col("seq_thru_2022"))) \
        .withColumn("split_token_is_ny", is_ny_token_udf(F.col("token_at_split_index"))) \
        .withColumn("first_2023_is_ny", is_ny_token_udf(F.col("first_token_from_2023"))) \
        .withColumn("last_2022_is_att", is_att_token_udf(F.col("last_token_thru_2022"))) \
        .withColumn("dobyr_thru_2022", extract_dobyr_udf(F.col("seq_thru_2022"))) \
        .withColumn("age_thru_2022", extract_age_udf(F.col("seq_thru_2022"))) \
        .withColumn("ny_count_thru_2022", count_ny_udf(F.col("seq_thru_2022"))) \
        .withColumn("computed_end_year_thru_2022", 
                    F.col("dobyr_thru_2022") + F.col("age_thru_2022") + F.col("ny_count_thru_2022")) \
        .withColumn("end_year_is_2022", F.col("computed_end_year_thru_2022") == 2022)
    
    # Cache for multiple operations
    df_split_checks.cache()
    
    # Diagnostic: Log sample rows showing ±5 tokens around the split index
    logging.info("%sDiagnostic: Sample of ±5 tokens around first_token_index_from_2023 (split token marked with [*...*]):", INDENT1)
    sample_contexts = df_split_checks.select(
        "enrollee_id", "first_token_index_from_2023", "tokens_around_split"
    ).limit(10).collect()
    for row in sample_contexts:
        logging.info("%senrollee_id=%s, index=%s, context: %s", 
                    INDENT2, row["enrollee_id"], row["first_token_index_from_2023"], row["tokens_around_split"])
    
    # Calculate check statistics
    check_stats = df_split_checks.agg(
        F.count("*").alias("total"),
        F.sum(F.when(F.col("split_token_is_ny"), 1).otherwise(0)).alias("split_token_ny_count"),
        F.sum(F.when(F.col("first_2023_is_ny"), 1).otherwise(0)).alias("first_2023_ny_count"),
        F.sum(F.when(F.col("last_2022_is_att"), 1).otherwise(0)).alias("last_2022_att_count"),
        F.sum(F.when(F.col("end_year_is_2022"), 1).otherwise(0)).alias("end_year_2022_count"),
        F.sum(F.when(F.col("computed_end_year_thru_2022").isNotNull(), 1).otherwise(0)).alias("valid_year_computation_count")
    ).collect()[0]
    
    total = check_stats["total"]
    
    # Check 3.1: Token at split index is <NY>
    split_ny_rate = (check_stats["split_token_ny_count"] / total) * 100
    split_not_ny_rate = 100 - split_ny_rate
    logging.info("%sCheck 3.1: Token at split index (first_token_index_from_2023) is <NY>:", INDENT1)
    logging.info("%sToken is <NY>: %s (%.2f%%)", INDENT2, f"{check_stats['split_token_ny_count']:,}", split_ny_rate)
    logging.info("%sToken is NOT <NY>: %s (%.2f%%)", INDENT2, f"{total - check_stats['split_token_ny_count']:,}", split_not_ny_rate)
    
    if split_not_ny_rate > 0:
        logging.warning("%sWARNING: %.2f%% of split tokens are not <NY>!", INDENT3, split_not_ny_rate)
        # Show examples
        not_ny_examples = df_split_checks.filter(~F.col("split_token_is_ny")) \
            .select("enrollee_id", "first_token_index_from_2023", "token_at_split_index") \
            .limit(5).collect()
        for row in not_ny_examples:
            logging.info("%sExample: enrollee_id=%s, index=%s, token=%s", 
                        INDENT3, row["enrollee_id"], row["first_token_index_from_2023"], row["token_at_split_index"])
    
    # Check 3.2: First token of event_seq_from_2023 is <NY>
    first_2023_ny_rate = (check_stats["first_2023_ny_count"] / total) * 100
    first_2023_not_ny_rate = 100 - first_2023_ny_rate
    logging.info("%sCheck 3.2: First token of event_seq_from_2023 is <NY>:", INDENT1)
    logging.info("%sFirst token is <NY>: %s (%.2f%%)", INDENT2, f"{check_stats['first_2023_ny_count']:,}", first_2023_ny_rate)
    logging.info("%sFirst token is NOT <NY>: %s (%.2f%%)", INDENT2, f"{total - check_stats['first_2023_ny_count']:,}", first_2023_not_ny_rate)
    
    if first_2023_not_ny_rate > 0:
        logging.warning("%sWARNING: %.2f%% of event_seq_from_2023 do not start with <NY>!", INDENT3, first_2023_not_ny_rate)
        not_ny_first_examples = df_split_checks.filter(~F.col("first_2023_is_ny")) \
            .select("enrollee_id", "first_token_from_2023", "event_seq_from_2023") \
            .limit(5).collect()
        for row in not_ny_first_examples:
            logging.info("%sExample: enrollee_id=%s, first_token=%s", 
                        INDENT3, row["enrollee_id"], row["first_token_from_2023"])
    
    # Check 3.3: Last token of seq_thru_2022 is <ATT-*>
    last_2022_att_rate = (check_stats["last_2022_att_count"] / total) * 100
    last_2022_not_att_rate = 100 - last_2022_att_rate
    logging.info("%sCheck 3.3: Last token of seq_thru_2022 is <ATT-*>:", INDENT1)
    logging.info("%sLast token is <ATT-*>: %s (%.2f%%)", INDENT2, f"{check_stats['last_2022_att_count']:,}", last_2022_att_rate)
    logging.info("%sLast token is NOT <ATT-*>: %s (%.2f%%)", INDENT2, f"{total - check_stats['last_2022_att_count']:,}", last_2022_not_att_rate)
    
    if last_2022_not_att_rate > 0:
        logging.warning("%sWARNING: %.2f%% of seq_thru_2022 do not end with <ATT-*>!", INDENT3, last_2022_not_att_rate)
        not_att_last_examples = df_split_checks.filter(~F.col("last_2022_is_att")) \
            .select("enrollee_id", "last_token_thru_2022") \
            .limit(5).collect()
        for row in not_att_last_examples:
            logging.info("%sExample: enrollee_id=%s, last_token=%s", 
                        INDENT3, row["enrollee_id"], row["last_token_thru_2022"])
    
    # Check 3.4: DOBYR + AGE + count(<NY>) = 2022 for seq_thru_2022
    valid_year_count = check_stats["valid_year_computation_count"]
    if valid_year_count > 0:
        end_year_2022_rate = (check_stats["end_year_2022_count"] / valid_year_count) * 100
        end_year_not_2022_rate = 100 - end_year_2022_rate
        logging.info("%sCheck 3.4: DOBYR + AGE + count(<NY>) = 2022 for seq_thru_2022:", INDENT1)
        logging.info("%sValid computations: %s", INDENT2, f"{valid_year_count:,}")
        logging.info("%sEquals 2022: %s (%.2f%%)", INDENT2, f"{check_stats['end_year_2022_count']:,}", end_year_2022_rate)
        logging.info("%sNot equals 2022: %s (%.2f%%)", INDENT2, f"{valid_year_count - check_stats['end_year_2022_count']:,}", end_year_not_2022_rate)
        
        if end_year_not_2022_rate > 0:
            logging.warning("%sWARNING: %.2f%% of seq_thru_2022 do not compute to 2022!", INDENT3, end_year_not_2022_rate)
            # Show distribution of computed years
            year_dist = df_split_checks.filter(~F.col("end_year_is_2022") & F.col("computed_end_year_thru_2022").isNotNull()) \
                .groupBy("computed_end_year_thru_2022") \
                .count() \
                .orderBy(F.col("count").desc()) \
                .limit(10).collect()
            logging.info("%sDistribution of computed end years (when not 2022):", INDENT3)
            for row in year_dist:
                logging.info("%sYear %s: %s cases", INDENT4, row["computed_end_year_thru_2022"], row["count"])
            
            # Show examples
            not_2022_examples = df_split_checks.filter(~F.col("end_year_is_2022") & F.col("computed_end_year_thru_2022").isNotNull()) \
                .select("enrollee_id", "dobyr_thru_2022", "age_thru_2022", "ny_count_thru_2022", "computed_end_year_thru_2022") \
                .limit(5).collect()
            for row in not_2022_examples:
                logging.info("%sExample: enrollee_id=%s, DOBYR=%s, AGE=%s, NY_count=%s, computed=%s", 
                            INDENT3, row["enrollee_id"], row["dobyr_thru_2022"], row["age_thru_2022"],
                            row["ny_count_thru_2022"], row["computed_end_year_thru_2022"])
    else:
        logging.warning("%sNo valid year computations available for seq_thru_2022", INDENT2)
    
    df_split_checks.unpersist()


@time_execution
def load_and_sample_trajectory(spark: SparkSession, processed_data_folder: str, sample_size: int, seed: int):
    """Load trajectory data and sample for sanity checks"""
    trajectory_path = os.path.join(processed_data_folder, "trajectory")
    logging.info("%sLoading trajectory from: %s", INDENT1, trajectory_path)
    
    trajectory_df = spark.read.parquet(trajectory_path)
    total_count = trajectory_df.count()
    logging.info("%sTotal trajectories: %s", INDENT1, f"{total_count:,}")
    
    # Calculate sample fraction
    if sample_size >= total_count:
        logging.info("%sSample size (%s) >= total count, using all data", INDENT1, f"{sample_size:,}")
        sampled_df = trajectory_df
        sample_count = total_count
    else:
        sample_fraction = sample_size / total_count
        logging.info("%sSampling %s rows (fraction: %.4f)", INDENT1, f"{sample_size:,}", sample_fraction)
        sampled_df = trajectory_df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        sample_count = sampled_df.count()
        logging.info("%sActual sample count: %s", INDENT1, f"{sample_count:,}")
    
    # Cache sampled data for multiple checks
    sampled_df.cache()
    
    return sampled_df, sample_count


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S5 SANITY CHECK STARTING...")
    
    # Parse arguments
    args = parse_args()
    
    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    
    # Log configuration
    logging.info("\nConfiguration:")
    logging.info("%sProcessed data folder: %s", INDENT1, processed_data_folder)
    logging.info("%sSample size: %s", INDENT1, f"{args.sample_size:,}")
    logging.info("%sRandom seed: %d", INDENT1, args.seed)
    
    # Create Spark session
    spark = SparkSession.builder.appName('ReClaim_Step5_SanityCheck').getOrCreate()
    logging.info("Spark session created: %s", spark.sparkContext.appName)
    
    # Load and sample trajectory data
    logging.info("********** Loading and Sampling Trajectory Data **********")
    sampled_df, sample_count = load_and_sample_trajectory(
        spark, processed_data_folder, args.sample_size, args.seed
    )
    
    # Run sanity checks
    check_missing_tokens(spark, sampled_df, sample_count)
    check_temporal_information(spark, sampled_df, sample_count)
    check_split_point(spark, sampled_df, sample_count)
    
    # Clean up
    sampled_df.unpersist()
    spark.stop()
    
    logging.info("Log file: %s", log_file)
    logging.info("S5 SANITY CHECK COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
