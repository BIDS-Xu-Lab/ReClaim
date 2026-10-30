#!/usr/bin/env python3
"""
Reclaim Data Sanity Check
Performs various sanity checks on trajectory data from S2.

Checks performed:
1. Missing token check: <DOBYR>, <SEX>, <AGE>
2. Temporal information check: start_year, end_year, num_years_passed comparison
3. Split point checks (for rows with first_token_index_from_cutoff available):
   - Token at split index is <NY>
   - First token of event_seq_from_cutoff is <NY>
   - Last token (before trailing <eos>) of seq_thru_cutoff is <ATT-*>
   - DOBYR + AGE + count(<NY>) = training_cutoff_year - 1 for seq_thru_cutoff
"""

import logging
import re
from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import BooleanType, IntegerType, StringType
import pyspark.sql.functions as F
from functools import reduce
import os
import sys
from glob import glob
import argparse
from pyspark.sql.window import Window

# Fix PySpark Python version mismatch - ensure workers use the same Python as driver
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from utils import setup_logging, time_execution, spark_log_contingency_table

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "
INDENT4 = "        "
INDENT5 = "          "


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run sanity checks on trajectory data')
    parser.add_argument("--external_val_data_dir", type=str,
                       default='external_val_data',
                       help='Directory containing external validation data')
    parser.add_argument("--training_cutoff_year", type=int, required=True,
                       help='Split cutoff year used in s2; seq_thru_cutoff is expected to end at training_cutoff_year - 1')
    return parser.parse_args()

def create_spark_session(app_name: str) -> SparkSession:
    """
    Create a SparkSession that works in both run modes:

    - Cluster mode (run_ynhh_full.sh): the stage is launched via
      spark-submit against a spark-start cluster, and SPARK_MASTER_URL is exported.
      In that case do NOT set master/memory here - the hardcoded local[*] config would
      override spark-submit's --master and silently fall back to a single node.
      Master, executor sizing, and shuffle partitions all come from spark-submit flags.

    - Standalone mode (run_ynhh_subset.sh, or an interactive session):
      plain `python s*.py` with no SPARK_MASTER_URL -> build a local[*] session
      sized for a single node.
    """
    if os.environ.get("SPARK_MASTER_URL"):
        spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        return spark

    driver_memory = "80g"
    max_result_size = "10g"
    # Shuffle partitions: typically 2-3x number of cores for better parallelism
    shuffle_partitions = "64"

    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", max_result_size) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    return spark

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


def get_last_token_before_eos(seq: str) -> str:
    """Get the last token, skipping a trailing <eos> (seq_thru_cutoff is now a self-contained <sos>...<eos> sequence)"""
    if seq is None or seq.strip() == "":
        return None
    tokens = [t for t in seq.strip().split(" ") if t != ""]
    while tokens and tokens[-1] == "<eos>":
        tokens.pop()
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
                .select("person_id", "start_year_seq", "end_year_seq", "num_years_passed_seq", 
                        "num_year_passed_dur", "year_delta", "total_duration") \
                .limit(10).collect()
            for row in high_delta_examples:
                logging.info("%sperson_id=%s, start=%s, end=%s, ny_count=%s, dur_years=%s, delta=%s, total_dur=%s", 
                            INDENT3, row["person_id"], row["start_year_seq"], row["end_year_seq"],
                            row["num_years_passed_seq"], row["num_year_passed_dur"], row["year_delta"],
                            row["total_duration"])
    else:
        logging.warning("%sNo valid comparisons available (all values NULL)", INDENT1)
    
    df_temporal.unpersist()


@time_execution
def check_split_point(_spark: SparkSession, df, sample_count: int, training_cutoff_year: int) -> None:
    """
    Check 3: Split point checks
    For rows where first_token_index_from_cutoff is available (not null):
    - Check if token at split index in seq is <NY>
    - Check if first token of event_seq_from_cutoff is <NY>
    - Check if last token (before trailing <eos>) of seq_thru_cutoff is <ATT-*>
    - Check if DOBYR + AGE + count(<NY> in seq_thru_cutoff) = training_cutoff_year - 1
    """
    logging.info("********** CHECK 3: Split Point Checks **********")

    # seq_thru_cutoff covers events through (training_cutoff_year - 1), so that is the expected end year
    expected_end_year = training_cutoff_year - 1

    # Filter to rows with first_token_index_from_cutoff available
    df_with_split = df.filter(F.col("first_token_index_from_cutoff").isNotNull())
    split_count = df_with_split.count()
    
    logging.info("%sRows with first_token_index_from_cutoff available: %s (out of %s sampled)", 
                 INDENT1, f"{split_count:,}", f"{sample_count:,}")
    
    if split_count == 0:
        logging.warning("%sNo rows with first_token_index_from_cutoff available, skipping split point checks", INDENT1)
        return
    
    # Register UDFs
    get_token_at_index_udf = F.udf(get_token_at_index, StringType())
    get_first_token_udf = F.udf(get_first_token, StringType())
    # seq_thru_cutoff is wrapped as <sos>...<eos>, so skip the trailing <eos> to reach the real last token
    get_last_token_udf = F.udf(get_last_token_before_eos, StringType())
    is_ny_token_udf = F.udf(is_ny_token, BooleanType())
    is_att_token_udf = F.udf(is_att_token, BooleanType())
    extract_dobyr_udf = F.udf(lambda seq: extract_token_value(seq, r'<DOBYR-(\d+)>'), IntegerType())
    extract_age_udf = F.udf(lambda seq: extract_token_value(seq, r'<AGE-(\d+)>'), IntegerType())
    count_ny_udf = F.udf(lambda seq: count_token_occurrences(seq, r'<NY>'), IntegerType())
    get_tokens_around_index_udf = F.udf(lambda seq, idx: get_tokens_around_index(seq, idx, window=5), StringType())
    
    # Add check columns
    df_split_checks = df_with_split \
        .withColumn("token_at_split_index", get_token_at_index_udf(F.col("seq"), F.col("first_token_index_from_cutoff"))) \
        .withColumn("tokens_around_split", get_tokens_around_index_udf(F.col("seq"), F.col("first_token_index_from_cutoff"))) \
        .withColumn("first_token_from_cutoff", get_first_token_udf(F.col("event_seq_from_cutoff"))) \
        .withColumn("last_token_thru_cutoff", get_last_token_udf(F.col("seq_thru_cutoff"))) \
        .withColumn("split_token_is_ny", is_ny_token_udf(F.col("token_at_split_index"))) \
        .withColumn("first_from_cutoff_is_ny", is_ny_token_udf(F.col("first_token_from_cutoff"))) \
        .withColumn("last_thru_cutoff_is_att", is_att_token_udf(F.col("last_token_thru_cutoff"))) \
        .withColumn("dobyr_thru_cutoff", extract_dobyr_udf(F.col("seq_thru_cutoff"))) \
        .withColumn("age_thru_cutoff", extract_age_udf(F.col("seq_thru_cutoff"))) \
        .withColumn("ny_count_thru_cutoff", count_ny_udf(F.col("seq_thru_cutoff"))) \
        .withColumn("computed_end_year_thru_cutoff", 
                    F.col("dobyr_thru_cutoff") + F.col("age_thru_cutoff") + F.col("ny_count_thru_cutoff")) \
        .withColumn("end_year_is_expected", F.col("computed_end_year_thru_cutoff") == expected_end_year)
    
    # Cache for multiple operations
    df_split_checks.cache()
    
    # Diagnostic: Log sample rows showing ±5 tokens around the split index
    logging.info("%sDiagnostic: Sample of ±5 tokens around first_token_index_from_cutoff (split token marked with [*...*]):", INDENT1)
    sample_contexts = df_split_checks.select(
        "person_id", "first_token_index_from_cutoff", "tokens_around_split"
    ).limit(10).collect()
    for row in sample_contexts:
        logging.info("%sperson_id=%s, index=%s, context: %s", 
                    INDENT2, row["person_id"], row["first_token_index_from_cutoff"], row["tokens_around_split"])
    
    # Calculate check statistics
    check_stats = df_split_checks.agg(
        F.count("*").alias("total"),
        F.sum(F.when(F.col("split_token_is_ny"), 1).otherwise(0)).alias("split_token_ny_count"),
        F.sum(F.when(F.col("first_from_cutoff_is_ny"), 1).otherwise(0)).alias("first_from_cutoff_ny_count"),
        F.sum(F.when(F.col("last_thru_cutoff_is_att"), 1).otherwise(0)).alias("last_thru_cutoff_att_count"),
        F.sum(F.when(F.col("end_year_is_expected"), 1).otherwise(0)).alias("end_year_expected_count"),
        F.sum(F.when(F.col("computed_end_year_thru_cutoff").isNotNull(), 1).otherwise(0)).alias("valid_year_computation_count")
    ).collect()[0]
    
    total = check_stats["total"]
    
    # Check 3.1: Token at split index is <NY>
    split_ny_rate = (check_stats["split_token_ny_count"] / total) * 100
    split_not_ny_rate = 100 - split_ny_rate
    logging.info("%sCheck 3.1: Token at split index (first_token_index_from_cutoff) is <NY>:", INDENT1)
    logging.info("%sToken is <NY>: %s (%.2f%%)", INDENT2, f"{check_stats['split_token_ny_count']:,}", split_ny_rate)
    logging.info("%sToken is NOT <NY>: %s (%.2f%%)", INDENT2, f"{total - check_stats['split_token_ny_count']:,}", split_not_ny_rate)
    
    if split_not_ny_rate > 0:
        logging.warning("%sWARNING: %.2f%% of split tokens are not <NY>!", INDENT3, split_not_ny_rate)
        # Show examples with ±5 tokens around split index
        not_ny_examples = df_split_checks.filter(~F.col("split_token_is_ny")) \
            .select("person_id", "first_token_index_from_cutoff", "token_at_split_index", "tokens_around_split") \
            .limit(5).collect()
        for row in not_ny_examples:
            logging.info("%sExample: person_id=%s, index=%s, token=%s",
                        INDENT3, row["person_id"], row["first_token_index_from_cutoff"], row["token_at_split_index"])
            logging.info("%s  Context (±5 tokens): %s", INDENT3, row["tokens_around_split"])

    # Check 3.2: First token of event_seq_from_cutoff is <NY>
    first_from_cutoff_ny_rate = (check_stats["first_from_cutoff_ny_count"] / total) * 100
    first_from_cutoff_not_ny_rate = 100 - first_from_cutoff_ny_rate
    logging.info("%sCheck 3.2: First token of event_seq_from_cutoff is <NY>:", INDENT1)
    logging.info("%sFirst token is <NY>: %s (%.2f%%)", INDENT2, f"{check_stats['first_from_cutoff_ny_count']:,}", first_from_cutoff_ny_rate)
    logging.info("%sFirst token is NOT <NY>: %s (%.2f%%)", INDENT2, f"{total - check_stats['first_from_cutoff_ny_count']:,}", first_from_cutoff_not_ny_rate)
    
    if first_from_cutoff_not_ny_rate > 0:
        logging.warning("%sWARNING: %.2f%% of event_seq_from_cutoff do not start with <NY>!", INDENT3, first_from_cutoff_not_ny_rate)
        not_ny_first_examples = df_split_checks.filter(~F.col("first_from_cutoff_is_ny")) \
            .select("person_id", "first_token_from_cutoff", "event_seq_from_cutoff") \
            .limit(5).collect()
        for row in not_ny_first_examples:
            logging.info("%sExample: person_id=%s, first_token=%s", 
                        INDENT3, row["person_id"], row["first_token_from_cutoff"])
    
    # Check 3.3: Last token (before trailing <eos>) of seq_thru_cutoff is <ATT-*>
    last_thru_cutoff_att_rate = (check_stats["last_thru_cutoff_att_count"] / total) * 100
    last_thru_cutoff_not_att_rate = 100 - last_thru_cutoff_att_rate
    logging.info("%sCheck 3.3: Last token (before <eos>) of seq_thru_cutoff is <ATT-*>:", INDENT1)
    logging.info("%sLast token is <ATT-*>: %s (%.2f%%)", INDENT2, f"{check_stats['last_thru_cutoff_att_count']:,}", last_thru_cutoff_att_rate)
    logging.info("%sLast token is NOT <ATT-*>: %s (%.2f%%)", INDENT2, f"{total - check_stats['last_thru_cutoff_att_count']:,}", last_thru_cutoff_not_att_rate)
    
    if last_thru_cutoff_not_att_rate > 0:
        logging.warning("%sWARNING: %.2f%% of seq_thru_cutoff do not end with <ATT-*> (before <eos>)!", INDENT3, last_thru_cutoff_not_att_rate)
        not_att_last_examples = df_split_checks.filter(~F.col("last_thru_cutoff_is_att")) \
            .select("person_id", "last_token_thru_cutoff") \
            .limit(5).collect()
        for row in not_att_last_examples:
            logging.info("%sExample: person_id=%s, last_token=%s", 
                        INDENT3, row["person_id"], row["last_token_thru_cutoff"])
    
    # Check 3.4: DOBYR + AGE + count(<NY>) = training_cutoff_year - 1 for seq_thru_cutoff
    valid_year_count = check_stats["valid_year_computation_count"]
    if valid_year_count > 0:
        end_year_expected_rate = (check_stats["end_year_expected_count"] / valid_year_count) * 100
        end_year_not_expected_rate = 100 - end_year_expected_rate
        logging.info("%sCheck 3.4: DOBYR + AGE + count(<NY>) = %d for seq_thru_cutoff:", INDENT1, expected_end_year)
        logging.info("%sValid computations: %s", INDENT2, f"{valid_year_count:,}")
        logging.info("%sEquals %d: %s (%.2f%%)", INDENT2, expected_end_year, f"{check_stats['end_year_expected_count']:,}", end_year_expected_rate)
        logging.info("%sNot equals %d: %s (%.2f%%)", INDENT2, expected_end_year, f"{valid_year_count - check_stats['end_year_expected_count']:,}", end_year_not_expected_rate)
        
        if end_year_not_expected_rate > 0:
            logging.warning("%sWARNING: %.2f%% of seq_thru_cutoff do not compute to %d!", INDENT3, end_year_not_expected_rate, expected_end_year)
            # Show distribution of computed years
            year_dist = df_split_checks.filter(~F.col("end_year_is_expected") & F.col("computed_end_year_thru_cutoff").isNotNull()) \
                .groupBy("computed_end_year_thru_cutoff") \
                .count() \
                .orderBy(F.col("count").desc()) \
                .limit(10).collect()
            logging.info("%sDistribution of computed end years (when not %d):", INDENT3, expected_end_year)
            for row in year_dist:
                logging.info("%sYear %s: %s cases", INDENT4, row["computed_end_year_thru_cutoff"], row["count"])
            
            # Show examples
            not_expected_examples = df_split_checks.filter(~F.col("end_year_is_expected") & F.col("computed_end_year_thru_cutoff").isNotNull()) \
                .select("person_id", "dobyr_thru_cutoff", "age_thru_cutoff", "ny_count_thru_cutoff", "computed_end_year_thru_cutoff") \
                .limit(5).collect()
            for row in not_expected_examples:
                logging.info("%sExample: person_id=%s, DOBYR=%s, AGE=%s, NY_count=%s, computed=%s", 
                            INDENT3, row["person_id"], row["dobyr_thru_cutoff"], row["age_thru_cutoff"],
                            row["ny_count_thru_cutoff"], row["computed_end_year_thru_cutoff"])
    else:
        logging.warning("%sNo valid year computations available for seq_thru_cutoff", INDENT2)
    
    df_split_checks.unpersist()


@time_execution
def load_trajectory(spark: SparkSession, external_val_data_dir: str):
    """Load trajectory data for sanity checks"""
    trajectory_path = os.path.join(external_val_data_dir, "trajectory")
    logging.info("%sLoading trajectory from: %s", INDENT1, trajectory_path)
    
    trajectory_df = spark.read.parquet(trajectory_path)
    total_count = trajectory_df.count()
    logging.info("%sTotal trajectories: %s", INDENT1, f"{total_count:,}")
    
    # Cache data for multiple checks
    trajectory_df.cache()
    
    return trajectory_df, total_count


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S5 SANITY CHECK STARTING...")
    
    # Parse arguments
    args = parse_args()

    # Log configuration
    logging.info("\nConfiguration:")
    logging.info("%sExternal validation data directory: %s", INDENT1, args.external_val_data_dir)
    logging.info("%sTraining cutoff year: %d", INDENT1, args.training_cutoff_year)
    
    # Create Spark session
    spark = create_spark_session('YNHH_Step5_SanityCheck')
    logging.info("Spark session created: %s", spark.sparkContext.appName)
    
    # Load trajectory data
    logging.info("********** Loading Trajectory Data **********")
    trajectory_df, total_count = load_trajectory(
        spark, args.external_val_data_dir
    )
    
    # Run sanity checks
    check_missing_tokens(spark, trajectory_df, total_count)
    check_temporal_information(spark, trajectory_df, total_count)
    check_split_point(spark, trajectory_df, total_count, args.training_cutoff_year)
    
    # Clean up
    trajectory_df.unpersist()
    spark.stop()
    
    logging.info("Log file: %s", log_file)
    logging.info("S5 SANITY CHECK COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()

