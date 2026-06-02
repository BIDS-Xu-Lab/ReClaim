#!/usr/bin/env python3
"""
Reclaim Data processing - Debug script for ATT token issues
Diagnoses why ATT tokens might exceed 12 months
"""

import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os
import argparse

from utils import setup_logging

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Debug ATT token issues in MarketScan data')
    parser.add_argument("--input_folder_path", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"), "v6", "marketscan_val"),
                       help='Base directory containing MarketScan processed data')            
    return parser.parse_args()


def check_att_distribution(spark: SparkSession, processed_data_folder: str) -> None:
    """Check the distribution of ATT values"""
    logging.info("=== Checking ATT Distribution ===")
    
    event_seq_path = os.path.join(processed_data_folder, "event_seq_per_timestamp")
    if not os.path.exists(event_seq_path):
        logging.warning("%sevent_seq_per_timestamp not found at %s", INDENT1, event_seq_path)
        return
    
    event_seq = spark.read.parquet(event_seq_path)
    
    # Get ATT distribution
    att_dist = event_seq.filter(F.col("time_to_next_timestamp").isNotNull()) \
        .groupBy("time_to_next_timestamp") \
        .count() \
        .orderBy(F.desc("time_to_next_timestamp"))
    
    logging.info("%sATT value distribution (sorted by value descending):", INDENT1)
    att_dist.show(30, truncate=False)
    
    # Count problematic ATT values
    problematic_count = event_seq.filter(F.col("time_to_next_timestamp") > 12).count()
    total_count = event_seq.filter(F.col("time_to_next_timestamp").isNotNull()).count()
    
    logging.info("%sTotal ATT tokens: %s", INDENT1, f"{total_count:,}")
    logging.info("%sATT > 12 count: %s", INDENT1, f"{problematic_count:,}")
    if total_count > 0:
        logging.info("%sATT > 12 percentage: %.4f%%", INDENT1, (problematic_count / total_count) * 100)
    
    return problematic_count > 0


def find_problematic_cases(spark: SparkSession, processed_data_folder: str) -> list:
    """Find specific cases where ATT > 12"""
    logging.info("=== Finding Cases with ATT > 12 ===")
    
    event_seq_path = os.path.join(processed_data_folder, "event_seq_per_timestamp")
    event_seq = spark.read.parquet(event_seq_path)
    
    problematic = event_seq.filter(F.col("time_to_next_timestamp") > 12) \
        .select("enrollee_id", "timestamp", "event_type", "time_to_next_timestamp", "next_timestamp") \
        .orderBy(F.desc("time_to_next_timestamp"))
    
    logging.info("%sSample of problematic cases (ATT > 12):", INDENT1)
    problematic.show(20, truncate=False)
    
    # Get list of problematic enrollee IDs
    problematic_enrollees = problematic.select("enrollee_id").distinct().collect()
    return [row.enrollee_id for row in problematic_enrollees[:10]]  # Return up to 10 for detailed analysis


def analyze_enrollee(spark: SparkSession, processed_data_folder: str, enrollee_id: str) -> None:
    """Analyze a specific enrollee to understand why ATT > 12"""
    logging.info("=== Analyzing enrollee: %s ===", enrollee_id)
    
    # Check anchor tokens
    anchor_path = os.path.join(processed_data_folder, "all_anchor_tokens")
    if os.path.exists(anchor_path):
        anchor_tokens = spark.read.parquet(anchor_path)
        enrollee_anchors = anchor_tokens.filter(F.col("enrollee_id") == enrollee_id) \
            .select("enrollee_id", "timestamp", "category", "token") \
            .orderBy("timestamp")
        
        logging.info("%sAnchor tokens (AGE/NY) for this enrollee:", INDENT1)
        enrollee_anchors.show(50, truncate=False)
        
        anchor_count = enrollee_anchors.count()
        logging.info("%sTotal anchor tokens: %d", INDENT1, anchor_count)
    
    # Check all events for this enrollee
    enrollment_path = os.path.join(processed_data_folder, "all_enrollment_tokens")
    claims_path = os.path.join(processed_data_folder, "all_claim_tokens")
    
    all_timestamps = None
    
    if os.path.exists(enrollment_path):
        enrollment = spark.read.parquet(enrollment_path)
        enroll_ts = enrollment.filter(F.col("enrollee_id") == enrollee_id) \
            .select("enrollee_id", "timestamp", F.lit("enrollment").alias("source"), "category") \
            .distinct()
        all_timestamps = enroll_ts
    
    if os.path.exists(claims_path):
        claims = spark.read.parquet(claims_path)
        claims_ts = claims.filter(F.col("enrollee_id") == enrollee_id) \
            .select("enrollee_id", "timestamp", F.lit("claims").alias("source"), "category") \
            .distinct()
        if all_timestamps is not None:
            all_timestamps = all_timestamps.union(claims_ts)
        else:
            all_timestamps = claims_ts
    
    if os.path.exists(anchor_path):
        anchor_tokens = spark.read.parquet(anchor_path)
        anchor_ts = anchor_tokens.filter(F.col("enrollee_id") == enrollee_id) \
            .select("enrollee_id", "timestamp", F.lit("anchor").alias("source"), "category") \
            .distinct()
        if all_timestamps is not None:
            all_timestamps = all_timestamps.union(anchor_ts)
    
    if all_timestamps is not None:
        logging.info("%sAll timestamps for this enrollee:", INDENT1)
        all_timestamps.orderBy("timestamp", "source").show(100, truncate=False)
    
    # Check duration
    duration_path = os.path.join(processed_data_folder, "duration")
    if os.path.exists(duration_path):
        duration = spark.read.parquet(duration_path)
        enrollee_duration = duration.filter(F.col("enrollee_id") == enrollee_id)
        logging.info("%sDuration info for this enrollee:", INDENT1)
        enrollee_duration.show(truncate=False)
        
        dur_data = enrollee_duration.collect()
        if dur_data:
            first_year = dur_data[0].first_event_year
            last_year = dur_data[0].last_event_year
            if first_year and last_year:
                expected_ny_years = list(range(first_year + 1, last_year + 1))
                logging.info("%sExpected NY tokens at years: %s", INDENT1, expected_ny_years)


def check_missing_anchor_tokens(spark: SparkSession, processed_data_folder: str) -> None:
    """Check if any enrollees with events are missing anchor tokens"""
    logging.info("=== Checking for Missing Anchor Tokens ===")
    
    enrollment_path = os.path.join(processed_data_folder, "all_enrollment_tokens")
    claims_path = os.path.join(processed_data_folder, "all_claim_tokens")
    anchor_path = os.path.join(processed_data_folder, "all_anchor_tokens")
    
    # Get all enrollees with events
    event_enrollees = None
    if os.path.exists(enrollment_path):
        enrollment = spark.read.parquet(enrollment_path)
        event_enrollees = enrollment.select("enrollee_id").distinct()
    
    if os.path.exists(claims_path):
        claims = spark.read.parquet(claims_path)
        claims_enrollees = claims.select("enrollee_id").distinct()
        if event_enrollees is not None:
            event_enrollees = event_enrollees.union(claims_enrollees).distinct()
        else:
            event_enrollees = claims_enrollees
    
    if event_enrollees is None:
        logging.warning("%sNo event tokens found", INDENT1)
        return
    
    event_count = event_enrollees.count()
    logging.info("%sTotal enrollees with events: %s", INDENT1, f"{event_count:,}")
    
    # Get all enrollees with anchor tokens
    if os.path.exists(anchor_path):
        anchor_tokens = spark.read.parquet(anchor_path)
        anchor_enrollees = anchor_tokens.select("enrollee_id").distinct()
        anchor_count = anchor_enrollees.count()
        logging.info("%sTotal enrollees with anchor tokens: %s", INDENT1, f"{anchor_count:,}")
        
        # Find missing
        missing_anchor = event_enrollees.join(anchor_enrollees, "enrollee_id", "left_anti")
        missing_count = missing_anchor.count()
        logging.info("%sEnrollees with events but NO anchor tokens: %s", INDENT1, f"{missing_count:,}")
        
        if missing_count > 0:
            logging.info("%sSample of enrollees missing anchor tokens:", INDENT1)
            missing_anchor.show(10, truncate=False)
    else:
        logging.warning("%sAnchor tokens file not found at %s", INDENT1, anchor_path)


def check_missing_dobyr(spark: SparkSession, processed_data_folder: str) -> None:
    """Check if any enrollees in duration are missing DOBYR tokens"""
    logging.info("=== Checking for Missing DOBYR Tokens ===")
    
    demo_path = os.path.join(processed_data_folder, "all_demographic_tokens")
    duration_path = os.path.join(processed_data_folder, "duration")
    
    if not os.path.exists(demo_path) or not os.path.exists(duration_path):
        logging.warning("%sRequired files not found", INDENT1)
        return
    
    demo = spark.read.parquet(demo_path)
    duration = spark.read.parquet(duration_path)
    
    # Enrollees with DOBYR
    dobyr_enrollees = demo.filter(F.col("category") == "DOBYR").select("enrollee_id").distinct()
    dobyr_count = dobyr_enrollees.count()
    logging.info("%sEnrollees with DOBYR token: %s", INDENT1, f"{dobyr_count:,}")
    
    # Enrollees in duration
    duration_enrollees = duration.select("enrollee_id").distinct()
    duration_count = duration_enrollees.count()
    logging.info("%sEnrollees in duration table: %s", INDENT1, f"{duration_count:,}")
    
    # Enrollees in duration but missing DOBYR
    missing_dobyr = duration_enrollees.join(dobyr_enrollees, "enrollee_id", "left_anti")
    missing_count = missing_dobyr.count()
    logging.info("%sEnrollees with events but NO DOBYR token: %s", INDENT1, f"{missing_count:,}")
    
    if missing_count > 0:
        logging.warning("%s*** BUG FOUND: %d enrollees have events but no DOBYR token! ***", INDENT1, missing_count)
        logging.info("%sThese enrollees will NOT have anchor tokens (AGE/NY), causing ATT > 12!", INDENT1)
        logging.info("%sSample:", INDENT1)
        missing_dobyr.show(10, truncate=False)


def check_multi_year_enrollees(spark: SparkSession, processed_data_folder: str) -> None:
    """Check enrollees spanning multiple years and their NY tokens"""
    logging.info("=== Checking Multi-Year Enrollees ===")
    
    duration_path = os.path.join(processed_data_folder, "duration")
    anchor_path = os.path.join(processed_data_folder, "all_anchor_tokens")
    
    if not os.path.exists(duration_path):
        logging.warning("%sDuration file not found", INDENT1)
        return
    
    duration = spark.read.parquet(duration_path)
    
    # Multi-year enrollees
    multi_year = duration.filter(F.col("first_event_year") < F.col("last_event_year"))
    multi_year_count = multi_year.count()
    logging.info("%sEnrollees spanning multiple years: %s", INDENT1, f"{multi_year_count:,}")
    
    if os.path.exists(anchor_path):
        anchor_tokens = spark.read.parquet(anchor_path)
        
        # Count NY tokens per enrollee
        ny_tokens = anchor_tokens.filter(F.col("category") == "NY")
        ny_per_enrollee = ny_tokens.groupBy("enrollee_id").count().withColumnRenamed("count", "ny_count")
        
        # Join with multi-year enrollees
        multi_year_with_ny = multi_year.join(ny_per_enrollee, "enrollee_id", "left") \
            .withColumn("expected_ny_count", F.col("last_event_year") - F.col("first_event_year")) \
            .withColumn("ny_count", F.coalesce(F.col("ny_count"), F.lit(0)))
        
        # Find mismatches
        mismatches = multi_year_with_ny.filter(F.col("ny_count") != F.col("expected_ny_count"))
        mismatch_count = mismatches.count()
        
        logging.info("%sMulti-year enrollees with NY token count mismatch: %s", INDENT1, f"{mismatch_count:,}")
        if mismatch_count > 0:
            logging.info("%sSample of mismatches:", INDENT1)
            mismatches.select("enrollee_id", "first_event_year", "last_event_year", 
                            "expected_ny_count", "ny_count").show(20, truncate=False)


def check_att_tokens_saved(spark: SparkSession, processed_data_folder: str) -> None:
    """Check the saved ATT tokens for out-of-range values"""
    logging.info("=== Checking Saved ATT Tokens ===")
    
    att_path = os.path.join(processed_data_folder, "all_att_tokens")
    if not os.path.exists(att_path):
        logging.warning("%sATT tokens file not found at %s", INDENT1, att_path)
        return
    
    att_tokens = spark.read.parquet(att_path)
    
    # Expected ATT values: <ATT-0> through <ATT-12> and empty string
    valid_att = [f"<ATT-{i}>" for i in range(13)] + [""]
    
    # Find invalid ATT values
    invalid_att = att_tokens.filter(~F.col("ATT").isin(valid_att))
    invalid_count = invalid_att.count()
    
    logging.info("%sInvalid ATT tokens (not in 0-12 range): %s", INDENT1, f"{invalid_count:,}")
    if invalid_count > 0:
        logging.info("%sSample of invalid ATT tokens:", INDENT1)
        invalid_att.groupBy("ATT").count().orderBy(F.desc("count")).show(20, truncate=False)


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S6 DEBUG - ATT TOKEN ANALYSIS STARTING...")
    
    # Parse arguments
    args = parse_args()
    
    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sProcessed data folder path: %s", INDENT1, processed_data_folder)
    
    # Create Spark session
    spark = SparkSession.builder.appName('MarketScan_Step6_Debug').getOrCreate()
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("DIAGNOSTIC 1: ATT Distribution")
    logging.info("=" * 60)
    has_problems = check_att_distribution(spark, processed_data_folder)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("DIAGNOSTIC 2: Check Saved ATT Tokens")
    logging.info("=" * 60)
    check_att_tokens_saved(spark, processed_data_folder)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("DIAGNOSTIC 3: Missing DOBYR Tokens (Root Cause Check)")
    logging.info("=" * 60)
    check_missing_dobyr(spark, processed_data_folder)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("DIAGNOSTIC 4: Missing Anchor Tokens")
    logging.info("=" * 60)
    check_missing_anchor_tokens(spark, processed_data_folder)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("DIAGNOSTIC 5: Multi-Year Enrollees NY Token Check")
    logging.info("=" * 60)
    check_multi_year_enrollees(spark, processed_data_folder)
    
    if has_problems:
        logging.info("")
        logging.info("=" * 60)
        logging.info("DIAGNOSTIC 6: Find Problematic Cases")
        logging.info("=" * 60)
        problematic_enrollees = find_problematic_cases(spark, processed_data_folder)
        
        if problematic_enrollees:
            logging.info("")
            logging.info("=" * 60)
            logging.info("DIAGNOSTIC 7: Detailed Analysis of Problematic Enrollees")
            logging.info("=" * 60)
            for i, enrollee_id in enumerate(problematic_enrollees[:3]):  # Analyze first 3
                logging.info("")
                logging.info("-" * 40)
                logging.info("Analyzing problematic enrollee %d of %d", i + 1, min(3, len(problematic_enrollees)))
                logging.info("-" * 40)
                analyze_enrollee(spark, processed_data_folder, enrollee_id)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("If ATT > 12 cases were found, check the diagnostics above.")
    logging.info("Most likely causes:")
    logging.info("%s1. Enrollees missing DOBYR tokens -> no anchor tokens generated", INDENT1)
    logging.info("%s2. Enrollees missing from demographic tokens but have events", INDENT1)
    logging.info("%s3. NY tokens not being generated for multi-year enrollees", INDENT1)
    logging.info("")
    
    # Stop Spark session
    spark.stop()
    logging.info("S6 DEBUG COMPLETED...")
    logging.info("Log file: %s", log_file)


if __name__ == "__main__":
    main()



