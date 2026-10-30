#!/usr/bin/env python3
"""
Reclaim Data processing
"""

import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os
import argparse
from pyspark.sql.window import Window


from utils import setup_logging, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "
INDENT4 = "        "
INDENT5 = "          "

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create demographic sequences from MarketScan data')
    parser.add_argument("--input_folder_path", type=str, required=True,
                       help='Base directory containing MarketScan processed data for downstream tasks')            
    parser.add_argument("--training_cutoff_year", type=int, required=True,
                       help='Split point: events with year >= this go into the from-cutoff portion (held out of seq_thru_cutoff)')
    return parser.parse_args()

@time_execution
def construct_demo_seq(spark: SparkSession, processed_data_folder: str) -> str:
    """all_demographic_tokens: DataFrame, 
    Construct demographic seq: <SEX> <DOBYR>
    """

    logging.info("%sConstruct demographic seq", INDENT1)
    demo_tokens = spark.read.parquet(os.path.join(processed_data_folder, "all_demographic_tokens"))

    demo_seq = demo_tokens \
        .groupBy("enrollee_id") \
        .agg(F.concat_ws(" ",
            F.array_sort(F.collect_list(F.struct(F.col("sub_order"), F.col("token")))).getItem("token")
        ).alias("demo_seq")) \
        .orderBy("enrollee_id")

    # save as demo_seq as required intermediate data, then re-read to break Spark lineage
    demo_seq_file_name = "demo_seq"
    demo_seq.write.mode("overwrite").parquet(os.path.join(processed_data_folder, demo_seq_file_name))
    logging.info("%sSaved demographic seq to: %s", INDENT1, os.path.join(processed_data_folder, demo_seq_file_name))

    return demo_seq_file_name

@time_execution
def construct_event_seq(
    spark: SparkSession,
    processed_data_folder: str,
    training_cutoff_year: int) -> str:
    """
    Construct event seq that follows demo seq: <AGE> <NY> <ERLST> <PLANTYP> <CAP> <EGEOLOC> <VT> <DX> <PROC> <RX> <DS> <LS> <COST> <ERLED> ... 
    """
    logging.info("%sConstruct event seq", INDENT1)
    
    # Read enrollment and claim tokens separately and union them here
    logging.info("%sReading and unioning enrollment and claim tokens...", INDENT2)
    all_enrollment_tokens = spark.read.parquet(os.path.join(processed_data_folder, "all_enrollment_tokens"))
    all_claim_tokens = spark.read.parquet(os.path.join(processed_data_folder, "all_claim_tokens"))
    all_event_tokens = all_enrollment_tokens.unionByName(all_claim_tokens)
    
    all_anchor_tokens = spark.read.parquet(os.path.join(processed_data_folder, "all_anchor_tokens"))

    logging.info("%sSort event tokens", INDENT1) 
    # Union demographic tokens, anchor tokens, and event tokens
    # Sort by original_date within timestamp to maintain date-level ordering within each week
    all_temporal_tokens_sorted = all_event_tokens \
        .unionByName(all_anchor_tokens) \
        .withColumn("category_priority",
            F.when(F.col("category") == "AGE", F.lit(0)) 
            .when(F.col("category") == "NY", F.lit(1))
            .when(F.col("category") == "ERLST", F.lit(2)) 
            .when(F.col("category") == "PLANTYP", F.lit(3))
            .when(F.col("category") == "CAP", F.lit(4))
            .when(F.col("category") == "EGEOLOC", F.lit(5))
            .when(F.col("category") == "VT", F.lit(6))
            .when(F.col("category") == "DX", F.lit(7))
            .when(F.col("category") == "PROC", F.lit(8))
            .when(F.col("category") == "RX", F.lit(9))
            .when(F.col("category") == "DS", F.lit(10))
            .when(F.col("category") == "LS", F.lit(11))
            .when(F.col("category") == "COST", F.lit(12))
            .when(F.col("category") == "ERLED", F.lit(13))
            .otherwise(F.lit(14))) \
        .withColumn("token_order_per_timestamp", 
            F.row_number().over(
                Window.partitionBy("enrollee_id", "timestamp").orderBy("payer_order", "event_type", "category_priority", "sub_order", "original_date", "original_token", "comb_order")))

    # ---- STAGE 1: Materialize sorted tokens to disk to break lineage ----
    # This isolates the heavy Window shuffle (partitionBy enrollee_id, timestamp) from downstream shuffles.
    # Without this, 3 shuffles chain in one DAG causing executors to OOM and MetadataFetchFailedException.
    intermediate_sorted_file = "_intermediate_sorted_tokens"
    logging.info("%sWriting intermediate sorted tokens to break DAG lineage...", INDENT1)
    all_temporal_tokens_sorted.select(
        "enrollee_id", "timestamp", "payer_order", "event_type", "token", "token_order_per_timestamp"
    ).write.mode("overwrite").parquet(os.path.join(processed_data_folder, intermediate_sorted_file))
    logging.info("%sSaved intermediate sorted tokens to: %s", INDENT1, os.path.join(processed_data_folder, intermediate_sorted_file))
    # Re-read from disk for fresh lineage
    all_temporal_tokens_sorted = spark.read.parquet(os.path.join(processed_data_folder, intermediate_sorted_file))
    logging.info("%sRe-loaded intermediate sorted tokens from parquet to start fresh DAG", INDENT1)
    
    # ---- STAGE 2: GroupBy + Window for lead/ATT ----
    logging.info("%sCreate event seq per timestamp", INDENT1)
    # Create window specification for inter-group processing.
    # Order must match the render order used when assembling the trajectory
    # (timestamp, payer_order, event_type) so that the inter-month <ATT-N> attaches to the
    # true last group of the month and <ATT-0> separates intra-month groups.
    window_spec = Window.partitionBy("enrollee_id").orderBy("timestamp", "payer_order", "event_type")

    # Use lead to get time_to_next_timestamp in months (since timestamps are month-based), put ATT after seq_per_timestamp
    # months_between returns a float, we cast to int to get whole months
    enrollee_event_seq_per_timestamp = all_temporal_tokens_sorted \
        .groupBy("enrollee_id", "timestamp", "payer_order", "event_type") \
        .agg(F.concat_ws(" ",
                F.array_sort(F.collect_list(F.struct(F.col("token_order_per_timestamp"), F.col("token")))).getItem("token")
            ).alias("seq_per_timestamp")) \
        .withColumn("time_to_next_timestamp", F.months_between(F.lead("timestamp").over(window_spec), F.col("timestamp")).cast("int")) \
        .withColumn("ATT", 
            F.when(F.col("time_to_next_timestamp").isNull(), F.lit("")) \
            .otherwise(F.concat(F.lit("<ATT-"), F.col("time_to_next_timestamp").cast("string"), F.lit(">")))) \
        .withColumn("seq_per_timestamp_with_ATT", F.concat_ws(" ", F.col("seq_per_timestamp"), F.col("ATT"))) \
        .withColumn("is_from_cutoff", F.year(F.col("timestamp")) >= training_cutoff_year)


    # save as seq_per_timestamp as required intermediate data, then re-read to break Spark lineage
    seq_per_timestamp_file_name = "event_seq_per_timestamp"
    enrollee_event_seq_per_timestamp.write.mode("overwrite").parquet(os.path.join(processed_data_folder, seq_per_timestamp_file_name))
    logging.info("%sSaved event seq per timestamp to: %s", INDENT1, os.path.join(processed_data_folder, seq_per_timestamp_file_name))

    return seq_per_timestamp_file_name


    
@time_execution
def construct_and_save_trajectory(
    spark: SparkSession,
    processed_data_folder: str,
    demo_seq_file_name: str,
    seq_per_timestamp_file_name: str
) -> None:
    """
    Construct trajectory
    """

    # Re-read from parquet to get fresh lineage
    enrollee_event_seq_per_timestamp = spark.read.parquet(os.path.join(processed_data_folder, seq_per_timestamp_file_name))
    logging.info("%sRe-loaded enrollee event sequence per timestamp from parquet to break lineage", INDENT2)

    
    att_tokens = enrollee_event_seq_per_timestamp \
        .withColumn("category", F.lit("ATT")) \
        .withColumn("token", F.col("ATT")) \
        .filter((F.col("time_to_next_timestamp") >= 0) & (F.col("time_to_next_timestamp") <= 12)) \
        .filter(F.col("token") != "<ATT->") \
        .select("category", "token")
        
    att_tokens_save_path = os.path.join(processed_data_folder, "all_att_tokens")
    att_tokens.write.mode("overwrite").parquet(att_tokens_save_path)
    logging.info("%sSaved all ATT tokens for later finalizing vocabulary to: %s", INDENT1, att_tokens_save_path)
    
    logging.info("%sGrouping by enrollee_id and is_from_cutoff to create separate thru-cutoff and from-cutoff sequences", INDENT2)
    patient_event_seq_by_period = enrollee_event_seq_per_timestamp \
        .groupBy("enrollee_id", "is_from_cutoff") \
        .agg(
            F.concat_ws(" ", 
                F.array_sort(F.collect_list(F.struct(F.col("timestamp"), F.col("payer_order"), F.col("event_type"), F.col("seq_per_timestamp_with_ATT")))).getItem("seq_per_timestamp_with_ATT")
            ).alias("period_seq")) \
        .withColumn("period_seq", F.trim(F.col("period_seq"))) \
        .withColumn("token_count_in_period", F.size(F.regexp_extract_all(F.col("period_seq"), F.lit("<[^>]+>"), 0)))
    
    logging.info("%sPivoting to get thru-cutoff and from-cutoff sequences as separate columns", INDENT2)
    patient_event_seq = patient_event_seq_by_period \
        .groupBy("enrollee_id") \
        .agg(
            F.max(F.when(~F.col("is_from_cutoff"), F.col("period_seq"))).alias("event_seq_thru_cutoff"),
            F.max(F.when(F.col("is_from_cutoff"), F.col("period_seq"))).alias("event_seq_from_cutoff"),
            F.max(F.when(~F.col("is_from_cutoff"), F.col("token_count_in_period"))).alias("event_token_count_thru_cutoff"),
            F.max(F.when(F.col("is_from_cutoff"), F.col("token_count_in_period"))).alias("event_token_count_from_cutoff")) \
        .withColumn("event_token_count_thru_cutoff", F.coalesce(F.col("event_token_count_thru_cutoff"), F.lit(0))) \
        .withColumn("event_token_count_from_cutoff", F.coalesce(F.col("event_token_count_from_cutoff"), F.lit(0)))
        # Note: event_seq_thru_cutoff and event_seq_from_cutoff are left as NULL when missing - concat_ws will skip NULLs 
    

    demo_seq = spark.read.parquet(os.path.join(processed_data_folder, demo_seq_file_name))
    duration = spark.read.parquet(os.path.join(processed_data_folder, "duration")).select("enrollee_id", "total_duration").distinct()

    trajectory = patient_event_seq \
        .join(demo_seq, on="enrollee_id", how="inner") \
        .join(duration, on="enrollee_id", how="left") \
        .withColumn("seq_thru_cutoff", F.concat_ws(" ",
            F.lit("<sos>"), 
            F.col("demo_seq"), 
            F.col("event_seq_thru_cutoff"),  # NULL values are skipped by concat_ws
            F.lit("<eos>"))) \
        .withColumn("seq", F.concat_ws(" ", 
            F.lit("<sos>"), 
            F.col("demo_seq"), 
            F.col("event_seq_thru_cutoff"),  # NULL values are skipped by concat_ws
            F.col("event_seq_from_cutoff"),  # NULL values are skipped by concat_ws
            F.lit("<eos>"))) \
        .withColumn("total_token_count", F.col("event_token_count_thru_cutoff") + F.col("event_token_count_from_cutoff") + 4) \
        .withColumn("token_count_thru_cutoff", F.col("event_token_count_thru_cutoff") + 3) \
        .withColumn("first_token_index_from_cutoff", 
            F.when(F.col("event_token_count_from_cutoff") == 0, F.lit(None))
            .when(F.col("event_token_count_thru_cutoff") == 0, F.lit(None))  # No historical context to predict from
            .otherwise(F.col("token_count_thru_cutoff")))  # 0-based: NY is always at this position  

    trajectory.cache()
    total_trajectories = trajectory.count()
    logging.info("%sConstructed %s trajectories:", INDENT2, f"{total_trajectories:,}")

    # Select final columns. No row-level filtering needed here: missing-DOBYR, missing-AGE,
    # age range, and enrollees with no pre-cutoff history are all already excluded in s0.
    trajectory_final = trajectory.select(
        "enrollee_id",
        "seq_thru_cutoff",
        "event_seq_from_cutoff",
        "seq",
        "first_token_index_from_cutoff",
        "total_token_count",
        "token_count_thru_cutoff",
        "event_token_count_from_cutoff",
        "total_duration"
    )

    logging.info("%sSample whole trajectories:", INDENT2)
    trajectory_final.select("enrollee_id", "first_token_index_from_cutoff", "seq").show(100, truncate=4096)
    logging.info("%sSample pretrain sequences:", INDENT2)
    trajectory_final.select("enrollee_id", "seq_thru_cutoff").show(100, truncate=4096)

    trajectory_save_path = os.path.join(processed_data_folder, "trajectory")
    trajectory_final.write.mode("overwrite").parquet(trajectory_save_path)
    logging.info("%sTotal %s trajectories saved to: %s", INDENT1, total_trajectories, trajectory_save_path)


    trajectory.unpersist() # done with trajectory



def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S2 CONSTRUCT SEQUENCES STARTING...")
    # Parse arguments
    args = parse_args()

    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    os.makedirs(processed_data_folder, exist_ok=True)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sProcessed data folder path: %s", INDENT1, processed_data_folder)
    logging.info("%sTraining cutoff year (events with year >= this are held out of seq_thru_cutoff): %s", INDENT1, args.training_cutoff_year)

    # Create Spark session
    spark = SparkSession.builder.appName('MarketScan_Step2_Construct_Sequences').getOrCreate()

    logging.info("********** 1 Construct demographic sequences **********")
    demo_seq_file_name = construct_demo_seq(spark, processed_data_folder)

    logging.info("********** 2 Construct event sequences per timestamp **********")
    seq_per_timestamp_file_name = construct_event_seq(spark, processed_data_folder, args.training_cutoff_year)

    logging.info("********** 3 Construct trajectory **********")
    construct_and_save_trajectory(spark, processed_data_folder, demo_seq_file_name, seq_per_timestamp_file_name)


    # Stop Spark session
    spark.stop()
    logging.info("S2 CONSTRUCT SEQUENCES COMPLETED SUCCESSFULLY...")
    logging.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
