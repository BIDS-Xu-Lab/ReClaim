#!/usr/bin/env python3
"""
Reclaim Data processing for external validation data (EHRSHOT)
"""

import logging
from pyspark.sql import SparkSession, DataFrame, Column
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
    parser = argparse.ArgumentParser(description='Create demographic sequences from MarketScan data')
    parser.add_argument("--external_val_data_dir", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_EHRSHOT_DATA_ROOT", "/path/to/processed_ehrshot_data_root"), "v6", "ehrshot"),
                       help='Directory containing external validation data') 
    return parser.parse_args()



def create_spark_session(app_name: str) -> SparkSession:
    """
    Create SparkSession configured for SBATCH resources:
    - 60 CPUs per task
    - 400GB memory per node
    - Single node setup, local mode
    """
    
    # # Resource allocation: 60 cores, 400GB memory per node
    # # Driver: Reserve ~40GB for driver (10% of total)
    # # Executor: Use remaining cores and memory
    # driver_memory = "4g"
    # executor_memory = "4g"  # Leave some overhead for system
    # max_result_size = "2g"
    # executor_cores = "4"
    # # Shuffle partitions: typically 2-3x number of cores for better parallelism
    # shuffle_partitions = "100"
    
    # spark = SparkSession.builder \
    #     .appName(app_name) \
    #     .master("local[*]") \
    #     .config("spark.driver.memory", driver_memory) \
    #     .config("spark.driver.maxResultSize", max_result_size) \
    #     .config("spark.executor.memory", executor_memory) \
    #     .config("spark.executor.cores", executor_cores) \
    #     .config("spark.sql.shuffle.partitions", shuffle_partitions) \
    #     .config("spark.sql.adaptive.enabled", "true") \
    #     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    #     .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    #     .getOrCreate()


    spark = (SparkSession.builder
        .appName('External_Validation_Data_Builder')
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")  # Increase driver memory
        .config("spark.executor.memory", "4g")  # Increase executor memory
        .config("spark.sql.shuffle.partitions", "100")  # Better parallelism
        .getOrCreate())
    return spark


@time_execution
def construct_demo_seq(spark: SparkSession, external_val_data_dir: str) -> str:
    """all_demographic_tokens: DataFrame, 
    Construct demographic seq: <SEX> <DOBYR>
    """

    logging.info("%sConstruct demographic seq", INDENT1)
    demo_tokens = spark.read.parquet(os.path.join(external_val_data_dir, "all_demographic_tokens"))

    demo_seq = demo_tokens \
        .groupBy("person_id") \
        .agg(F.concat_ws(" ",
            F.array_sort(F.collect_list(F.struct(F.col("sub_order"), F.col("token")))).getItem("token")
        ).alias("demo_seq")) \
        .orderBy("person_id")

    # save as demo_seq as required intermediate data, then re-read to break Spark lineage
    demo_seq_file_name = "demo_seq"
    demo_seq.write.mode("overwrite").parquet(os.path.join(external_val_data_dir, demo_seq_file_name))
    logging.info("%sSaved demographic seq to: %s", INDENT1, os.path.join(external_val_data_dir, demo_seq_file_name))

    return demo_seq_file_name

@time_execution
def construct_event_seq(
    spark: SparkSession,
    external_val_data_dir: str) -> str:
    """
    Construct even seq that follows demo seq: <AGE> <NY> <ERLST> <PLANTYP> <CAP> <EGEOLOC> <VT> <DX> <PROC> <RX> <DS> <LS> <COST> <ERLED> ... 
    """
    logging.info("%sConstruct event seq", INDENT1)
    
    # Read enrollment and claim tokens separately and union them here
    logging.info("%sReading and unioning enrollment and claim tokens...", INDENT2)
    all_enrollment_tokens = spark.read.parquet(os.path.join(external_val_data_dir, "all_enrollment_tokens"))
    all_claim_tokens = spark.read.parquet(os.path.join(external_val_data_dir, "all_claim_tokens"))
    all_event_tokens = all_enrollment_tokens.unionByName(all_claim_tokens)
    
    all_anchor_tokens = spark.read.parquet(os.path.join(external_val_data_dir, "all_anchor_tokens"))

    logging.info("%sSort event tokens", INDENT1) 
    # Union demographic tokens, anchor tokens, and event tokens
    # Sort by original_date within timestamp to maintain date-level ordering within each week
    all_temporal_tokens_sorted = all_event_tokens \
        .unionByName(all_anchor_tokens) \
        .repartition("person_id") \
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
                Window.partitionBy("person_id", "timestamp").orderBy("event_type", "category_priority", "sub_order", "original_date", "original_token", "comb_order")))



    logging.info("%sCreate event seq per timestamp", INDENT1)
    # Create window specification for inter-timestamp processing
    window_spec = Window.partitionBy("person_id").orderBy("timestamp")

    # Use lead to get time_to_next_timestamp in months (since timestamps are month-based), put ATT after seq_per_timestamp
    # months_between returns a float, we cast to int to get whole months
    enrollee_event_seq_per_timestamp = all_temporal_tokens_sorted \
        .groupBy("person_id", "timestamp", "event_type") \
        .agg(F.concat_ws(" ",
                F.array_sort(F.collect_list(F.struct(F.col("token_order_per_timestamp"), F.col("token")))).getItem("token")
            ).alias("seq_per_timestamp")) \
        .orderBy("person_id", "timestamp", "event_type") \
        .withColumn("event_order", F.row_number().over(window_spec)) \
        .withColumn("next_timestamp", F.lead("timestamp").over(window_spec)) \
        .withColumn("time_to_next_timestamp", F.months_between(F.lead("timestamp").over(window_spec), F.col("timestamp")).cast("int")) \
        .withColumn("ATT", 
            F.when(F.col("time_to_next_timestamp").isNull(), F.lit("")) \
            .otherwise(F.concat(F.lit("<ATT-"), F.col("time_to_next_timestamp").cast("string"), F.lit(">")))) \
        .withColumn("seq_per_timestamp_with_ATT", F.concat_ws(" ", F.col("seq_per_timestamp"), F.col("ATT"))) \
        .withColumn("is_from_2023", F.year(F.col("timestamp")) >= 2023)


    # save as seq_per_timestamp as required intermediate data, then re-read to break Spark lineage
    seq_per_timestamp_file_name = "event_seq_per_timestamp"
    enrollee_event_seq_per_timestamp.write.mode("overwrite").parquet(os.path.join(external_val_data_dir, seq_per_timestamp_file_name))
    logging.info("%sSaved event seq per timestamp to: %s", INDENT1, os.path.join(external_val_data_dir, seq_per_timestamp_file_name))

    return seq_per_timestamp_file_name


    
@time_execution
def filter_trajectory(trajectory: DataFrame) -> DataFrame:
    """
    Filter trajectory based on seq:
        - exclude seq with missing dobyr: <DOBYR-MISSING>
        - exclude seq with missing age: <AGE-MISSING>
        - exclude seq with age < 10 or age > 110
    """
    logging.info("%sFiltering trajectories...", INDENT2)
    
    trajectory_filtered = trajectory \
        .filter(~F.col("seq").contains("<DOBYR-MISSING>")) \
        .filter(~F.col("seq").contains("<AGE-MISSING>")) \
        .withColumn(
            "age_value",
            F.regexp_extract(F.col("seq"), r"<AGE-(\d+)>", 1).cast("int")) \
        .filter((F.col("age_value") >= 10) & (F.col("age_value") <= 110)) \
        .drop("age_value")

    # select required column
    trajectory_final = trajectory_filtered.select(
        "person_id",
        "seq_thru_2022", 
        "event_seq_from_2023", 
        "seq", 
        "first_token_index_from_2023", 
        "total_token_count", 
        "token_count_thru_2022", 
        "event_token_count_from_2023",
        "total_duration"
    )

    return trajectory_final




@time_execution
def construct_and_save_trajectory(
    spark: SparkSession,
    external_val_data_dir: str,
    demo_seq_file_name: str,
    seq_per_timestamp_file_name: str
) -> None:
    """
    Construct trajectory
    """

    # Re-read from parquet to get fresh lineage
    enrollee_event_seq_per_timestamp = spark.read.parquet(os.path.join(external_val_data_dir, seq_per_timestamp_file_name))
    logging.info("%sRe-loaded enrollee event sequence per timestamp from parquet to break lineage", INDENT2)

    
    att_tokens = enrollee_event_seq_per_timestamp \
        .withColumn("category", F.lit("ATT")) \
        .withColumn("token", F.col("ATT")) \
        .filter((F.col("time_to_next_timestamp") >= 0) & (F.col("time_to_next_timestamp") <= 12)) \
        .filter(F.col("token") != "<ATT->") \
        .select("category", "token")
        
    att_tokens_save_path = os.path.join(external_val_data_dir, "all_att_tokens")
    att_tokens.write.mode("overwrite").parquet(att_tokens_save_path)
    logging.info("%sSaved all ATT tokens for later finalizing vocabulary to: %s", INDENT1, att_tokens_save_path)
    
    logging.info("%sGrouping by person_id and is_from_2023 to create separate thru-2022 and from-2023 sequences", INDENT2)
    patient_event_seq_by_period = enrollee_event_seq_per_timestamp \
        .groupBy("person_id", "is_from_2023") \
        .agg(
            F.concat_ws(" ", 
                F.array_sort(F.collect_list(F.struct(F.col("timestamp"), F.col("event_type"), F.col("seq_per_timestamp_with_ATT")))).getItem("seq_per_timestamp_with_ATT")
            ).alias("period_seq")) \
        .withColumn("period_seq", F.trim(F.col("period_seq"))) \
        .withColumn("token_count_in_period", F.size(F.regexp_extract_all(F.col("period_seq"), F.lit("<[^>]+>"), 0)))
    
    logging.info("%sPivoting to get pre-2023 and post-2022 sequences as separate columns", INDENT2)
    patient_event_seq = patient_event_seq_by_period \
        .groupBy("person_id") \
        .agg(
            F.max(F.when(~F.col("is_from_2023"), F.col("period_seq"))).alias("event_seq_thru_2022"),
            F.max(F.when(F.col("is_from_2023"), F.col("period_seq"))).alias("event_seq_from_2023"),
            F.max(F.when(~F.col("is_from_2023"), F.col("token_count_in_period"))).alias("event_token_count_thru_2022"),
            F.max(F.when(F.col("is_from_2023"), F.col("token_count_in_period"))).alias("event_token_count_from_2023")) \
        .withColumn("event_token_count_thru_2022", F.coalesce(F.col("event_token_count_thru_2022"), F.lit(0))) \
        .withColumn("event_token_count_from_2023", F.coalesce(F.col("event_token_count_from_2023"), F.lit(0)))
        # Note: event_seq_thru_2022 and event_seq_from_2023 are left as NULL when missing - concat_ws will skip NULLs 
    

    demo_seq = spark.read.parquet(os.path.join(external_val_data_dir, demo_seq_file_name))
    duration = spark.read.parquet(os.path.join(external_val_data_dir, "duration")).select("person_id", "total_duration").distinct()

    trajectory = patient_event_seq \
        .join(demo_seq, on="person_id", how="inner") \
        .join(duration, on="person_id", how="left") \
        .withColumn("seq_thru_2022", F.concat_ws(" ",
            F.col("demo_seq"), 
            F.col("event_seq_thru_2022"))) \
        .withColumn("seq", F.concat_ws(" ", 
            F.lit("<sos>"), 
            F.col("demo_seq"), 
            F.col("event_seq_thru_2022"),  # NULL values are skipped by concat_ws
            F.col("event_seq_from_2023"),  # NULL values are skipped by concat_ws
            F.lit("<eos>"))) \
        .withColumn("total_token_count", F.col("event_token_count_thru_2022") + F.col("event_token_count_from_2023") + 4) \
        .withColumn("token_count_thru_2022", F.col("event_token_count_thru_2022") + 3) \
        .withColumn("first_token_index_from_2023", 
            F.when(F.col("event_token_count_from_2023") == 0, F.lit(None))
            .when(F.col("event_token_count_thru_2022") == 0, F.lit(None))  # No historical context to predict from
            .otherwise(F.col("token_count_thru_2022")))  # 0-based: NY is always at this position  

    trajectory.cache()
    logging.info("%sConstricted %s trajectories:", INDENT2, f"{trajectory.count():,}")
    

    trajectory_final = filter_trajectory(trajectory)

    # Cache trajectory since it's used for show, count, and write

    logging.info("%sSample trajectories:", INDENT2)
    trajectory_final.select("person_id", "first_token_index_from_2023", "seq").show(100, truncate=4096)
    
    total_trajectories = trajectory_final.count()
    trajectory_save_path = os.path.join(external_val_data_dir, "trajectory")
    trajectory_final.write.mode("overwrite").parquet(trajectory_save_path)
    logging.info("%sTotal %s filtered trajectories saved to: %s", INDENT1, total_trajectories, trajectory_save_path)


    trajectory.unpersist() # done with trajectory



def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S2 CONSTRUCT SEQUENCES STARTING...")

    # Parse arguments
    args = parse_args()
    os.makedirs(args.external_val_data_dir, exist_ok=True)

    # Create Spark session    
    spark = create_spark_session('EHRSHOT_Step2_Construct_Sequences')
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sExternal validation data directory: %s", INDENT1, args.external_val_data_dir)

    logging.info("********** 1 Construct demographic sequences **********")
    demo_seq_file_name = construct_demo_seq(spark, args.external_val_data_dir)

    logging.info("********** 2 Construct event sequences per timestamp **********")
    seq_per_timestamp_file_name = construct_event_seq(spark, args.external_val_data_dir)

    logging.info("********** 3 Construct trajectory **********")
    construct_and_save_trajectory(spark, args.external_val_data_dir, demo_seq_file_name, seq_per_timestamp_file_name)


    # Stop Spark session
    spark.stop()
    logging.info("S2 CONSTRUCT SEQUENCES COMPLETED SUCCESSFULLY...")
    logging.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
