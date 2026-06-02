import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"  # Suppress NumExpr thread warning
import argparse
import logging
from typing import List, Union
from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from utils import setup_logging, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

DEFAULT_PROCESSED_MARKETSCAN_DATA_ROOT = os.environ.get(
    "PROCESSED_MARKETSCAN_DATA_ROOT",
    "/path/to/processed_marketscan_data_root",
)


def resolve_marketscan_split_path(args) -> Union[str, List[str]]:
    """Resolve explicit split folder(s), or derive them from the processed root."""
    if args.marketscan_split_dir:
        paths = [path.strip() for path in args.marketscan_split_dir.split(",") if path.strip()]
        return paths if len(paths) > 1 else paths[0]

    if args.test_split == 'both':
        return [
            f"{args.reclaim_data_dir}/v{args.version}/marketscan_test",
            f"{args.reclaim_data_dir}/v{args.version}/marketscan_val",
        ]
    return f"{args.reclaim_data_dir}/v{args.version}/marketscan_{args.test_split}"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Construct previous sequence for cohort data')
    parser.add_argument('--reclaim_data_dir', type=str, 
                       default=DEFAULT_PROCESSED_MARKETSCAN_DATA_ROOT,
                       help='Processed MarketScan root used to derive split folders when --marketscan_split_dir is not set')
    parser.add_argument('--marketscan_split_dir', type=str,
                       default=os.environ.get("MARKETSCAN_SPLIT_DIR"),
                       help='Explicit MarketScan split folder. Use a comma-separated list for multiple folders.')
    parser.add_argument('--version', type=int, 
                       default=6, # 4, 5, 6
                       help='Data version')
    parser.add_argument('--test_split', type=str, 
                       default='test', choices=['test', 'val', 'both'],
                       help='Which split to use: test, val, or both')
    return parser.parse_args()

def create_spark_session(app_name: str) -> SparkSession:
    """
    Create SparkSession configured for SBATCH resources:
    - 60 CPUs per task
    - 400GB memory per node
    - Single node setup, local mode
    """
    
    # Resource allocation: 60 cores, 400GB memory per node
    # Driver: Reserve ~40GB for driver (10% of total)
    # Executor: Use remaining cores and memory
    driver_memory = "50g"
    executor_memory = "350g"  # Leave some overhead for system
    executor_cores = "10"  # 10 cores per executor, allows ~6 executors
    max_result_size = "10g"  # Increase for large result sets
    # Shuffle partitions: typically 2-3x number of cores for better parallelism
    shuffle_partitions = "240"  # 4x 60 cores
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", max_result_size) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    return spark


def read_and_union_parquet(spark: SparkSession, test_split_folder_path: Union[str, List[str]], subpath: str) -> DataFrame:
    """Read parquet from one or more split folder paths and union the results."""
    paths = test_split_folder_path if isinstance(test_split_folder_path, list) else [test_split_folder_path]
    dfs = [spark.read.parquet(os.path.join(p, subpath)) for p in paths]
    return reduce(DataFrame.unionByName, dfs)


@time_execution
def construct_and_save_pre_entry_seq(
    spark: SparkSession,
    test_split_folder_path: Union[str, List[str]],
    final_cohort_dir: str,
    version: int
) -> None:
    """
    Construct pre entry-date event sequence 
    """
    # read final cohort data
    cohort_df = spark.read.csv(os.path.join(final_cohort_dir, "cohort.csv"), header=True, inferSchema=True)
    demo_seq = read_and_union_parquet(spark, test_split_folder_path, "processed_data/demo_seq")
    

    if version == 6:
        logging.info("%sConstructing pre entry event sequence for v6 data", INDENT2)
        # read enrollee event sequence per timestamp
        enrollee_event_seq_per_timestamp = read_and_union_parquet(spark, test_split_folder_path, "processed_data/event_seq_per_timestamp")
        pre_entry_event_seq = enrollee_event_seq_per_timestamp \
            .join(cohort_df, enrollee_event_seq_per_timestamp.enrollee_id == cohort_df.pid, "inner") \
            .withColumn("is_before_entry", F.when(F.col("timestamp") < F.col("entry_date"), F.lit(True)).otherwise(F.lit(False))) \
            .filter(F.col("is_before_entry")) \
            .groupBy("enrollee_id") \
            .agg(
                F.concat_ws(" ", 
                    F.array_sort(F.collect_list(F.struct(F.col("timestamp"), F.col("event_type"), F.col("seq_per_timestamp_with_ATT")))).getItem("seq_per_timestamp_with_ATT")
                ).alias("pre_entry_event_seq")) \
            .withColumn("pre_entry_event_seq", F.trim(F.col("pre_entry_event_seq"))) 
        
        pre_entry_seq = pre_entry_event_seq \
            .join(demo_seq, on="enrollee_id", how="inner") \
            .withColumn("pre_entry_seq", F.concat_ws(" ", 
                F.lit("<sos>"), 
                F.col("demo_seq"), 
                F.col("pre_entry_event_seq"))) \
            .withColumn("pre_entry_seq", F.trim(F.col("pre_entry_seq"))) \
            .withColumnRenamed("enrollee_id", "pid") \
            .select("pid", "pre_entry_seq") \
            .withColumn("token_count", F.size(F.regexp_extract_all(F.col("pre_entry_seq"), F.lit("<[^>]+>"), 0)))

    if version == 7:
        logging.info("%sConstructing pre entry event sequence for v7 data", INDENT2)
        # process age anchor token separately
        all_anchor_tokens = read_and_union_parquet(spark, test_split_folder_path, "processed_data/all_anchor_tokens") \
            .withColumn("month_of_year", F.month(F.col("timestamp"))) \
            .withColumn("seq_unit", F.col("token")) \
            .withColumn("seq_priority", F.lit(0)) \
            .select("enrollee_id", "timestamp", "seq_unit", "seq_priority")
        
        # read enrollee event sequence per timestamp
        enrollee_event_seq_per_timestamp = read_and_union_parquet(spark, test_split_folder_path, "processed_data/event_seq_per_timestamp") \
            .withColumn("seq_priority", F.lit(1)) 

        pre_entry_event_seq = enrollee_event_seq_per_timestamp \
            .select("enrollee_id", "timestamp", "seq_unit", "seq_priority") \
            .unionByName(all_anchor_tokens) \
            .join(cohort_df, enrollee_event_seq_per_timestamp.enrollee_id == cohort_df.pid, "inner") \
            .withColumn("is_before_entry", F.when(F.col("timestamp") < F.col("entry_date"), F.lit(True)).otherwise(F.lit(False))) \
            .filter(F.col("is_before_entry")) \
            .groupBy("enrollee_id") \
            .agg(
                F.concat_ws(" ", 
                    F.array_sort(F.collect_list(F.struct(F.col("timestamp"), F.col("seq_priority"), F.col("seq_unit")))).getItem("seq_unit")
                ).alias("pre_entry_event_seq")) \
            .withColumn("pre_entry_event_seq", F.trim(F.col("pre_entry_event_seq"))) 
        
        pre_entry_seq = pre_entry_event_seq \
            .join(demo_seq, on="enrollee_id", how="inner") \
            .withColumn("pre_entry_seq", F.concat_ws(" ", 
                F.lit("<sos>"), 
                F.col("demo_seq"), 
                F.col("pre_entry_event_seq"))) \
            .withColumn("pre_entry_seq", F.trim(F.col("pre_entry_seq"))) \
            .withColumnRenamed("enrollee_id", "pid") \
            .select("pid", "pre_entry_seq") \
            .withColumn("token_count", F.size(F.regexp_extract_all(F.col("pre_entry_seq"), F.lit("<[^>]+>"), 0)))

    # Log token count distribution
    logging.info("%sToken count distribution for pre_entry_seq:", INDENT2)
    
    # Calculate statistics
    token_stats = pre_entry_seq.agg(
        F.mean("token_count").alias("mean"),
        F.expr("percentile_approx(token_count, 0.5)").alias("median"),
        F.min("token_count").alias("min"),
        F.max("token_count").alias("max"),
        F.stddev("token_count").alias("stddev"),
        F.count("*").alias("total_count"),
        F.sum(F.when(F.col("token_count") > 4096, 1).otherwise(0)).alias("count_over_4096")
    ).collect()[0]
    
    logging.info("%sMean token count: %.2f", INDENT3, token_stats["mean"])
    logging.info("%sMedian token count: %.2f", INDENT3, token_stats["median"])
    logging.info("%sMin token count: %d", INDENT3, token_stats["min"])
    logging.info("%sMax token count: %d", INDENT3, token_stats["max"])
    logging.info("%sStd dev: %.2f", INDENT3, token_stats["stddev"])
    logging.info("%sTotal sequences: %s", INDENT3, f"{token_stats['total_count']:,}")
    logging.info("%sSequences with token count > 4096: %s (%.2f%%)", 
                 INDENT3, 
                 f"{token_stats['count_over_4096']:,}", 
                 100.0 * token_stats['count_over_4096'] / token_stats['total_count'])

    # Log first 10 rows of pre_entry_seq
    logging.info("%sFirst 5 rows of pre_entry_seq:", INDENT2)
    pre_entry_seq_rows = pre_entry_seq.select("pid", "pre_entry_seq").limit(5).collect()
    for i, row in enumerate(pre_entry_seq_rows, 1):
        logging.info("%sPre_entry_seq %s: %s", INDENT3, i, row['pre_entry_seq'])

    # Join with cohort_df to preserve all cohort columns
    logging.info("%sJoining pre_entry_seq with cohort data to preserve all columns", INDENT2)
    cohort_with_pre_entry_seq = pre_entry_seq.join(cohort_df, on="pid", how="inner")
    
    # Log counts
    logging.info("%sCohort count: %s", INDENT2, f"{cohort_df.count():,}")
    logging.info("%sPre_entry_seq count: %s", INDENT2, f"{pre_entry_seq.count():,}")
    logging.info("%sFinal cohort_with_pre_entry_seq count: %s", INDENT2, f"{cohort_with_pre_entry_seq.count():,}")

    # Save the result to a csv file (convert to pandas for single file output)
    output_path = os.path.join(final_cohort_dir, "cohort_with_pre_entry_seq.csv")
    logging.info("%sSaving results to: %s", INDENT2, output_path)
    cohort_with_pre_entry_seq.toPandas().to_csv(output_path, index=False)



   


@time_execution
def main():
    """Main function to read token.parquet and visit.parquet files and log sample data"""
    # Setup logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("CONSTRUCT PREVSEQ STARTING...")
    
    # Parse arguments
    args = parse_args()

    working_dir = os.getcwd()
    test_split_folder_path = resolve_marketscan_split_path(args)
    final_cohort_dir = os.path.join(working_dir, 'intermediate', f'v{args.version}', 'final_cohort')
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sWorking directory: %s", INDENT1, working_dir)
    logging.info("%sFinal cohort folder: %s", INDENT1, final_cohort_dir)

    # Create Spark session in local mode
    spark = create_spark_session("construct_pre_entry_seq")
    logging.info("%sSpark session created in local mode", INDENT1)
    
    try:
        # Create Spark session
        logging.info("%sConstructing pre entry sequence", INDENT1)
        construct_and_save_pre_entry_seq(spark, test_split_folder_path, final_cohort_dir, args.version)
        
        logging.info("CONSTRUCT PREVSEQ COMPLETED SUCCESSFULLY...")
        logging.info("Log file: %s", log_file)
        
    except Exception as e:
        logging.error("Script failed with error: %s", str(e))
        raise
    finally:
        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            logging.info("Spark session stopped")


if __name__ == "__main__":
    main()
