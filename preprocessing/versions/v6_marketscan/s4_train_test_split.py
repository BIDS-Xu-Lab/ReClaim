#!/usr/bin/env python3
"""
Reclaim Data Train/Val/Test Split
Splits trajectory data into train, validation, and test sets.

Hybrid approach:
1. Use HuggingFace datasets to split enrollee_ids (lightweight, deterministic)
2. Save overall_split mapping
3. Use Spark to join and write splits in parallel (efficient I/O)

Split strategy:
- If dataset size > 10 million: fixed 1M val, 1M test, rest for train
- If dataset size <= 10 million: 80% train, 10% val, 10% test
"""

import logging
import os
import argparse
from datasets import load_dataset, Dataset, disable_caching
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from utils import setup_logging, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

# Split thresholds
SIZE_THRESHOLD = 10_000_000  # 10 million
FIXED_HOLDOUT_SIZE = 1_000_000  # 1 million each for val and test

# disable cache for limited disk quota
disable_caching()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Split trajectory data into train/val/test sets')
    parser.add_argument("--input_folder_path", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"), "v6", "marketscan_val"),
                       help='Base directory containing processed data')
    parser.add_argument("--seed", type=int, default=66,
                       help='Random seed for reproducibility')
    parser.add_argument("--num_proc", type=int, default=16,
                       help='Number of processes for HuggingFace parallel loading')
    parser.add_argument("--num_partitions", type=int, default=500,
                       help='Number of output partitions for train set (val/test use 1/5 of this)')
    return parser.parse_args()



@time_execution
def load_enrollee_ids(processed_data_folder: str, num_proc: int) -> Dataset:
    """Load only enrollee_ids from trajectory parquet files using HuggingFace datasets"""
    trajectory_path = os.path.join(processed_data_folder, "trajectory", "*.parquet")
    logging.info("%sLoading enrollee_ids from: %s", INDENT1, trajectory_path)
    logging.info("%sUsing %d processes for parallel loading", INDENT2, num_proc)
    
    # Load only the enrollee_id column (much faster than loading full data)
    dataset = load_dataset(
        "parquet", 
        data_files=trajectory_path, 
        split="train",
        columns=["enrollee_id"],  # Only load enrollee_id column
        num_proc=num_proc
    )
    
    total_count = len(dataset)
    logging.info("%sTotal enrollees loaded: %s", INDENT1, f"{total_count:,}")
    
    # Sort by enrollee_id to ensure reproducible order
    logging.info("%sSorting by enrollee_id for reproducibility...", INDENT1)
    dataset = dataset.sort("enrollee_id")
    
    return dataset


@time_execution
def create_split_assignments(
    dataset: Dataset, 
    seed: int,
    processed_data_folder: str
) -> str:
    """
    Split enrollee_ids into train/val/test sets using HuggingFace datasets.
    Saves the split assignments to a parquet file.
    
    Strategy:
    - If size > 10M: fixed 1M val, 1M test, rest for train
    - If size <= 10M: 80% train, 10% val, 10% test
    
    Args:
        dataset: HuggingFace Dataset with enrollee_ids
        seed: Random seed for reproducibility
        processed_data_folder: Path to save overall_split
    
    Returns:
        Path to the saved overall_split parquet file
    """
    total_size = len(dataset)
    logging.info("%sTotal samples: %s", INDENT1, f"{total_size:,}")
    
    if total_size > SIZE_THRESHOLD:
        # Fixed holdout: 1M val, 1M test
        train_size = total_size - 2 * FIXED_HOLDOUT_SIZE
        val_size = FIXED_HOLDOUT_SIZE
        test_size = FIXED_HOLDOUT_SIZE
        logging.info("%sUsing FIXED holdout strategy (size > %s)", INDENT1, f"{SIZE_THRESHOLD:,}")
        logging.info("%sTrain size: %s, Val size: %s, Test size: %s", INDENT2, f"{train_size:,}", f"{val_size:,}", f"{test_size:,}")
        
        # First split: train vs (val + test)
        holdout_size = val_size + test_size
        train_rest = dataset.train_test_split(test_size=holdout_size, seed=seed)
        train = train_rest["train"]
        rest = train_rest["test"]
        
        # Second split: val vs test (50/50 of holdout)
        val_test = rest.train_test_split(test_size=0.5, seed=seed)
        val = val_test["train"]
        test = val_test["test"]
        
    else:
        # Ratio-based: 8:1:1
        logging.info("%sUsing RATIO-based strategy (size <= %s)", INDENT1, f"{SIZE_THRESHOLD:,}")
        logging.info("%sRatios: train=0.80, val=0.10, test=0.10", INDENT2)
        
        # First split: train (80%) vs (val + test) (20%)
        train_rest = dataset.train_test_split(train_size=0.8, seed=seed)
        train = train_rest["train"]
        rest = train_rest["test"]
        
        # Second split: val vs test (50/50 of remaining 20%)
        val_test = rest.train_test_split(test_size=0.5, seed=seed)
        val = val_test["train"]
        test = val_test["test"]
    
    # Log counts
    train_count = len(train)
    val_count = len(val)
    test_count = len(test)
    total = train_count + val_count + test_count
    
    logging.info("%sSplit results:", INDENT1)
    logging.info("%sTrain: %s (%.2f%%)", INDENT2, f"{train_count:,}", (train_count / total) * 100)
    logging.info("%sVal: %s (%.2f%%)", INDENT2, f"{val_count:,}", (val_count / total) * 100)
    logging.info("%sTest: %s (%.2f%%)", INDENT2, f"{test_count:,}", (test_count / total) * 100)

    split_assignment_path = os.path.join(processed_data_folder, "split_assignments")
    os.makedirs(split_assignment_path, exist_ok = True)
    logging.info("%sSave split assignment separately to: %s", INDENT2,split_assignment_path)

    
    train_split_path = os.path.join(split_assignment_path, "train_split_ids.parquet")
    train.to_parquet(train_split_path)
    logging.info("%sWriting train split (%s rows) to %s", INDENT3, f"{train_count:,}", train_split_path)
    
    val_split_path = os.path.join(split_assignment_path, "val_split_ids.parquet")
    val.to_parquet(val_split_path)
    logging.info("%sWriting val split (%s rows) to %s", INDENT3, f"{val_count:,}", val_split_path)

    test_split_path = os.path.join(split_assignment_path, "test_split_ids.parquet")
    test.to_parquet(test_split_path)
    logging.info("%sWriting test split (%s rows) to %s", INDENT3, f"{test_count:,}", test_split_path)


    return split_assignment_path




@time_execution
def load_and_join_with_spark(
    spark: SparkSession,
    processed_data_folder: str,
    split_assignment_path: str
) -> tuple:
    """
    Load full trajectory data and join with split assignments using Spark for better I/O efficiency.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load full trajectory data
    trajectory_path = os.path.join(processed_data_folder, "trajectory")
    logging.info("%sLoading full trajectory from: %s", INDENT1, trajectory_path)
    trajectory_df = spark.read.parquet(trajectory_path).cache()
    logging.info("%sLoaded %s trajectories", INDENT1, f"{trajectory_df.count():,}")
    
    # Load split assignments (small: only enrollee_id + split columns)
    logging.info("%sJoining trajectory with each split ...", INDENT1)
    # train
    train_ids = spark.read.parquet(os.path.join(split_assignment_path,"train_split_ids.parquet"))
    train_df = trajectory_df.join(train_ids, on = "enrollee_id", how='inner').cache()
    # val
    val_ids = spark.read.parquet(os.path.join(split_assignment_path,"val_split_ids.parquet"))
    val_df = trajectory_df.join(val_ids, on = "enrollee_id", how='inner').cache()
    # test
    test_ids = spark.read.parquet(os.path.join(split_assignment_path,"test_split_ids.parquet"))
    test_df = trajectory_df.join(test_ids, on = "enrollee_id", how='inner').cache()

    # create overall_split by combining all ids with split labels
    logging.info("%sCreating overall_split with split labels...", INDENT1)
    train_ids_w_split = train_ids.withColumn("split", F.lit("train"))
    val_ids_w_split = val_ids.withColumn("split", F.lit("val"))
    test_ids_w_split = test_ids.withColumn("split", F.lit("test"))
    overall_split = train_ids_w_split.union(val_ids_w_split).union(test_ids_w_split)
    # save overall splot
    overall_split_path = os.path.join(split_assignment_path, "overall_split")
    overall_split.write.mode("overwrite").parquet(overall_split_path)
    logging.info("%sOverall split data save to %s", INDENT2, overall_split_path)
    
    logging.info("%sLoaded split data and convert them into spark data frame", INDENT1)

    # unpersist trajectory afterwards
    trajectory_df.unpersist()
    
    return train_df, val_df, test_df


@time_execution
def log_split_statistics(train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
    """Log statistics for each split using Spark aggregations"""
    
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logging.info("%s%s set statistics:", INDENT1, split_name.capitalize())
        
        # Compute all stats in a single pass using Spark aggregations
        stats = df.agg(
            F.count("*").alias("count"),
            F.avg("total_token_count").alias("avg_tokens"),
            F.percentile_approx("total_token_count", 0.5).alias("median_tokens"),
            F.min("total_token_count").alias("min_tokens"),
            F.max("total_token_count").alias("max_tokens"),
            F.avg("total_duration").alias("avg_duration_days"),
            F.avg("token_count_thru_2022").alias("avg_tokens_thru_2022"),
            F.avg("event_token_count_from_2023").alias("avg_event_tokens_from_2023"),
            F.sum(F.when(F.col("event_token_count_from_2023") > 0, 1).otherwise(0)).alias("has_2023_data")
        ).collect()[0]
        
        logging.info("%sCount: %s", INDENT2, f"{stats['count']:,}")
        logging.info("%sAvg tokens: %.1f, Median: %.1f, Min: %d, Max: %d", 
                     INDENT2, stats['avg_tokens'], stats['median_tokens'], 
                     stats['min_tokens'], stats['max_tokens'])
        logging.info("%sAvg duration: %.1f days", INDENT2, stats['avg_duration_days'] or 0)
        logging.info("%sAvg tokens thru 2022: %.1f", INDENT2, stats['avg_tokens_thru_2022'])
        logging.info("%sAvg event tokens from 2023: %.1f", INDENT2, stats['avg_event_tokens_from_2023'])
        logging.info("%sHas 2023 data: %s (%.2f%%)", INDENT2, 
                     f"{stats['has_2023_data']:,}", 
                     (stats['has_2023_data'] / stats['count']) * 100)
        
        # Log first and last 10 enrollee_ids for reproducibility verification
        first_10_rows = df.select("enrollee_id").orderBy("enrollee_id").limit(10).collect()
        first_10_ids = [row["enrollee_id"] for row in first_10_rows]
        logging.info("%sFirst 10 enrollee_ids (sorted): %s", INDENT2, first_10_ids)
        last_10_rows = df.select("enrollee_id").orderBy("enrollee_id", ascending=False).limit(10).collect()
        last_10_ids = [row["enrollee_id"] for row in last_10_rows]
        logging.info("%sLast 10 enrollee_ids (sorted): %s", INDENT2, last_10_ids)



@time_execution
def save_splits(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    final_data_folder: str,
    num_partitions: int
):
    """Save train/val/test splits as parquet files using Spark parallel writes"""
    
    logging.info("%sSaving splits to final_data_folder: %s", INDENT1, final_data_folder)
    logging.info("%sUsing %d partitions for train, %d for val/test", INDENT2, 
                 num_partitions, max(num_partitions // 5, 1))
    
    train_path = os.path.join(final_data_folder, "train")
    val_path = os.path.join(final_data_folder, "val")
    test_path = os.path.join(final_data_folder, "test")
    
    # Repartition and write in parallel
    train_df.repartition(num_partitions).write.mode("overwrite").parquet(train_path)
    logging.info("%sTrain saved to: %s", INDENT2, train_path)
    
    val_partitions = max(num_partitions // 5, 1)
    val_df.repartition(val_partitions).write.mode("overwrite").parquet(val_path)
    logging.info("%sVal saved to: %s", INDENT2, val_path)
    
    test_partitions = max(num_partitions // 5, 1)
    test_df.repartition(test_partitions).write.mode("overwrite").parquet(test_path)
    logging.info("%sTest saved to: %s", INDENT2, test_path)


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S4 TRAIN/VAL/TEST SPLIT STARTING...")
    
    # Parse arguments
    args = parse_args()
    
    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    final_data_folder = os.path.join(args.input_folder_path, "final_data")
    
    # Log configuration
    logging.info("\nConfiguration:")
    logging.info("%sProcessed data folder: %s", INDENT1, processed_data_folder)
    logging.info("%sFinal data folder: %s", INDENT1, final_data_folder)
    logging.info("%sSeed: %d", INDENT1, args.seed)
    logging.info("%sSize threshold: %s", INDENT1, f"{SIZE_THRESHOLD:,}")
    logging.info("%sFixed holdout size (if >threshold): %s each for val/test", INDENT1, f"{FIXED_HOLDOUT_SIZE:,}")
    logging.info("%sNum proc (HuggingFace): %d", INDENT1, args.num_proc)
    logging.info("%sNum partitions (Spark output): %d", INDENT1, args.num_partitions)
    
    # Create Spark session
    spark = SparkSession.builder.appName('ReClaim_Step4_TrainTestSplit').getOrCreate()
    logging.info("Spark session created: %s", spark.sparkContext.appName)
    
    # Load only enrollee_ids (fast)
    logging.info("********** 1.1 Loading enrollee_ids with HF datasets**********")
    enrollee_dataset = load_enrollee_ids(processed_data_folder, args.num_proc)
    
    # Create and save split assignments
    logging.info("********** 1.2 Creating split assignments with HF datasets **********")
    overall_split_path = create_split_assignments(
        enrollee_dataset, 
        args.seed,
        processed_data_folder,
    )
    
    # Free memory from HuggingFace dataset
    del enrollee_dataset
    
    # Load full trajectory and join with split assignments
    logging.info("********** 2.1 Loading trajectory and joining with splits with Spark **********")
    train_df, val_df, test_df = load_and_join_with_spark(
        spark, processed_data_folder, overall_split_path
    )
    
    # Log statistics
    logging.info("********** 2.2 Computing split statistics with Spark **********")
    log_split_statistics(train_df, val_df, test_df)
    
    # Save splits using Spark parallel writes
    logging.info("********** 2.3 Saving splits with Spark **********")
    save_splits(train_df, val_df, test_df, final_data_folder, args.num_partitions)
    
    # Clean up
    train_df.unpersist()
    val_df.unpersist()
    test_df.unpersist()
    spark.stop()
    

    logging.info("Log file: %s", log_file)
    logging.info("S4 TRAIN/VAL/TEST SPLIT COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
