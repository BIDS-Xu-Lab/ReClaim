#!/usr/bin/env python3
"""
Script to prepare evaluation data from trajectory parquet files.
Processes patient sequences, normalizes ICD codes, and saves to CSV.
"""

from pyspark.sql import SparkSession
import argparse
import os
from pathlib import Path
import random
import pandas as pd
from pyspark.sql.functions import rand
from pyspark.sql.functions import col, udf, concat_ws
from data_ops import get_instruct_seq
from pyspark.sql.types import BooleanType

def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation data from trajectory parquet files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory containing processed_data folder",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for CSV file"
    )
    parser.add_argument(
        "--traj_file",
        type=str,
        default="processed_data/trajectory.parquet",
        help="Path to trajectory parquet file relative to data_dir (default: processed_data/trajectory.parquet)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="processed_data/overall_split.parquet",
        help="Path to split parquet file relative to data_dir (default: processed_data/overall_split.parquet)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to use (default: test)",
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=None,
        help="Limit number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--demo_end_index",
        type=int,
        default=3,
        help="Index where demographics end (default: 3)",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="visit",
        choices=["visit", "first_year", "demo", "random"],
        help="Split by visit (default: visit)",
    )
    parser.add_argument(
        "--instruct_token",
        type=str,
        default="<INSTRUCT-DX>",
        choices=["<INSTRUCT-DX>", "<INSTRUCT-COST>"],
        help="Instruction token (default: <INSTRUCT-DX>, or <INSTRUCT-COST>)",
    )
    parser.add_argument(
        "--instruct_position",
        type=str,
        default="before",
        choices=["before", "after"],
        help="Instruction position (default: before, or after)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="traj_train_pd.parquet",
        help="Output parquet filename (default: traj_test_pd.parquet)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24, # or None
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--java_home",
        type=str,
        default="***->[REPLACE WITH YOUR PATH]<-***",
        help="Replace the default placeholder with your Java home directory for Spark",
    )
    parser.add_argument(
        "--spark_driver_memory",
        type=str,
        default="500g",
        help="Spark driver memory (default: 500g)",
    )
    parser.add_argument(
        "--spark_driver_max_result_size",
        type=str,
        default="100g",
        help="Spark driver max result size (default: 100g)",
    )
    parser.add_argument(
        "--spark_master",
        type=str,
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )
    parser.add_argument(
        "--filter_sex",
        action="store_true",
        default=True,
        help="Filter sequences to include only those with SEX-1 or SEX-2 (default: True)",
    )
    parser.add_argument(
        "--no_filter_sex",
        dest="filter_sex",
        action="store_false",
        help="Disable sex filtering",
    )
    parser.add_argument(
        "--filter_dobyr",
        action="store_true",
        default=True,
        help="Filter sequences to include only those with DOBYR (default: True)",
    )
    parser.add_argument(
        "--no_filter_dobyr",
        dest="filter_dobyr",
        action="store_false",
        help="Disable DOBYR filtering",
    )
    parser.add_argument(
        "--min_att_tokens",
        type=int,
        default=2,
        help="Minimum number of ATT tokens to keep a sequence (default: 2)",
    )
    parser.add_argument(
        "--min_ny_tokens",
        type=int,
        default=0,
        help="Minimum number of NY tokens to keep a sequence (default: 0)",
    )
    parser.add_argument(
        "--filter_age_missing",
        action="store_true",
        default=True,
        help="Filter out sequences with AGE-MISSING (default: True)",
    )
    parser.add_argument(
        "--no_filter_age_missing",
        dest="filter_age_missing",
        action="store_false",
        help="Disable AGE-MISSING filtering",
    )
    parser.add_argument(
        "--filter_att_lt",
        action="store_true",
        default=True,
        help="Filter out sequences with ATT-LT (default: True)",
    )
    parser.add_argument(
        "--no_filter_att_lt",
        dest="filter_att_lt",
        action="store_false",
        help="Disable ATT-LT filtering",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Set Java home
    if args.java_home:
        os.environ["JAVA_HOME"] = args.java_home

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = (
        SparkSession.builder.appName("prepare_posttrain_data")
        .master(args.spark_master)
        .config("spark.driver.memory", args.spark_driver_memory)
        .config("spark.driver.maxResultSize", args.spark_driver_max_result_size)
        .getOrCreate()
    )

    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

    spark.conf.set("spark.sql.pivotMaxValues", "10000000")
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    # Construct file paths
    traj_path = os.path.join(args.data_dir, args.traj_file)
    split_path = os.path.join(args.data_dir, args.split_file)

    print(f"Loading trajectory data from: {traj_path}")
    print(f"Loading split data from: {split_path}")

    # Load data
    traj_df = spark.read.parquet(traj_path)
    split_df = spark.read.parquet(split_path)

    print(f"Loaded trajectory data: {traj_df.count()} rows")
    print(f"Loaded split data: {split_df.count()} rows")

    # Set Spark seed if provided, to ensure deterministic sampling when using limit()
    if args.seed is not None:
        print(f"Setting Spark execution seed to: {args.seed}")
        spark.conf.set("spark.sql.execution.seed", str(args.seed))

    # Build query
    print(f"Filtering for split: {args.split_name}")

    traj_train_df = traj_df.join(   
        split_df.filter(col("split") == args.split_name), on="enrollee_id", how="semi" 
    )
    print("Training split size", traj_train_df.count())

    # Apply limit if specified
    if args.data_size:

        print("random sampling ~1 million rows")

        fraction = 0.01
        traj_train_df = traj_train_df.sample(withReplacement=False, fraction=fraction, seed=args.seed)
        traj_train_df = traj_train_df.cache()

        print(traj_train_df.count())

    # Apply filters
    if args.filter_sex:
        print("Filtering for SEX-1 or SEX-2")
        traj_train_df = traj_train_df.filter(
            col("seq_thru_cutoff").contains("<SEX-1>") | col("seq_thru_cutoff").contains("<SEX-2>")
        )
        print("after sex filter: ", traj_train_df.count())

    if args.filter_dobyr:
        print("Filtering for DOBYR")
        traj_train_df = traj_train_df.filter(col("seq_thru_cutoff").contains("<DOBYR"))
        print("after dobyr filter: ", traj_train_df.count())

    if args.filter_age_missing:
        print("Filtering out AGE-MISSING")
        traj_train_df = traj_train_df.filter(~col("seq_thru_cutoff").contains("<AGE-MISSING>"))
        print("after age missing filter: ", traj_train_df.count())

    if args.filter_att_lt:
        print("Filtering out ATT-LT")
        traj_train_df = traj_train_df.filter(~col("seq_thru_cutoff").contains("<ATT-LT>"))
        print("after att lt filter: ", traj_train_df.count())
        
    print(f"Filtering out sequences with less than {args.min_att_tokens} ATT tokens")
    # Define a UDF for the ATT token filtering
    def enough_att_tokens(seq, min_att_tokens):
        tokens = seq.split(" ")
        att_tokens = [token for token in tokens if token.startswith("<ATT-")]
        return len(att_tokens) >= min_att_tokens
    enough_att_tokens_udf = udf(lambda seq: enough_att_tokens(seq, args.min_att_tokens), BooleanType())
    traj_train_df = traj_train_df.filter(enough_att_tokens_udf(col("seq_thru_cutoff")))
    print("after att tokens filter: ", traj_train_df.count())
    
    print(f"Filtering out sequences with less than {args.min_ny_tokens} NY tokens")
    # Define a UDF for the NY token filtering
    def enough_ny_tokens(seq, min_ny_tokens):
        tokens = seq.split(" ")
        ny_tokens = [token for token in tokens if token.startswith("<NY")]
        return len(ny_tokens) >= min_ny_tokens
    enough_ny_tokens_udf = udf(lambda seq: enough_ny_tokens(seq, args.min_ny_tokens), BooleanType())
    traj_train_df = traj_train_df.filter(enough_ny_tokens_udf(col("seq_thru_cutoff")))
    print("after ny tokens filter: ", traj_train_df.count())

    # Convert to pandas
    print("Converting to pandas DataFrame...")
    traj_test_pd = traj_train_df.select("enrollee_id", "seq_thru_cutoff").toPandas()
    print(f"Loaded {len(traj_test_pd)} samples from trajectory data")
    
    print("Generating instruction sequences...")
    icd_code_map_file = (
        Path(__file__).resolve().parents[4]
        / "pre-training/qwen_training/files/icd_code_map_with_category.csv"
    )
    icd_code_map = pd.read_csv(icd_code_map_file)
    icd_codes = icd_code_map["ICD code"].tolist()
    instruct_results = traj_test_pd['seq_thru_cutoff'].apply(lambda x: get_instruct_seq(x,
                                                                                        demo_end_index=args.demo_end_index, 
                                                                                        split_by=args.split_by, 
                                                                                        instruct_token=args.instruct_token, 
                                                                                        instruct_position=args.instruct_position,
                                                                                        disease_set=icd_codes)
                                                                                        )
    print("Number of sequence before filtering None: ", len(instruct_results))
    mask = instruct_results.notnull()

    instruct_results = instruct_results[mask]
    traj_test_pd = traj_test_pd[mask]

    print("Number of None results after filtering: ", instruct_results.isnull().sum())
    print("Number of sequence after filtering None: ", len(instruct_results))
    
    traj_test_pd['instruct_seq'] = instruct_results.apply(lambda res: res[0])
    traj_test_pd['split_point'] = instruct_results.apply(lambda res: res[1])
    # record disease in resulted trajectory
    traj_test_pd['predict_icd'] = instruct_results.apply(lambda res: res[2])

    traj_test_pd = traj_test_pd[traj_test_pd['split_point'] < args.seq_len]
    print(f"Number of sequence after filtering split_point >= {args.seq_len}: ", len(traj_test_pd))

    inst_len = traj_test_pd['instruct_seq'].str.split().str.len()
    traj_test_pd = traj_test_pd[inst_len <= args.seq_len]
    print(f"Number of sequence after filtering full instruct_seq length > {args.seq_len}: ", len(traj_test_pd))

    # random select the final number of samples
    if len(traj_test_pd) < args.data_size:
        raise ValueError(
            f"Only {len(traj_test_pd)} valid sequences remain after filtering, "
            f"fewer than the requested data_size={args.data_size}. "
            f"Lower --data_size or relax the upstream filters."
        )
    traj_test_pd = traj_test_pd.sample(n=args.data_size, random_state=args.seed)
    print(f"Randomly selected {len(traj_test_pd)} final samples")

    # calculate the prediction icd coverage
    all_icds_series = traj_test_pd['predict_icd'].explode().dropna()
    icd_counts = all_icds_series.value_counts()
    total_unique_covered = len(icd_counts)

    print(traj_test_pd['instruct_seq'].iloc[0])
    print(traj_test_pd['split_point'].iloc[0])
    print(traj_test_pd['predict_icd'].iloc[0])

    print("total unique icd covered:", total_unique_covered)
    print("total icd of interest:", len(icd_codes))

    # Save to parquet
    output_path = os.path.join(args.output_dir, args.output_file)
    print(f"Saving to: {output_path}")
    traj_test_pd.to_parquet(output_path, index=False)
    print(f"Successfully saved {len(traj_test_pd)} samples to {output_path}")

    # Stop Spark session
    spark.stop()
    print("Spark session stopped")

if __name__ == "__main__":
    main()
