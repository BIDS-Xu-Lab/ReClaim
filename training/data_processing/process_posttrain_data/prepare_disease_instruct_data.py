#!/usr/bin/env python3
"""
Script to prepare evaluation data from trajectory parquet files.
Processes patient sequences, normalizes ICD codes, and saves to CSV.
"""

from pyspark.sql import SparkSession
import argparse
import os
import random
import pandas as pd
from pyspark.sql.functions import col, udf, concat_ws
from data_ops import get_instruct_seq_dx, get_instruct_seq_cost
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
        "--instruct_type",
        type=str,
        default="split",
        choices=["split", "mask"],
        help="Type of instruction (default: disease)",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="visit",
        choices=["visit", "first_year", "demo", "year", "random"],
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
        default=42, # or None
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--java_home",
        type=str,
        default="/gpfs/radev/apps/avx512/software/Java/21.0.2",
        help="Path to Java home directory (default: /gpfs/radev/apps/avx512/software/Java/21.0.2)",
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

    # Set Java home
    if args.java_home:
        os.environ["JAVA_HOME"] = args.java_home

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Spark session
    print("Initializing Spark session...")
    spark = (
        SparkSession.builder.appName("prepare_eval_data")
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

    # Build query
    print(f"Filtering for split: {args.split_name}")
    traj_train_df = traj_df.join(
        split_df.filter(col("split") == args.split_name), on="enrollee_id", how="semi" 
    )
    print("Training split size", len(traj_train_df.collect()))

    # Set Spark seed if provided, to ensure deterministic sampling when using limit()
    if args.seed is not None:
        print(f"Setting Spark execution seed to: {args.seed}")
        spark.conf.set("spark.sql.execution.seed", str(args.seed))

    holdout_test_split_ids_path = "***->[REPLACE WITH YOUR PATH]<-***/test_split_ids.parquet"
    def filter_holdout_test_ids(df, spark_session, holdout_test_split_ids_path):
        """Load holdout test split IDs from parquet and filter them out of the PySpark DataFrame.

        Args:
            df: PySpark DataFrame with an 'enrollee_id' column.
            spark_session: Active SparkSession.
            holdout_test_split_ids_path: Path to a parquet file containing test split IDs.

        Returns:
            PySpark DataFrame with holdout test enrollee_ids removed.
        """
        print(f"Loading holdout test split IDs from: {holdout_test_split_ids_path}")
        holdout_ids_df = spark_session.read.parquet(holdout_test_split_ids_path).select("enrollee_id")
        print(f"Loaded {holdout_ids_df.count()} holdout test IDs")

        initial_count = df.count()
        df = df.join(holdout_ids_df, on="enrollee_id", how="anti")
        filtered_count = df.count()
        removed = initial_count - filtered_count
        pct = 100 * filtered_count / initial_count if initial_count > 0 else 0
        print(f"Holdout filter: {initial_count} -> {filtered_count} (removed {removed}, kept {pct:.1f}%)")
        return df

    traj_train_df = filter_holdout_test_ids(traj_train_df, spark, holdout_test_split_ids_path)

    # Apply filters
    if args.filter_sex:
        print("Filtering for SEX-1 or SEX-2")
        traj_train_df = traj_train_df.filter(
            col("seq_thru_2022").contains("<SEX-1>") | col("seq_thru_2022").contains("<SEX-2>")
        )
        print("after sex filter: ", traj_train_df.count())

    if args.filter_dobyr:
        print("Filtering for DOBYR")
        traj_train_df = traj_train_df.filter(col("seq_thru_2022").contains("<DOBYR"))
        print("after dobyr filter: ", traj_train_df.count())

    if args.filter_age_missing:
        print("Filtering out AGE-MISSING")
        traj_train_df = traj_train_df.filter(~col("seq_thru_2022").contains("<AGE-MISSING>"))
        print("after age missing filter: ", traj_train_df.count())

    if args.filter_att_lt:
        print("Filtering out ATT-LT")
        traj_train_df = traj_train_df.filter(~col("seq_thru_2022").contains("<ATT-LT>"))
        print("after att lt filter: ", traj_train_df.count())
        
    print(f"Filtering out sequences with less than {args.min_att_tokens} ATT tokens")
    # Define a UDF for the ATT token filtering
    def enough_att_tokens(seq, min_att_tokens):
        tokens = seq.split(" ")
        att_tokens = [token for token in tokens if token.startswith("<ATT-")]
        return len(att_tokens) >= min_att_tokens
    enough_att_tokens_udf = udf(lambda seq: enough_att_tokens(seq, args.min_att_tokens), BooleanType())
    traj_train_df = traj_train_df.filter(enough_att_tokens_udf(col("seq_thru_2022")))
    print("after att tokens filter: ", traj_train_df.count())
    
    print(f"Filtering out sequences with less than {args.min_ny_tokens} NY tokens")
    # Define a UDF for the NY token filtering
    def enough_ny_tokens(seq, min_ny_tokens):
        tokens = seq.split(" ")
        ny_tokens = [token for token in tokens if token.startswith("<NY")]
        return len(ny_tokens) >= min_ny_tokens
    enough_ny_tokens_udf = udf(lambda seq: enough_ny_tokens(seq, args.min_ny_tokens), BooleanType())
    traj_train_df = traj_train_df.filter(enough_ny_tokens_udf(col("seq_thru_2022")))
    print("after ny tokens filter: ", traj_train_df.count())

    # Convert to pandas
    print("Converting to pandas DataFrame...")
    traj_test_pd = traj_train_df.select("enrollee_id", "seq_thru_2022").toPandas()
    print(f"Loaded {len(traj_test_pd)} samples from trajectory data")
    
    print("Generating instruction sequences...")
    icd_code_map_file = "***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_all_qwen_training/files/icd_code_map_with_category.csv"
    icd_code_map = pd.read_csv(icd_code_map_file)
    icd_codes = icd_code_map["ICD code"].tolist()
    instruct_results = traj_test_pd['seq_thru_2022'].apply(lambda x: get_instruct_seq_dx(x,
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

    if args.instruct_type == "split":
        traj_test_pd['split_point'] = instruct_results.apply(lambda res: res[1])

        traj_test_pd = traj_test_pd[traj_test_pd['split_point'] < 4096]
        print("Number of sequence after filtering split_point >= 4096: ", len(traj_test_pd))

    elif args.instruct_type == "mask":
        traj_test_pd['dx_positions'] = instruct_results.apply(lambda res: res[1])

    # random select the final number of samples
    traj_test_pd = traj_test_pd.sample(n=args.data_size, random_state=args.seed)
    print(f"Randomly selected {len(traj_test_pd)} final samples")

    print(traj_test_pd['instruct_seq'].iloc[0])
    
    if args.instruct_type == "split":
        print(traj_test_pd['split_point'].iloc[0])
    elif args.instruct_type == "mask":
        print(traj_test_pd['dx_positions'].iloc[0])

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
