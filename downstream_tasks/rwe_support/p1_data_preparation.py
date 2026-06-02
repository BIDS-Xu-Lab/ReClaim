#!/usr/bin/env python3
"""
Drug category patient cohort definition processing
Step 1: Create drug name to token mapping
Step 2: Define patient cohorts by drug categories
"""

import os
import argparse
import logging
from typing import List, Union
from functools import reduce
from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql import functions as F
from utils import setup_logging, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

# Drug class lists
GLP1_LIST = ["albiglutide", "dulaglutide", "exenatide", "liraglutide", 
            "lixisenatide", "semaglutide", "tirzepatide"]
SGLT2_LIST = ["canagliflozin", "dapagliflozin", "empagliflozin", 
             "ertugliflozin", "sotagliflozin", "bexagliflozin"]
DPP4_LIST = ["sitagliptin", "saxagliptin", "alogliptin", "linagliptin"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DRUG_MAPPING_FILE = os.path.join(
    REPO_ROOT,
    "preprocessing",
    "vocab_mapping",
    "mapping",
    "ndc_drug_to_rxnorm_ingredient_mapping.csv",
)
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
    parser = argparse.ArgumentParser(description='Create drug mapping and define patient cohorts by drug categories')
    parser.add_argument('--drug_mapping_file', type=str, 
                       default=os.environ.get("DRUG_MAPPING_FILE", DEFAULT_DRUG_MAPPING_FILE),
                       help='Drug mapping file')
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


# =============================================================================
# Step 1: Drug Class Identification
# =============================================================================

@time_execution
def load_and_preprocess_drug_mapping(spark: SparkSession, mapping_df_path: str) -> DataFrame:
    """
    Load mapping data and select only the needed columns.
    Also lower-case the drug name columns for downstream ease.
    """
    logging.info("%sLoading drug mapping data from: %s", INDENT1, mapping_df_path)
    drug_mapping_df = spark.read.csv(mapping_df_path, header=True, inferSchema=True) \
        .select(
            'ndc_drug_code', 
            'ndc_drug_name', 
            'rxnorm_drug_name', 
            'rxnorm_ingredient_name'
        ).dropDuplicates()
    # Lowercase name columns
    drug_mapping_df = drug_mapping_df.withColumn("ndc_drug_name", F.lower(F.col("ndc_drug_name"))) \
           .withColumn("rxnorm_drug_name", F.lower(F.col("rxnorm_drug_name"))) \
           .withColumn("rxnorm_ingredient_name", F.lower(F.col("rxnorm_ingredient_name")))
    return drug_mapping_df

@time_execution
def create_drug_class_flags(drug_mapping_df: DataFrame) -> DataFrame:
    """
    Create drug class flags for GLP-1, SGLT-2, and DPP-4 drugs
    by searching for patterns in ndc_drug_name, rxnorm_drug_name, and rxnorm_ingredient_name.
    """
    logging.info("%sCreating drug class flags", INDENT1)

    logging.info("%sGLP-1 drugs: %s items", INDENT2, len(GLP1_LIST))
    logging.info("%sSGLT-2 drugs: %s items", INDENT2, len(SGLT2_LIST))
    logging.info("%sDPP4 drugs: %s items", INDENT2, len(DPP4_LIST))

    # Create regex patterns for each drug category
    glp1_pattern = "|".join(GLP1_LIST)
    sglt2_pattern = "|".join(SGLT2_LIST)
    dpp4_pattern = "|".join(DPP4_LIST)

    # Create drug category flags by checking all three variables for a match
    drug_mapping_flagged = drug_mapping_df.withColumn(
        "glp1",
        F.col("ndc_drug_name").rlike(glp1_pattern) |
        F.col("rxnorm_drug_name").rlike(glp1_pattern) |
        F.col("rxnorm_ingredient_name").rlike(glp1_pattern)
    ).withColumn(
        "sglt2",
        F.col("ndc_drug_name").rlike(sglt2_pattern) |
        F.col("rxnorm_drug_name").rlike(sglt2_pattern) |
        F.col("rxnorm_ingredient_name").rlike(sglt2_pattern)
    ).withColumn(
        "dpp4",
        F.col("ndc_drug_name").rlike(dpp4_pattern) |
        F.col("rxnorm_drug_name").rlike(dpp4_pattern) |
        F.col("rxnorm_ingredient_name").rlike(dpp4_pattern)
    )

    # Log counts for each category
    glp1_count = drug_mapping_flagged.filter(F.col("glp1")).count()
    sglt2_count = drug_mapping_flagged.filter(F.col("sglt2")).count()
    dpp4_count = drug_mapping_flagged.filter(F.col("dpp4")).count()

    logging.info("%sGLP-1 matches: %s", INDENT2, glp1_count)
    logging.info("%sSGLT-2 matches: %s", INDENT2, sglt2_count)
    logging.info("%sDPP-4 matches: %s", INDENT2, dpp4_count)

    # Check for multiple class matches
    multiple_classes = drug_mapping_flagged.withColumn("match_count", 
        F.col("glp1").cast("int") + F.col("sglt2").cast("int") + F.col("dpp4").cast("int")) \
        .filter(F.col("match_count") >= 2)
    

    multiple_classes_count = multiple_classes.count()
    
    logging.info("%sFound %s rows that match multiple categories", INDENT2, multiple_classes_count)
    
    if multiple_classes_count > 0:
        logging.info("%sMultiple class matches (first 100):", INDENT2)
        for row in multiple_classes.limit(100).collect():
            logging.info("%sNDC Drug Code: %s, NDC Drug Name: %s, RxNorm Drug Name: %s, RxNorm Ingredient Name: %s, GLP1: %s, SGLT2: %s, DPP4: %s", 
                        INDENT3, row['ndc_drug_code'], row['ndc_drug_name'], row['rxnorm_drug_name'], row['rxnorm_ingredient_name'],
                        row['glp1'], row['sglt2'], row['dpp4'])

    return drug_mapping_flagged



# =============================================================================
# Step 2: Cohort identification
# =============================================================================

def read_and_union_parquet(spark: SparkSession, test_split_folder_path: Union[str, List[str]], subpath: str) -> DataFrame:
    """Read parquet from one or more split folder paths and union the results."""
    paths = test_split_folder_path if isinstance(test_split_folder_path, list) else [test_split_folder_path]
    dfs = []
    for p in paths:
        full_path = os.path.join(p, subpath)
        logging.info("%sReading from: %s", INDENT2, full_path)
        dfs.append(spark.read.parquet(full_path))
    return reduce(DataFrame.unionByName, dfs)


@time_execution
def load_test_tokens(
    spark: SparkSession, 
    test_split_folder_path: Union[str, List[str]], 
) -> DataFrame:
    """
    Load drug claims token data for test split(s), extracting NDC code from original_token.
    If test_split_folder_path is a list, reads from all paths and unions the results.
    """

    logging.info("%sLoading test split data", INDENT1)
    
    all_claim_tokens_df = read_and_union_parquet(spark, test_split_folder_path, 'processed_data/all_claim_tokens') \
        .select('enrollee_id', 'timestamp', 'event_type', 'category', 'value', 'token', 'original_token') \
        .distinct()

    all_claim_tokens_df.cache()

    all_tokens_count = all_claim_tokens_df.count()
    logging.info("%sAll tokens dataframe row count: %s", INDENT2, all_tokens_count)
    
    return all_claim_tokens_df



@time_execution
def extract_and_save_cohort(
    all_claim_tokens_df: DataFrame, 
    drug_mapping_flagged: DataFrame,
    output_dir: str
) -> DataFrame:
    """
    Extract cohort of patients taking GLP-1, SGLT-2, or DPP-4 drug classes
    """
    logging.info("%sExtracting cohort of patients taking GLP-1, SGLT-2, or DPP-4 drug classes", INDENT1)    
    
    # Load claim tokens and extract NDC code as ndc_drug_code
    selected_drug_token_df = all_claim_tokens_df \
        .filter(F.col("category") == "RX") \
        .filter(F.col("timestamp") >= "2018-01-01") \
        .filter(F.col("timestamp") <= "2024-12-31") \
        .withColumn("ndc_drug_code", F.regexp_extract(F.col("original_token"), r"<NDCNUM-(.+)>", 1))
    
    logging.info("%sSelected drug token dataframe row count: %s", INDENT2, selected_drug_token_df.count())
    
    cohort_df = selected_drug_token_df.join(
        drug_mapping_flagged,
        selected_drug_token_df['ndc_drug_code'] == drug_mapping_flagged['ndc_drug_code'],
        "inner"
    ).filter(F.col("glp1") | F.col("sglt2") | F.col("dpp4"))

    logging.info("%sCohort dataframe row count: %s", INDENT2, cohort_df.count())
    
    cohort_df = cohort_df.select("enrollee_id", "glp1", "sglt2", "dpp4").dropDuplicates()

    logging.info("%sAnalyzing drug class distribution in cohort", INDENT1)
    
    drug_class_count = cohort_df \
        .groupBy("enrollee_id") \
        .agg(
            F.max(F.col("glp1").cast("int")).alias("has_glp1"),
            F.max(F.col("sglt2").cast("int")).alias("has_sglt2"),
            F.max(F.col("dpp4").cast("int")).alias("has_dpp4")) \
        .withColumn("drug_combo", F.concat(F.col("has_glp1"), F.col("has_sglt2"), F.col("has_dpp4"))) \
        .groupBy("drug_combo") \
        .agg(F.count("*").alias("count")) \
        .collect()
    
    # Convert to dictionary for easy lookup
    combination_counts = {row["drug_combo"]: row["count"] for row in drug_class_count}
    
    # Calculate metrics from combinations (e.g., "100" = GLP1 only)
    overall_enrollees_in_cohort = sum(combination_counts.values())
    glp1_enrollees_in_cohort = sum(v for k, v in combination_counts.items() if k[0] == "1")
    sglt2_enrollees_in_cohort = sum(v for k, v in combination_counts.items() if k[1] == "1")
    dpp4_enrollees_in_cohort = sum(v for k, v in combination_counts.items() if k[2] == "1")
    
    glp1_only_enrollees_in_cohort = combination_counts.get("100", 0)
    sglt2_only_enrollees_in_cohort = combination_counts.get("010", 0)
    dpp4_only_enrollees_in_cohort = combination_counts.get("001", 0)
    
    logging.info("%sOverall enrollees in cohort: %s", INDENT2, f"{overall_enrollees_in_cohort:,}")
    logging.info("%sGLP1 enrollees in cohort: %s", INDENT2, f"{glp1_enrollees_in_cohort:,}")
    logging.info("%sSGLT2 enrollees in cohort: %s", INDENT2, f"{sglt2_enrollees_in_cohort:,}")
    logging.info("%sDPP4 enrollees in cohort: %s", INDENT2, f"{dpp4_enrollees_in_cohort:,}")
    logging.info("%sGLP1-only enrollees in cohort: %s", INDENT2, f"{glp1_only_enrollees_in_cohort:,}")
    logging.info("%sSGLT2-only enrollees in cohort: %s", INDENT2, f"{sglt2_only_enrollees_in_cohort:,}")
    logging.info("%sDPP4-only enrollees in cohort: %s", INDENT2, f"{dpp4_only_enrollees_in_cohort:,}")


    # Save cohort data to parquet
    logging.info("%sSaving cohort data to parquet", INDENT1)
    os.makedirs(output_dir, exist_ok=True)
    cohort_df.write.mode("overwrite").parquet(os.path.join(output_dir, "selected_cohort"))
    logging.info("%sSuccessfully saved results to %s", INDENT2, os.path.join(output_dir, "selected_cohort"))

    return cohort_df


def format_icd10_code(code_col: Column) -> Column:
    """
    Format ICD-10 codes with decimal point inserted after the third character.
    """
    return F.regexp_replace(code_col, r"^(.{3})(.+)$", r"$1.$2")



@time_execution
def extract_and_save_required_tables(
    spark: SparkSession,
    all_claim_tokens_df: DataFrame,
    cohort_df: DataFrame,
    drug_mapping_df: DataFrame,
    test_split_folder_path: Union[str, List[str]],
    output_path: str
) -> None:
    """
    Extract required tables from cohort dataframe
    """
    logging.info("%sExtracting required tables from cohort dataframe", INDENT1)

    logging.info("%sCreating dx_df (diagnosis data)", INDENT2)
    dx_df = (all_claim_tokens_df \
        .filter(F.col("category") == "DX")
        .filter(~(F.col("value").rlike(r"^SECONDARY|^PRINCIPAL")))
        .join(cohort_df, "enrollee_id", "inner")
        .select("enrollee_id", "timestamp", "value")
        .withColumn("value_formatted", format_icd10_code(F.col("value")))
        .withColumnRenamed("enrollee_id", "pid")
        .withColumnRenamed("timestamp", "dx_date")
        .withColumnRenamed("value_formatted", "code")
        .dropDuplicates())

    dx_df_count = dx_df.count()
    dx_df_pid = dx_df.select("pid").distinct().count()
    # save dx_df
    logging.info("%sSaving %s rows, %s unique patients to dx_df", INDENT2, dx_df_count, dx_df_pid)
    dx_df.write.mode("overwrite").parquet(os.path.join(output_path, "df_processed", "dx_df"))

    logging.info("%sCreating med_df (medication data)", INDENT2)
    med_df = (all_claim_tokens_df 
        .filter(F.col("category") == "RX")
        .withColumn("ndc_drug_code", F.regexp_extract(F.col("original_token"), r"<NDCNUM-(.+)>", 1))
        .join(cohort_df, on="enrollee_id", how="inner")
        .join(drug_mapping_df, on="ndc_drug_code", how="inner")
        .select("enrollee_id", "timestamp", "ndc_drug_code","ndc_drug_name")
        .withColumnRenamed("enrollee_id", "pid")
        .withColumnRenamed("timestamp", "start_date")
        .withColumnRenamed("ndc_drug_name", "medication_name")
        .dropDuplicates())

    med_df_count = med_df.count()
    med_df_pid = med_df.select("pid").distinct().count()
    # save med_df
    logging.info("%sSaving %s rows, %s unique patients to med_df", INDENT2, med_df_count, med_df_pid)
    med_df.write.mode("overwrite").parquet(os.path.join(output_path, "df_processed", "med_df"))

    logging.info("%sCreating core_df (visit data)", INDENT2)
    core_df = (all_claim_tokens_df \
        .filter(F.col("category") == "VT")
        .join(cohort_df, on="enrollee_id", how="inner")
        .select("enrollee_id", "timestamp", "value")
        .withColumnRenamed("enrollee_id", "pid")
        .withColumnRenamed("timestamp", "visit_date")
        .withColumn("patient_class", F.when(F.col("value") == "outpatient", "Outpatient")
                             .when(F.col("value") == "inpatient", "Inpatient")
                             .when(F.col("value") == "pharmacy", "Outpatient")
                             .otherwise("Unknown"))
        .dropDuplicates())

    # save core_df
    core_df_count = core_df.count()
    core_df_pid = core_df.select("pid").distinct().count()
    logging.info("%sSaving %s rows, %s unique patients to core_df", INDENT2, core_df_count, core_df_pid)
    core_df.write.mode("overwrite").parquet(os.path.join(output_path, "df_processed", "core_df"))

    logging.info("%sCreating demo_df (demographics data)", INDENT2)
    demo_tokens_df = read_and_union_parquet(spark, test_split_folder_path, 'processed_data/all_demographic_tokens')
    demo_df = (demo_tokens_df
        .join(cohort_df, on="enrollee_id", how="inner")
        .select("enrollee_id", "category", "value")
        .dropDuplicates()
        .groupBy("enrollee_id")
        .pivot("category")
        .agg(F.first(F.col("value")))
        .withColumn("birth_date", F.to_date(F.concat(F.col("DOBYR").cast("string"), F.lit("-01-01")), "yyyy-MM-dd"))
        .withColumnRenamed("enrollee_id", "pid")
        .withColumn("gender", F.when(F.col("SEX") == "1", "Male")
                             .when(F.col("SEX") == "2", "Female")
                             .otherwise("Unknown"))
        .select("pid", "birth_date", "gender")
        .dropDuplicates())

    # save demo_df
    demo_df_count = demo_df.count()
    demo_df_pid = demo_df.select("pid").distinct().count()
    logging.info("%sSaving %s rows, %s unique patients to demo_df", INDENT2, demo_df_count, demo_df_pid)
    demo_df.write.mode("overwrite").parquet(os.path.join(output_path, "df_processed", "demo_df"))






def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("COHORT EXTRACTION STARTING...")
    
    # Parse arguments
    args = parse_args()
    
    # Create paths
    working_dir = os.getcwd()
    test_split_folder_path = resolve_marketscan_split_path(args)
    intermediate_result_path = os.path.join(working_dir, 'intermediate', f'v{args.version}')
    os.makedirs(intermediate_result_path, exist_ok=True)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sWorking directory: %s", INDENT1, working_dir)
    logging.info("%sTest split folder: %s", INDENT1, test_split_folder_path)
    
    # Create Spark session in local mode
    spark = create_spark_session("cohort_extraction")
    logging.info("%sSpark session created in local mode", INDENT1)
    
    try:
        # =====================================================================
        # Step 1: Drug Name to Token Mapping
        # =====================================================================
        logging.info("********** 1 Load and preprocess drug mapping data **********")
        drug_mapping_df = load_and_preprocess_drug_mapping(spark, args.drug_mapping_file)
        
        logging.info("********** 2 Create drug class flags **********")
        drug_mapping_flagged = create_drug_class_flags(drug_mapping_df)
        
        # =====================================================================
        # Step 2: Cohort Extraction
        # =====================================================================
        logging.info("********** 3 Load test claim tokens **********")
        all_claim_tokens_df = load_test_tokens( spark, test_split_folder_path)
        logging.info("********** 4 Extract and save cohort **********")
        cohort_df = extract_and_save_cohort(all_claim_tokens_df, drug_mapping_flagged, intermediate_result_path)
        logging.info("********** 5 Extract and save required tables **********")
        extract_and_save_required_tables(spark, all_claim_tokens_df, cohort_df, drug_mapping_df, test_split_folder_path, intermediate_result_path)
        
        logging.info("COHORT EXTRACTION COMPLETED SUCCESSFULLY...")
        all_claim_tokens_df.unpersist()
        logging.info("Log file: %s", log_file)
        
    except Exception as e:
        logging.error("Script failed with error: %s", str(e))
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped")


if __name__ == "__main__":
    main()
