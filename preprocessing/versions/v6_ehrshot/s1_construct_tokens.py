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
    parser.add_argument("--omop_dir", type=str, 
                       default=os.path.join(os.environ.get("EHRSHOT_DATA_ROOT", "/path/to/ehrshot_data_root"), "ehrshot_omop"),
                       help='Base directory containing EHRSHOT Raw data')
    parser.add_argument("--mapping_dir", type=str, 
                       default='../../vocab_mapping/mapping',
                       help='Directory containing the mapping files')
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


####################################
## Enrollment period processing ##
####################################

@time_execution
def construct_demographic_and_enrollment_tokens(spark: SparkSession, omop_dir: str, external_val_data_dir: str, excluded_personids_df: DataFrame) -> DataFrame:
    """
    Aggregate enrollment detail table to get the continuous enrollment period for the specified payer type.
    
    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        external_val_data_dir: Directory containing the external validation data
        excluded_personids_df: DataFrame of person IDs column containing person IDs to exclude (uses anti-join)

    Returns:
        demographic_tokens: DataFrame containing the demographic tokens
        enrollment_tokens: DataFrame containing the enrollment tokens

    """
    # ************* Demographic ************* #
    logging.info("%sExtracting demographic information", INDENT2)
    # DOBYR can't be missing by design since it is used to anchor temporal information
    # Using window function to ensure deterministic selection of first record
    # Order by enrollment_start_date, then DOBYR and SEX for tie-breaking determinism
    person_df = spark.read.parquet(os.path.join(omop_dir, "person"))
    demographic_base = person_df \
        .filter(F.col("year_of_birth").isNotNull()) \
        .withColumn("DOBYR", F.col("year_of_birth").cast("string")) \
        .withColumn("SEX", F.when(F.col("gender_concept_id") == 8507, "1") \
            .when(F.col("gender_concept_id") == 8532, "2") \
            .otherwise("MISSING")) \
        .select("person_id", "DOBYR", "SEX") \
        .distinct()
    
    # Create demographic tokens (no mapping/hierarchical splitting, so no original_token/comb_order needed)
    # SEX token
    sex_tokens = demographic_base \
        .withColumn("category", F.lit("SEX")) \
        .withColumn("value", F.col("SEX")) \
        .withColumn("token", F.concat(F.lit("<SEX-"), F.col("SEX"), F.lit(">"))) \
        .withColumn("sub_order", F.lit(0)) \
        .select("person_id", "category", "value", "token", "sub_order")
    
    # DOBYR token
    dobyr_tokens = demographic_base \
        .withColumn("category", F.lit("DOBYR")) \
        .withColumn("value", F.col("DOBYR")) \
        .withColumn("token", F.concat(F.lit("<DOBYR-"), F.col("DOBYR"), F.lit(">"))) \
        .withColumn("sub_order", F.lit(1)) \
        .select("person_id", "category", "value", "token", "sub_order")
    
    # Union DOBYR and SEX tokens
    demographic_tokens = dobyr_tokens.union(sex_tokens)

    # ************* Enrollment ************* #

    logging.info("%sConstructing enrollment tokens", INDENT2)

    enrollment_period_collapsed_df = spark.read.parquet(os.path.join(external_val_data_dir, "ehrshot_payer_plan_period_collapsed"))

    # check enrollment number per person_id
    enrollment_number_per_person_id = enrollment_period_collapsed_df \
        .groupBy("person_id") \
        .agg(
            F.count("enrollment_start_date").alias("enrollment_number"),
            F.mean("enrollment_duration_days").alias("enrollment_duration_days_mean"),
            F.sum("plans_merged").alias("plans_merged_sum"),
        )
    # check distribution of enrollment number per person_id
    enrollment_number_per_person_id.select("enrollment_number", "enrollment_duration_days_mean", "plans_merged_sum").describe().show()

    
    enrollment_period_df = enrollment_period_collapsed_df \
        .join(excluded_personids_df, on="person_id", how = "left_anti") \
        .withColumn("enrollment_start_date", F.trunc(F.col("enrollment_start_date"), "month")) \
        .withColumn("CAP", 
            F.when(F.col("CAP").isNull() | (F.col("CAP").cast("string") == ''), "MISSING")
                .otherwise(F.col("CAP").cast("string"))) \
        .withColumn("EGEOLOC", F.lit("62")) \
        .withColumn("PLANTYP", 
            F.when(F.col("PLANTYP").isNull() | (F.col("PLANTYP").cast("string") == ''), "MISSING")
                .otherwise(F.col("PLANTYP").cast("string")))

    selected_plan_variables = ['PLANTYP','CAP','EGEOLOC']

    logging.info("%sConstructing enrollment start tokens", INDENT2)
    enrollment_start_event_type = 1 # 0-indexed, 1st order among event types if in same timestamp 
    enrollment_end_event_type = 5 # 0-indexed, 5th order among event types if in same timestamp 
    id_cols = ["person_id", "timestamp", "event_type"]

    # construct enrollment start tokens
    # ERLST token: <ERLST-CCAE>, <ERLST-MDCR>, <ERLST-MDCD>
    # Plan variable tokens: <PLANTYP-X>, <CAP-X>, <EGEOLOC-X>
    # Use week-based timestamp (first day of the week), keep original_date for sorting
    enrollment_start_tokens = enrollment_period_df \
        .withColumn("original_date", F.col("enrollment_start_date")) \
        .withColumn("timestamp", F.trunc(F.col("enrollment_start_date"), "month")) \
        .withColumn("event_type", F.lit(enrollment_start_event_type)) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["PAYER"] + selected_plan_variables,
            variableColumnName="category",
            valueColumnName="value") \
        .withColumn("value",
            # For PAYER rows: keep original payer value (CCAE/MDCR/MDCD)
            F.when(F.col("category") == "PAYER", F.col("value")).otherwise(F.col("value"))) \
        .withColumn("category",
            # For PAYER rows: change category to ERLST
            F.when(F.col("category") == "PAYER", F.lit("ERLST")).otherwise(F.col("category"))) \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    # construct enrollment end tokens
    # ERLED token: <ERLED-CCAE>, <ERLED-MDCR>, <ERLED-MDCD>
    # Use week-based timestamp (first day of the week), keep original_date for sorting
    enrollment_end_tokens = enrollment_period_df \
        .withColumn("original_date", F.col("enrollment_end_date")) \
        .withColumn("timestamp", F.trunc(F.col("enrollment_end_date"), "month")) \
        .withColumn("event_type", F.lit(enrollment_end_event_type)) \
        .withColumn("category", F.lit("ERLED")) \
        .withColumn("value", F.col("PAYER")) \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    # union the enrollment start and end tokens
    enrollment_tokens = enrollment_start_tokens.union(enrollment_end_tokens)

    return demographic_tokens, enrollment_tokens



def format_and_split_ICD10_token(code_col: Column, nomap_value: str) -> Column:
    """
    Format and split raw ICD-10 codes into MAJOR, MINOR, and SUFFIX component tokens.
    
    Takes unformatted ICD-10 codes (without decimal) and outputs space-separated tokens.
    MAJOR = first 3 characters (category code)
    MINOR = characters 4+ excluding trailing letters (subcategory digits)
    SUFFIX = trailing 1-3 uppercase letters (extension)
    
    Args:
        code_col: Spark Column containing raw ICD-10 codes without decimal (e.g., 'E1165', 'S72001A')
        nomap_value: The NOMAP value string to preserve as-is
    
    Returns:
        Column: Space-separated string of component tokens
    
    Examples:
        'E1165'    -> '<DX-MAJOR_E11> <DX-MINOR_65>'
        'S72001A'  -> '<DX-MAJOR_S72> <DX-MINOR_001> <DX-SUFFIX_A>'
        'A01234B'  -> '<DX-MAJOR_A01> <DX-MINOR_234> <DX-SUFFIX_B>'
        'J06'      -> '<DX-MAJOR_J06>'
        'E11'      -> '<DX-MAJOR_E11>'
        'NOMAP'    -> '<DX-NOMAP>'
        NULL/''    -> '<DX-MISSING>'
    """
    # Trim leading and trailing whitespace
    code = F.trim(code_col)
    
    # Extract MAJOR: first 3 characters
    major = F.substring(code, 1, 3)
    
    # Extract REST: everything after first 3 characters (remove first 3 chars)
    rest = F.regexp_replace(code, r"^.{3}", "")
    
    # Extract SUFFIX: trailing 1-3 uppercase letters from REST
    suffix = F.regexp_extract(rest, r"([A-Z]{1,3})$", 1)
    
    # Extract MINOR: REST minus trailing letters
    minor = F.regexp_replace(rest, r"[A-Z]{1,3}$", "")
    
    # Build component tokens (null if component is empty - concat_ws will skip nulls)
    major_token = F.concat(F.lit("<DX-MAJOR_"), major, F.lit(">"))
    minor_token = F.when(F.length(minor) > 0, F.concat(F.lit("<DX-MINOR_"), minor, F.lit(">")))
    suffix_token = F.when(F.length(suffix) > 0, F.concat(F.lit("<DX-SUFFIX_"), suffix, F.lit(">")))
    
    # Combine tokens - concat_ws naturally skips null values
    result = F.when(code.isNull() | (code == ""), F.lit("<DX-MISSING>")) \
        .when(code == nomap_value, F.lit("<DX-NOMAP>")) \
        .otherwise(F.concat_ws(" ", major_token, minor_token, suffix_token))
    
    return result


def add_principal_secondary_markers(df: DataFrame, id_cols: list, category: str) -> DataFrame:
    """
    Add marker tokens to differentiate principal from secondary diagnoses/procedures.
    Principal codes (from DX1/PROC1) get <category-PRINCIPAL> marker.
    Secondary codes (from DX2+/PROC2+) get <category-SECONDARY> marker.
    
    Input: is_principal (boolean) - True for principal (DX1/PROC1), False for secondary (DX2+/PROC2+)
    
    Output: sub_order (int) for token sorting (only meaningful for DX and PROC categories):
        - 0 = PRINCIPAL marker token
        - 1 = principal diagnosis/procedure codes (from DX1/PROC1)
        - 2 = SECONDARY marker token
        - 3 = secondary diagnosis/procedure codes (from DX2+/PROC2+)
    
    Args:
        df: DataFrame with columns: *id_cols, original_date, is_principal (boolean), 
            category, value, token, original_token, comb_order
        id_cols: List of ID column names (should NOT include original_date)
        category: Either "DX" or "PROC"
    
    Returns:
        DataFrame with columns: *id_cols, original_date, category, value, token, original_token, comb_order, sub_order
    
    Note: id_cols should be ["enrollee_id", "timestamp"] - NOT including original_date.
          The function expects original_date as a separate column and will use the earliest
          original_date for the marker tokens to ensure only ONE marker per (enrollee_id, timestamp).
    """
    
    # Convert is_principal to sub_order: True -> 1, False -> 3
    df_adjusted = df \
        .withColumn("sub_order", 
            F.when(F.col("is_principal"), F.lit(1)).otherwise(F.lit(3)))
    
    # Create principal marker tokens (ONE per enrollee_id + timestamp, using earliest original_date)
    # sub_order=0 for PRINCIPAL marker
    # Group by id_cols only (not original_date) to ensure single marker per timestamp
    principal_markers = df_adjusted \
        .filter(F.col("is_principal")) \
        .groupBy(*id_cols) \
        .agg(F.min("original_date").alias("original_date")) \
        .withColumn("category", F.lit(category)) \
        .withColumn("value", F.lit("PRINCIPAL")) \
        .withColumn("token", F.lit(f"<{category}-PRINCIPAL>")) \
        .withColumn("original_token", F.lit(f"<{category}-PRINCIPAL>")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0))
    
    # Create secondary marker tokens (ONE per enrollee_id + timestamp, using earliest original_date)
    # sub_order=2 for SECONDARY marker
    secondary_markers = df_adjusted \
        .filter(~F.col("is_principal")) \
        .groupBy(*id_cols) \
        .agg(F.min("original_date").alias("original_date")) \
        .withColumn("category", F.lit(category)) \
        .withColumn("value", F.lit("SECONDARY")) \
        .withColumn("token", F.lit(f"<{category}-SECONDARY>")) \
        .withColumn("original_token", F.lit(f"<{category}-SECONDARY>")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(2))
    
    # Union all and drop is_principal (replaced by sub_order)
    result = df_adjusted \
        .drop("is_principal") \
        .unionByName(principal_markers) \
        .unionByName(secondary_markers)
    
    return result


def apply_token_mapping(spark: SparkSession, df_long: DataFrame, category: str, mapping_dir: str, id_cols: list) -> DataFrame:
    """
    Apply token level mapping for Procedure, Drug and ICD codes. 
        - For ICD codes,it's mostly one-to-one mapping, so we can apply the mapping directly.
        - For Procedure and Drug codes, it contains hierarchical mapping, so we need to explode the token and update the combination order.
    Args:
        df_long: DataFrame with tokens in long format
        category: Type of mapping (e.g., 'PROC', 'RX', 'DX')
        mapping_dir: Directory containing the mapping files
        id_cols: ID columns
    WHEN 
    Returns:
        DataFrame: DataFrame with mapped tokens and combination order
    """

    logging.info("%sApplying hierarchical token mapping for %s", INDENT2, category)
    if category == 'DX':
        mapping_df = spark.read.csv(os.path.join(mapping_dir, "icd9_to_icd10_mapping.csv"), header=True) 
        df_to_map = df_long 
        nomap_token = "<DX-NOMAP>"
        nomap_value = "NOMAP"
    
    elif category == 'PROC':
        mapping_df = spark.read.csv(os.path.join(mapping_dir, "proc_to_snomed_mapping.csv"), header=True) 
        df_to_map = df_long 
        nomap_token = "<PROC-NOMAP>"
        nomap_value = "NOMAP"

    elif category == 'RX':
        mapping_df = spark.read.csv(os.path.join(mapping_dir, "rxnorm_drug_to_rxnorm_ingredient_mapping.csv"), header=True)
        df_to_map = df_long
        nomap_token = "<RX-NOMAP>"
        nomap_value = "NOMAP"
    else:
        raise ValueError(f"Invalid category: {category}")

    # Apply mapping
    mapping_var = ['source_token', 'target_token', 'target_value', 'source_vocab']
    mapped_var = ['category', 'original_token', 'mapped_value', 'mapped_token', 'source_vocab']


    df_mapped = df_to_map.join(
        F.broadcast(mapping_df.select(*mapping_var)), 
        df_to_map.token == mapping_df.source_token, 
        "left")
    
    # For DX category, consider DXVER and target_token to determine mapping:
    # DXVER == "0": original is ICD10, always keep original value/token
    # DXVER == "9": original is ICD9, use mapped value if available, otherwise keep original
    # DXVER is NULL: infer from mapping - if mapping found assume ICD9, otherwise assume ICD10
    # 
    # Cases:
    # 1. DXVER == "0": ICD10, keep original, source_vocab = "ICD10"
    # 2. DXVER == "9" and target NOT NULL: ICD9 mapped, use target, source_vocab = "ICD9"
    # 3. DXVER == "9" and target is NULL: ICD9 not in mapping, keep original, source_vocab = "ICD9"
    # 4. DXVER is NULL and target NOT NULL: assume ICD9, use target, source_vocab = "ICD9"
    # 5. DXVER is NULL and target is NULL: assume ICD10, keep original, source_vocab = "ICD10"
    if category == 'DX':
        df_mapped = df_mapped \
            .withColumn("mapped_value", 
                F.when(F.col("DXVER") == "0", F.col("value"))  # Case 1: ICD10, keep original
                .when((F.col("DXVER") == "9") & F.col("target_token").isNotNull(), F.col("target_value"))  # Case 2: ICD9 with mapping
                .when((F.col("DXVER") == "9") & F.col("target_token").isNull(), F.col("value"))  # Case 3: ICD9 no mapping, keep original
                .when(F.col("DXVER").isNull() & F.col("target_token").isNotNull(), F.col("target_value"))  # Case 4: NULL DXVER, has mapping
                .when(F.col("DXVER").isNull() & F.col("target_token").isNull(), F.col("value"))  # Case 5: NULL DXVER, no mapping (infer ICD9)
                .otherwise(nomap_value)) \
            .withColumn("source_vocab", 
                F.when(F.col("DXVER") == "0", F.lit("ICD10CM"))  # Case 1: ICD10
                .when(F.col("DXVER") == "9", F.lit("ICD9CM"))  # Cases 2 & 3: ICD9 (regardless of mapping success)
                .when(F.col("DXVER").isNull() & F.col("target_token").isNotNull(), F.col("source_vocab"))  # Case 4: infer ICD9 from mapping
                .when(F.col("DXVER").isNull() & F.col("target_token").isNull(), F.lit("ICD9CM"))  # Case 5: infer ICD9 (no mapping)
                .otherwise(F.col("source_vocab"))) \
            .withColumn("mapped_token", format_and_split_ICD10_token(F.col("mapped_value"), nomap_value)) \
            .withColumn("original_token", F.col("mapped_token"))

        # log the formatted ICD10CM token distribution
        # logging.info("%sLogging formatted ICD10CM token distribution", INDENT3)
        # spark_log_contingency_table(df_mapped, "mapped_token", top_n=100)
    else:
        df_mapped = df_mapped \
            .withColumn("mapped_value", F.when(F.col("target_value").isNotNull(), F.col("target_value")).otherwise(nomap_value)) \
            .withColumn("mapped_token", F.when(F.col("target_token").isNotNull(), F.col("target_token")).otherwise(nomap_token))
    
    df_mapped = df_mapped \
        .select(*id_cols, *mapped_var) \
        .distinct()

    # log the mapping rate
    source_token_count = df_mapped.count()
    nomap_count = df_mapped.filter(F.col("mapped_token") == nomap_token).count()
    if source_token_count > 0:
        nomap_rate = (nomap_count / source_token_count) * 100
        mapping_rate = 100 - nomap_rate
        logging.info("%s%.2f%% mapped successfully; %.2f%% not mapped", INDENT1, mapping_rate, nomap_rate)

    
    # some extra logging for ICD DXVER distribution
    if category == 'DX':
        logging.info("%sExtra diagnostic logging for DXVER distribution", INDENT3)
        total_count = df_mapped.count()
        
        # Count by DXVER value
        dxver_0_count = df_mapped.filter(F.col("DXVER") == "0").count()
        dxver_9_count = df_mapped.filter(F.col("DXVER") == "9").count()
        dxver_other_count = total_count - dxver_0_count - dxver_9_count
        
        logging.info("%sDXVER=0 (ICD10): %s (%.2f%%)", INDENT3, f"{dxver_0_count:,}", (dxver_0_count / total_count * 100) if total_count > 0 else 0)
        logging.info("%sDXVER=9 (ICD9): %s (%.2f%%)", INDENT3, f"{dxver_9_count:,}", (dxver_9_count / total_count * 100) if total_count > 0 else 0)
        logging.info("%sDXVER=other/null: %s (%.2f%%)", INDENT3, f"{dxver_other_count:,}", (dxver_other_count / total_count * 100) if total_count > 0 else 0)
    
    
    # log the source vocab
    logging.info("%sCheck the distribution of the source vocab", INDENT2)
    spark_log_contingency_table(df_mapped, "source_vocab")

    # Explode the token column and update combination order for mapped data
    df_mapped_final = df_mapped \
        .drop("comb_order") \
        .withColumn("token_array", F.split(F.trim("mapped_token"), "\\s+")) \
        .select("*", F.posexplode("token_array").alias("comb_order", "token")) \
        .withColumnRenamed("mapped_value", "value") \
        .select(*id_cols, "category", "original_token", "value", "token", "comb_order")

    return df_mapped_final




@time_execution
def construct_inpatient_visit_tokens(spark: SparkSession, omop_dir: str, mapping_dir: str, excluded_personids_df: DataFrame) -> DataFrame:
    """
    Processing inpatient visit tokens from OMOP data.
    Combines VT/COST/LS/DS tokens with DX and PROC tokens for inpatient visits.
    
    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        mapping_dir: Directory containing the mapping files
        excluded_personids_df: DataFrame of person IDs to exclude (uses anti-join)
    Returns:
        DataFrame containing the inpatient visit tokens
    """
    logging.info("%sConstructing inpatient visit tokens", INDENT2)
    inpatient_event_type = 4  # 0-indexed, 4th order among event types if in same timestamp
    
    # Read parquet files once
    logging.info("Reading visit occurrence data...")
    visit_occurrence = spark.read.parquet(f'{omop_dir}/visit_occurrence/*.parquet') \
        .filter((F.col("visit_start_DATE") >= F.lit("2008-01-01").cast("date")) & (F.col("visit_start_DATE") <= F.lit("2025-03-31").cast("date")))
    logging.info("Reading condition occurrence data...")
    condition_occurrence = spark.read.parquet(f'{omop_dir}/condition_occurrence/*.parquet')
    logging.info("Reading procedure occurrence data...")
    procedure_occurrence = spark.read.parquet(f'{omop_dir}/procedure_occurrence/*.parquet')
    logging.info("Reading concept data...")
    concept = spark.read.parquet(f'{omop_dir}/concept/*.parquet')

    # Create inpatient visits DataFrame with transformations
    # Filter for inpatient visits only: visit_concept_id in [262, 9201]
    # Exclude person_ids from excluded_personids_df
    logging.info("Processing inpatient visits data...")
    inpatient_visits = visit_occurrence \
        .join(excluded_personids_df, on="person_id", how="left_anti") \
        .filter(F.col('visit_concept_id').isin([262, 9201])) \
        .withColumn('original_date', F.col('visit_start_DATE')) \
        .withColumn('timestamp', F.trunc(F.col('visit_start_DATE'), 'month')) \
        .withColumn('event_type', F.lit(inpatient_event_type)) \
        .withColumn('VT', F.lit('inpatient')) \
        .withColumn('DAYS', F.datediff(F.col('visit_end_DATE'), F.col('visit_start_DATE'))) \
        .withColumn('LS', F.when(F.col('DAYS').isNull(), F.lit('MISSING')).when(F.col('DAYS') < 7, F.lit('0')).otherwise(F.lit('1'))) \
        .withColumn('DS', F.lit('MISSING')) \
        .select(
            'person_id',
            'visit_occurrence_id',
            'original_date',
            'timestamp',
            'event_type',
            'visit_start_DATE',
            'visit_end_DATE',
            'visit_concept_id',
            'DAYS',
            'VT',
            'LS',
            'DS'
        )

    # define id columns
    id_cols = ["person_id", "timestamp", "event_type"]

    logging.info("%sProcessing tokens: VT/COST aggregated, LS/DS keep all distinct values per month", INDENT3)
    
    # VT and COST: aggregated (VT first-by-date, COST as MISSING, one token per timestamp)
    # Keep min original_date for sorting within timestamp
    vt_cost_agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT")  # first by date
    ]
    
    inpatient_vt_cost_tokens = inpatient_visits.groupBy(*id_cols) \
        .agg(*vt_cost_agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", F.lit("MISSING")) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["VT", "COST"],
            variableColumnName="category",
            valueColumnName="value") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()
    
    # LS and DS: keep all distinct values per month (unpivot like PROC/DX approach)
    # Keep first original_date for each unique (LS, DS) value for sorting
    inpatient_ls_ds_long = inpatient_visits \
        .select(*id_cols, "original_date", "LS", "DS") \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["LS", "DS"],
            variableColumnName="category",
            valueColumnName="value")
    
    # Deduplicate keeping first occurrence by original_date
    ls_ds_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    inpatient_ls_ds_tokens = inpatient_ls_ds_long \
        .withColumn("row_num", F.row_number().over(ls_ds_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")
    
    # Union VT/COST tokens with LS/DS tokens
    inpatient_agg_tokens = inpatient_vt_cost_tokens \
        .union(inpatient_ls_ds_tokens) \
        .distinct()

    logging.info("%sProcessing tokens to be mapped/standardized and deduplicated: PROC, DX", INDENT3)

    # ==================== PROC tokens ====================
    logging.info("Joining inpatient visits with procedures and concepts...")
    visit_procedure = inpatient_visits \
        .join(
            procedure_occurrence,
            (inpatient_visits['person_id'] == procedure_occurrence['person_id']) & 
            (inpatient_visits['visit_occurrence_id'] == procedure_occurrence['visit_occurrence_id']),
            'inner') \
        .join(
            concept,
            F.col('procedure_source_concept_id') == concept['concept_id'],
            'inner') \
        .filter(F.col('vocabulary_id').isin(['CPT4', 'ICD10PCS', 'ICD9CM'])) \
        .withColumn('category', F.lit('PROC')) \
        .withColumn('value', F.col('concept_code')) \
        .filter(F.col('value').isNotNull() & (F.col('value') != "")) \
        .select(
            inpatient_visits['person_id'],
            inpatient_visits['timestamp'],
            inpatient_visits['event_type'],
            inpatient_visits['original_date'],
            'category',
            'value'
        )

    # Build PROC tokens
    # Since EHRSHOT doesn't have a principal indicator for procedures, we mark all as principal
    logging.info("Building PROC tokens...")
    # Deduplicate keeping first occurrence by original_date
    proc_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    inpatient_proc_long = visit_procedure \
        .withColumn("is_principal", F.lit(True)) \
        .withColumn("row_num", F.row_number().over(proc_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn('token', F.concat(F.lit('<'), F.col('category'), F.lit('-'), F.col('value'), F.lit('>'))) \
        .withColumn('original_token', F.col('token')) \
        .withColumn('comb_order', F.lit(0))

    # Apply token mapping for procedure (include is_principal and original_date in id_cols to preserve through mapping)
    proc_id_cols = id_cols + ["original_date", "is_principal"]
    inpatient_proc_mapped = apply_token_mapping(spark, inpatient_proc_long, "PROC", mapping_dir, proc_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    inpatient_proc_tokens = add_principal_secondary_markers(inpatient_proc_mapped, id_cols, "PROC") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")

    # ==================== DX tokens ====================
    logging.info("Joining inpatient visits with conditions and concepts...")
    visit_condition = inpatient_visits \
        .join(
            condition_occurrence,
            (inpatient_visits['person_id'] == condition_occurrence['person_id']) & 
            (inpatient_visits['visit_occurrence_id'] == condition_occurrence['visit_occurrence_id']),
            'inner') \
        .join(
            concept,
            F.col('condition_source_concept_id') == concept['concept_id'],
            'inner') \
        .filter(F.col('vocabulary_id').isin(['ICD9CM', 'ICD10CM'])) \
        .withColumn('is_principal',
            F.when(F.col('condition_status_concept_id') == 32902, F.lit(True)).otherwise(F.lit(False))) \
        .withColumn('category', F.lit('DX')) \
        .withColumn('DXVER', F.when(F.col('vocabulary_id') == 'ICD9CM', F.lit('9')).when(F.col('vocabulary_id') == 'ICD10CM', F.lit('0')).otherwise(F.lit(None))) \
        .withColumn('value', F.regexp_replace(F.col('concept_code'), r'\.', '')) \
        .select(
            inpatient_visits['person_id'],
            inpatient_visits['timestamp'],
            inpatient_visits['event_type'],
            inpatient_visits['original_date'],
            'is_principal',
            'category',
            'DXVER',
            'value'
        )

    # Build DX tokens
    logging.info("Building DX tokens...")
    # Deduplicate keeping first occurrence by original_date
    dx_dedup_window = Window.partitionBy(*id_cols, "DXVER", "is_principal", "category", "value").orderBy("original_date")
    inpatient_dx_long = visit_condition \
        .withColumn("row_num", F.row_number().over(dx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn('token', F.concat(F.lit('<'), F.col('category'), F.lit('-'), F.col('value'), F.lit('>'))) \
        .withColumn('original_token', F.col('token')) \
        .withColumn('comb_order', F.lit(0))

    # Apply token mapping for DX (DXVER needed for mapping logic, is_principal and original_date for tracking)
    dx_id_cols = id_cols + ["original_date", "DXVER", "is_principal"]
    inpatient_dx_mapped = apply_token_mapping(spark, inpatient_dx_long, "DX", mapping_dir, dx_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    inpatient_dx_tokens = add_principal_secondary_markers(inpatient_dx_mapped, id_cols, "DX") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    # ==================== Merge all tokens ====================
    # Ordering columns (for DX/PROC categories):
    #   - original_date: for sorting within same timestamp (month)
    #   - sub_order: 0=PRINCIPAL marker, 1=principal codes, 2=SECONDARY marker, 3=secondary codes
    #   - comb_order: hierarchical token order within each sub_order group
    # For other categories (VT, COST, LS, DS), sub_order=0 (placeholder, not used for sorting)
    inpatient_tokens = inpatient_agg_tokens.union(inpatient_proc_tokens).union(inpatient_dx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    logging.info("Inpatient visit tokens construction completed!")
    return inpatient_tokens


@time_execution
def construct_outpatient_visit_tokens(spark: SparkSession, omop_dir: str, mapping_dir: str, excluded_personids_df: DataFrame) -> DataFrame:
    """
    Processing outpatient visit tokens from OMOP data.
    Combines VT/COST tokens with DX and PROC tokens for outpatient visits.
    
    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        mapping_dir: Directory containing the mapping files
        excluded_personids_df: DataFrame of person IDs to exclude (uses anti-join)
    Returns:
        DataFrame containing the outpatient visit tokens
    """
    logging.info("%sConstructing outpatient visit tokens", INDENT2)
    outpatient_event_type = 2  # 0-indexed, 2nd order among event types if in same timestamp
    
    # Read parquet files once
    logging.info("Reading visit occurrence data...")
    visit_occurrence = spark.read.parquet(f'{omop_dir}/visit_occurrence/*.parquet') \
        .filter((F.col("visit_start_DATE") >= F.lit("2008-01-01").cast("date")) & (F.col("visit_start_DATE") <= F.lit("2025-03-31").cast("date")))
    logging.info("Reading condition occurrence data...")
    condition_occurrence = spark.read.parquet(f'{omop_dir}/condition_occurrence/*.parquet')
    logging.info("Reading procedure occurrence data...")
    procedure_occurrence = spark.read.parquet(f'{omop_dir}/procedure_occurrence/*.parquet')
    logging.info("Reading concept data...")
    concept = spark.read.parquet(f'{omop_dir}/concept/*.parquet')

    # Create outpatient visits DataFrame with transformations
    # Filter for outpatient visits only: visit_concept_id in [9202, 581477, 9203]
    # Exclude person_ids from excluded_personids_df
    logging.info("Processing outpatient visits data...")
    outpatient_visits = visit_occurrence \
        .join(excluded_personids_df, on="person_id", how="left_anti") \
        .filter(F.col('visit_concept_id').isin([9202, 581477, 9203])) \
        .withColumn('original_date', F.col('visit_start_DATE')) \
        .withColumn('timestamp', F.trunc(F.col('visit_start_DATE'), 'month')) \
        .withColumn('event_type', F.lit(outpatient_event_type)) \
        .withColumn('VT', F.lit('outpatient')) \
        .select(
            'person_id',
            'visit_occurrence_id',
            'original_date',
            'timestamp',
            'event_type',
            'visit_start_DATE',
            'visit_end_DATE',
            'visit_concept_id',
            'VT'
        )

    # define id columns
    id_cols = ["person_id", "timestamp", "event_type"]

    logging.info("%sProcessing tokens to be aggregated: VT, COST", INDENT3)
    
    # VT and COST: aggregated (VT first-by-date, COST as MISSING, one token per timestamp)
    # Keep min original_date for sorting within timestamp
    vt_cost_agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT")  # first by date
    ]
    
    outpatient_vt_cost_tokens = outpatient_visits.groupBy(*id_cols) \
        .agg(*vt_cost_agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", F.lit("MISSING")) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["VT", "COST"],
            variableColumnName="category",
            valueColumnName="value") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    logging.info("%sProcessing tokens to be mapped/standardized and deduplicated: PROC, DX", INDENT3)

    # ==================== PROC tokens ====================
    logging.info("Joining outpatient visits with procedures and concepts...")
    visit_procedure = outpatient_visits \
        .join(
            procedure_occurrence,
            (outpatient_visits['person_id'] == procedure_occurrence['person_id']) & 
            (outpatient_visits['visit_occurrence_id'] == procedure_occurrence['visit_occurrence_id']),
            'inner') \
        .join(
            concept,
            F.col('procedure_source_concept_id') == concept['concept_id'],
            'inner') \
        .filter(F.col('vocabulary_id').isin(['CPT4', 'ICD10PCS', 'ICD9CM'])) \
        .withColumn('category', F.lit('PROC')) \
        .withColumn('value', F.col('concept_code')) \
        .filter(F.col('value').isNotNull() & (F.col('value') != "")) \
        .select(
            outpatient_visits['person_id'],
            outpatient_visits['timestamp'],
            outpatient_visits['event_type'],
            outpatient_visits['original_date'],
            'category',
            'value'
        )

    # Build PROC tokens
    # Since EHRSHOT doesn't have a principal indicator for procedures, we mark all as principal
    logging.info("Building PROC tokens...")
    # Deduplicate keeping first occurrence by original_date
    proc_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    outpatient_proc_long = visit_procedure \
        .withColumn("is_principal", F.lit(True)) \
        .withColumn("row_num", F.row_number().over(proc_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn('token', F.concat(F.lit('<'), F.col('category'), F.lit('-'), F.col('value'), F.lit('>'))) \
        .withColumn('original_token', F.col('token')) \
        .withColumn('comb_order', F.lit(0))

    # Apply token mapping for procedure (include is_principal and original_date in id_cols to preserve through mapping)
    proc_id_cols = id_cols + ["original_date", "is_principal"]
    outpatient_proc_mapped = apply_token_mapping(spark, outpatient_proc_long, "PROC", mapping_dir, proc_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    outpatient_proc_tokens = add_principal_secondary_markers(outpatient_proc_mapped, id_cols, "PROC") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")

    # ==================== DX tokens ====================
    logging.info("Joining outpatient visits with conditions and concepts...")
    visit_condition = outpatient_visits \
        .join(
            condition_occurrence,
            (outpatient_visits['person_id'] == condition_occurrence['person_id']) & 
            (outpatient_visits['visit_occurrence_id'] == condition_occurrence['visit_occurrence_id']),
            'inner') \
        .join(
            concept,
            F.col('condition_source_concept_id') == concept['concept_id'],
            'inner') \
        .filter(F.col('vocabulary_id').isin(['ICD9CM', 'ICD10CM'])) \
        .withColumn('is_principal',
            F.when(F.col('condition_status_concept_id') == 32902, F.lit(True)).otherwise(F.lit(False))) \
        .withColumn('category', F.lit('DX')) \
        .withColumn('DXVER', F.when(F.col('vocabulary_id') == 'ICD9CM', F.lit('9')).when(F.col('vocabulary_id') == 'ICD10CM', F.lit('0')).otherwise(F.lit(None))) \
        .withColumn('value', F.regexp_replace(F.col('concept_code'), r'\.', '')) \
        .select(
            outpatient_visits['person_id'],
            outpatient_visits['timestamp'],
            outpatient_visits['event_type'],
            outpatient_visits['original_date'],
            'is_principal',
            'category',
            'DXVER',
            'value'
        )

    # Build DX tokens
    logging.info("Building DX tokens...")
    # Deduplicate keeping first occurrence by original_date
    dx_dedup_window = Window.partitionBy(*id_cols, "DXVER", "is_principal", "category", "value").orderBy("original_date")
    outpatient_dx_long = visit_condition \
        .withColumn("row_num", F.row_number().over(dx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn('token', F.concat(F.lit('<'), F.col('category'), F.lit('-'), F.col('value'), F.lit('>'))) \
        .withColumn('original_token', F.col('token')) \
        .withColumn('comb_order', F.lit(0))

    # Apply token mapping for DX (DXVER needed for mapping logic, is_principal and original_date for tracking)
    dx_id_cols = id_cols + ["original_date", "DXVER", "is_principal"]
    outpatient_dx_mapped = apply_token_mapping(spark, outpatient_dx_long, "DX", mapping_dir, dx_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    outpatient_dx_tokens = add_principal_secondary_markers(outpatient_dx_mapped, id_cols, "DX") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    # ==================== Merge all tokens ====================
    # Ordering columns (for DX/PROC categories):
    #   - original_date: for sorting within same timestamp (month)
    #   - sub_order: 0=PRINCIPAL marker, 1=principal codes, 2=SECONDARY marker, 3=secondary codes
    #   - comb_order: hierarchical token order within each sub_order group
    # For other categories (VT, COST), sub_order=0 (placeholder, not used for sorting)
    outpatient_tokens = outpatient_vt_cost_tokens.union(outpatient_proc_tokens).union(outpatient_dx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    logging.info("Outpatient visit tokens construction completed!")
    return outpatient_tokens


@time_execution
def construct_pharmacy_visit_tokens(spark: SparkSession, omop_dir: str, mapping_dir: str, excluded_personids_df: DataFrame) -> DataFrame:
    """
    Processing pharmacy visit tokens from OMOP data.
    Combines VT/COST tokens with RX tokens for pharmacy visits.
    
    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        mapping_dir: Directory containing the mapping files
        excluded_personids_df: DataFrame of person IDs to exclude (uses anti-join)
    Returns:
        DataFrame containing the pharmacy visit tokens
    """
    logging.info("%sConstructing pharmacy visit tokens", INDENT2)
    pharmacy_event_type = 3  # 0-indexed, 3rd order among event types if in same timestamp
    
    # Read parquet files once
    logging.info("Reading visit occurrence data...")
    visit_occurrence = spark.read.parquet(f'{omop_dir}/visit_occurrence/*.parquet') \
        .filter((F.col("visit_start_DATE") >= F.lit("2008-01-01").cast("date")) & (F.col("visit_start_DATE") <= F.lit("2025-03-31").cast("date")))
    logging.info("Reading drug exposure data...")
    drug_exposure = spark.read.parquet(f'{omop_dir}/drug_exposure/*.parquet')
    logging.info("Reading concept data...")
    concept = spark.read.parquet(f'{omop_dir}/concept/*.parquet')

    # Create pharmacy visits DataFrame with transformations
    # Filter for pharmacy visits: visit_concept_id in [9202, 581477, 9203, 581458]
    # Exclude person_ids from excluded_personids_df
    # First join with drug_exposure to filter out visits without valid drugs (drug_source_concept_id is NULL or 0)
    logging.info("Processing pharmacy visits data...")
    logging.info("Joining with drug_exposure to filter visits with valid drugs...")

    pharmacy_visits = visit_occurrence \
        .join(excluded_personids_df, on="person_id", how="left_anti") \
        .filter(F.col('visit_concept_id').isin([9202, 581477, 9203, 581458])) \
        .join(
            drug_exposure.select('person_id', 'visit_occurrence_id', 'drug_source_concept_id', 'drug_source_value').distinct(),
            on=['person_id', 'visit_occurrence_id'],
            how='inner') \
        .filter(F.col('drug_source_concept_id').isNotNull() & (F.col('drug_source_concept_id') != 0)) \
        .withColumn('original_date', F.col('visit_start_DATE')) \
        .withColumn('timestamp', F.trunc(F.col('visit_start_DATE'), 'month')) \
        .withColumn('event_type', F.lit(pharmacy_event_type)) \
        .withColumn('VT', F.lit('pharmacy')) \
        .select(
            'person_id',
            'visit_occurrence_id',
            'original_date',
            'timestamp',
            'event_type',
            'visit_start_DATE',
            'visit_end_DATE',
            'visit_concept_id',
            'VT',
            'drug_source_concept_id',
            'drug_source_value'
        )

    # define id columns
    id_cols = ["person_id", "timestamp", "event_type"]

    logging.info("%sProcessing tokens to be aggregated: VT, COST", INDENT3)
    
    # VT and COST: aggregated (VT first-by-date, COST as MISSING, one token per timestamp)
    # Keep min original_date for sorting within timestamp
    vt_cost_agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT")  # first by date
    ]
    
    pharmacy_vt_cost_tokens = pharmacy_visits.groupBy(*id_cols) \
        .agg(*vt_cost_agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", F.lit("MISSING")) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["VT", "COST"],
            variableColumnName="category",
            valueColumnName="value") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    logging.info("%sProcessing tokens to be mapped/standardized and deduplicated: RX", INDENT3)

    # ==================== RX tokens ====================
    # pharmacy_visits already has drug info from the initial join, now filter by RxNorm vocabulary
    logging.info("Filtering pharmacy visits by RxNorm vocabulary...")
    visit_rx = pharmacy_visits \
        .join(
            concept,
            pharmacy_visits['drug_source_concept_id'] == concept['concept_id'],
            'inner') \
        .filter(F.col('vocabulary_id') == 'RxNorm') \
        .withColumn('category', F.lit('RX')) \
        .withColumn('value', F.col('concept_code')) \
        .filter(F.col('value').isNotNull() & (F.col('value') != "")) \
        .select(
            'person_id',
            'timestamp',
            'event_type',
            'original_date',
            'category',
            'value'
        )

    # Build RX tokens
    logging.info("Building RX tokens...")
    # Deduplicate keeping first occurrence by original_date
    rx_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    pharmacy_rx_long = visit_rx \
        .withColumn("row_num", F.row_number().over(rx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn('token', F.concat(F.lit('<'), F.col('category'), F.lit('-'), F.col('value'), F.lit('>'))) \
        .withColumn('original_token', F.col('token')) \
        .withColumn('comb_order', F.lit(0))

    # Apply token mapping for RX
    rx_id_cols = id_cols + ["original_date"]
    pharmacy_rx_mapped = apply_token_mapping(spark, pharmacy_rx_long, "RX", mapping_dir, rx_id_cols) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order")
    
    # Add sub_order=0 for RX tokens (no principal/secondary distinction for drugs)
    pharmacy_rx_tokens = pharmacy_rx_mapped \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")

    # ==================== Merge all tokens ====================
    # original_date included for sorting within same timestamp (month)
    pharmacy_tokens = pharmacy_vt_cost_tokens.unionByName(pharmacy_rx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    logging.info("Pharmacy visit tokens construction completed!")
    return pharmacy_tokens


###################
## Anchor Tokens ##
###################

@time_execution
def construct_anchor_tokens(all_demographic_tokens: DataFrame, all_event_tokens: DataFrame) -> DataFrame:
    """
    This function creates two types of "anchor" tokens that serve as temporal reference points in a sequence:
    - Age tokens - mark the person's age at the start of their first event year
        - Age token's value is the difference between the first event's Year-01-01 and DOBYR-01-01
        - We'll put the age token's timestamp to the first date of the year of the person's first event (Year-01-01)
    - Newyear tokens - mark each January 1st from (first_event_year + 1) to last_event_year
        - Skips first_event_year since AGE token already marks that year
        - NY token on Jan 1 of last_event_year will naturally come before any events later in that year
    
    Args:
        all_demographic_tokens: DataFrame containing demographic tokens (must include DOBYR)
        all_event_tokens: DataFrame containing all event tokens
    
    Returns:
        duration: DataFrame with first/last event dates and total duration per person
        all_anchor_tokens: DataFrame containing AGE and NY anchor tokens
    """

    logging.info("%sConstructing anchor tokens (AGE and NY)", INDENT1)

    anchor_event_type = 0  # 0-indexed, 0th order among event types (AGE/NY come first)
    id_cols = ["person_id", "timestamp", "event_type"]

    # Extract DOBYR from demographic tokens (filter for DOBYR category and get value)
    dobyr_df = all_demographic_tokens.filter(F.col("category") == "DOBYR") \
        .select("person_id", F.col("value").cast("int").alias("DOBYR")).distinct()
    
    # first and last event dates
    duration = all_event_tokens.groupBy("person_id") \
        .agg(
            F.min("timestamp").alias("first_event_date"),
            F.max("timestamp").alias("last_event_date")) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("last_event_year", F.year(F.col("last_event_date"))) \
        .withColumn("total_duration", F.datediff(F.col("last_event_date"), F.col("first_event_date")))
    
    # age_token
    # Use month-based timestamp (first day of the month containing Jan 1)
    # Keep original_date (Jan 1) for sorting within timestamp
    # Use inner join to only include persons with events (avoid NULL first_event_year)
    age_tokens = dobyr_df.join(duration, on="person_id", how="inner") \
        .withColumn("age", F.col("first_event_year") - F.col("DOBYR")) \
        .withColumn("original_date", F.make_date(F.col("first_event_year"), F.lit(1), F.lit(1))) \
        .withColumn("timestamp", F.trunc(F.col("original_date"), "month")) \
        .withColumn("event_type", F.lit(anchor_event_type)) \
        .withColumn("category", F.lit("AGE")) \
        .withColumn("value", F.col("age").cast("string")) \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    spark_log_contingency_table(age_tokens, "value")

    # newyear tokens - generate NY token for each year-01-01 between first and last event
    # Skip first_event_year (since AGE token serves as the first year marker)
    # Include up to last_event_year (NY token on Jan 1 will come before any events later in that year)
    # Filter: only include persons with events spanning multiple years (first_event_year < last_event_year)
    # Use inner join to only include persons with events
    year_range_df = dobyr_df.join(duration, on="person_id", how="inner") \
        .select("person_id", "first_event_year", "last_event_year") \
        .filter(F.col("first_event_year") < F.col("last_event_year"))
    
    # Create exploded years using sequence, starting from first_event_year + 1
    # Use month-based timestamp (first day of the month containing Jan 1)
    # Keep original_date (Jan 1) for sorting within timestamp
    newyear_tokens = year_range_df \
        .withColumn("year_seq", F.expr("sequence(first_event_year + 1, last_event_year, 1)")) \
        .withColumn("year", F.explode("year_seq")) \
        .withColumn("original_date", F.make_date(F.col("year"), F.lit(1), F.lit(1))) \
        .withColumn("timestamp", F.trunc(F.col("original_date"), "month")) \
        .withColumn("event_type", F.lit(anchor_event_type)) \
        .withColumn("category", F.lit("NY")) \
        .withColumn("value", F.lit("NY")) \
        .withColumn("token", F.lit("<NY>")) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()
    
    # Combine age and newyear tokens
    all_anchor_tokens = age_tokens.union(newyear_tokens)

    return duration, all_anchor_tokens


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S1 CONSTRUCT TOKENS STARTING...")
    # Parse arguments
    args = parse_args()


    os.makedirs(args.external_val_data_dir, exist_ok=True)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sOMOP directory: %s", INDENT1, args.omop_dir)
    logging.info("%sExternal validation data directory: %s", INDENT1, args.external_val_data_dir)
    logging.info("%sMapping directory: %s", INDENT1, args.mapping_dir)

    # Create Spark session    
    spark = create_spark_session('EHRSHOT_Step1_Construct_Tokens')
    
    # excluded person IDs
    excluded_personids_df = spark.read.parquet(os.path.join(args.external_val_data_dir, "excluded_personids")).select("person_id").distinct().cache()
    logging.info("%sExcluded person IDs: %s", INDENT1, f"{excluded_personids_df.count():,}")



    logging.info("********** 1 Constructing demographic and enrollment tokens **********")
    all_demographic_tokens, all_enrollment_tokens = construct_demographic_and_enrollment_tokens(
        spark, args.omop_dir, args.external_val_data_dir, excluded_personids_df)
    
    # Save demographic tokens
    logging.info("%sSaving demographic tokens...", INDENT1)
    all_demographic_tokens.write.mode("overwrite").parquet(os.path.join(args.external_val_data_dir, "all_demographic_tokens"))
    
    # Save enrollment tokens
    logging.info("%sSaving enrollment tokens...", INDENT1)
    all_enrollment_tokens.write.mode("overwrite").parquet(os.path.join(args.external_val_data_dir, "all_enrollment_tokens"))

    logging.info("********** 2 Constructing all event tokens (enrollment + visit tokens) **********")
    
    # Construct visit tokens for each visit type
    logging.info("%sConstructing inpatient visit tokens...", INDENT1)
    inpatient_tokens = construct_inpatient_visit_tokens(spark, args.omop_dir, args.mapping_dir, excluded_personids_df)
    
    logging.info("%sConstructing outpatient visit tokens...", INDENT1)
    outpatient_tokens = construct_outpatient_visit_tokens(spark, args.omop_dir, args.mapping_dir, excluded_personids_df)
    
    logging.info("%sConstructing pharmacy visit tokens...", INDENT1)
    pharmacy_tokens = construct_pharmacy_visit_tokens(spark, args.omop_dir, args.mapping_dir, excluded_personids_df)
    
    # Union all visit tokens
    logging.info("%sUnioning all visit tokens...", INDENT1)
    all_claim_tokens = inpatient_tokens \
        .unionByName(outpatient_tokens) \
        .unionByName(pharmacy_tokens)
    # save all claim tokens
    logging.info("%sSaving all claim tokens...", INDENT1)
    all_claim_tokens.write.mode("overwrite").parquet(os.path.join(args.external_val_data_dir, "all_claim_tokens"))
    
  
    logging.info("********** 3 Constructing anchor tokens (AGE and NY) **********")
    # Re-read from disk for fresh lineage to avoid OOM errors
    logging.info("%sRe-loading from disk for fresh lineage...", INDENT1)
    all_demographic_tokens_reload = spark.read.parquet(os.path.join(args.external_val_data_dir, "all_demographic_tokens"))
    all_enrollment_tokens_reload = spark.read.parquet(os.path.join(args.external_val_data_dir, "all_enrollment_tokens"))
    all_claim_tokens_reload = spark.read.parquet(os.path.join(args.external_val_data_dir, "all_claim_tokens"))

    # Union enrollment and claim tokens just for computing duration/anchor tokens
    all_event_tokens = all_enrollment_tokens_reload.unionByName(all_claim_tokens_reload)
    
    duration, all_anchor_tokens = construct_anchor_tokens(all_demographic_tokens_reload, all_event_tokens)
    duration.write.mode("overwrite").parquet(os.path.join(args.external_val_data_dir, "duration"))
    all_anchor_tokens.write.mode("overwrite").parquet(os.path.join(args.external_val_data_dir, "all_anchor_tokens"))

    
    # Stop Spark
    spark.stop()
    logging.info("S1 CONSTRUCT TOKENS COMPLETED SUCCESSFULLY...")
    logging.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
