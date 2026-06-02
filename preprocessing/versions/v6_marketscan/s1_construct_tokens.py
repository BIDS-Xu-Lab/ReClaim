#!/usr/bin/env python3
"""
Reclaim Data processing
"""

import logging
from pyspark.sql import SparkSession, DataFrame, Column
import pyspark.sql.functions as F
from functools import reduce
import os
from glob import glob
import argparse
from pyspark.sql.window import Window


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
    parser.add_argument("--base_dir", type=str, 
                       default=os.environ.get("RAW_MARKETSCAN_DATA_ROOT", "/path/to/raw_marketscan_data_root"),
                       help='Base directory containing MarketScan Raw data')
    parser.add_argument("--input_folder_path", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"), "v6", "marketscan_val"),
                       help='Base directory containing MarketScan processed data for downstream tasks')            
    parser.add_argument("--payer_list", nargs='+', default=['CCAE', 'MDCR', 'MDCD'], 
                       help='List of payers to process')
    parser.add_argument("--table_list", nargs='+', default=['t','i','o','d'], 
                       help='List of table types to process (t=enrollment_detail, i=inpatient, o=outpatient, d=pharmacy)')
    parser.add_argument("--mapping_dir", type=str, 
                       default='../../vocab_mapping/mapping',
                       help='Directory containing the mapping files')
    return parser.parse_args()

###################################
## Data loading helper functions ##
###################################

def get_file_path(payer: str, table: str, base_dir: str) -> list:
    """Find all Parquet files matching dataset and table across years"""
    if payer == 'MDCD':
        search_pattern = f"Medicaid_{table}*.snappy.parquet"
    else:
        search_pattern = f"{payer}_{table.upper()}*.snappy.parquet"
    
    file_path = os.path.join(base_dir, payer, search_pattern)
    files = glob(file_path)
    logging.info("%s[%s-%s] Found %d files. Example: %s", INDENT1, payer, table, len(files), files[0])
    return sorted(files)


def build_file_path_dict(payer_list: list, table_list: list, base_dir: str) -> dict:
    """Build dictionary mapping payers and tables to their file paths"""
    
    table_file_path_dict = {}
    total_files = 0
    for payer in payer_list:
        table_file_path_dict[payer] = {}
        for table in table_list:
            paths = get_file_path(payer, table, base_dir)
            table_file_path_dict[payer][table] = paths
            total_files += len(paths)

    logging.info("%sTotal files found: %d", INDENT1, total_files)
    return table_file_path_dict


####################################
## Enrollment period processing ##
####################################

@time_execution
def construct_demographic_and_enrollment_tokens(spark: SparkSession, table_file_path_dict: dict, payer_category: str, excluded_enrollids_df: DataFrame) -> DataFrame:
    """
    Aggregate enrollment detail table to get the continuous enrollment period for the specified payer type.
    
    Args:
        spark: SparkSession instance
        table_file_path_dict: Dictionary mapping datasets and tables to file paths
        payer_category: Either "CCAE_MDCR" or "MDCD", need to combine CCAE and MDCR enrollment data together since they are linkable
        excluded_enrollids_df: DataFrame of enrollee IDs column containing enrollee IDs to exclude (uses anti-join)

    Returns:
        enrollment_period_df: Dataframe containing the continuous enrollment coverage information per enrollee
            variables: ENROLID, PAYER, plan_start_date, plan_end_date, *selected_variables's first appearance in the period, COVERED_DAYS
    """
    
    # Validate payer_type
    logging.info("%sPayer category: %s", INDENT2, payer_category)

    # Load enrollment data based on payer type
    if payer_category == "CCAE_MDCR":
        ccae_enrollment_df = spark.read.parquet(*table_file_path_dict['CCAE']['t']).withColumn("PAYER", F.lit("CCAE"))
        mdcr_enrollment_df = spark.read.parquet(*table_file_path_dict['MDCR']['t']).withColumn("PAYER", F.lit("MDCR"))
        
        logging.info("%sLoaded enrollment data", INDENT3)
        
        enrollment_df_filtered = ccae_enrollment_df.union(mdcr_enrollment_df) \
            .join(excluded_enrollids_df, on="ENROLID", how = "left_anti") \
            .withColumn("enrollee_id", F.concat(F.lit('CR'), F.col("ENROLID"))) \
            .withColumn("enrollment_start_date", F.trunc(F.col("DTSTART"), "month")) \
            .withColumn("CAP", 
                F.when(F.col("DATATYP").cast("string").isin("1", "3"), "0")
                 .when(F.col("DATATYP").cast("string").isin("2", "4"), "1")
                 .otherwise("MISSING")) \
            .withColumn("EGEOLOC", 
                F.when(F.col("EGEOLOC").isNull() | (F.col("EGEOLOC").cast("string") == ''), "MISSING")
                 .otherwise(F.col("EGEOLOC").cast("string"))) \
            .withColumn("PLANTYP", 
                F.when(F.col("PLANTYP").isNull() | (F.col("PLANTYP").cast("string") == ''), "MISSING")
                 .otherwise(F.col("PLANTYP").cast("string")))

        selected_plan_variables = ['PLANTYP','CAP','EGEOLOC']
        # For CCAE_MDCR, partition by both ENROLID and PAYER
        window_spec = Window.partitionBy("ENROLID", "PAYER").orderBy("enrollment_start_date")
    else:  # MDCD
        
        enrollment_df_filtered = spark.read.parquet(*table_file_path_dict['MDCD']['t']).withColumn("PAYER", F.lit("MDCD")) \
            .join(excluded_enrollids_df, on="ENROLID", how = "left_anti") \
            .withColumn("enrollee_id", F.concat(F.lit('MD'), F.col("ENROLID"))) \
            .withColumn("enrollment_start_date", F.trunc(F.col("DTSTART"), "month")) \
            .withColumn("CAP", 
                F.when(F.col("CAP").cast("string") == "0", "0")
                 .when(F.col("CAP").cast("string") == "1", "1")
                 .otherwise("MISSING")) \
            .withColumn("EGEOLOC", F.lit("MISSING")) \
            .withColumn("PLANTYP", 
                F.when(F.col("PLANTYP").isNull() | (F.col("PLANTYP").cast("string") == ''), "MISSING")
                 .otherwise(F.col("PLANTYP").cast("string")))

        selected_plan_variables = ['PLANTYP','CAP','EGEOLOC']
        # For MDCD, partition only by ENROLID
        window_spec = Window.partitionBy("ENROLID").orderBy("enrollment_start_date")
    
    
    # ************* Demographic ************* #
    logging.info("%sExtracting demographic information", INDENT2)
    # DOBYR can't be missing by design since it is used to anchor temporal information
    # Using window function to ensure deterministic selection of first record
    # Order by enrollment_start_date, then DOBYR and SEX for tie-breaking determinism
    first_enrollment_window = Window.partitionBy("enrollee_id").orderBy("enrollment_start_date","DOBYR","SEX")
    demographic_base = enrollment_df_filtered \
        .withColumn("_rn", F.row_number().over(first_enrollment_window)) \
        .filter(F.col("_rn") == 1) \
        .select("enrollee_id", "DOBYR", "SEX") \
        .filter(F.col("DOBYR").isNotNull()) \
        .withColumn("DOBYR", F.col("DOBYR").cast("string")) \
        .withColumn("SEX", F.coalesce(F.col("SEX").cast("string"), F.lit("MISSING")))
    
    
    # Create demographic tokens (no mapping/hierarchical splitting, so no original_token/comb_order needed)
    # SEX token
    sex_tokens = demographic_base \
        .withColumn("category", F.lit("SEX")) \
        .withColumn("value", F.col("SEX")) \
        .withColumn("token", F.concat(F.lit("<SEX-"), F.col("SEX"), F.lit(">"))) \
        .withColumn("sub_order", F.lit(0)) \
        .select("enrollee_id", "category", "value", "token", "sub_order")
    
    # DOBYR token
    dobyr_tokens = demographic_base \
        .withColumn("category", F.lit("DOBYR")) \
        .withColumn("value", F.col("DOBYR")) \
        .withColumn("token", F.concat(F.lit("<DOBYR-"), F.col("DOBYR"), F.lit(">"))) \
        .withColumn("sub_order", F.lit(1)) \
        .select("enrollee_id", "category", "value", "token", "sub_order")
    
    # Union DOBYR and SEX tokens
    demographic_tokens = dobyr_tokens.union(sex_tokens)

    
    

    # ************* Enrollment ************* #
    # Identify enrollment breaks using window functions (assume full month enrollment if data exist in the month)
    logging.info("%sIdentifying enrollment breaks using window functions", INDENT2)
    enrollment_break_df = enrollment_df_filtered \
        .withColumn("prev_enrollment_start_date", F.lag("enrollment_start_date", 1).over(window_spec)) \
        .withColumn("is_new_enrollment",
                    F.when((F.col("prev_enrollment_start_date").isNull()) | 
                           (F.months_between(F.col("enrollment_start_date"), F.col("prev_enrollment_start_date")) > 1), 1)
                    .otherwise(0)) \
        .withColumn("enrollment", F.sum("is_new_enrollment").over(window_spec))

    logging.info("%sAggregating to get enrollment periods with first month's claims", INDENT2)
    agg_expressions = [
        F.min("enrollment_start_date").alias("enrollment_start_date"),
        F.last_day(F.max("enrollment_start_date")).alias("enrollment_end_date")
    ] + [F.first(col).alias(col) for col in selected_plan_variables]
    
    enrollment_period_df = enrollment_break_df.groupBy("enrollee_id", "PAYER", "enrollment") \
        .agg(*agg_expressions) \
        .withColumn("covered_days", F.datediff(F.col("enrollment_end_date"), F.col("enrollment_start_date")) + 1) \
        .orderBy("enrollee_id", "enrollment", "PAYER")
    

    logging.info("%sConstructing enrollment start tokens", INDENT2)
    enrollment_start_event_type = 1 # 0-indexed, 1st order among event types if in same timestamp 
    enrollment_end_event_type = 5 # 0-indexed, 5th order among event types if in same timestamp 
    id_cols = ["enrollee_id", "timestamp", "event_type"]

    # construct enrollment start tokens
    # ERLST token: <ERLST-CCAE>, <ERLST-MDCR>, <ERLST-MDCD>
    # Plan variable tokens: <PLANTYP-X>, <CAP-X>, <EGEOLOC-X>
    # Use month-based timestamp (first day of the month), keep original_date for sorting
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
    # Use month-based timestamp (first day of the month), keep original_date for sorting
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





@time_execution
def construct_all_demographic_and_enrollment_tokens(spark: SparkSession, table_file_path_dict: dict, excluded_ccae_mdcr_df: DataFrame, excluded_mdcd_df: DataFrame) -> DataFrame:
    """
    Construct demographic and enrollment tokens for all payers
    """
    
    logging.info("%sConstructing demographic and enrollment tokens for CCAE_MDCR", INDENT1)
    ccae_mdcr_demographic_tokens, ccae_mdcr_enrollment_tokens = construct_demographic_and_enrollment_tokens(spark, table_file_path_dict, 
                                                            payer_category="CCAE_MDCR", 
                                                            excluded_enrollids_df=excluded_ccae_mdcr_df)
    
    logging.info("%sConstructing demographic and enrollment tokens for MDCD", INDENT1)    
    mdcd_demographic_tokens, mdcd_enrollment_tokens = construct_demographic_and_enrollment_tokens(spark, table_file_path_dict, 
                                                        payer_category="MDCD", 
                                                        excluded_enrollids_df=excluded_mdcd_df)
    # union the demographic and enrollment data
    all_demographic_tokens = ccae_mdcr_demographic_tokens.union(mdcd_demographic_tokens)
    all_enrollment_tokens = ccae_mdcr_enrollment_tokens.union(mdcd_enrollment_tokens)

    return all_demographic_tokens, all_enrollment_tokens

                                            

############################
## Claim event processing ##
############################

def encode_cost(cost_col: Column) -> Column:
    """
    Encode aggregated COST/GROSS_PAY (per visit-date) as a 2-digit mantissa/exponent string.

    Args:
        cost_col: Spark Column containing cost values
    
    Returns:
        Column: Encoded cost as 2-digit string

    Examples:
        NULL -> 'MISSING'
        negative_value -> "00"
        0 -> '00'
        940 -> '92'
        950 -> '13'
        7988 -> '83'
        5e6 -> '56'
        1e10 -> '99' (clipped)
    """

    # work with absolute value only for positive cost
    v = cost_col.cast("double")

    # compute exponent b = floor(log10(v)), avoid log10(0)
    safe_v = F.when(v<=1, F.lit(1.0)).otherwise(v)
    b = F.floor(F.log10(safe_v))

    # compute mantissa m = v/10^b and round to get digit a
    m = safe_v / F.pow(F.lit(10.0), b)
    a_raw = F.round(m).cast("int")

    # adjust rounding -> a=10, eg, 950 -> m=9.5, round to 10 -> should be 1*10^3 = 13
    a_adj = F.when(a_raw >= 10, F.lit(1)).otherwise(a_raw)
    b_adj = F.when(a_raw >= 10, b+1).otherwise(b)

    # clip value where exponent > 9 (values >= 950 million), actually not exist in the whole data
    a_clipped = F.when(b_adj>9, F.lit(9)).otherwise(a_adj)
    b_clipped = F.when(b_adj>9, F.lit(9)).otherwise(b_adj)

    # concatenate to form 2-digit code
    encoded = F.concat(a_clipped.cast("string"), b_clipped.cast("string"))

    return F.when(cost_col.isNull(), F.lit("MISSING")) \
           .when(cost_col <= 0, F.lit("00")) \
            .otherwise(encoded)





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
        original_proc_count = df_long.count()
        logging.info("%sFor procedure, drop HCPCS, only keep ICD9CM, ICD10PCS and CPT; starting from %s procedure codes", INDENT2, f"{original_proc_count:,}")
        # hcpcs_regex = "^[A-Z][0-9]{4}" 
        df_to_map = df_long \
            .filter(~F.col("value").rlike(r"^[A-Z][0-9]{4}$")) \
            .filter(
                F.col("value").rlike(r'^[0-9]{5}|[0-9]{4}[A-Z0-9]$') | 
                F.col("value").rlike(r'[A-Z0-9]{7}$') |
                F.col("value").rlike(r'^(E\d{3}(\.\d)?|V\d{2}(\.\d{1,2})?|\d{3}(\.\d{1,2})?)$') |
                F.col("value").rlike(r'^\d{2}(\.\d{1,2})?$') 
            )
        
        filtered_proc_count = df_to_map.count()
        logging.info("%s... %s procedure codes left after filtering", INDENT2, f"{filtered_proc_count:,}")
        nomap_token = "<PROC-NOMAP>"
        nomap_value = "NOMAP"

    elif category == 'RX':
        mapping_df = spark.read.csv(os.path.join(mapping_dir, "ndc_drug_to_rxnorm_ingredient_mapping.csv"), header=True)
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
def construct_table_i_tokens(spark: SparkSession, table_i_wide_filtered: DataFrame, mapping_dir: str) -> DataFrame:
    """
    Processing inpatient admissiontable
    Args:
        table_i_wide_filtered_df: DataFrame containing the inpatient table
        mapping_dir: Directory containing the mapping files
    Returns:
        table_i_wide_filtered_df: DataFrame containing the inpatient table
    """

    logging.info("%sConstructing inpatient tokens", INDENT2)
    inpatient_dx_col = [f"DX{i}" for i in range(1, 16)]
    inpatient_proc_col = [f"PROC{i}" for i in range(1, 16)]
    inpatient_cost_col = ["TOTPAY"]
    inpatient_date_col = ["ADMDATE"]
    inpatient_other_col = ["DAYS","DSTATUS","DXVER"]
    inpatient_event_type = 4 # 0-indexed, 4th order among event types if in same timestamp 
    

    # Unify variables:
    # LS: long stay: length of stay >= 7 days
    # DS: discharge status: 0: expired, 1: still inpatient, 2: transfer, 3: discharge home / self-care, 4: other alive status
    # Use month-based timestamp (first day of the month), keep original_date for sorting within month
    table_i_wide_selected = table_i_wide_filtered.select("enrollee_id", *inpatient_dx_col, *inpatient_proc_col, *inpatient_cost_col, *inpatient_other_col, *inpatient_date_col) \
        .withColumn("original_date", F.col("ADMDATE")) \
        .withColumn("timestamp", F.trunc(F.col("ADMDATE"), "month")) \
        .withColumn("event_type", F.lit(inpatient_event_type)) \
        .withColumn("VT", F.lit("inpatient")) \
        .withColumn("LS",F.when(F.col("DAYS").isNull(), F.lit("MISSING")).when(F.col("DAYS") < 7, F.lit("0")).otherwise(F.lit("1"))) \
        .withColumn("DS",
            F.when(F.col("DSTATUS").isNull(), F.lit("MISSING"))
            .when((F.to_date(F.col("timestamp")) >= F.to_date(F.lit("2016-01-01"))) & F.col("DSTATUS").isin(20,21,40,41,42,87), F.lit("MISSING"))
            .when((F.to_date(F.col("timestamp")) < F.to_date(F.lit("2016-01-01"))) & F.col("DSTATUS").isin(20,40,41,42), F.lit(0))
            .when((F.col("DSTATUS") == 9) | F.col("DSTATUS").between(30,39), F.lit(1))
            .when(F.col("DSTATUS").isin(
                2,3,4,5,21,87,43,51,61,62,63,64,65,66,69,70,71,72,
                82,83,84,85,88,89,90,91,92,93,94,95,99
            ), F.lit(2))
            .when(F.col("DSTATUS").isin(1,6,8,50,81,86), F.lit(3))
            .when(F.col("DSTATUS").between(10,19), F.lit(4))
            .otherwise(F.lit("MISSING"))) \
        .withColumnRenamed("TOTPAY", "COST") \
        .select("enrollee_id", "event_type", "timestamp", "original_date", "VT", "DXVER", *inpatient_dx_col, *inpatient_proc_col, "LS", "DS", "COST")

    # define id columns
    id_cols= ["enrollee_id", "timestamp",  "event_type"]

    logging.info("%sProcessing tokens: VT/COST aggregated, LS/DS keep all distinct values per month", INDENT3)
    
    # VT and COST: aggregated (VT first-by-date, COST summed, one token per timestamp)
    # Keep min original_date for sorting within timestamp
    vt_cost_agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT"),  # first by date
        F.sum("COST").alias("COST")
    ]
    
    table_i_vt_cost_tokens = table_i_wide_selected.groupBy(*id_cols) \
        .agg(*vt_cost_agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", encode_cost(F.col("COST"))) \
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
    table_i_ls_ds_long = table_i_wide_selected \
        .select(*id_cols, "original_date", "LS", "DS") \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=["LS", "DS"],
            variableColumnName="category",
            valueColumnName="value")
    
    # Deduplicate keeping first occurrence by original_date
    ls_ds_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    table_i_ls_ds_tokens = table_i_ls_ds_long \
        .withColumn("row_num", F.row_number().over(ls_ds_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")
    
    # Union VT/COST tokens with LS/DS tokens
    table_i_agg_tokens = table_i_vt_cost_tokens \
        .union(table_i_ls_ds_tokens) \
        .distinct()
    
    logging.info("%sProcessing tokens to be mapped/standardized and deduplicated: PROC, DX", INDENT3)
    
    # Unpivot PROC columns with principal/secondary tracking
    # is_principal: True for PROC1 (principal), False for PROC2+ (secondary)
    # Keep original_date for sorting within timestamp; deduplicate keeping first by original_date
    table_i_proc_long = table_i_wide_selected.select(*id_cols, "original_date", *inpatient_proc_col) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=inpatient_proc_col,
            variableColumnName="proc_col",
            valueColumnName="value") \
        .filter(F.col("value").isNotNull() & (F.col("value") != "")) \
        .withColumn("category", F.lit("PROC")) \
        .withColumn("is_principal", F.col("proc_col") == "PROC1")
    
    # Deduplicate keeping first occurrence by original_date (instead of arbitrary .distinct())
    proc_dedup_window = Window.partitionBy(*id_cols, "is_principal", "category", "value").orderBy("original_date")
    table_i_proc_long = table_i_proc_long \
        .withColumn("row_num", F.row_number().over(proc_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0))
    
    # apply token mapping for procedure (include is_principal and original_date in id_cols to preserve through mapping)
    proc_id_cols = id_cols + ["original_date", "is_principal"]
    table_i_proc_mapped = apply_token_mapping(spark, table_i_proc_long, "PROC", mapping_dir, proc_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    table_i_proc_tokens = add_principal_secondary_markers(table_i_proc_mapped, id_cols, "PROC") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")


    # Unpivot DX columns with principal/secondary tracking, need to keep DXVER for later mapping
    # is_principal: True for DX1 (principal), False for DX2+ (secondary)
    # Keep original_date for sorting within timestamp; deduplicate keeping first by original_date
    table_i_dx_long = table_i_wide_selected.select(*id_cols, "original_date", "DXVER", *inpatient_dx_col) \
        .unpivot(
            ids=id_cols + ["original_date", "DXVER"],
            values=inpatient_dx_col,
            variableColumnName="dx_col",
            valueColumnName="value") \
        .filter(F.col("value").isNotNull() & (F.col("value") != "")) \
        .withColumn("category", F.lit("DX")) \
        .withColumn("is_principal", F.col("dx_col") == "DX1")
    
    # Deduplicate keeping first occurrence by original_date (instead of arbitrary .distinct())
    dx_dedup_window = Window.partitionBy(*id_cols, "DXVER", "is_principal", "category", "value").orderBy("original_date")
    table_i_dx_long = table_i_dx_long \
        .withColumn("row_num", F.row_number().over(dx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0))

    # apply token mapping for icd9 dx codes (DXVER needed for mapping logic, is_principal and original_date for tracking)
    dx_id_cols = id_cols + ["original_date", "DXVER", "is_principal"]
    table_i_dx_mapped = apply_token_mapping(spark, table_i_dx_long, "DX", mapping_dir, dx_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    table_i_dx_tokens = add_principal_secondary_markers(table_i_dx_mapped, id_cols, "DX") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()


    # merge all tokens
    # Ordering columns (for DX/PROC categories):
    #   - original_date: for sorting within same timestamp (month)
    #   - sub_order: 0=PRINCIPAL marker, 1=principal codes, 2=SECONDARY marker, 3=secondary codes
    #   - comb_order: hierarchical token order within each sub_order group
    # For other categories (VT, COST, LS, DS), sub_order=0 (placeholder, not used for sorting)
    table_i_tokens = table_i_agg_tokens.union(table_i_proc_tokens).union(table_i_dx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    return table_i_tokens
    


@time_execution
def construct_table_o_tokens(spark: SparkSession, table_o_wide_filtered: DataFrame, mapping_dir: str) -> DataFrame:
    """
    Processing outpatient services table
    Args:
        spark: SparkSession instance
        table_o_wide_filtered: DataFrame containing the outpatient table
        mapping_dir: Directory containing the mapping files
    Returns:
        DataFrame containing the processed outpatient tokens
    """

    outpatient_dx_col = [f"DX{i}" for i in range(1, 5)] # four diagnosis columns for outpatient DX1- DX4
    outpatient_proc_col = ["PROC1"] # only one procedure column for outpatient
    outpatient_cost_col = ["PAY"]  # Outpatient uses PAY instead of TOTPAY
    outpatient_date_col = ["SVCDATE"]
    outpatient_other_col = ["DXVER"]
    outpatient_event_type = 2 # 0-indexed, 2nd order among event types if in same timestamp 
    

    # Unify variables for outpatient (no DAYS, LS, DS like inpatient)
    # Use month-based timestamp (first day of the month), keep original_date for sorting within month
    table_o_wide_selected = table_o_wide_filtered.select("enrollee_id", *outpatient_dx_col, *outpatient_proc_col, *outpatient_cost_col, *outpatient_other_col, *outpatient_date_col) \
        .withColumn("original_date", F.col("SVCDATE")) \
        .withColumn("timestamp", F.trunc(F.col("SVCDATE"), "month")) \
        .withColumn("event_type", F.lit(outpatient_event_type)) \
        .withColumn("VT", F.lit("outpatient")) \
        .withColumnRenamed("PAY", "COST") \
        .select("enrollee_id", "event_type", "timestamp", "original_date", "VT", "DXVER", *outpatient_dx_col, *outpatient_proc_col, "COST")

    # define id columns
    id_cols = ["enrollee_id", "timestamp", "event_type"]

    logging.info("%sProcessing tokens to be aggregated and appear only once per timestamp/unit: VT, COST", INDENT3)
    # For month-based aggregation: COST is summed, VT takes first occurrence by date
    # Keep min original_date for sorting within timestamp
    agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT"),  # first by date
        F.sum("COST").alias("COST")
    ]

    # sub_order: placeholder for non-DX/PROC categories
    table_o_agg_tokens = table_o_wide_selected.groupBy(*id_cols) \
        .agg(*agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", encode_cost(F.col("COST"))) \
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
    # Unpivot PROC columns with principal/secondary tracking
    # is_principal: True for PROC1 (principal), False for PROC2+ (secondary)
    # Keep original_date for sorting within timestamp; deduplicate keeping first by original_date
    table_o_proc_long = table_o_wide_selected.select(*id_cols, "original_date", *outpatient_proc_col) \
        .unpivot(
            ids=id_cols + ["original_date"],
            values=outpatient_proc_col,
            variableColumnName="proc_col",
            valueColumnName="value") \
        .filter(F.col("value").isNotNull() & (F.col("value") != "")) \
        .withColumn("category", F.lit("PROC")) \
        .withColumn("is_principal", F.col("proc_col") == "PROC1")
    
    # Deduplicate keeping first occurrence by original_date
    proc_dedup_window = Window.partitionBy(*id_cols, "is_principal", "category", "value").orderBy("original_date")
    table_o_proc_long = table_o_proc_long \
        .withColumn("row_num", F.row_number().over(proc_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0))
    
    # apply token mapping for procedure (include is_principal and original_date in id_cols to preserve through mapping)
    proc_id_cols = id_cols + ["original_date", "is_principal"]
    table_o_proc_mapped = apply_token_mapping(spark, table_o_proc_long, "PROC", mapping_dir, proc_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    table_o_proc_tokens = add_principal_secondary_markers(table_o_proc_mapped, id_cols, "PROC") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")


    # Unpivot DX columns with principal/secondary tracking, need to keep DXVER for later mapping
    # is_principal: True for DX1 (principal), False for DX2+ (secondary)
    # Keep original_date for sorting within timestamp; deduplicate keeping first by original_date
    table_o_dx_long = table_o_wide_selected.select(*id_cols, "original_date", "DXVER", *outpatient_dx_col) \
        .unpivot(
            ids=id_cols + ["original_date", "DXVER"],
            values=outpatient_dx_col,
            variableColumnName="dx_col",
            valueColumnName="value") \
        .filter(F.col("value").isNotNull() & (F.col("value") != "")) \
        .withColumn("category", F.lit("DX")) \
        .withColumn("is_principal", F.col("dx_col") == "DX1")
    
    # Deduplicate keeping first occurrence by original_date
    dx_dedup_window = Window.partitionBy(*id_cols, "DXVER", "is_principal", "category", "value").orderBy("original_date")
    table_o_dx_long = table_o_dx_long \
        .withColumn("row_num", F.row_number().over(dx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0))

    # apply token mapping for icd9 dx codes (DXVER needed for mapping logic, is_principal and original_date for tracking)
    dx_id_cols = id_cols + ["original_date", "DXVER", "is_principal"]
    table_o_dx_mapped = apply_token_mapping(spark, table_o_dx_long, "DX", mapping_dir, dx_id_cols) \
        .select(*id_cols, "original_date", "is_principal", "category", "value", "token", "original_token", "comb_order")
    
    # Add principal/secondary marker tokens with sub_order column
    # Note: pass id_cols WITHOUT original_date - function handles original_date separately
    table_o_dx_tokens = add_principal_secondary_markers(table_o_dx_mapped, id_cols, "DX") \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()


    # merge all tokens
    # Ordering columns (for DX/PROC categories):
    #   - original_date: for sorting within same timestamp (month)
    #   - sub_order: 0=PRINCIPAL marker, 1=principal codes, 2=SECONDARY marker, 3=secondary codes
    #   - comb_order: hierarchical token order within each sub_order group
    # For other categories (VT, COST), sub_order=0 (placeholder, not used for sorting)
    table_o_tokens = table_o_agg_tokens.union(table_o_proc_tokens).union(table_o_dx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    return table_o_tokens



@time_execution
def construct_table_d_tokens(spark: SparkSession, table_d_wide_filtered: DataFrame, mapping_dir: str) -> DataFrame:
    """
    Processing outpatient pharmaceuticals table d
    
    Since there's only one NDCNUM column per row, no unpivot is needed.
    We take distinct NDCNUM per id_cols, apply RX mapping, and union with VT/COST tokens.
    
    Args:
        spark: SparkSession instance
        table_d_wide_filtered: DataFrame containing the outpatient pharmaceuticals table
        mapping_dir: Directory containing the mapping files
    Returns:
        DataFrame containing the processed pharmacy tokens
    """

    rx_ndc_col = "NDCNUM"
    rx_date_col = "SVCDATE"
    rx_cost_col = "PAY"
    pharmacy_event_type = 3 # 0-indexed, 3rd order among event types if in same timestamp 

    # Select and unify variables
    # Use month-based timestamp (first day of the month), keep original_date for sorting within month
    table_d_wide_selected = table_d_wide_filtered.select("enrollee_id", rx_ndc_col, rx_cost_col, rx_date_col) \
        .withColumn("original_date", F.col(rx_date_col)) \
        .withColumn("timestamp", F.trunc(F.col(rx_date_col), "month")) \
        .withColumn("event_type", F.lit(pharmacy_event_type)) \
        .withColumn("VT", F.lit("pharmacy")) \
        .withColumnRenamed(rx_cost_col, "COST") \
        .select("enrollee_id", "event_type", "timestamp", "original_date", "VT", rx_ndc_col, "COST")

    # define id columns
    id_cols = ["enrollee_id", "timestamp", "event_type"]

    logging.info("%sProcessing tokens to be aggregated and appear only once per timestamp/unit: VT, COST", INDENT3)
    # For month-based aggregation: COST is summed, VT takes first occurrence by date
    # Keep min original_date for sorting within timestamp
    agg_expressions = [
        F.min("original_date").alias("original_date"),  # earliest date in the month for sorting
        F.min(F.struct(F.col("original_date"), F.col("VT"))).getField("VT").alias("VT"),  # first by date
        F.sum("COST").alias("COST")
    ]

    table_d_agg_tokens = table_d_wide_selected.groupBy(*id_cols) \
        .agg(*agg_expressions) \
        .withColumn("VT", F.when(F.col("VT").isNull(), F.lit("MISSING")).otherwise(F.col("VT"))) \
        .withColumn("COST", encode_cost(F.col("COST"))) \
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
    
    # Keep original_date for sorting within timestamp; deduplicate keeping first by original_date
    table_d_rx_long = table_d_wide_selected.select(*id_cols, "original_date", rx_ndc_col) \
        .filter(F.col(rx_ndc_col).isNotNull() & (F.col(rx_ndc_col) != "")) \
        .withColumn("category", F.lit("RX")) \
        .withColumn("value", F.col(rx_ndc_col))
    
    # Deduplicate keeping first occurrence by original_date
    rx_dedup_window = Window.partitionBy(*id_cols, "category", "value").orderBy("original_date")
    table_d_rx_long = table_d_rx_long \
        .withColumn("row_num", F.row_number().over(rx_dedup_window)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .withColumn("token", F.concat(F.lit("<"), F.lit("NDCNUM"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0))

    # Apply token mapping for RX (NDC to RxNorm), include original_date in id_cols
    rx_id_cols = id_cols + ["original_date"]
    table_d_rx_mapped = apply_token_mapping(spark, table_d_rx_long, "RX", mapping_dir, rx_id_cols) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order")

    # Add sub_order=0 for RX tokens (no principal/secondary distinction for drugs)
    table_d_rx_tokens = table_d_rx_mapped \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order")

    #############################
    ## Merge all tokens
    #############################
    # original_date included for sorting within same timestamp (month)
    table_d_tokens = table_d_agg_tokens.unionByName(table_d_rx_tokens) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()

    return table_d_tokens



@time_execution
def construct_all_claim_tokens(spark: SparkSession, table_file_path_dict: dict, mapping_dir: str, excluded_ccae_mdcr_df: DataFrame, excluded_mdcd_df: DataFrame) -> DataFrame:
    """
    Processing all claim tables (inpatient, outpatient, pharmacy) for all payers
    """
    enrollee_id_prefix_map = {'CCAE': 'CR', 'MDCR': 'CR', 'MDCD': 'MD'} # CCAE and MDCR are linkable and share the same patient id prefix 'CR'
    table_tokens_list = []
    logging.info("%sProcessing all claim tables for all payers", INDENT1)
    for payer in ['CCAE', 'MDCR', 'MDCD']:
        # select appropriate exclusion dataframe based on the payer
        excluded_enrollids_df = excluded_ccae_mdcr_df if payer in ['CCAE','MDCR'] else excluded_mdcd_df
        for table in ['i', 'o', 'd']:
            # filter out excluded enrollees and add patient id prefix
            table_wide_filtered_df = spark.read.parquet(*table_file_path_dict[payer][table]) \
                .join(excluded_enrollids_df, on="ENROLID", how = "left_anti") \
                .withColumn("enrollee_id", F.concat(F.lit(enrollee_id_prefix_map[payer]), F.col("ENROLID")))
            logging.info("%sLoaded and filtered %s - %s, start processing...", INDENT1, payer, table)
            
            if table == 'i':
                table_tokens = construct_table_i_tokens(spark, table_wide_filtered_df, mapping_dir)
            elif table == 'o':
                table_tokens = construct_table_o_tokens(spark, table_wide_filtered_df, mapping_dir)
            elif table == 'd':
                table_tokens = construct_table_d_tokens(spark, table_wide_filtered_df, mapping_dir)
            else:
                raise ValueError(f"Unknown table type: {table}")
            
            table_tokens_list.append(table_tokens)

    all_claim_tokens = reduce(lambda x, y: x.unionByName(y), table_tokens_list)

    return all_claim_tokens



###################
## Anchor Tokens ##
###################


@time_execution
def construct_anchor_tokens(all_demographic_tokens: DataFrame, all_event_tokens: DataFrame) -> DataFrame:
    """
    This function creates two types of "anchor" tokens that serve as temporal reference points in a sequence:
    - Age tokens - mark the enrollee's age at the start of their first event year
        - Age token's value is the difference between the first event's Year-01-01 and DOBYR-01-01
        - We'll put the age token's timestamp to the first date of the year of the enrollee's first event (Year-01-01)
    - Newyear tokens - mark each January 1st from (first_event_year + 1) to last_event_year
        - Skips first_event_year since AGE token already marks that year
        - NY token on Jan 1 of last_event_year will naturally come before any events later in that year
    """

    logging.info("%sConstructing anchor tokens (AGE and NY)", INDENT1)

    anchor_event_type = 0 # 0-indexed, 0th order among event types (AGE/NY come first)
    id_cols = ["enrollee_id", "timestamp", "event_type"]

    # Extract DOBYR from demographic tokens (filter for DOBYR category and get value)
    dobyr_df = all_demographic_tokens.filter(F.col("category") == "DOBYR") \
        .select("enrollee_id", F.col("value").cast("int").alias("DOBYR")).distinct()
    # first and last event dates
    duration = all_event_tokens.groupBy("enrollee_id") \
        .agg(
            F.min("timestamp").alias("first_event_date"),
            F.max("timestamp").alias("last_event_date")) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("last_event_year", F.year(F.col("last_event_date"))) \
        .withColumn("total_duration", F.datediff(F.col("last_event_date"), F.col("first_event_date")))
    
    # age_token
    # Use month-based timestamp (first day of the month containing Jan 1)
    # Keep original_date (Jan 1) for sorting within timestamp
    age_tokens = dobyr_df.join(duration, on="enrollee_id", how="left") \
        .withColumn("age", F.col("first_event_year") - F.col("DOBYR")) \
        .withColumn("original_date", F.make_date(F.col("first_event_year"), F.lit(1), F.lit(1))) \
        .withColumn("timestamp", F.trunc(F.col("original_date"), "month")) \
        .withColumn("event_type", F.lit(anchor_event_type)) \
        .withColumn("category", F.lit("AGE")) \
        .withColumn("value", F.col("age")) \
        .withColumn("token", F.concat(F.lit("<"), F.col("category"), F.lit("-"), F.col("value"), F.lit(">"))) \
        .withColumn("original_token", F.col("token")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*id_cols, "original_date", "category", "value", "token", "original_token", "comb_order", "sub_order") \
        .distinct()


    # newyear tokens - generate NY token for each year-01-01 between first and last event
    # Skip first_event_year (since AGE token serves as the first year marker)
    # Include up to last_event_year (NY token on Jan 1 will come before any events later in that year)
    # Filter: only include enrollees with events spanning multiple years (first_event_year < last_event_year)
    year_range_df = dobyr_df.join(duration, on="enrollee_id", how="left") \
        .select("enrollee_id", "first_event_year", "last_event_year") \
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
        .withColumn("value", F.lit("")) \
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

    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    os.makedirs(processed_data_folder, exist_ok=True)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sBase directory: %s", INDENT1, args.base_dir)
    logging.info("%sProcessed data folder path: %s", INDENT1, processed_data_folder)
    logging.info("%sPayers: %s", INDENT1, args.payer_list)
    logging.info("%sTables: %s", INDENT1, args.table_list)
    logging.info("%sMapping directory: %s", INDENT1, args.mapping_dir)

    # Create Spark session
    spark = SparkSession.builder.appName('MarketScan_Step1_Construct_Tokens').getOrCreate()
    
    # Build file path dictionary
    table_file_path_dict = build_file_path_dict(args.payer_list, args.table_list, args.base_dir)
    
    # excluded enrollees
    excluded_ccae_mdcr_df = spark.read.parquet(os.path.join(processed_data_folder, "excluded_enrollids_ccae_mdcr")).select("ENROLID").distinct().cache()
    excluded_mdcd_df = spark.read.parquet(os.path.join(processed_data_folder, "excluded_enrollids_mdcd")).select("ENROLID").distinct().cache()
    logging.info("%sExcluded CCAE/MDCR enrollees: %s; Excluded MDCD enrollees: %s", INDENT1, f"{excluded_ccae_mdcr_df.count():,}",  f"{excluded_mdcd_df.count():,}")

    logging.info("********** 1 Constructing all demographic and enrollment tokens **********")
    all_demographic_tokens, all_enrollment_tokens = construct_all_demographic_and_enrollment_tokens(spark, table_file_path_dict, excluded_ccae_mdcr_df, excluded_mdcd_df)
    
    # Save demographic tokens
    all_demographic_tokens.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "all_demographic_tokens"))
    
    # Save enrollment tokens separately (union with claim tokens deferred to s2)
    logging.info("%sSaving enrollment tokens...", INDENT1)
    all_enrollment_tokens.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "all_enrollment_tokens"))

    logging.info("********** 2 Constructing all claim tokens **********")
    all_claim_tokens = construct_all_claim_tokens(spark, table_file_path_dict, args.mapping_dir, excluded_ccae_mdcr_df, excluded_mdcd_df)
    
    # Save claim tokens separately (union with enrollment tokens deferred to s2)
    logging.info("%sSaving claim tokens...", INDENT1)
    all_claim_tokens.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "all_claim_tokens"))

    logging.info("********** 3 Constructing anchor tokens (AGE and NY) **********")
    # Re-read from disk for fresh lineage
    logging.info("%sRe-loading from disk for fresh lineage to avoid OOM errors...", INDENT1)
    all_demographic_tokens_reload = spark.read.parquet(os.path.join(processed_data_folder, "all_demographic_tokens"))
    all_enrollment_tokens_reload = spark.read.parquet(os.path.join(processed_data_folder, "all_enrollment_tokens"))
    all_claim_tokens_reload = spark.read.parquet(os.path.join(processed_data_folder, "all_claim_tokens"))
    
    # Union enrollment and claim tokens just for computing duration/anchor tokens
    all_event_tokens = all_enrollment_tokens_reload.unionByName(all_claim_tokens_reload)
    
    duration, all_anchor_tokens = construct_anchor_tokens(all_demographic_tokens_reload, all_event_tokens)
    duration.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "duration"))
    all_anchor_tokens.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "all_anchor_tokens"))

    
    # Stop Spark
    spark.stop()
    logging.info("S1 CONSTRUCT TOKENS COMPLETED SUCCESSFULLY...")
    logging.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
