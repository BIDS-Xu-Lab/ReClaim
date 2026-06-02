#!/usr/bin/env python3
"""
Reclaim Data Filter
Creates demographic sequences and vocabulary from first patient visits
"""


import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import os
from glob import glob
import argparse


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
    parser.add_argument("--base_dir", type=str, 
                       default=os.environ.get("RAW_MARKETSCAN_DATA_ROOT", "/path/to/raw_marketscan_data_root"),
                       help='Base directory containing MarketScan Raw data')
    parser.add_argument("--input_folder_path", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_MARKETSCAN_DATA_ROOT", "/path/to/processed_marketscan_data_root"), "v6", "marketscan_val"),
                       help='Base directory containing MarketScan processed data for downstream tasks')            
    parser.add_argument("--payer_list", nargs='+', default=['CCAE', 'MDCR', 'MDCD'], 
                       help='List of payers to process')
    parser.add_argument("--table_list", nargs='+', default=['t', 'i', 'd', 'o'], 
                       help='List of table types to process (t=detail, i=inpatient, d=pharmacy, o=outpatient)')
    parser.add_argument("--min_mem_days", type=int, default=180, help='Minimum total member days required for a patient to be included')
    parser.add_argument("--min_claims_duration_days", type=int, default=30, help='Minimum claims duration required for a patient to be included')
    parser.add_argument("--min_age", type=int, default=10, help='Minimum age at first event required for a patient to be included')
    parser.add_argument("--max_age", type=int, default=110, help='Maximum age at first event required for a patient to be included')
    return parser.parse_args()


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
    """Build dictionary mapping datasets and tables to their file paths"""
    
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


@time_execution
def ccae_mdcr_filter(spark: SparkSession, table_file_path_dict: dict, min_mem_days: int, min_claims_duration_days: int, min_age: int, max_age: int, processed_data_folder: str):
    """
    Filter combined CCAE and MDCR data in the following steps:
    1. Filter enrollees based on the EIDFLAG (EIDFLAG describe the quality of the ENROLID assignment)
    2. Filter enrollees based on the Drug data availability (RX)
    3. Filter enrollees based on the minimum member days (combined across both datasets)
    4. Filter enrollees based on the minimum claims duration (max visit date - min visit date, combined across both datasets)
    5. Filter enrollees based on minimum age at first event

    Then save the excluded enrollees to processed data folder
    """
    logging.info("CCAE + MDCR Combined Filtering:")
    
    # Load enrollment data from both datasets (include DOBYR for age calculation)
    ccae_enrollment_df = spark.read.parquet(*table_file_path_dict['CCAE']['t']).select("ENROLID", "RX", "MEMDAYS", "DTSTART", "DOBYR")
    mdcr_enrollment_df = spark.read.parquet(*table_file_path_dict['MDCR']['t']).select("ENROLID", "RX", "MEMDAYS", "DTSTART", "DOBYR")
    ccae_mdcr_enrollment_df = ccae_enrollment_df.union(mdcr_enrollment_df)
    
    total_enrollids = ccae_mdcr_enrollment_df.select("ENROLID").distinct().count()
    ccae_enrollids = ccae_enrollment_df.select("ENROLID").distinct().count()
    mdcr_enrollids = mdcr_enrollment_df.select("ENROLID").distinct().count()
    overlapping_enrollids = ccae_enrollids + mdcr_enrollids - total_enrollids
    
    logging.info("%sCCAE ENROLIDs: %s", INDENT1, f"{ccae_enrollids:,}")
    logging.info("%sMDCR ENROLIDs: %s", INDENT1, f"{mdcr_enrollids:,}")
    logging.info("%sOverlapping ENROLIDs: %s", INDENT1, f"{overlapping_enrollids:,}")
    logging.info("%sTotal unique ENROLIDs: %s", INDENT1, f"{total_enrollids:,}")

    # Load claims data from both datasets
    ccae_i_df = spark.read.parquet(*table_file_path_dict['CCAE']['i']).select("ENROLID", "EIDFLAG", "ADMDATE").withColumnRenamed("ADMDATE", "VISIT_DATE")
    ccae_o_df = spark.read.parquet(*table_file_path_dict['CCAE']['o']).select("ENROLID", "EIDFLAG", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")
    ccae_d_df = spark.read.parquet(*table_file_path_dict['CCAE']['d']).select("ENROLID", "EIDFLAG", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")
    
    mdcr_i_df = spark.read.parquet(*table_file_path_dict['MDCR']['i']).select("ENROLID", "EIDFLAG", "ADMDATE").withColumnRenamed("ADMDATE", "VISIT_DATE")
    mdcr_o_df = spark.read.parquet(*table_file_path_dict['MDCR']['o']).select("ENROLID", "EIDFLAG", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")
    mdcr_d_df = spark.read.parquet(*table_file_path_dict['MDCR']['d']).select("ENROLID", "EIDFLAG", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")
    
    ccae_mdcr_claims_df = ccae_i_df.union(ccae_o_df).union(ccae_d_df).union(mdcr_i_df).union(mdcr_o_df).union(mdcr_d_df)

    # Filter enrollees based on the EIDFLAG (EIDFLAG describe the quality of the ENROLID assignment)
    invalid_enrollids_df = ccae_mdcr_claims_df.filter(
        F.col("EIDFLAG") != 1
    ).select("ENROLID").distinct()
    
    logging.info("%sInvalid ENROLIDs (EIDFLAG != 1): %s", INDENT1, f"{invalid_enrollids_df.count():,}")

    # Filter enrollees based on the Drug data availability (RX)
    missing_rx_df = ccae_mdcr_enrollment_df.filter(
        F.col("RX") != 1
    ).select("ENROLID").distinct()
    
    logging.info("%sENROLIDs with missing RX: %s", INDENT1, f"{missing_rx_df.count():,}")
    
    # Filter enrollees based on the minimum member days (aggregated across both datasets)
    min_mem_days_df = ccae_mdcr_enrollment_df.groupBy("ENROLID").agg(F.sum("MEMDAYS").alias("TOTAL_MEMDAYS"))
    insufficient_mem_days_df = min_mem_days_df.filter(
        F.col("TOTAL_MEMDAYS") < min_mem_days
    ).select("ENROLID").distinct()
    
    logging.info("%sENROLIDs with insufficient member days: %s", INDENT1, f"{insufficient_mem_days_df.count():,}")

    # Filter enrollees based on the minimum claims duration (max visit date - min visit date, across both datasets)
    min_claims_duration_df = ccae_mdcr_claims_df.groupBy("ENROLID").agg(F.max("VISIT_DATE").alias("MAX_VISIT_DATE"), F.min("VISIT_DATE").alias("MIN_VISIT_DATE"))
    insufficient_claims_duration_df = min_claims_duration_df.filter(
        F.datediff(F.col("MAX_VISIT_DATE"), F.col("MIN_VISIT_DATE")) < min_claims_duration_days
    ).select("ENROLID").distinct()
    logging.info("%sENROLIDs with insufficient claims duration: %s", INDENT1, f"{insufficient_claims_duration_df.count():,}")

    # Filter enrollees based on minimum age at first event
    # Get DOBYR from first enrollment record (ordered by DTSTART then ENROLID for determinism)
    # Using window function to ensure deterministic selection of first record
    first_enrollment_window = Window.partitionBy("ENROLID").orderBy("DTSTART","DOBYR")
    dobyr_df = ccae_mdcr_enrollment_df \
        .withColumn("_rn", F.row_number().over(first_enrollment_window)) \
        .filter(F.col("_rn") == 1) \
        .select("ENROLID", "DOBYR") \
        .join(
            ccae_mdcr_enrollment_df.groupBy("ENROLID").agg(F.min("DTSTART").alias("MIN_DTSTART")),
            on="ENROLID",
            how="inner"
        )
    
    # Calculate first_event_date = min(MIN_VISIT_DATE, MIN_DTSTART) and age
    age_df = dobyr_df \
        .join(min_claims_duration_df.select("ENROLID", "MIN_VISIT_DATE"), on="ENROLID", how="left") \
        .withColumn("first_event_date", 
            F.when(F.col("MIN_VISIT_DATE").isNull(), F.col("MIN_DTSTART"))
             .when(F.col("MIN_DTSTART").isNull(), F.col("MIN_VISIT_DATE"))
             .otherwise(F.least(F.col("MIN_VISIT_DATE"), F.col("MIN_DTSTART")))) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("age", F.col("first_event_year") - F.col("DOBYR"))
    
    underage_enrollids_df = age_df.filter(F.col("age") < min_age).select("ENROLID").distinct()
    logging.info("%sENROLIDs with age < %d: %s", INDENT1, min_age, f"{underage_enrollids_df.count():,}")

    overage_enrollids_df = age_df.filter(F.col("age") > max_age).select("ENROLID").distinct()
    logging.info("%sENROLIDs with age > %d: %s", INDENT1, max_age, f"{overage_enrollids_df.count():,}")

    # union all the excluded enrollees
    excluded_enrollids_df = invalid_enrollids_df.union(missing_rx_df).union(insufficient_mem_days_df).union(insufficient_claims_duration_df).union(underage_enrollids_df).union(overage_enrollids_df).distinct()
    logging.info("%sTotal ENROLIDs to be excluded: %s, %.2f%%", INDENT1, f"{excluded_enrollids_df.count():,}", excluded_enrollids_df.count() / total_enrollids * 100)

    # save the excluded enrollees to parquet file
    excluded_enrollids_df.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "excluded_enrollids_ccae_mdcr"))



@time_execution
def mdcd_filter(spark: SparkSession, table_file_path_dict: dict, min_mem_days: int = 180, min_claims_duration_days: int = 30, min_age: int = 10, max_age: int = 110, processed_data_folder: str = None):

    """
    Filter MDCD data in the following steps:
    1. Filter enrollees based on the Drug data availability (DRUGCOVG)
    2. Filter enrollees based on the minimum member days
    3. Filter enrollees based on the minimum claims duration (max visit date - min visit date)
    4. Filter enrollees based on the Medicare dual eligibility status (MEDICARE)
    5. Filter enrollees based on minimum age at first event

    Then save the excluded enrollees to processed data folder
    """
    logging.info("MDCD Filtering:")
    # enrollment data (include DOBYR for age calculation)
    mdcd_enrollment_df = spark.read.parquet(*table_file_path_dict['MDCD']['t']).select("ENROLID", "DRUGCOVG", "MEMDAYS", "MEDICARE", "DTSTART", "DOBYR")
    mdcd_total_enrollids = mdcd_enrollment_df.select("ENROLID").distinct().count()
    logging.info("%sTotal ENROLIDs: %s", INDENT1, f"{mdcd_total_enrollids:,}")
    # claims data
    mdcd_i_df = spark.read.parquet(*table_file_path_dict['MDCD']['i']).select("ENROLID", "ADMDATE").withColumnRenamed("ADMDATE", "VISIT_DATE")
    mdcd_o_df = spark.read.parquet(*table_file_path_dict['MDCD']['o']).select("ENROLID", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")
    mdcd_d_df = spark.read.parquet(*table_file_path_dict['MDCD']['d']).select("ENROLID", "SVCDATE").withColumnRenamed("SVCDATE", "VISIT_DATE")

    mdcd_claims_df = mdcd_i_df.union(mdcd_o_df).union(mdcd_d_df)
    
    # Filter enrollees based on the Drug data availability (DRUGCOVG)   
    missing_drugcovg_df = mdcd_enrollment_df.filter(
        F.col("DRUGCOVG") != 1
    ).select("ENROLID").distinct()
    
    logging.info("%sENROLIDs with missing DRUGCOVG: %s", INDENT1, f"{missing_drugcovg_df.count():,}")
    
    # Filter enrollees based on the minimum member days
    min_mem_days_df = mdcd_enrollment_df.groupBy("ENROLID").agg(F.sum("MEMDAYS").alias("TOTAL_MEMDAYS"))
    insufficient_mem_days_df = min_mem_days_df.filter(
        F.col("TOTAL_MEMDAYS") < min_mem_days
    ).select("ENROLID").distinct()
    logging.info("%sENROLIDs with insufficient member days: %s", INDENT1, f"{insufficient_mem_days_df.count():,}")
    
    # Filter enrollees based on the minimum claims duration (max visit date - min visit date)
    min_claims_duration_df = mdcd_claims_df.groupBy("ENROLID").agg(F.max("VISIT_DATE").alias("MAX_VISIT_DATE"), F.min("VISIT_DATE").alias("MIN_VISIT_DATE"))
    insufficient_claims_duration_df = min_claims_duration_df.filter(
        F.datediff(F.col("MAX_VISIT_DATE"), F.col("MIN_VISIT_DATE")) < min_claims_duration_days
    ).select("ENROLID").distinct()
    logging.info("%sENROLIDs with insufficient claims duration: %s", INDENT1, f"{insufficient_claims_duration_df.count():,}")

    # Filter enrollees based on the Medicare dual eligibility status (MEDICARE) to avoid data leakage
    dual_eligible_enrollids_df = mdcd_enrollment_df.filter(
        F.col("MEDICARE") == 1
    ).select("ENROLID").distinct()
    logging.info("%sENROLIDs with Medicare dual eligibility: %s", INDENT1, f"{dual_eligible_enrollids_df.count():,}")

    # Filter enrollees based on minimum age at first event
    # Get DOBYR from first enrollment record (ordered by DTSTART then ENROLID for determinism)
    # Using window function to ensure deterministic selection of first record
    first_enrollment_window = Window.partitionBy("ENROLID").orderBy("DTSTART","DOBYR")
    dobyr_df = mdcd_enrollment_df \
        .withColumn("_rn", F.row_number().over(first_enrollment_window)) \
        .filter(F.col("_rn") == 1) \
        .select("ENROLID", "DOBYR") \
        .join(
            mdcd_enrollment_df.groupBy("ENROLID").agg(F.min("DTSTART").alias("MIN_DTSTART")),
            on="ENROLID",
            how="inner"
        )
    
    # Calculate first_event_date = min(MIN_VISIT_DATE, MIN_DTSTART) and age
    age_df = dobyr_df \
        .join(min_claims_duration_df.select("ENROLID", "MIN_VISIT_DATE"), on="ENROLID", how="left") \
        .withColumn("first_event_date", 
            F.when(F.col("MIN_VISIT_DATE").isNull(), F.col("MIN_DTSTART"))
             .when(F.col("MIN_DTSTART").isNull(), F.col("MIN_VISIT_DATE"))
             .otherwise(F.least(F.col("MIN_VISIT_DATE"), F.col("MIN_DTSTART")))) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("age", F.col("first_event_year") - F.col("DOBYR"))
    
    underage_enrollids_df = age_df.filter(F.col("age") < min_age).select("ENROLID").distinct()
    logging.info("%sENROLIDs with age < %d: %s", INDENT1, min_age, f"{underage_enrollids_df.count():,}")

    overage_enrollids_df = age_df.filter(F.col("age") > max_age).select("ENROLID").distinct()
    logging.info("%sENROLIDs with age > %d: %s", INDENT1, max_age, f"{overage_enrollids_df.count():,}")

    # union all the excluded enrollees    
    mdcd_excluded_enrollids_df = missing_drugcovg_df.union(insufficient_mem_days_df).union(insufficient_claims_duration_df).union(dual_eligible_enrollids_df).union(underage_enrollids_df).union(overage_enrollids_df).distinct()
    logging.info("%sTotal ENROLIDs to be excluded: %s, %.2f%%", INDENT1, f"{mdcd_excluded_enrollids_df.count():,}", mdcd_excluded_enrollids_df.count() / mdcd_total_enrollids * 100)

    # save the excluded enrollees to parquet file
    mdcd_excluded_enrollids_df.write.mode("overwrite").parquet(os.path.join(processed_data_folder, "excluded_enrollids_mdcd"))
    



def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S0 DATA FILTERING STARTING...")
    # Parse arguments
    args = parse_args()

    # Create output paths
    processed_data_folder = os.path.join(args.input_folder_path, "processed_data")
    os.makedirs(processed_data_folder, exist_ok=True)
    
    # Log configuration
    logging.info("\nConfiguration:")
    logging.info("%sBase directory: %s", INDENT1, args.base_dir)
    logging.info("%sProcessed data folder path: %s", INDENT1, processed_data_folder)
    logging.info("%sPayers: %s", INDENT1, args.payer_list)
    logging.info("%sTables: %s", INDENT1, args.table_list)
    logging.info("%sMinimum member days: %s", INDENT1, args.min_mem_days)
    logging.info("%sMinimum claims duration days: %s", INDENT1, args.min_claims_duration_days)
    logging.info("%sMinimum age: %s", INDENT1, args.min_age)
    logging.info("%sMaximum age: %s", INDENT1, args.max_age)
    
    
    # Create Spark session
    spark = SparkSession.builder.appName('MarketScan_Step0_Data_Filtering').getOrCreate()
    
    # Build file path dictionary
    table_file_path_dict = build_file_path_dict(args.payer_list, args.table_list, args.base_dir)
    
    # Filter data
    ccae_mdcr_filter(spark, table_file_path_dict, args.min_mem_days, args.min_claims_duration_days, args.min_age, args.max_age, processed_data_folder)
    mdcd_filter(spark, table_file_path_dict, args.min_mem_days, args.min_claims_duration_days, args.min_age, args.max_age, processed_data_folder)
    


    # Stop Spark
    spark.stop()
    logging.info("Log file: %s", log_file)
    logging.info("S0 DATA FILTERING COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
