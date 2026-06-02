#!/usr/bin/env python3
"""
Reclaim Data processing for external validation data (EHRSHOT)
"""

import logging
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import os
import sys
import argparse
from pyspark.sql.window import Window
import pandas as pd

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
    parser = argparse.ArgumentParser(description='Finalize vocabulary for EHRSHOT external validation data')
    parser.add_argument("--external_val_data_dir", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_EHRSHOT_DATA_ROOT", "/path/to/processed_ehrshot_data_root"), "v6", "ehrshot"),
                       help='Directory containing external validation data')
    parser.add_argument("--dictionary_dir", type=str, 
                       default=os.environ.get("MARKETSCAN_DICTIONARY_DIR", "/path/to/marketscan_dictionary_dir"),
                       help='Directory containing the Marketscan dictionary files')
    return parser.parse_args()


def create_spark_session(app_name: str) -> SparkSession:
    """
    Create SparkSession configured for local resources
    """
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate())
    return spark

@time_execution
def finalize_and_save_vocabulary(
    spark: SparkSession, 
    vocab_from_data: DataFrame, 
    dictionary_dir: str,
    external_val_data_dir: str) -> None:
    """
    Finalize vocabulary by augmenting vocab from data with all possible values by design.
    """
    logging.info("%sFinalizing vocabulary", INDENT1)

    # Union intra and inter timestamp vocab
    logging.info("%sTotal vocabulary from data: %s", INDENT1, f"{vocab_from_data.count():,}")

    # ********** Define all possible values by design ********** ##
    # Note:  DOBYR, AGE, NY are non-missing values
    # DOBYR: 1900-2024
    dobyr_full = [{"token": f"<DOBYR-{i}>", "category": "DOBYR"} for i in range(1900, 2015)]
    dobyr_full_df = spark.createDataFrame(dobyr_full).select("category", "token")
    
    # AGE: 10-110
    age_full = [{"token": f"<AGE-{i}>", "category": "AGE"} for i in range(10,111)]
    age_full_df = spark.createDataFrame(age_full).select("category", "token")
    
    # NY:
    ny_full = [{"token": "<NY>", "category": "NY"}]
    ny_full_df = spark.createDataFrame(ny_full).select("category", "token")
    
    # SEX: 1=Male, 2=Female, MISSING
    sex_full = [{"token": f"<SEX-{i}>", "category": "SEX"} for i in ["1", "2", "MISSING"]]
    sex_full_df = spark.createDataFrame(sex_full).select("category", "token")
    
    # ERLST and ERLED tokens: <ERLST-CCAE>, <ERLST-MDCR>, <ERLST-MDCD>, etc.
    planstart_full = [{"token": f"<ERLST-{payer}>", "category": "ERLST"} for payer in ["CCAE", "MDCR", "MDCD", "MISSING"]]
    planstart_full_df = spark.createDataFrame(planstart_full).select("category", "token")
    
    planend_full = [{"token": f"<ERLED-{payer}>", "category": "ERLED"} for payer in ["CCAE", "MDCR", "MDCD", "MISSING"]]
    planend_full_df = spark.createDataFrame(planend_full).select("category", "token")
    
    # PLANTYP: original data value + MISSING
    plantyp_unique_value =  pd.read_csv(os.path.join(dictionary_dir, "PLANTYP.csv"))['VALUE'].dropna().apply(lambda x: f"{int(x)}")
    plantyp_full = [{"token": f"<PLANTYP-{i}>", "category": "PLANTYP"} for i in plantyp_unique_value] + \
                   [{"token": "<PLANTYP-MISSING>", "category": "PLANTYP"}]
    plantyp_full_df = spark.createDataFrame(plantyp_full).select("category", "token")
    
    # CAP: 0=Non-capitated, 1=Capitated, MISSING
    cap_full = [{"token": f"<CAP-{i}>", "category": "CAP"} for i in ["0", "1", "MISSING"]]
    cap_full_df = spark.createDataFrame(cap_full).select("category", "token")
    
    # EGEOLOC: original data value + MISSING
    egeoloc_unique_value = pd.read_csv(os.path.join(dictionary_dir, "EGEOLOC.csv"))['VALUE'].dropna().apply(lambda x: f"{int(x):02d}")
    egeoloc_full = [{"token": f"<EGEOLOC-{i}>", "category": "EGEOLOC"} for i in egeoloc_unique_value] + \
                   [{"token": "<EGEOLOC-MISSING>", "category": "EGEOLOC"}]
    egeoloc_full_df = spark.createDataFrame(egeoloc_full).select("category", "token")
    
    # VT: Visit types
    vt_full = [{"token": f"<VT-{vt}>", "category": "VT"} for vt in ["inpatient", "outpatient", "pharmacy", "MISSING"]]
    vt_full_df = spark.createDataFrame(vt_full).select("category", "token")
    
    # DS: Discharge status (0-4 + MISSING)
    ds_full = [{"token": f"<DS-{i}>", "category": "DS"} for i in range(5)] + \
              [{"token": "<DS-MISSING>", "category": "DS"}]
    ds_full_df = spark.createDataFrame(ds_full).select("category", "token")
    
    # LS: Length of stay (0=<7days, 1=>=7days, MISSING)
    ls_full = [{"token": f"<LS-{i}>", "category": "LS"} for i in ["0", "1", "MISSING"]]
    ls_full_df = spark.createDataFrame(ls_full).select("category", "token")
    
    # COST: 00-99  + MISSING
    cost_full = [{"token": f"<COST-{i:02d}>", "category": "COST"} for i in range(100)] + \
                [{"token": "<COST-MISSING>", "category": "COST"}]
    cost_full_df = spark.createDataFrame(cost_full).select("category", "token")
    
    # ATT: 0-12 months (time between events in months)
    att_full = [{"token": f"<ATT-{i}>", "category": "ATT"} for i in range(13)]
    att_full_df = spark.createDataFrame(att_full).select("category", "token")
    
    # DX, PROC, RX: Add MISSING and NOMAP tokens (actual codes come from data/mapping)
    dx_special = [
        {"token": "<DX-MISSING>", "category": "DX"},
        {"token": "<DX-NOMAP>", "category": "DX"},
        {"token": "<DX-PRINCIPAL>", "category": "DX"},
        {"token": "<DX-SECONDARY>", "category": "DX"}
    ]
    dx_special_df = spark.createDataFrame(dx_special).select("category", "token")
    
    proc_special = [
        {"token": "<PROC-MISSING>", "category": "PROC"},
        {"token": "<PROC-NOMAP>", "category": "PROC"},
        {"token": "<PROC-PRINCIPAL>", "category": "PROC"},
        {"token": "<PROC-SECONDARY>", "category": "PROC"},
        {"token": "<PROC-COMBSTART>", "category": "PROC"},
        {"token": "<PROC-COMBEND>", "category": "PROC"},
    ]
    proc_special_df = spark.createDataFrame(proc_special).select("category", "token")
    
    rx_special = [
        {"token": "<RX-MISSING>", "category": "RX"},
        {"token": "<RX-NOMAP>", "category": "RX"},
        {"token": "<RX-COMBSTART>", "category": "RX"},
        {"token": "<RX-COMBEND>", "category": "RX"}
    ]
    rx_special_df = spark.createDataFrame(rx_special).select("category", "token")
    

     # Instruct tokens for specified downstream tasks: <INSTRUCT-COST>, <INSTRUCT-DX>>
    instruct_tokens = [
        {"token": "<INSTRUCT-COST>", "category": "INSTRUCT"},
        {"token": "<INSTRUCT-DX>", "category": "INSTRUCT"}
    ]
    instruct_tokens_df = spark.createDataFrame(instruct_tokens).select("category", "token")

    # Union all theoretical vocab
    vocab_by_design = sex_full_df \
        .union(dobyr_full_df) \
        .union(age_full_df) \
        .union(ny_full_df) \
        .union(planstart_full_df) \
        .union(planend_full_df) \
        .union(plantyp_full_df) \
        .union(cap_full_df) \
        .union(egeoloc_full_df) \
        .union(vt_full_df) \
        .union(ds_full_df) \
        .union(ls_full_df) \
        .union(cost_full_df) \
        .union(att_full_df) \
        .union(dx_special_df) \
        .union(proc_special_df) \
        .union(rx_special_df) \
        .union(instruct_tokens_df)
    
    # Full join vocab_from_data and vocab_by_design, fill missing token_freq with 0
    final_vocab = vocab_from_data \
        .join(vocab_by_design, on=["category", "token"], how="full") \
        .withColumn("token_freq", F.coalesce(F.col("token_freq"), F.lit(0))) \
        .distinct()
    
    # Cache final_vocab since it's used for count, logging, and save
    final_vocab.cache()
    
    logging.info("%sFinal vocabulary size: %s", INDENT1, f"{final_vocab.count():,}")
    logging.info("%sFinal Vocabulary by category:", INDENT1)
    spark_log_contingency_table(final_vocab, "category")

    # Save vocabulary using Spark's native write (avoids expensive toPandas collection to driver)
    vocab_save_path = os.path.join(external_val_data_dir, "vocab")
    final_vocab.write.mode("overwrite").parquet(vocab_save_path)
    logging.info("%sFinal Vocabulary saved to: %s", INDENT1, vocab_save_path)

    final_vocab.unpersist()




def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S3 FINALIZE VOCAB STARTING...")
    
    # Parse arguments
    args = parse_args()
    os.makedirs(args.external_val_data_dir, exist_ok=True)
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sExternal validation data directory: %s", INDENT1, args.external_val_data_dir)
    logging.info("%sDictionary directory: %s", INDENT1, args.dictionary_dir)

    # Create Spark session
    spark = create_spark_session('EHRSHOT_Step3_Finalize_Vocab')

    # Read combined tokens from data
    logging.info("%sCombining all tokens from data...", INDENT1)
    combined_tokens_from_data = (
        spark.read.parquet(os.path.join(args.external_val_data_dir, "all_demographic_tokens")).select("category", "token")
        .unionByName(spark.read.parquet(os.path.join(args.external_val_data_dir, "all_enrollment_tokens")).select("category", "token"))
        .unionByName(spark.read.parquet(os.path.join(args.external_val_data_dir, "all_claim_tokens")).select("category", "token"))
        .unionByName(spark.read.parquet(os.path.join(args.external_val_data_dir, "all_anchor_tokens")).select("category", "token"))
        .unionByName(spark.read.parquet(os.path.join(args.external_val_data_dir, "all_att_tokens")).select("category", "token"))
    )

    # Calculate vocabulary from data
    logging.info("%sCalculating vocabulary and frequency from data...", INDENT1)
    vocab_from_data = combined_tokens_from_data \
        .groupBy("category", "token") \
        .agg(F.count("*").alias("token_freq")) \
        .orderBy("category", F.desc("token_freq")) \
        .select("category", "token", "token_freq")

    # Finalize vocabulary
    logging.info("%sFinalizing vocabulary by fullfilling all possible values by design...", INDENT1)
    finalize_and_save_vocabulary(spark, vocab_from_data, args.dictionary_dir, args.external_val_data_dir)

    # Stop Spark session
    spark.stop()
    logging.info("S3 FINALIZE VOCAB COMPLETED SUCCESSFULLY...")
    logging.info("Log file: %s", log_file)

if __name__ == "__main__":
    main()
