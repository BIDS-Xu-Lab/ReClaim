#!/usr/bin/env python3
"""
Reclaim Data Filter
Creates demographic sequences and vocabulary from first patient visits
"""

# Set up environment BEFORE importing pyspark
import os
import sys

# Ensure Java is in PATH (required for Spark)
java_home = os.environ.get('JAVA_HOME', '/usr/local/openjdk-11')
os.environ['PATH'] = os.path.join(java_home, 'bin') + os.pathsep + os.environ.get('PATH', '')

# Fix PySpark Python version mismatch - ensure workers use the same Python as driver
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Now import pyspark and other modules
import logging
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType
from datetime import date
import pyspark.sql.functions as F
from glob import glob
import argparse
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
                       default=os.path.join(os.environ.get("YNHH_DATA_ROOT", "/path/to/ynhh_data_root"), "ynhh_omop"),
                       help='Base directory containing YNHH OMOP data')
    parser.add_argument("--mapping_dir", type=str, 
                       default='../../vocab_mapping/mapping',
                       help='Directory containing the mapping files')
    parser.add_argument("--external_val_data_dir", type=str, 
                       default=os.path.join(os.environ.get("PROCESSED_YNHH_DATA_ROOT", "/path/to/processed_ynhh_data_root"), "v6", "ynhh"),
                       help='Directory containing external validation data')            
    parser.add_argument("--min_mem_days", type=int, default=180, help='Minimum (enrollment) member days required for a patient to be included')
    parser.add_argument("--min_duration_days", type=int, default=30, help='Minimum claims duration required for a patient to be included')
    parser.add_argument("--min_age", type=int, default=10, help='Minimum age at first event required for a patient to be included')
    parser.add_argument("--max_age", type=int, default=110, help='Maximum age at first event required for a patient to be included')
    parser.add_argument("--sample_size", type=int, default=110000, help='Sample size for the external validation dataset')
    parser.add_argument("--seed", type=int, default=66, help='Random seed for the external validation dataset')
    return parser.parse_args()



def create_spark_session(app_name: str) -> SparkSession:
    driver_memory = "80g"
    max_result_size = "10g"
    # Shuffle partitions: typically 2-3x number of cores for better parallelism
    shuffle_partitions = "64"
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", max_result_size) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    return spark

def merge_overlapping_periods_monthly(payer_plan_period_df: DataFrame):
    """
    Merge overlapping or adjacent date ranges for each person_id + PAYER combination.
    Keeps original dates but uses 1 month gap as threshold for separation.
    PLANTYP and CAP become features of the enrollment (take first by earliest start date).
    
    Result: For each person_id, PAYER-specific enrollments that are 
    sequential, chronological, and non-overlapping.
    """
    
    # Step 0: Drop enrollments shorter than 30 days (also handles end < start)
    payer_plan_period_df = payer_plan_period_df.filter(
        F.datediff(F.col("payer_plan_period_end_date"), F.col("payer_plan_period_start_date")) >= 30
    )
    
    # Step 1: Define window partitioned by person_id + PAYER only, ordered by start date
    window_spec = Window.partitionBy("person_id", "PAYER") \
                        .orderBy("payer_plan_period_start_date")
    
    # Step 2: Calculate the cumulative maximum end date up to the previous row
    payer_plan_period_df_with_max = payer_plan_period_df.withColumn(
        "prev_max_end",
        F.max("payer_plan_period_end_date").over(
            window_spec.rowsBetween(Window.unboundedPreceding, -1)
        )
    )
    
    # Step 3: Flag when a new enrollment starts (gap > 1 month detected)
    payer_plan_period_df_with_flag = payer_plan_period_df_with_max.withColumn(
        "months_gap",
        F.months_between(
            F.col("payer_plan_period_start_date"), 
            F.col("prev_max_end")
        )
    ).withColumn(
        "is_new_enrollment",
        F.when(
            F.col("prev_max_end").isNull() |  # First row
            (F.col("months_gap") > 1),         # Gap > 1 month
            1
        ).otherwise(0)
    )
    
    # Step 4: Create group ID by cumulative sum of the flags
    payer_plan_period_df_with_group = payer_plan_period_df_with_flag.withColumn(
        "enrollment_group_id",
        F.sum("is_new_enrollment").over(window_spec)
    )
    
    # Step 5: Aggregate to get merged enrollment periods
    # Take first PLANTYP and CAP (by earliest start date) using first() on ordered window
    payer_plan_period_collapsed = payer_plan_period_df_with_group.groupBy(
        "person_id", "PAYER", "enrollment_group_id"
    ).agg(
        F.first("PLANTYP").alias("PLANTYP"),  # first by start date (due to prior ordering)
        F.first("CAP").alias("CAP"),
        F.min("payer_plan_period_start_date").alias("enrollment_start_date"),
        F.max("payer_plan_period_end_date").alias("enrollment_end_date"),
        F.count("*").alias("plans_merged")
    ).drop("enrollment_group_id")
    
    # Step 6: Calculate duration in days
    payer_plan_period_collapsed = payer_plan_period_collapsed.withColumn(
        "enrollment_duration_days",
        F.datediff(F.col("enrollment_end_date"), F.col("enrollment_start_date"))
    )
    
    # Step 7: Coalesce NULL/None values to "MISSING" for PAYER, PLANTYP, CAP
    # Cast CAP to string first since it may be inferred as BIGINT
    payer_plan_period_collapsed = payer_plan_period_collapsed \
        .withColumn("PAYER", F.coalesce(F.col("PAYER").cast("string"), F.lit("MISSING"))) \
        .withColumn("PLANTYP", F.coalesce(F.col("PLANTYP").cast("string"), F.lit("MISSING"))) \
        .withColumn("CAP", F.coalesce(F.col("CAP").cast("string"), F.lit("MISSING")))
    
    return payer_plan_period_collapsed



def sample_eligible_personids(eligible_personids: DataFrame, sample_size: int, seed: int, external_val_data_dir: str) -> DataFrame:
    """
    Sample eligible person IDs using PySpark and save as parquet.
    
    Args:
        eligible_personids: Spark DataFrame containing eligible person_ids
        sample_size: Number of person_ids to sample
        seed: Random seed for reproducibility
        external_val_data_dir: Directory to save the sampled dataset
        
    Returns:
        Spark DataFrame containing sampled person_ids (sorted by person_id)
    """
    eligible_count = eligible_personids.count()
    logging.info("%sTotal eligible Person IDs: %s", INDENT1, f"{eligible_count:,}")
    
    sample_fraction = round( sample_size / eligible_count, 2)
    sampled_personids = eligible_personids \
        .orderBy("person_id") \
        .sample(withReplacement=False, fraction=sample_fraction, seed=seed) \
        .orderBy("person_id")
    
    # Log sampling info and first/last 10 person_ids
    first_10 = [row["person_id"] for row in sampled_personids.limit(10).collect()]
    last_10 = [row["person_id"] for row in sampled_personids.orderBy(F.col("person_id").desc()).limit(10).collect()][::-1]
    logging.info("%sSampled %s person IDs (seed=%d)", INDENT1, f"{sampled_personids.count():,}", seed)
    logging.info("%sFirst 10 sampled person IDs: %s", INDENT1, first_10)
    logging.info("%sLast 10 sampled person IDs: %s", INDENT1, last_10)
    
    # Save as parquet
    output_path = os.path.join(external_val_data_dir, "sampled_eligible_personids")
    sampled_personids.write.mode("overwrite").parquet(output_path)
    logging.info("%sSaved sampled eligible person IDs to %s", INDENT1, output_path)
    
    return sampled_personids


@time_execution
def ynhh_filter(spark: SparkSession, omop_dir: str, mapping_dir: str, min_mem_days: int, min_duration_days: int, min_age: int, max_age: int, external_val_data_dir: str, sample_size: int, seed: int):
    """
    Filter YNHH data in the following steps:
    1. Filter enrollees based on the minimum member days
    2. Filter enrollees based on the minimum duration (max visit date - min visit date)
    3. Filter enrollees based on minimum age at first event

    """
    logging.info("YNHH Filtering:")
    
    # Load enrollment data from both datasets (include DOBYR for age calculation)
    ynhh_person = spark.read.parquet(os.path.join(omop_dir, "person")).select("person_id", "gender_concept_id", "date_of_birth")    
    total_personids = ynhh_person.select("person_id").distinct().count()
    logging.info("%sTotal unique person IDs: %s", INDENT1, f"{total_personids:,}")

    # Load payer plan period data
    ynhh_payer_plan_period = spark.read.parquet(os.path.join(omop_dir, "payer_plan_period")) \
        .withColumn("payer_plan_period_end_date", F.coalesce(F.col("payer_plan_period_end_date"), F.lit("2025-03-31").cast("date"))) \
        .filter((F.col("payer_plan_period_start_date") >= F.lit("2008-01-01").cast("date")) & (F.col("payer_plan_period_end_date") >= F.lit("2008-01-01").cast("date"))) \
        .filter((F.col("payer_plan_period_start_date") <= F.lit("2025-03-31").cast("date")) & (F.col("payer_plan_period_end_date") <= F.lit("2025-03-31").cast("date")))

    plantype_marketscan_mapping = spark.read.csv(os.path.join(mapping_dir, "payer_concept_annotated.csv"), header=True, inferSchema=True) \
        .withColumn("PLANTYP", F.coalesce(F.col("PLANTYP"), F.lit("MISSING"))) \
        .withColumn("CAP", F.coalesce(F.col("CAP"), F.lit("MISSING"))) \
        .withColumn("PAYER", F.coalesce(F.col("PAYER"), F.lit("MISSING"))) \
        .select("payer_concept_id", "PAYER", "PLANTYP", "CAP")
    
    plantype_marketscan_mapping.show(10, truncate=False)

    ynhh_payer_plan_period_mapped = (
        ynhh_payer_plan_period
        .join(plantype_marketscan_mapping, on="payer_concept_id", how="left")
        .select("person_id", "PAYER", "PLANTYP", "CAP", "payer_concept_id", "payer_plan_period_start_date", "payer_plan_period_end_date")
        .withColumn("PAYER", F.coalesce(F.col("PAYER"), F.lit("MISSING"))) \
        .withColumn("PLANTYP", F.coalesce(F.col("PLANTYP"), F.lit("MISSING"))) \
        .withColumn("CAP", F.coalesce(F.col("CAP"), F.lit("MISSING")))
    )

    ynhh_payer_plan_period_mapped.show(10, truncate=False)
    # Log distribution of PAYER, PLANTYP, CAP, payer_concept_id
    logging.info("%sLogging distribution of PAYER", INDENT1)
    spark_log_contingency_table(ynhh_payer_plan_period_mapped, "PAYER")
    
    logging.info("%sLogging distribution of PLANTYP", INDENT1)
    spark_log_contingency_table(ynhh_payer_plan_period_mapped, "PLANTYP")
    
    logging.info("%sLogging distribution of CAP", INDENT1)
    spark_log_contingency_table(ynhh_payer_plan_period_mapped, "CAP")
    
    logging.info("%sLogging distribution of payer_concept_id", INDENT1)
    spark_log_contingency_table(ynhh_payer_plan_period_mapped, "payer_concept_id")

    # collapse overlapping/duplicated enrollment periods for each person_id + PAYER + PLANTYP combination
    ynhh_payer_plan_period_collapsed = merge_overlapping_periods_monthly(ynhh_payer_plan_period_mapped)
    ynhh_payer_plan_period_collapsed.write.mode("overwrite").parquet(os.path.join(external_val_data_dir, "ynhh_payer_plan_period_collapsed"))

    #######################
    ### start filtering ### 
    #######################
    personids_with_enrollment = ynhh_payer_plan_period_collapsed.select("person_id").distinct()
    
    # Filter out person_ids with no enrollment records (0 member days)
    personids_without_enrollment = ynhh_person.select("person_id").distinct() \
        .join(personids_with_enrollment, on="person_id", how="left_anti")
    logging.info("%sPerson IDs with no enrollment records (0 member days): %s", INDENT1, f"{personids_without_enrollment.count():,}")
    
    # Filter out person_ids with insufficient member days (less than min_mem_days)
    personids_with_low_mem_days = ynhh_payer_plan_period_collapsed.groupBy("person_id").agg(F.sum("enrollment_duration_days").alias("TOTAL_MEMDAYS")) \
        .filter(F.col("TOTAL_MEMDAYS") < min_mem_days) \
        .select("person_id").distinct()
    logging.info("%sPerson IDs with enrollment but insufficient member days (< %d): %s", INDENT1, min_mem_days, f"{personids_with_low_mem_days.count():,}")
    
    # Union both groups
    insufficient_mem_days_personids = personids_without_enrollment.union(personids_with_low_mem_days).distinct()
    logging.info("%sTotal Person IDs with insufficient member days: %s", INDENT1, f"{insufficient_mem_days_personids.count():,}")

    # total visit start and end dates
    visit_start_end = spark.read.parquet(os.path.join(omop_dir, "visit_occurrence")) \
        .withColumn("visit_start_date", F.col("visit_start_datetime").cast("date")) \
        .withColumn("visit_end_date", F.col("visit_end_datetime").cast("date")) \
        .filter((F.col("visit_start_date") >= F.lit("2008-01-01").cast("date")) & (F.col("visit_end_date") >= F.lit("2008-01-01").cast("date"))) \
        .filter((F.col("visit_start_date") <= F.lit("2025-03-31").cast("date")) & (F.col("visit_end_date") <= F.lit("2025-03-31").cast("date"))) \
        .groupBy("person_id").agg(F.min("visit_start_date").alias("MIN_VISIT_DATE"), F.max("visit_end_date").alias("MAX_VISIT_DATE")) 
    
    # insufficient visit duration
    insufficient_visit_duration_personids = visit_start_end \
        .filter(F.datediff(F.col("MAX_VISIT_DATE"), F.col("MIN_VISIT_DATE")) < min_duration_days) \
        .select("person_id").distinct()
    logging.info("%sPerson IDs with insufficient EHR visit duration: %s", INDENT1, f"{insufficient_visit_duration_personids.count():,}")
    
    # age at first event
    age_at_first_event = ynhh_person \
        .join(ynhh_payer_plan_period_collapsed, on="person_id", how="left") \
        .join(visit_start_end, on="person_id", how="left") \
        .withColumn("first_event_date", 
            F.when(F.col("MIN_VISIT_DATE").isNull(), F.col("enrollment_start_date"))
             .when(F.col("enrollment_start_date").isNull(), F.col("MIN_VISIT_DATE"))
             .otherwise(F.least(F.col("MIN_VISIT_DATE"), F.col("enrollment_start_date")))) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("year_of_birth", F.year(F.col("date_of_birth"))) \
        .withColumn("age", F.col("first_event_year") - F.col("year_of_birth"))

    spark_log_contingency_table(age_at_first_event, "age")

    underage_personids = age_at_first_event.filter(F.col("age") < min_age).select("person_id").distinct()
    logging.info("%sPerson IDs with age < %d: %s", INDENT1, min_age, f"{underage_personids.count():,}")
    overage_personids = age_at_first_event.filter(F.col("age") > max_age).select("person_id").distinct()
    logging.info("%sPerson IDs with age > %d: %s", INDENT1, max_age, f"{overage_personids.count():,}")


    # union all the excluded enrollees
    excluded_personids = insufficient_mem_days_personids.union(insufficient_visit_duration_personids).union(underage_personids).union(overage_personids).distinct()
    logging.info("%sTotal Person IDs to be excluded: %s, %.2f%%", INDENT1, f"{excluded_personids.count():,}", excluded_personids.count() / total_personids * 100)

    # save the excluded enrollees to parquet file
    excluded_personids.write.mode("overwrite").parquet(os.path.join(external_val_data_dir, "excluded_personids"))

    # Find eligible person_ids by left_anti join with excluded_personids
    eligible_personids = ynhh_person.select("person_id").distinct() \
        .join(excluded_personids, on="person_id", how="left_anti")
    
    # Sample eligible person_ids and save as HF Dataset
    sample_eligible_personids(eligible_personids, sample_size, seed, external_val_data_dir)


def test_merge_overlapping_periods_monthly(spark: SparkSession):
    """Test the merge_overlapping_periods_monthly function with sample data."""
    logging.info("Testing the merge_overlapping_periods_monthly function...")
    schema = StructType([
        StructField("person_id", StringType(), True),
        StructField("PAYER", StringType(), True),
        StructField("PLANTYP", StringType(), True),
        StructField("CAP", StringType(), True),
        StructField("payer_plan_period_start_date", DateType(), True),
        StructField("payer_plan_period_end_date", DateType(), True),
    ])

    test_data = [
        ("P1", "PayerA", "HMO", "C1", date(2010, 1, 15), date(2012, 6, 30)),   # Plan A
        ("P1", "PayerA", "PPO", "C2", date(2011, 3, 1), date(2013, 9, 15)),    # Plan B
        ("P1", "PayerA", "HMO", "C1", date(2013, 5, 1), date(2014, 2, 28)),    # Plan C
        ("P1", "PayerA", "EPO", "C3", date(2014, 3, 20), date(2015, 1, 31)),   # Plan D
        ("P1", "PayerA", "PPO", "C2", date(2016, 5, 1), date(2017, 8, 15)),    # Plan E
        ("P1", "PayerA", "HMO", "C1", date(2017, 6, 1), date(2018, 12, 31)),   # Plan F
    ]

    df = spark.createDataFrame(test_data, schema)

    result = merge_overlapping_periods_monthly(df)
    result.orderBy("person_id", "PAYER", "enrollment_start_date").show(truncate=False)


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging(level=logging.INFO)
    logging.info("S0 DATA FILTERING STARTING...")
    # Parse arguments
    args = parse_args()

    # Create output paths
    os.makedirs(args.external_val_data_dir, exist_ok=True)
    
    # Log configuration
    logging.info("\nConfiguration:")
    logging.info("%sOMOP directory: %s", INDENT1, args.omop_dir)
    logging.info("%sExternal validation data directory: %s", INDENT1, args.external_val_data_dir)
    logging.info("%sMinimum member days: %s", INDENT1, args.min_mem_days)
    logging.info("%sMinimum visit duration days: %s", INDENT1, args.min_duration_days)
    logging.info("%sMinimum age: %s", INDENT1, args.min_age)
    logging.info("%sMaximum age: %s", INDENT1, args.max_age)
    logging.info("%sSample size: %s", INDENT1, args.sample_size)
    logging.info("%sRandom seed: %s", INDENT1, args.seed)
    
    
    # Create Spark session
    spark = create_spark_session('YNHH_Step0_Data_Filtering')

    # Test the merge function
    test_merge_overlapping_periods_monthly(spark)
    
    # Filter data
    ynhh_filter(spark, args.omop_dir, args.mapping_dir, args.min_mem_days, args.min_duration_days, args.min_age, args.max_age, args.external_val_data_dir, args.sample_size, args.seed)
    


    # Stop Spark
    spark.stop()
    logging.info("Log file: %s", log_file)
    logging.info("YNHH DATA FILTERING COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
