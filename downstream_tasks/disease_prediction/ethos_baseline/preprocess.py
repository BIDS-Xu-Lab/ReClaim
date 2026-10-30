#!/usr/bin/env python3
"""
ETHOS Data Preprocessing

Converts the EHRSHOT OMOP CDM tables into the patient timelines an ETHOS checkpoint reads, in four
steps that run in order and write their intermediates to --out_dir.

Pipeline: OMOP -> RxNorm/ATC crosswalk -> cohort filter -> tokens -> timelines (HDF5)

The vocabulary is frozen: it was built when the model was trained, so a token string it does not
contain cannot be encoded. Such tokens are dropped and the drop rate is logged.

No code here is taken from the ETHOS repository, https://github.com/ipolharvard/ethos-paper, but
every rule that decides how a code reaches its tokens follows that repository's tokenizer rather
than the more obvious reading, because the model was trained on what the tokenizer produced. Each
such rule is commented where it applies. The reference tables that name a code are the tokenizer's
own, and --ref_dir points at the ethos/data directory of a checkout rather than at a copy kept here.
ETHOS is described in Renc, Jia, Samir, Was, Li, Bates and Sitek, "Zero shot health trajectory
prediction using transformer", npj Digital Medicine (2024), https://doi.org/10.1038/s41746-024-01235-0

ETHOS reads its own vocabulary rather than ReClaim's, which is why this baseline tokenizes EHRSHOT
itself instead of reading traj_test_pd.parquet. The cohort is what the two share, and that is applied
later, by transform_predefined_cohort.py.

    python preprocess.py --omop_dir <ehrshot_omop> --vocab <vocab.pkl> --out_dir <work_dir>
"""

import logging
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType, IntegerType
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# PySpark workers must run the same interpreter as the driver
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from utils import (setup_logging, time_execution, spark_log_contingency_table, omop_table_path,
                   omop_table_exists)
from vocabulary import (Vocabulary, SeparatorInjector, TOKEN_DTYPE, TIME_DTYPE, SECONDS_IN_YEAR)

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "

# Vocabulary objects are cached per process, so a worker reads the pickle at most once
_VOCABULARY_CACHE = {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Turn the EHRSHOT OMOP tables into ETHOS timelines')
    parser.add_argument("--omop_dir", type=str, default=os.environ.get("EHRSHOT_OMOP_DIR"),
                       required="EHRSHOT_OMOP_DIR" not in os.environ,
                       help='Base directory containing EHRSHOT OMOP tables')
    parser.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "processed"),
                       help='Directory the intermediates and the timelines are written to')
    parser.add_argument("--vocab", type=str, default=os.environ.get("ETHOS_VOCAB"),
                       required="ETHOS_VOCAB" not in os.environ,
                       help='Vocabulary the checkpoint was trained with. Not distributed here')
    parser.add_argument("--ref_dir", type=str, default=os.environ.get("ETHOS_REF_DIR"),
                       required="ETHOS_REF_DIR" not in os.environ,
                       help="The ETHOS repository's ethos/data directory, holding the code system "
                            "reference tables. Not distributed here")
    parser.add_argument("--timelines", type=str, default=None,
                       help='Timelines file to write (default: <out_dir>/timelines.hdf5)')
    parser.add_argument("--date_start", type=str, default="2008-01-01",
                       help='Earliest visit date to tokenize')
    parser.add_argument("--date_end", type=str, default="2025-03-31",
                       help='Latest visit date to tokenize')
    parser.add_argument("--min_duration_days", type=int, default=30,
                       help='Fewest days between a patient first and last visit')
    parser.add_argument("--min_age", type=int, default=10,
                       help='Youngest age at the first event')
    parser.add_argument("--max_age", type=int, default=110,
                       help='Oldest age at the first event')
    args = parser.parse_args()
    args.timelines = args.timelines or os.path.join(args.out_dir, "timelines.hdf5")
    args.crosswalk = os.path.join(args.out_dir, "ehrshot_rxnorm_to_atc.csv.gz")
    return args


def create_spark_session(app_name: str) -> SparkSession:
    """
    Create a SparkSession configured for local resources.

    Dates are read as instants in UTC, so that an age computed from them is the same no matter which
    timezone the machine keeps or whether daylight saving is in effect on the day of an event.

    Args:
        app_name: Name the session is reported under

    Returns:
        SparkSession: The session every step of this script runs on
    """
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate())
    return spark


def load_vocabulary(vocab_path: str) -> Vocabulary:
    """
    Load the vocabulary, caching it per process.

    Args:
        vocab_path: Path to the vocabulary pickle

    Returns:
        Vocabulary instance
    """
    if vocab_path not in _VOCABULARY_CACHE:
        _VOCABULARY_CACHE[vocab_path] = Vocabulary(vocab_path)
    return _VOCABULARY_CACHE[vocab_path]


# =============================================================================
# RxNorm to ATC crosswalk
# =============================================================================

RXNORM_TO_ATC_RELATIONSHIPS = ["RxNorm - ATC pr lat", "RxNorm - ATC pr up"]


@time_execution
def build_rxnorm_to_atc(spark: SparkSession, omop_dir: str) -> DataFrame:
    """
    Build the crosswalk from an RxNorm drug code to its ATC 5th level code.

    The concept table supplies both sides of the bridge and the relationship table connects them.
    Relationships that have been retired are left out.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP tables

    Returns:
        DataFrame containing rxnorm_code and atc_code, empty when the tables cannot bridge the two
    """
    concept = spark.read.parquet(omop_table_path(omop_dir, "concept")) \
        .select("concept_id", "vocabulary_id", "concept_class_id", "concept_code")

    rxnorm_concepts = concept \
        .filter(F.col("vocabulary_id") == "RxNorm") \
        .select(F.col("concept_id").alias("rxnorm_concept_id"),
                F.col("concept_code").alias("rxnorm_code"))
    atc_concepts = concept \
        .filter((F.col("vocabulary_id") == "ATC") & (F.col("concept_class_id") == "ATC 5th")) \
        .select(F.col("concept_id").alias("atc_concept_id"),
                F.col("concept_code").alias("atc_code"))
    logging.info("%sRxNorm concepts: %s | ATC 5th level concepts: %s", INDENT1,
                 f"{rxnorm_concepts.count():,}", f"{atc_concepts.count():,}")

    if not omop_table_exists(omop_dir, "concept_relationship"):
        logging.warning("%sconcept_relationship is not present, so the crosswalk stays empty and "
                        "drugs will be left out of the timelines", INDENT1)
        return spark.createDataFrame([], schema="rxnorm_code string, atc_code string")

    relationships = spark.read.parquet(omop_table_path(omop_dir, "concept_relationship")) \
        .filter(F.col("relationship_id").isin(RXNORM_TO_ATC_RELATIONSHIPS)) \
        .filter(F.col("invalid_reason").isNull()) \
        .select("concept_id_1", "concept_id_2")
    relationship_count = relationships.count()
    logging.info("%sRxNorm to ATC relationships: %s", INDENT1, f"{relationship_count:,}")

    if relationship_count == 0:
        logging.warning("%sNo RxNorm to ATC relationship is recorded, so the crosswalk stays empty "
                        "and drugs will be left out of the timelines", INDENT1)
        return spark.createDataFrame([], schema="rxnorm_code string, atc_code string")

    crosswalk = relationships \
        .join(rxnorm_concepts, F.col("concept_id_1") == F.col("rxnorm_concept_id"), "inner") \
        .join(atc_concepts, F.col("concept_id_2") == F.col("atc_concept_id"), "inner") \
        .select("rxnorm_code", "atc_code") \
        .distinct()

    logging.info("%sCrosswalk rows: %s (distinct RxNorm codes: %s)", INDENT1,
                 f"{crosswalk.count():,}", f"{crosswalk.select('rxnorm_code').distinct().count():,}")
    return crosswalk


# =============================================================================
# Selecting the patients that can be tokenized
# =============================================================================

GENDER_MALE = 8507


GENDER_FEMALE = 8532


SEX_MALE_TOKEN = "SEX_M"


SEX_FEMALE_TOKEN = "SEX_F"


MIN_ENROLLMENT_DAYS = 30


@time_execution
def construct_person_context(spark: SparkSession, omop_dir: str) -> DataFrame:
    """
    Build the static context of every patient whose timeline can be tokenized.

    A patient is tokenizable when two conditions hold:
    1. year_of_birth is known, since it anchors all temporal information in the timeline
    2. the gender maps onto one of the two SEX tokens contained in the model vocabulary

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data

    Returns:
        DataFrame containing person_id, sex_token and year_of_birth
    """
    logging.info("%sExtracting demographic information", INDENT1)

    ehrshot_person = spark.read.parquet(omop_table_path(omop_dir, "person")) \
        .select("person_id", "gender_concept_id", "year_of_birth")
    total_personids = ehrshot_person.select("person_id").distinct().count()
    logging.info("%sTotal unique person IDs: %s", INDENT1, f"{total_personids:,}")

    # Log how gender is distributed before it is narrowed down to male/female
    spark_log_contingency_table(ehrshot_person, "gender_concept_id")

    # year_of_birth can't be missing by design since it is used to anchor temporal information,
    # and a gender outside male/female has no token to encode it with, so both are dropped here.
    # ReClaim keeps an unusable gender and writes it as a missing value instead, which the model
    # vocabulary has no room for; the two cohorts still agree as long as the count below is zero.
    with_birth_year = ehrshot_person.filter(F.col("year_of_birth").isNotNull())
    with_birth_year_personids = with_birth_year.select("person_id").distinct().count()
    logging.info("%sPerson IDs without a birth year: %s", INDENT1,
                 f"{total_personids - with_birth_year_personids:,}")

    person_context = with_birth_year \
        .withColumn("sex_token",
            F.when(F.col("gender_concept_id") == GENDER_MALE, F.lit(SEX_MALE_TOKEN))
             .when(F.col("gender_concept_id") == GENDER_FEMALE, F.lit(SEX_FEMALE_TOKEN))) \
        .filter(F.col("sex_token").isNotNull()) \
        .select("person_id", "sex_token", F.col("year_of_birth").cast("int").alias("year_of_birth")) \
        .distinct()

    tokenizable_personids = person_context.count()
    logging.info("%sPerson IDs with a birth year but no male/female gender: %s", INDENT1,
                 f"{with_birth_year_personids - tokenizable_personids:,}")
    logging.info("%sTokenizable person IDs (birth year known, gender male/female): %s, %.2f%%",
                 INDENT1, f"{tokenizable_personids:,}", tokenizable_personids / total_personids * 100)
    spark_log_contingency_table(person_context, "sex_token")

    return person_context


@time_execution
def find_enrollment_start_dates(spark: SparkSession, omop_dir: str, date_start: str,
                                date_end: str) -> DataFrame:
    """
    Read the insurance enrollment periods a first event can be dated by.

    A payer never becomes a token, since insurance is not part of a patient timeline, but a patient
    can be enrolled long before their first recorded visit, and ReClaim dates the first event at
    the earlier of the two. A period counts when it lies inside the date window and lasts at least
    MIN_ENROLLMENT_DAYS, which are the two conditions ReClaim imposes as well.

    ReClaim goes on to merge periods of the same payer that sit less than a month apart, replacing
    each run of periods with its earliest start. That leaves the earliest start of a patient where
    it was, so the age a patient is excluded on is unaffected; only their latest start can move,
    and with it the upper age bound, which no EHRSHOT patient comes near. Merging is therefore left
    out, and with it the payer taxonomy it would need.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        date_start: Earliest enrollment date to keep
        date_end: Latest enrollment date to keep

    Returns:
        DataFrame containing person_id and enrollment_start_date, one row per period
    """
    if not omop_table_exists(omop_dir, "payer_plan_period"):
        logging.info("%sNo payer_plan_period table, dating first events by visits alone", INDENT1)
        return spark.createDataFrame([], "person_id long, enrollment_start_date date")

    enrollment_start_dates = spark.read.parquet(omop_table_path(omop_dir, "payer_plan_period")) \
        .filter((F.col("payer_plan_period_start_DATE") >= F.lit(date_start).cast("date")) & (F.col("payer_plan_period_end_DATE") >= F.lit(date_start).cast("date"))) \
        .filter((F.col("payer_plan_period_start_DATE") <= F.lit(date_end).cast("date")) & (F.col("payer_plan_period_end_DATE") <= F.lit(date_end).cast("date"))) \
        .filter(F.datediff(F.col("payer_plan_period_end_DATE"), F.col("payer_plan_period_start_DATE")) >= MIN_ENROLLMENT_DAYS) \
        .select("person_id", F.col("payer_plan_period_start_DATE").alias("enrollment_start_date")) \
        .distinct()

    logging.info("%sPerson IDs with a usable enrollment period: %s", INDENT1,
                 f"{enrollment_start_dates.select('person_id').distinct().count():,}")

    return enrollment_start_dates


@time_execution
def find_excluded_personids(spark: SparkSession, omop_dir: str, date_start: str, date_end: str,
                            min_duration_days: int, min_age: int, max_age: int,
                            person_context: DataFrame) -> DataFrame:
    """
    Find the patients excluded from the cohort in the following steps:
    1. Filter patients based on the minimum duration (max visit date - min visit date)
    2. Filter patients based on the age at first event, dated at the earlier of the first visit
       and the start of an insurance enrollment

    Only visits that both start and end inside the date window are considered, so records with a
    missing or out-of-window end date do not contribute to the visit span.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        date_start: Earliest visit date to keep
        date_end: Latest visit date to keep
        min_duration_days: Minimum visit duration required for a patient to be included
        min_age: Minimum age at first event required for a patient to be included
        max_age: Maximum age at first event required for a patient to be included
        person_context: DataFrame of tokenizable patients, provides year_of_birth

    Returns:
        DataFrame containing the person_id column of the excluded patients
    """
    logging.info("EHRSHOT Filtering:")

    total_personids = person_context.count()

    # total visit start and end dates
    visit_start_end = spark.read.parquet(omop_table_path(omop_dir, "visit_occurrence")) \
        .filter((F.col("visit_start_DATE") >= F.lit(date_start).cast("date")) & (F.col("visit_end_DATE") >= F.lit(date_start).cast("date"))) \
        .filter((F.col("visit_start_DATE") <= F.lit(date_end).cast("date")) & (F.col("visit_end_DATE") <= F.lit(date_end).cast("date"))) \
        .groupBy("person_id").agg(F.min("visit_start_DATE").alias("MIN_VISIT_DATE"), F.max("visit_end_DATE").alias("MAX_VISIT_DATE"))

    # insufficient visit duration
    insufficient_visit_duration_personids = visit_start_end \
        .filter(F.datediff(F.col("MAX_VISIT_DATE"), F.col("MIN_VISIT_DATE")) < min_duration_days) \
        .select("person_id").distinct()
    logging.info("%sPerson IDs with insufficient EHR visit duration: %s", INDENT1, f"{insufficient_visit_duration_personids.count():,}")

    # age at first event, which a patient reaches at whichever comes first, their first visit in
    # the date window or the start of an insurance enrollment. A patient carries one row per
    # enrollment period, so an age outside the range on any of them excludes them, as in ReClaim.
    enrollment_start_dates = find_enrollment_start_dates(spark, omop_dir, date_start, date_end)

    age_at_first_event = person_context \
        .join(enrollment_start_dates, on="person_id", how="left") \
        .join(visit_start_end, on="person_id", how="left") \
        .withColumn("first_event_date",
            F.when(F.col("MIN_VISIT_DATE").isNull(), F.col("enrollment_start_date"))
             .when(F.col("enrollment_start_date").isNull(), F.col("MIN_VISIT_DATE"))
             .otherwise(F.least(F.col("MIN_VISIT_DATE"), F.col("enrollment_start_date")))) \
        .withColumn("first_event_year", F.year(F.col("first_event_date"))) \
        .withColumn("age", F.col("first_event_year") - F.col("year_of_birth"))

    spark_log_contingency_table(age_at_first_event, "age")

    underage_personids = age_at_first_event.filter(F.col("age") < min_age).select("person_id").distinct()
    logging.info("%sPerson IDs with age < %d: %s", INDENT1, min_age, f"{underage_personids.count():,}")
    overage_personids = age_at_first_event.filter(F.col("age") > max_age).select("person_id").distinct()
    logging.info("%sPerson IDs with age > %d: %s", INDENT1, max_age, f"{overage_personids.count():,}")

    # union all the excluded patients
    excluded_personids = insufficient_visit_duration_personids.union(underage_personids).union(overage_personids).distinct()
    excluded_count = excluded_personids.count()
    logging.info("%sTotal Person IDs to be excluded: %s, %.2f%% of the tokenizable patients",
                 INDENT1, f"{excluded_count:,}", excluded_count / total_personids * 100)

    return excluded_personids


# =============================================================================
# Turning clinical records into tokens
# =============================================================================

INPATIENT_VISITS = [262, 9201]


OUTPATIENT_VISITS = [9202, 581477, 9203]


PHARMACY_VISITS = [9202, 581477, 9203, 581458]


ED_VISITS = [9203]


ADMISSION_START_EVENT_TYPE = 0


DIAGNOSIS_EVENT_TYPE = 1


PROCEDURE_EVENT_TYPE = 2


MEDICATION_EVENT_TYPE = 3


ADMISSION_END_EVENT_TYPE = 4


ICD_CM_VOCABULARIES = ['ICD9CM', 'ICD10CM']


ICD_PCS_VOCABULARIES = ['ICD10PCS', 'ICD9CM', 'ICD9Proc']


ICD9_PROCEDURE_VOCABULARIES = ['ICD9CM', 'ICD9Proc']


RX_VOCABULARIES = ['RxNorm']


PRINCIPAL_CONDITION_STATUS = 32902


BRACKET_EPSILON_SECONDS = 1e-6 * SECONDS_IN_YEAR


ZERO_LENGTH_SECONDS = 2 * BRACKET_EPSILON_SECONDS


TOKEN_COLUMNS = ["person_id", "timestamp", "event_type", "category", "value", "token",
                 "original_token", "comb_order", "sub_order"]


def keep_one_row_per_key(df: DataFrame, key_col: str, order_by: list) -> DataFrame:
    """
    Keep one row per key, the first under the given ordering.

    Args:
        df: DataFrame holding the key column
        key_col: Column whose duplicates are collapsed
        order_by: Columns deciding which row of a key survives

    Returns:
        DataFrame with one row per key
    """
    return df \
        .withColumn("row_num", F.row_number().over(Window.partitionBy(key_col).orderBy(*order_by))) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num")


def pick_shortest_value(df: DataFrame, key_col: str, value_col: str) -> DataFrame:
    """
    Keep one row per key: the one with the shortest value, ties broken alphabetically.

    A reference table names some codes more than once, and the vocabulary holds the shortest of the
    competing names. The ETHOS tokenizer does the same in its translation_base.py and mimic/atc.py.

    Args:
        df: DataFrame holding the key and value columns
        key_col: Column whose duplicates are collapsed
        value_col: Column the shortest value is picked from

    Returns:
        DataFrame with one row per key
    """
    return keep_one_row_per_key(
        df.filter(F.col(key_col).isNotNull() & F.col(value_col).isNotNull()),
        key_col, [F.length(F.col(value_col)), F.col(value_col)])


def load_icd9_to_icd10_mapping(spark: SparkSession, mapping_path: str, target_col: str) -> DataFrame:
    """
    Load the ICD-9 to ICD-10 code mapping, keeping the target the file lists first.

    A translation table offers several ICD-10 codes for most ICD-9 codes, and the vocabulary holds
    whichever one the file lists first, not the shortest, as the ETHOS tokenizer's translation_base.py
    does.

    Row order is the file's own: gzip cannot be split, so Spark reads the table as one partition, and
    monotonically_increasing_id() rises down a partition.

    Args:
        spark: SparkSession instance
        mapping_path: Path to the mapping file
        target_col: Name to give the ICD-10 column

    Returns:
        DataFrame containing icd_9 and the renamed ICD-10 target column
    """
    mapping_df = spark.read.csv(mapping_path, header=True, inferSchema=False) \
        .select("icd_9", "icd_10") \
        .withColumn("file_order", F.monotonically_increasing_id()) \
        .filter(F.col("icd_9").isNotNull() & F.col("icd_10").isNotNull())
    return keep_one_row_per_key(mapping_df, "icd_9", [F.col("file_order")]) \
        .drop("file_order") \
        .withColumnRenamed("icd_10", target_col)


def load_icd_cm_title_mapping(spark: SparkSession, ref_dir: str) -> DataFrame:
    """
    Load the mapping from a dotless ICD-10-CM code to the title the vocabulary spells it out with.

    Args:
        spark: SparkSession instance
        ref_dir: Directory containing the code system reference tables

    Returns:
        DataFrame containing icd_category and icd_title
    """
    order_df = spark.read.csv(os.path.join(ref_dir, "icd10cm-order-Jan-2021.csv.gz"),
                              header=True, inferSchema=False).select("code", "long")
    return pick_shortest_value(order_df, "code", "long") \
        .withColumnRenamed("code", "icd_category") \
        .withColumnRenamed("long", "icd_title")


def load_atc_name_mapping(spark: SparkSession, ref_dir: str) -> DataFrame:
    """
    Load the mapping from an ATC code prefix to the drug group name the vocabulary uses.

    Args:
        spark: SparkSession instance
        ref_dir: Directory containing the code system reference tables

    Returns:
        DataFrame containing atc_group and atc_name
    """
    atc_df = spark.read.csv(os.path.join(ref_dir, "atc_coding.csv.gz"),
                            header=True, inferSchema=False).select("atc_code", "atc_name")
    return pick_shortest_value(atc_df, "atc_code", "atc_name") \
        .withColumnRenamed("atc_code", "atc_group")


def load_rxnorm_to_atc_mapping(spark: SparkSession, crosswalk_path: str) -> DataFrame:
    """
    Load the crosswalk from an RxNorm drug code to its ATC 5th level code.

    Args:
        spark: SparkSession instance
        crosswalk_path: Path to the crosswalk file

    Returns:
        DataFrame containing rxnorm_code and atc_code, empty when the crosswalk is missing
    """
    if not os.path.exists(crosswalk_path):
        logging.warning("%sRxNorm to ATC crosswalk not found at %s, medications will be skipped",
                        INDENT1, crosswalk_path)
        return spark.createDataFrame([], schema="rxnorm_code string, atc_code string")

    crosswalk_df = spark.read.csv(crosswalk_path, header=True, inferSchema=False) \
        .select("rxnorm_code", "atc_code")
    return pick_shortest_value(crosswalk_df, "rxnorm_code", "atc_code")


def load_vocabulary_tokens(spark: SparkSession, vocab: Vocabulary) -> DataFrame:
    """
    List every token string the vocabulary contains, so tokens outside it can be dropped.

    Args:
        spark: SparkSession instance
        vocab: Vocabulary instance

    Returns:
        DataFrame containing a single token column
    """
    return spark.createDataFrame([(str(token),) for token in vocab.stoi], schema="token string")


def load_code_whitelist(spark: SparkSession, vocab: Vocabulary, code_system: str) -> DataFrame:
    """
    List the source codes of a code system that the vocabulary was built with.

    A code outside this list has no token spelling in the vocabulary, so it is dropped before the
    code is split into token pieces.

    Args:
        spark: SparkSession instance
        vocab: Vocabulary instance
        code_system: Name the vocabulary registered the codes under

    Returns:
        DataFrame containing a single code column
    """
    codes = sorted(str(code) for code in vocab.get_q_storage(code_system).values())
    logging.info("%s%s codes registered in the vocabulary: %s", INDENT1, code_system, f"{len(codes):,}")
    return spark.createDataFrame([(code,) for code in codes], schema="code string")


def format_and_split_icd_cm_token(code_col: Column, title_col: Column) -> Column:
    """
    Spell a dotless ICD-10-CM code out as its token pieces.

    The three pieces are the three parts an ICD-10-CM code is built from: the category, emitted as its
    title, the up to three characters narrowing it to an etiology, a site and a severity, and the
    7th-character extension. A piece that would be empty is left out, so a 3-character code emits only
    its title.

    Both piece names sit a position away from what they hold, and are kept as the vocabulary spells
    them, which is how the ETHOS tokenizer's translation_base.py splits a code.

    Args:
        code_col: Column containing dotless ICD-10-CM codes (e.g. 'E1165', 'S72001A')
        title_col: Column containing the title of the code's 3-character category

    Returns:
        Column containing the token pieces of the code, in the order they are read in

    Examples:
        'E1165'   with title 'Type 2 diabetes ...' -> ['ICD_Type 2 diabetes ...', 'ICD_4-5_65']
        'S72001A' with title 'Fracture of femur'   -> ['ICD_Fracture of femur', 'ICD_4-5_001', 'ICD_6-_A']
    """
    category_token = F.concat(F.lit("ICD_"), title_col)
    minor_token = F.when(F.length(F.substring(code_col, 4, 3)) > 0,
                         F.concat(F.lit("ICD_4-5_"), F.substring(code_col, 4, 3)))
    suffix_token = F.when(F.length(F.substring(code_col, 7, 100)) > 0,
                          F.concat(F.lit("ICD_6-_"), F.substring(code_col, 7, 100)))

    # A title is written out in words, so the pieces are kept as separate array elements rather than
    # joined into one string, where the spaces inside a title could no longer be told apart
    return F.array_compact(F.array(category_token, minor_token, suffix_token))


def format_and_split_icd_pcs_token(code_col: Column) -> Column:
    """
    Spell an ICD-10-PCS code out as one token piece per code position.

    Each of the seven positions of a procedure code carries its own meaning, so each character is
    emitted as its own token.

    Args:
        code_col: Column containing ICD-10-PCS codes (e.g. '0DTJ4ZZ')

    Returns:
        Column containing the seven token pieces of the code, in the order they are read in
    """
    position_tokens = [
        F.when(F.length(F.substring(code_col, position, 1)) > 0,
               F.concat(F.lit("ICD_PCS_"), F.substring(code_col, position, 1)))
        for position in range(1, 8)
    ]
    return F.array_compact(F.array(*position_tokens))


def format_and_split_atc_token(code_col: Column, name_col: Column) -> Column:
    """
    Spell an ATC 5th level code out as its token pieces.

    ATC is a five level hierarchy in which the length of the code encodes the level: 1 character is
    the anatomical main group, 3 the therapeutic subgroup, 4 the pharmacological subgroup, 5 the
    chemical subgroup and 7 the chemical substance. The three pieces emitted here are the name of
    the therapeutic subgroup, the character that narrows it to the pharmacological subgroup, and the
    remaining characters that identify the substance, so together they rebuild the full code.

    Args:
        code_col: Column containing ATC 5th level codes (e.g. 'A10BA02')
        name_col: Column containing the name of the code's 3-character therapeutic subgroup

    Returns:
        Column containing the token pieces of the code, in the order they are read in

    Examples:
        'A10BA02' with name 'drugs used in diabetes'
            -> ['ATC_drugs used in diabetes', 'ATC_4_B', 'ATC_SUFFIX_A02']
    """
    group_token = F.concat(F.lit("ATC_"), name_col)
    subgroup_token = F.when(F.length(F.substring(code_col, 4, 1)) > 0,
                            F.concat(F.lit("ATC_4_"), F.substring(code_col, 4, 1)))
    substance_token = F.when(F.length(F.substring(code_col, 5, 100)) > 0,
                             F.concat(F.lit("ATC_SUFFIX_"), F.substring(code_col, 5, 100)))

    # A drug group is written out in words, so its spaces have to survive as part of one token
    return F.array_compact(F.array(group_token, subgroup_token, substance_token))


def explode_token_pieces(df: DataFrame, id_cols: list) -> DataFrame:
    """
    Lay the token pieces of a code out as one row per piece.

    The position of a piece inside its code is kept as comb_order, so the pieces of one code stay
    together and in order when the timeline is assembled.

    Args:
        df: DataFrame containing a token_pieces column holding the pieces of one code
        id_cols: Columns to carry through

    Returns:
        DataFrame with one row per token piece, carrying comb_order and token
    """
    return df \
        .filter(F.col("token_pieces").isNotNull() & (F.size(F.col("token_pieces")) > 0)) \
        .select("*", F.posexplode(F.col("token_pieces")).alias("comb_order", "token")) \
        .select(*id_cols, "comb_order", "token")


def drop_tokens_outside_vocabulary(df: DataFrame, vocabulary_tokens_df: DataFrame, category: str) -> DataFrame:
    """
    Drop the token pieces the vocabulary does not contain and log how many were lost.

    Args:
        df: DataFrame containing a token column
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        category: Token category, used for logging

    Returns:
        DataFrame containing only tokens the vocabulary can encode
    """
    total_count = df.count()
    kept_df = df.join(F.broadcast(vocabulary_tokens_df), on="token", how="left_semi")
    kept_count = kept_df.count()

    if total_count > 0:
        kept_rate = kept_count / total_count * 100
        logging.info("%s%s: %.2f%% of token pieces are in the vocabulary; %.2f%% dropped",
                     INDENT1, category, kept_rate, 100 - kept_rate)
    return kept_df


@time_execution
def load_visits(spark: SparkSession, omop_dir: str, date_start: str, date_end: str,
                kept_personids_df: DataFrame, visit_concept_ids: list) -> DataFrame:
    """
    Load the visits of the retained patients that fall inside the date window.

    A visit whose end date is missing or precedes its start date is treated as ending on the day it
    started, so that it still contributes its clinical records.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        date_start: Earliest visit date to keep
        date_end: Latest visit date to keep
        kept_personids_df: DataFrame of the person IDs that survived the cohort filters
        visit_concept_ids: OMOP visit_concept_id values to keep

    Returns:
        DataFrame containing person_id, visit_occurrence_id, visit_concept_id, visit_start_date,
        visit_end_date and the timestamp of the visit
    """
    visit_occurrence = spark.read.parquet(omop_table_path(omop_dir, "visit_occurrence")) \
        .join(F.broadcast(kept_personids_df), on="person_id", how="left_semi") \
        .filter(F.col('visit_concept_id').isin(visit_concept_ids)) \
        .filter((F.col("visit_start_DATE") >= F.lit(date_start).cast("date")) & (F.col("visit_start_DATE") <= F.lit(date_end).cast("date")))

    return visit_occurrence \
        .withColumn("visit_start_date", F.col("visit_start_DATE").cast("date")) \
        .withColumn("visit_end_date",
            F.when(F.col("visit_end_DATE").isNull() | (F.col("visit_end_DATE") < F.col("visit_start_DATE")),
                   F.col("visit_start_DATE").cast("date"))
             .otherwise(F.col("visit_end_DATE").cast("date"))) \
        .withColumn("timestamp", F.col("visit_start_date").cast("timestamp")) \
        .select("person_id", "visit_occurrence_id", "visit_concept_id",
                "visit_start_date", "visit_end_date", "timestamp")


@time_execution
def construct_diagnosis_tokens(spark: SparkSession, omop_dir: str, ref_dir: str,
                               vocabulary_tokens_df: DataFrame, icd_cm_whitelist_df: DataFrame,
                               visits: DataFrame) -> DataFrame:
    """
    Construct the diagnosis tokens recorded during the given visits.

    Diagnoses are read as their source codes, so ICD-9-CM codes are translated to ICD-10-CM before
    the code is spelled out. Codes the vocabulary was not built with are dropped. A principal
    diagnosis is ordered ahead of the secondary ones recorded at the same instant.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        icd_cm_whitelist_df: DataFrame listing the diagnosis codes the vocabulary was built with
        visits: DataFrame of visits whose diagnoses are tokenized

    Returns:
        DataFrame containing the diagnosis tokens
    """
    logging.info("%sConstructing diagnosis tokens", INDENT2)
    id_cols = ["person_id", "timestamp", "event_type", "category", "value", "original_token", "sub_order"]

    condition_occurrence = spark.read.parquet(omop_table_path(omop_dir, "condition_occurrence"))
    concept = spark.read.parquet(omop_table_path(omop_dir, "concept")) \
        .select("concept_id", "concept_code", "vocabulary_id")

    # Resolve the diagnosis to the code that was actually recorded, and note which ICD edition it is
    visit_condition = visits \
        .join(condition_occurrence, on=["person_id", "visit_occurrence_id"], how="inner") \
        .join(concept, F.col("condition_source_concept_id") == F.col("concept_id"), "inner") \
        .filter(F.col("vocabulary_id").isin(ICD_CM_VOCABULARIES)) \
        .withColumn("category", F.lit("DX")) \
        .withColumn("event_type", F.lit(DIAGNOSIS_EVENT_TYPE)) \
        .withColumn("original_token", F.col("concept_code")) \
        .withColumn("icd_version", F.when(F.col("vocabulary_id") == "ICD9CM", F.lit("9"))
                                   .when(F.col("vocabulary_id") == "ICD10CM", F.lit("0"))) \
        .withColumn("source_code", F.regexp_replace(F.col("concept_code"), r"\.", "")) \
        .withColumn("sub_order",
            F.when(F.col("condition_status_concept_id") == PRINCIPAL_CONDITION_STATUS, F.lit(0)).otherwise(F.lit(1))) \
        .filter(F.col("source_code").isNotNull() & (F.col("source_code") != "")) \
        .select("person_id", "timestamp", "event_type", "category", "original_token",
                "icd_version", "source_code", "sub_order") \
        .distinct()

    spark_log_contingency_table(visit_condition, "icd_version")

    # ICD-10-CM codes are already in the target edition, ICD-9-CM codes have to be translated and are
    # dropped when the translation table has no entry for them
    icd9_to_icd10_df = load_icd9_to_icd10_mapping(
        spark, os.path.join(ref_dir, "icd_cm_9_to_10_mapping.csv.gz"), "icd_10_cm")
    diagnosis_mapped = visit_condition \
        .join(F.broadcast(icd9_to_icd10_df), F.col("source_code") == F.col("icd_9"), "left") \
        .withColumn("value",
            F.when(F.col("icd_version") == "0", F.col("source_code"))
             .otherwise(F.col("icd_10_cm"))) \
        .filter(F.col("value").isNotNull())

    # Keep the codes the vocabulary was built with, then spell each one out
    icd_cm_titles_df = load_icd_cm_title_mapping(spark, ref_dir)
    diagnosis_tokens = diagnosis_mapped \
        .join(F.broadcast(icd_cm_whitelist_df), F.col("value") == F.col("code"), "left_semi") \
        .withColumn("icd_category", F.substring(F.col("value"), 1, 3)) \
        .join(F.broadcast(icd_cm_titles_df), on="icd_category", how="inner") \
        .withColumn("token_pieces", format_and_split_icd_cm_token(F.col("value"), F.col("icd_title")))

    diagnosis_tokens = explode_token_pieces(diagnosis_tokens, id_cols)
    diagnosis_tokens = drop_tokens_outside_vocabulary(diagnosis_tokens, vocabulary_tokens_df, "DX")

    return diagnosis_tokens.select(*TOKEN_COLUMNS).distinct()


@time_execution
def construct_procedure_tokens(spark: SparkSession, omop_dir: str, ref_dir: str,
                               vocabulary_tokens_df: DataFrame, icd_pcs_whitelist_df: DataFrame,
                               visits: DataFrame) -> DataFrame:
    """
    Construct the procedure tokens recorded during the given visits.

    Procedures are spelled out position by position from their ICD-10-PCS code, so ICD-9 procedure
    codes are translated first and codes from billing code systems that have no ICD-10-PCS
    equivalent are dropped.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        icd_pcs_whitelist_df: DataFrame listing the procedure codes the vocabulary was built with
        visits: DataFrame of visits whose procedures are tokenized

    Returns:
        DataFrame containing the procedure tokens
    """
    logging.info("%sConstructing procedure tokens", INDENT2)
    id_cols = ["person_id", "timestamp", "event_type", "category", "value", "original_token", "sub_order"]

    procedure_occurrence = spark.read.parquet(omop_table_path(omop_dir, "procedure_occurrence"))
    concept = spark.read.parquet(omop_table_path(omop_dir, "concept")) \
        .select("concept_id", "concept_code", "vocabulary_id")

    visit_procedure = visits \
        .join(procedure_occurrence, on=["person_id", "visit_occurrence_id"], how="inner") \
        .join(concept, F.col("procedure_source_concept_id") == F.col("concept_id"), "inner") \
        .withColumn("category", F.lit("PROC")) \
        .withColumn("event_type", F.lit(PROCEDURE_EVENT_TYPE)) \
        .withColumn("original_token", F.col("concept_code")) \
        .withColumn("source_code", F.regexp_replace(F.col("concept_code"), r"\.", "")) \
        .withColumn("sub_order", F.lit(0)) \
        .filter(F.col("source_code").isNotNull() & (F.col("source_code") != ""))

    # Report which code systems the procedures arrive in before the unusable ones are dropped
    spark_log_contingency_table(visit_procedure, "vocabulary_id")

    visit_procedure = visit_procedure \
        .filter(F.col("vocabulary_id").isin(ICD_PCS_VOCABULARIES)) \
        .withColumn("is_icd9", F.col("vocabulary_id").isin(ICD9_PROCEDURE_VOCABULARIES)) \
        .select("person_id", "timestamp", "event_type", "category", "original_token",
                "is_icd9", "source_code", "sub_order") \
        .distinct()

    icd9_to_icd10_df = load_icd9_to_icd10_mapping(
        spark, os.path.join(ref_dir, "icd_pcs_9_to_10_mapping.csv.gz"), "icd_10_pcs")
    procedure_mapped = visit_procedure \
        .join(F.broadcast(icd9_to_icd10_df), F.col("source_code") == F.col("icd_9"), "left") \
        .withColumn("value",
            F.when(F.col("is_icd9"), F.col("icd_10_pcs")).otherwise(F.col("source_code"))) \
        .filter(F.col("value").isNotNull())

    procedure_tokens = procedure_mapped \
        .join(F.broadcast(icd_pcs_whitelist_df), F.col("value") == F.col("code"), "left_semi") \
        .withColumn("token_pieces", format_and_split_icd_pcs_token(F.col("value")))

    procedure_tokens = explode_token_pieces(procedure_tokens, id_cols)
    procedure_tokens = drop_tokens_outside_vocabulary(procedure_tokens, vocabulary_tokens_df, "PROC")

    return procedure_tokens.select(*TOKEN_COLUMNS).distinct()


@time_execution
def construct_medication_tokens(spark: SparkSession, omop_dir: str, ref_dir: str, crosswalk_path: str,
                                vocabulary_tokens_df: DataFrame, visits: DataFrame) -> DataFrame:
    """
    Construct the medication tokens dispensed on the given visits.

    Drugs are recorded as RxNorm codes, while the vocabulary spells medications out along the ATC
    hierarchy, so each drug is first crosswalked to its ATC 5th level code. Drugs without a
    crosswalk entry are dropped.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        crosswalk_path: Path to the RxNorm to ATC crosswalk
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        visits: DataFrame of visits whose drugs are tokenized

    Returns:
        DataFrame containing the medication tokens
    """
    logging.info("%sConstructing medication tokens", INDENT2)
    id_cols = ["person_id", "timestamp", "event_type", "category", "value", "original_token", "sub_order"]

    drug_exposure = spark.read.parquet(omop_table_path(omop_dir, "drug_exposure"))
    concept = spark.read.parquet(omop_table_path(omop_dir, "concept")) \
        .select("concept_id", "concept_code", "vocabulary_id")

    visit_drug = visits \
        .join(drug_exposure, on=["person_id", "visit_occurrence_id"], how="inner") \
        .filter(F.col("drug_source_concept_id").isNotNull() & (F.col("drug_source_concept_id") != 0)) \
        .join(concept, F.col("drug_source_concept_id") == F.col("concept_id"), "inner") \
        .filter(F.col("vocabulary_id").isin(RX_VOCABULARIES)) \
        .withColumn("category", F.lit("RX")) \
        .withColumn("event_type", F.lit(MEDICATION_EVENT_TYPE)) \
        .withColumn("original_token", F.col("concept_code")) \
        .withColumn("sub_order", F.lit(0)) \
        .filter(F.col("concept_code").isNotNull() & (F.col("concept_code") != "")) \
        .select("person_id", "timestamp", "event_type", "category", "original_token",
                F.col("concept_code").alias("rxnorm_source_code"), "sub_order") \
        .distinct()

    rxnorm_to_atc_df = load_rxnorm_to_atc_mapping(spark, crosswalk_path)
    atc_names_df = load_atc_name_mapping(spark, ref_dir)

    medication_mapped = visit_drug \
        .join(F.broadcast(rxnorm_to_atc_df), F.col("rxnorm_source_code") == F.col("rxnorm_code"), "left") \
        .withColumn("value", F.col("atc_code"))

    crosswalked_count = medication_mapped.filter(F.col("value").isNotNull()).count()
    total_count = medication_mapped.count()
    if total_count > 0:
        crosswalk_rate = crosswalked_count / total_count * 100
        logging.info("%sRX: %.2f%% of drugs crosswalked to an ATC code; %.2f%% dropped",
                     INDENT1, crosswalk_rate, 100 - crosswalk_rate)

    medication_tokens = medication_mapped \
        .filter(F.col("value").isNotNull()) \
        .withColumn("atc_group", F.substring(F.col("value"), 1, 3)) \
        .join(F.broadcast(atc_names_df), on="atc_group", how="inner") \
        .withColumn("token_pieces", format_and_split_atc_token(F.col("value"), F.col("atc_name")))

    medication_tokens = explode_token_pieces(medication_tokens, id_cols)
    medication_tokens = drop_tokens_outside_vocabulary(medication_tokens, vocabulary_tokens_df, "RX")

    return medication_tokens.select(*TOKEN_COLUMNS).distinct()


@time_execution
def construct_admission_bracket_tokens(visits: DataFrame, admission_prefix: str) -> DataFrame:
    """
    Bracket a stay with its opening and closing token.

    The opening bracket sits at the start of the stay and the closing one just before the discharge
    instant, so everything recorded during the stay falls between them. How long the stay lasted is
    left for the two brackets to say through the time between them: a stay that ends on the day it
    began closes moments after it opens, and a longer one closes a day or more later.

    Args:
        visits: DataFrame of the visits that make up this admission type
        admission_prefix: Token prefix of the admission type (e.g. 'INPATIENT_ADMISSION')

    Returns:
        DataFrame containing the admission bracket tokens
    """
    logging.info("%sConstructing %s bracket tokens", INDENT2, admission_prefix)

    # Stay length in whole days, which is all a pair of visit dates can tell, and the length used for
    # placing the closing bracket, never shorter than the gap kept between the brackets of a stay that
    # ends on the day it began
    stays = visits \
        .withColumn("length_seconds", F.datediff(F.col("visit_end_date"), F.col("visit_start_date")) * F.lit(86400)) \
        .withColumn("bracket_length_seconds", F.greatest(F.col("length_seconds"), F.lit(ZERO_LENGTH_SECONDS))) \
        .withColumn("start_timestamp", F.col("timestamp")) \
        .withColumn("end_timestamp", F.timestamp_seconds(
            F.unix_timestamp(F.col("timestamp")) + F.col("bracket_length_seconds") - F.lit(BRACKET_EPSILON_SECONDS))) \
        .select("person_id", "start_timestamp", "end_timestamp", "length_seconds")

    admission_start_tokens = stays \
        .withColumn("timestamp", F.col("start_timestamp")) \
        .withColumn("event_type", F.lit(ADMISSION_START_EVENT_TYPE)) \
        .withColumn("category", F.lit("VT")) \
        .withColumn("value", F.lit(f"{admission_prefix}_START")) \
        .withColumn("token", F.col("value")) \
        .withColumn("original_token", F.col("value")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*TOKEN_COLUMNS)

    admission_end_tokens = stays \
        .withColumn("timestamp", F.col("end_timestamp")) \
        .withColumn("event_type", F.lit(ADMISSION_END_EVENT_TYPE)) \
        .withColumn("category", F.lit("VT")) \
        .withColumn("value", F.lit(f"{admission_prefix}_END")) \
        .withColumn("token", F.col("value")) \
        .withColumn("original_token", F.col("value")) \
        .withColumn("comb_order", F.lit(0)) \
        .withColumn("sub_order", F.lit(0)) \
        .select(*TOKEN_COLUMNS)

    # How the stays are distributed over whole days, which is what the brackets end up expressing
    spark_log_contingency_table(
        stays.withColumn("length_days", (F.col("length_seconds") / F.lit(86400)).cast("int")),
        "length_days")

    return admission_start_tokens.unionByName(admission_end_tokens)


@time_execution
def construct_inpatient_visit_tokens(spark: SparkSession, omop_dir: str, ref_dir: str,
                                     vocabulary_tokens_df: DataFrame, icd_cm_whitelist_df: DataFrame,
                                     icd_pcs_whitelist_df: DataFrame, kept_personids_df: DataFrame,
                                     date_start: str, date_end: str) -> DataFrame:
    """
    Construct the tokens of the inpatient visits: the admission brackets that delimit each stay,
    together with the diagnoses and procedures recorded during it.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        icd_cm_whitelist_df: DataFrame listing the diagnosis codes the vocabulary was built with
        icd_pcs_whitelist_df: DataFrame listing the procedure codes the vocabulary was built with
        kept_personids_df: DataFrame of the person IDs that survived the cohort filters
        date_start: Earliest visit date to tokenize
        date_end: Latest visit date to tokenize

    Returns:
        DataFrame containing the inpatient visit tokens
    """
    logging.info("%sConstructing inpatient visit tokens", INDENT1)

    inpatient_visits = load_visits(spark, omop_dir, date_start, date_end,
                                   kept_personids_df, INPATIENT_VISITS).cache()
    logging.info("%sInpatient visits: %s", INDENT2, f"{inpatient_visits.count():,}")

    admission_tokens = construct_admission_bracket_tokens(inpatient_visits, "INPATIENT_ADMISSION")
    diagnosis_tokens = construct_diagnosis_tokens(spark, omop_dir, ref_dir, vocabulary_tokens_df,
                                                  icd_cm_whitelist_df, inpatient_visits)
    procedure_tokens = construct_procedure_tokens(spark, omop_dir, ref_dir, vocabulary_tokens_df,
                                                  icd_pcs_whitelist_df, inpatient_visits)

    inpatient_tokens = admission_tokens.unionByName(diagnosis_tokens).unionByName(procedure_tokens)

    logging.info("Inpatient visit tokens construction completed!")
    return inpatient_tokens


@time_execution
def construct_outpatient_visit_tokens(spark: SparkSession, omop_dir: str, ref_dir: str,
                                      vocabulary_tokens_df: DataFrame, icd_cm_whitelist_df: DataFrame,
                                      icd_pcs_whitelist_df: DataFrame, kept_personids_df: DataFrame,
                                      date_start: str, date_end: str) -> DataFrame:
    """
    Construct the tokens of the outpatient visits: the diagnoses and procedures recorded during
    ambulatory and emergency room visits.

    Outpatient visits are points in time rather than stays, so they carry no brackets of their own.
    The diagnoses and procedures of an emergency room visit are built here and end up inside the
    emergency brackets built from the same visit.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        icd_cm_whitelist_df: DataFrame listing the diagnosis codes the vocabulary was built with
        icd_pcs_whitelist_df: DataFrame listing the procedure codes the vocabulary was built with
        kept_personids_df: DataFrame of the person IDs that survived the cohort filters
        date_start: Earliest visit date to tokenize
        date_end: Latest visit date to tokenize

    Returns:
        DataFrame containing the outpatient visit tokens
    """
    logging.info("%sConstructing outpatient visit tokens", INDENT1)

    outpatient_visits = load_visits(spark, omop_dir, date_start, date_end,
                                    kept_personids_df, OUTPATIENT_VISITS).cache()
    logging.info("%sOutpatient visits: %s", INDENT2, f"{outpatient_visits.count():,}")

    diagnosis_tokens = construct_diagnosis_tokens(spark, omop_dir, ref_dir, vocabulary_tokens_df,
                                                  icd_cm_whitelist_df, outpatient_visits)
    procedure_tokens = construct_procedure_tokens(spark, omop_dir, ref_dir, vocabulary_tokens_df,
                                                  icd_pcs_whitelist_df, outpatient_visits)

    outpatient_tokens = diagnosis_tokens.unionByName(procedure_tokens)

    logging.info("Outpatient visit tokens construction completed!")
    return outpatient_tokens


@time_execution
def construct_ed_visit_tokens(spark: SparkSession, omop_dir: str, kept_personids_df: DataFrame,
                              date_start: str, date_end: str) -> DataFrame:
    """
    Construct the tokens of the emergency room visits: the brackets that delimit the emergency stay.

    The diagnoses, procedures and drugs of an emergency room visit are built by the outpatient and
    pharmacy streams, which read the same visits, so they are not repeated here.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        kept_personids_df: DataFrame of the person IDs that survived the cohort filters
        date_start: Earliest visit date to tokenize
        date_end: Latest visit date to tokenize

    Returns:
        DataFrame containing the emergency visit tokens
    """
    logging.info("%sConstructing emergency visit tokens", INDENT1)

    ed_visits = load_visits(spark, omop_dir, date_start, date_end, kept_personids_df, ED_VISITS).cache()
    logging.info("%sEmergency visits: %s", INDENT2, f"{ed_visits.count():,}")

    ed_tokens = construct_admission_bracket_tokens(ed_visits, "ED_ADMISSION")

    logging.info("Emergency visit tokens construction completed!")
    return ed_tokens


@time_execution
def construct_pharmacy_visit_tokens(spark: SparkSession, omop_dir: str, ref_dir: str, crosswalk_path: str,
                                    vocabulary_tokens_df: DataFrame, kept_personids_df: DataFrame,
                                    date_start: str, date_end: str) -> DataFrame:
    """
    Construct the tokens of the pharmacy visits: the drugs dispensed on a visit.

    Args:
        spark: SparkSession instance
        omop_dir: Directory containing the OMOP data
        ref_dir: Directory containing the code system reference tables
        crosswalk_path: Path to the RxNorm to ATC crosswalk
        vocabulary_tokens_df: DataFrame listing every token the vocabulary contains
        kept_personids_df: DataFrame of the person IDs that survived the cohort filters
        date_start: Earliest visit date to tokenize
        date_end: Latest visit date to tokenize

    Returns:
        DataFrame containing the pharmacy visit tokens
    """
    logging.info("%sConstructing pharmacy visit tokens", INDENT1)

    pharmacy_visits = load_visits(spark, omop_dir, date_start, date_end,
                                  kept_personids_df, PHARMACY_VISITS).cache()
    logging.info("%sPharmacy visits: %s", INDENT2, f"{pharmacy_visits.count():,}")

    pharmacy_tokens = construct_medication_tokens(spark, omop_dir, ref_dir, crosswalk_path,
                                                  vocabulary_tokens_df, pharmacy_visits)

    logging.info("Pharmacy visit tokens construction completed!")
    return pharmacy_tokens


# =============================================================================
# Assembling one timeline per patient
# =============================================================================

BIRTH_ANCHOR_MONTHS = 6


TOKEN_ORDER_COLUMNS = ["timestamp", "event_type", "sub_order", "comb_order", "token"]


TIMELINE_SCHEMA = StructType([
    StructField("person_id", LongType()),
    StructField("times", ArrayType(DoubleType())),
    StructField("tokens", ArrayType(IntegerType())),
])


def load_vocabulary_token_ids(spark: SparkSession, vocab: Vocabulary) -> DataFrame:
    """
    List every token string of the vocabulary together with the ID it is encoded as.

    Args:
        spark: SparkSession instance
        vocab: Vocabulary instance

    Returns:
        DataFrame containing token and token_id
    """
    return spark.createDataFrame([(str(token), int(token_id)) for token, token_id in vocab.stoi.items()],
                                 schema="token string, token_id int")


@time_execution
def construct_event_seq(spark: SparkSession, out_dir: str, vocabulary_token_ids_df: DataFrame) -> DataFrame:
    """
    Turn the token strings of every patient into encoded events carrying the patient's age.

    The age of the patient at an event is what places that event in time, so the timestamp of every
    token is measured against the middle of the patient's birth year and expressed in years.

    Args:
        spark: SparkSession instance
        out_dir: Directory holding the pipeline outputs
        vocabulary_token_ids_df: DataFrame mapping a token string to its token ID

    Returns:
        DataFrame containing one encoded event per row, with the ordering columns kept
    """
    logging.info("%sLoading claim tokens and static context", INDENT1)
    all_claim_tokens = spark.read.parquet(os.path.join(out_dir, "all_claim_tokens"))
    person_context = spark.read.parquet(os.path.join(out_dir, "person_context"))

    # Only patients with a static context can be handed to the model, since the context opens their
    # timeline
    event_seq = all_claim_tokens \
        .join(person_context.select("person_id", "year_of_birth"), on="person_id", how="inner")

    # The date a patient's age is measured from, and the age in years at each event
    birth_anchor = F.add_months(F.make_date(F.col("year_of_birth"), F.lit(1), F.lit(1)),
                                BIRTH_ANCHOR_MONTHS).cast("timestamp")
    event_seq = event_seq \
        .withColumn("birth_anchor", birth_anchor) \
        .withColumn("age_at_event",
            (F.col("timestamp").cast("double") - F.col("birth_anchor").cast("double")) / F.lit(SECONDS_IN_YEAR))

    # Tokens the vocabulary cannot encode have no place in a timeline
    total_count = event_seq.count()
    event_seq = event_seq.join(F.broadcast(vocabulary_token_ids_df), on="token", how="inner")
    encoded_count = event_seq.count()
    logging.info("%sEncoded %s of %s tokens (%.2f%%)", INDENT1, f"{encoded_count:,}", f"{total_count:,}",
                 encoded_count / total_count * 100 if total_count > 0 else 0.0)

    return event_seq.select("person_id", "token_id", "age_at_event", *TOKEN_ORDER_COLUMNS)


def build_timeline_assembler(vocab_path: str):
    """
    Build the function that assembles one patient's events into a finished timeline.

    Ordering the events of a patient, spelling out the gaps between them with separator tokens and
    closing the timeline all happen per patient, so they are done in one pass over that patient's
    events.

    Args:
        vocab_path: Path to the vocabulary pickle

    Returns:
        A function mapping a patient's events to a single row holding the finished timeline
    """
    def assemble_timeline(events: pd.DataFrame) -> pd.DataFrame:
        vocab = load_vocabulary(vocab_path)
        injector = SeparatorInjector(vocab)

        ordered = events.sort_values(TOKEN_ORDER_COLUMNS, kind="stable")
        times = ordered["age_at_event"].to_numpy(dtype=TIME_DTYPE)
        tokens = ordered["token_id"].to_numpy(dtype=TOKEN_DTYPE)
        times, tokens = injector(times, tokens)

        return pd.DataFrame([{
            "person_id": int(ordered["person_id"].iloc[0]),
            "times": [float(t) for t in times],
            "tokens": [int(t) for t in tokens],
        }])

    return assemble_timeline


@time_execution
def construct_trajectory(event_seq: DataFrame, vocab_path: str) -> DataFrame:
    """
    Assemble the events of every patient into one timeline per patient.

    Args:
        event_seq: DataFrame of encoded events carrying the patient's age
        vocab_path: Path to the vocabulary pickle

    Returns:
        DataFrame containing one row per patient, holding the timeline times and tokens
    """
    logging.info("%sAssembling per-patient timelines with separator tokens", INDENT1)
    assemble_timeline = build_timeline_assembler(vocab_path)
    return event_seq.groupBy("person_id").applyInPandas(assemble_timeline, schema=TIMELINE_SCHEMA)


@time_execution
def save_trajectory(spark: SparkSession, out_dir: str, timelines_path: str, trajectory: DataFrame,
                    vocab: Vocabulary) -> None:
    """
    Concatenate the patient timelines and write them, with the static context, to HDF5.

    The timelines are stored back to back in one long array, so each patient is located by the index
    its timeline starts at. Patients are stored in ascending patient order, which keeps a run
    reproducible.

    Args:
        spark: SparkSession instance
        out_dir: Directory holding the pipeline outputs
        timelines_path: HDF5 file to write
        trajectory: DataFrame containing one timeline per patient
        vocab: Vocabulary instance, used to encode the static context tokens
    """
    person_context = spark.read.parquet(os.path.join(out_dir, "person_context"))

    # The static context that opens a timeline, and the birth year the calendar year is recovered from
    context_rows = person_context.select("person_id", "sex_token", "year_of_birth").collect()
    sex_token_of = {row["person_id"]: row["sex_token"] for row in context_rows}
    year_of_birth_of = {row["person_id"]: int(row["year_of_birth"]) for row in context_rows}

    all_times, all_tokens = [], []
    patient_ids, patient_offsets, patient_context, age_reference = [], [], [], []
    offset = 0

    for row in trajectory.orderBy("person_id").toLocalIterator():
        person_id = row["person_id"]
        all_times.append(np.asarray(row["times"], dtype=TIME_DTYPE))
        all_tokens.append(np.asarray(row["tokens"], dtype=TOKEN_DTYPE))

        patient_offsets.append(offset)
        offset += len(row["tokens"])
        patient_ids.append(person_id)
        patient_context.append([int(vocab.encode(sex_token_of[person_id]))])
        age_reference.append(year_of_birth_of[person_id])

    if not patient_ids:
        raise RuntimeError("No patient timeline could be assembled, so there is nothing to write.")

    times = np.concatenate(all_times).astype(TIME_DTYPE)
    tokens = np.concatenate(all_tokens).astype(TOKEN_DTYPE)

    os.makedirs(os.path.dirname(timelines_path) or ".", exist_ok=True)
    with h5py.File(timelines_path, "w") as f:
        f.create_dataset("times", data=times)
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("patient_data_offsets", data=np.asarray(patient_offsets, dtype=np.int64))
        f.create_dataset("patient_ids", data=np.asarray(patient_ids))
        f.create_dataset("patient_context", data=np.asarray(patient_context, dtype=TOKEN_DTYPE))
        f.create_dataset("age_reference", data=np.asarray(age_reference, dtype=np.int32))

    logging.info("%sTimelines written: %s patients, %s tokens", INDENT1,
                 f"{len(patient_ids):,}", f"{len(tokens):,}")
    logging.info("%sTokens per patient: min %s, mean %.1f, max %s", INDENT1,
                 f"{min(len(t) for t in all_tokens):,}",
                 len(tokens) / len(patient_ids),
                 f"{max(len(t) for t in all_tokens):,}")


# =============================================================================================
# The four steps, run in order by main()
# =============================================================================================

def build_crosswalk(spark, args):
    """Step 1: bridge RxNorm to ATC, which is how the vocabulary names a drug."""
    logging.info("********** 1 Building the RxNorm to ATC crosswalk **********")
    # A small table every later stage broadcasts to its workers, so it is kept as one file rather
    # than spread over parts
    crosswalk = build_rxnorm_to_atc(spark, args.omop_dir).toPandas()
    os.makedirs(os.path.dirname(args.crosswalk) or ".", exist_ok=True)
    crosswalk.to_csv(args.crosswalk, index=False, compression="gzip")
    logging.info("%sWrote the crosswalk: %s rows", INDENT1, f"{len(crosswalk):,}")


def filter_patients(spark, args):
    """Step 2: select the patients that can be tokenized and record their static context."""
    logging.info("********** 2 Selecting the patients that can be tokenized **********")
    person_context = construct_person_context(spark, args.omop_dir)
    person_context.write.mode("overwrite").parquet(os.path.join(args.out_dir, "person_context"))

    # Re-read from disk for fresh lineage, to avoid recomputing the person table
    person_context = spark.read.parquet(os.path.join(args.out_dir, "person_context"))
    excluded = find_excluded_personids(
        spark, args.omop_dir, args.date_start, args.date_end,
        args.min_duration_days, args.min_age, args.max_age, person_context)
    excluded.write.mode("overwrite").parquet(os.path.join(args.out_dir, "excluded_personids"))

    kept = person_context.join(
        spark.read.parquet(os.path.join(args.out_dir, "excluded_personids")),
        on="person_id", how="left_anti")
    logging.info("%sPerson IDs kept after exclusions: %s", INDENT1,
                 f"{kept.select('person_id').distinct().count():,}")


def construct_tokens(spark, args, vocab):
    """Step 3: turn the clinical records of those patients into tokens of the vocabulary."""
    logging.info("********** 3 Turning clinical records into tokens **********")
    # The vocabulary decides which tokens can be produced at all, and which source codes were
    # registered when it was built
    vocabulary_tokens_df = load_vocabulary_tokens(spark, vocab).cache()
    icd_cm_whitelist_df = load_code_whitelist(spark, vocab, "ICD_CM_CODE").cache()
    icd_pcs_whitelist_df = load_code_whitelist(spark, vocab, "ICD_PCS_CODE").cache()

    person_context_df = spark.read.parquet(os.path.join(args.out_dir, "person_context"))
    excluded_df = spark.read.parquet(
        os.path.join(args.out_dir, "excluded_personids")).select("person_id").distinct()
    kept_personids_df = person_context_df.select("person_id").distinct() \
        .join(excluded_df, on="person_id", how="left_anti").cache()
    logging.info("%sPerson IDs to tokenize: %s", INDENT1, f"{kept_personids_df.count():,}")

    inpatient_tokens = construct_inpatient_visit_tokens(
        spark, args.omop_dir, args.ref_dir, vocabulary_tokens_df,
        icd_cm_whitelist_df, icd_pcs_whitelist_df, kept_personids_df,
        args.date_start, args.date_end)
    outpatient_tokens = construct_outpatient_visit_tokens(
        spark, args.omop_dir, args.ref_dir, vocabulary_tokens_df,
        icd_cm_whitelist_df, icd_pcs_whitelist_df, kept_personids_df,
        args.date_start, args.date_end)
    ed_tokens = construct_ed_visit_tokens(
        spark, args.omop_dir, kept_personids_df, args.date_start, args.date_end)
    pharmacy_tokens = construct_pharmacy_visit_tokens(
        spark, args.omop_dir, args.ref_dir, args.crosswalk, vocabulary_tokens_df,
        kept_personids_df, args.date_start, args.date_end)

    all_claim_tokens = inpatient_tokens \
        .unionByName(outpatient_tokens) \
        .unionByName(ed_tokens) \
        .unionByName(pharmacy_tokens) \
        .select(*TOKEN_COLUMNS) \
        .distinct()
    all_claim_tokens.write.mode("overwrite").parquet(os.path.join(args.out_dir, "all_claim_tokens"))

    all_claim_tokens = spark.read.parquet(os.path.join(args.out_dir, "all_claim_tokens"))
    logging.info("%sTotal claim tokens: %s", INDENT1, f"{all_claim_tokens.count():,}")
    logging.info("%sPerson IDs with tokens: %s", INDENT1,
                 f"{all_claim_tokens.select('person_id').distinct().count():,}")
    spark_log_contingency_table(all_claim_tokens, "category")


def construct_timelines(spark, args, vocab):
    """Step 4: assemble one chronological timeline per patient and write them to HDF5."""
    logging.info("********** 4 Assembling the timelines **********")
    vocabulary_token_ids_df = load_vocabulary_token_ids(spark, vocab).cache()
    event_seq = construct_event_seq(spark, args.out_dir, vocabulary_token_ids_df).cache()
    logging.info("%sEvents to place in a timeline: %s", INDENT1, f"{event_seq.count():,}")
    logging.info("%sPatients with events: %s", INDENT1,
                 f"{event_seq.select('person_id').distinct().count():,}")
    trajectory = construct_trajectory(event_seq, args.vocab)
    save_trajectory(spark, args.out_dir, args.timelines, trajectory, vocab)


def main():
    """Main execution function."""
    log_file = setup_logging(level=logging.INFO)
    logging.info("PREPROCESS STARTING...")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("\nConfiguration:")
    logging.info("%sOMOP directory: %s", INDENT1, args.omop_dir)
    logging.info("%sOutput directory: %s", INDENT1, args.out_dir)
    logging.info("%sVocabulary: %s", INDENT1, args.vocab)
    logging.info("%sReference directory: %s", INDENT1, args.ref_dir)
    logging.info("%sTimelines file: %s", INDENT1, args.timelines)
    logging.info("%sVisit date window: %s to %s", INDENT1, args.date_start, args.date_end)
    logging.info("%sMinimum visit duration days: %s", INDENT1, args.min_duration_days)
    logging.info("%sMinimum enrollment duration days: %s", INDENT1, MIN_ENROLLMENT_DAYS)
    logging.info("%sAge at the first event: %s to %s", INDENT1, args.min_age, args.max_age)

    spark = create_spark_session('EHRSHOT_Preprocess')
    vocab = load_vocabulary(args.vocab)
    logging.info("%sTokens in the vocabulary: %s", INDENT1, f"{len(vocab):,}")

    build_crosswalk(spark, args)
    filter_patients(spark, args)
    construct_tokens(spark, args, vocab)
    construct_timelines(spark, args, vocab)

    spark.stop()
    logging.info("Log file: %s", log_file)
    logging.info("PREPROCESS COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
