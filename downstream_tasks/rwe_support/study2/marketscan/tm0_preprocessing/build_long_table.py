#!/usr/bin/env python3
"""
build_long_table.py  --  hdPS occurrence-level long table for the union TTE cohort.

Baseline claims of cohort members -> LONG table (person_id | event_code | domain), one row per
code OCCURRENCE (recurrence tiers need multiplicity -- NO dropDuplicates). Five domains:
inpatient_dx, inpatient_proc, outpatient_dx, outpatient_proc, drug.

  dx    : 3-char ICD-10 category (decimal-stripped), index day INCLUDED
  proc  : CPT/HCPCS/ICD-10-PCS code as-is,          index day EXCLUDED
  drug  : NDC -> rxnorm ingredient name,            index day EXCLUDED

Dropped from the candidate pool (defining codes):
  - the 17 study-class ingredients + any ingredient of a dual-class combo NDC (exposure)
  - 3-char categories of the 6 primary outcome code sets (outcome)
  NCO codes stay: they are baseline-morbidity proxies, outcomes only post-index.

Output: intermediate/hdps_long/<cohort_id>.parquet (Spark directory), read per emulation by
select_covariates.R.
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging

import yaml
from pyspark.sql import functions as F

import codesets_tte as cs
from marketscan_io import MarketScanIO, create_spark_session, strip_decimal
from utils import setup_logging, time_execution

INDENT1, INDENT2 = "  ", "    "

DX_DOMAIN = {"inpatient": "inpatient_dx", "outpatient": "outpatient_dx"}
PROC_DOMAIN = {"inpatient": "inpatient_proc", "outpatient": "outpatient_proc"}

DEFAULT_VOCAB = os.environ.get("VOCAB_NDC", "")   # NDC->RxNorm-ingredient map; set in env.sh


def parse_args():
    p = argparse.ArgumentParser(description="TTE hdPS long table")
    p.add_argument("--config", default="config/tte_glp1_psych.yaml")
    p.add_argument("--raw_root", default=os.environ.get("MS_RAW_SUBSET", ""))
    p.add_argument("--stratum", default="POOL")
    p.add_argument("--vocab_ndc", default=DEFAULT_VOCAB)
    p.add_argument("--cohort_root", default="intermediate/cohorts",
                   help="root of the cohort.csvs")
    p.add_argument("--out_root", default="intermediate/hdps_long")
    return p.parse_args()


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("BUILD HDPS LONG TABLE STARTING...")
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    baseline_days = int(cfg["windows"]["baseline_days"])

    from build_cohort import era_years
    roots = [os.path.join(args.raw_root, "MarketScan_val"),
             os.path.join(args.raw_root, "MarketScan_test")]   # held-out split = val+test

    cohort_csv = os.path.join(args.cohort_root, "cohort.csv")
    spark = create_spark_session(f"tm0_long_{cfg['cohort_id']}_{args.stratum}")
    try:
        io = MarketScanIO(spark, raw_roots=roots, payer=args.stratum, era_years=era_years(cfg))
        logging.info("%sReading cohort: %s", INDENT1, cohort_csv)
        cohort = (spark.read.csv(cohort_csv, header=True, inferSchema=True)
                  # person_id is a namespaced STRING (CR<enrolid> / MD<enrolid>); keep it a
                  # string, every io frame is keyed the same way.
                  .select(F.col("person_id").cast("string").alias("ENROLID"), "index_date")
                  .withColumn("index_date", F.to_date("index_date"))
                  .withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days)))
        ids = cohort.select("ENROLID").distinct()

        # ---- diagnosis rows (index-day INCLUDED); 3-char ICD-10 category ----
        dx = io.dx_long_with_setting(person_ids=ids)
        dx = (dx.join(cohort, "ENROLID", "inner")
              .filter((F.col("dx_date") >= F.col("baseline_start")) &
                      (F.col("dx_date") <= F.col("index_date")))
              .withColumn("event_code", F.substring(F.upper(strip_decimal(F.col("code"))), 1, 3))
              .withColumn("domain",
                          F.when(F.col("setting") == "inpatient", F.lit("inpatient_dx"))
                          .otherwise(F.lit("outpatient_dx"))))
        dx_rows = dx.select("ENROLID", "event_code", "domain")

        # ---- procedure rows (index-day EXCLUDED); code as-is ----
        pr = io.proc_long_with_setting(person_ids=ids)
        pr = (pr.join(cohort, "ENROLID", "inner")
              .filter((F.col("proc_date") >= F.col("baseline_start")) &
                      (F.col("proc_date") < F.col("index_date")))
              .withColumn("event_code", F.col("code"))
              .withColumn("domain",
                          F.when(F.col("setting") == "inpatient", F.lit("inpatient_proc"))
                          .otherwise(F.lit("outpatient_proc"))))
        pr_rows = pr.select("ENROLID", "event_code", "domain")

        # ---- drug rows (index-day EXCLUDED); ingredient-level token ----
        dr = io.drug_long(args.vocab_ndc)
        dr = (dr.join(cohort, "ENROLID", "inner")
              .filter((F.col("svc_date") >= F.col("baseline_start")) &
                      (F.col("svc_date") < F.col("index_date")))
              .withColumn("domain", F.lit("drug")))
        dr_rows = dr.select("ENROLID", "event_code", "domain")

        long_df = dx_rows.unionByName(pr_rows).unionByName(dr_rows)

        # ---- drop exposure- and outcome-defining codes ----
        drop_ing = sorted({i.lower() for ings in cs.EXPOSURE_INGREDIENTS.values() for i in ings})
        long_df = long_df.filter(~((F.col("domain") == "drug") &
                                   (F.lower(F.col("event_code")).isin(drop_ing))))
        out_keys = cs.outcome_drop_keys_3char()
        long_df = long_df.filter(~(F.col("domain").rlike("_dx$") &
                                   F.col("event_code").isin(out_keys)))
        logging.info("%sDropped defining codes: %d exposure ingredients, %d outcome 3-char cats",
                     INDENT2, len(drop_ing), len(out_keys))

        long_df = (long_df.withColumnRenamed("ENROLID", "person_id")
                   .filter(F.col("event_code").isNotNull() & (F.col("event_code") != "")))

        os.makedirs(args.out_root, exist_ok=True)
        tag = cfg["cohort_id"]
        out_parquet = os.path.join(args.out_root, f"{tag}.parquet")
        logging.info("%sDomain occurrence counts:", INDENT1)
        for r in long_df.groupBy("domain").count().orderBy("domain").collect():
            logging.info("%s%-16s %s", INDENT2, r["domain"], f"{r['count']:,}")
        long_df.write.mode("overwrite").parquet(out_parquet)
        logging.info("Wrote %s. Log: %s", out_parquet, log_file)
        logging.info("BUILD HDPS LONG TABLE COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
