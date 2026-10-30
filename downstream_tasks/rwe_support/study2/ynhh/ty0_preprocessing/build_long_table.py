#!/usr/bin/env python3
"""
build_long_table.py -- hdPS occurrence-level long table for the union TTE cohort (ty0_2).

One row per code OCCURRENCE (recurrence tiers need multiplicity: no dropDuplicates), five
setting-aware domains over the 365d baseline:

  dx    : standard condition_concept_id as-is, index day INCLUDED
  proc  : procedure_concept_id as-is, index day EXCLUDED
  drug  : ingredient concept NAME via drug_strength (adapter.drug_long), index day EXCLUDED

Dropped defining codes: the 17 study ingredients (drug domain) and the primary-outcome
standard concept_ids (dx domains, from resolve_codes's dx_concept_map OUT_* rows). NCO
concepts stay in the pool (baseline-morbidity proxies; outcomes count only post-index).

Input: intermediate/cohorts/<cell>/cohort.csv. Output: intermediate/hdps_long/<cell>.parquet.
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging

from pyspark.sql import functions as F

import codesets_tte as cs
from datasources import make_adapter, get_source
from utils import setup_logging, time_execution
from cohort_helpers import load_config, create_spark_session

INDENT1, INDENT2 = "  ", "    "


def parse_args():
    p = argparse.ArgumentParser(description="TTE OMOP hdPS long table")
    p.add_argument("--config", default="config/tte_glp1_psych.yaml")
    p.add_argument("--source", default="ynhh")
    p.add_argument("--codes_resolved", default="intermediate/codes_resolved")
    p.add_argument("--cohort_root", default="intermediate/cohorts",
                   help="Root of the build_cohort cohort.csvs")
    p.add_argument("--out_root", default="intermediate/hdps_long")
    p.add_argument("--cell", default=None,
                   help="Cohort cell name for input/output paths (default: --source), e.g. ynhh_src")
    return p.parse_args()


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("BUILD HDPS LONG (OMOP) STARTING...")
    args = parse_args()
    cfg = load_config(args.config, args.source)
    baseline_days = int(cfg["windows"]["baseline_days"])

    src = get_source(args.source)
    cohort_csv = os.path.join(args.cohort_root, args.cell or args.source, "cohort.csv")
    spark = create_spark_session(f"t2_{cfg['cohort_id']}_{args.source}",
                                 profile=src.get("spark_profile", "bigmem"))
    try:
        import pandas as pd
        adapter = make_adapter(args.source, spark, stratum=args.source,
                               codes_resolved_root=args.codes_resolved, cfg=cfg)
        logging.info("%sReading cohort: %s", INDENT1, cohort_csv)
        cohort = (spark.read.csv(cohort_csv, header=True, inferSchema=True)
                  .select(F.col("person_id").cast("long").alias("ENROLID"), "index_date")
                  .withColumn("index_date", F.to_date("index_date"))
                  .withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days)))
        ids = cohort.select("ENROLID").distinct()

        # ---- dx rows (index day INCLUDED); event_code = standard concept_id as-is ----
        dx = adapter.dx_long_with_setting(person_ids=ids)
        dx = (dx.join(cohort, "ENROLID", "inner")
              .filter((F.col("dx_date") >= F.col("baseline_start")) &
                      (F.col("dx_date") <= F.col("index_date")))
              .withColumn("event_code", adapter.dx_long_event_code(F.col("code")))
              .withColumn("domain",
                          F.when(F.col("setting") == "inpatient", F.lit("inpatient_dx"))
                          .otherwise(F.lit("outpatient_dx"))))
        dx_rows = dx.select("ENROLID", "event_code", "domain")

        # ---- proc rows (index day EXCLUDED) ----
        pr = adapter.proc_long_with_setting()
        pr = (pr.join(cohort, "ENROLID", "inner")
              .filter((F.col("proc_date") >= F.col("baseline_start")) &
                      (F.col("proc_date") < F.col("index_date")))
              .withColumn("event_code", F.col("code"))
              .withColumn("domain",
                          F.when(F.col("setting") == "inpatient", F.lit("inpatient_proc"))
                          .otherwise(F.lit("outpatient_proc"))))
        pr_rows = pr.select("ENROLID", "event_code", "domain")

        # ---- drug rows (index day EXCLUDED); ingredient names via drug_strength ----
        dr = adapter.drug_long(ids)
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
        dxmap = pd.read_csv(os.path.join(args.codes_resolved, "dx_concept_map.csv"), dtype=str)
        out_keys = sorted(dxmap[dxmap.set_id.str.startswith("OUT_")].concept_id.unique())
        long_df = long_df.filter(~(F.col("domain").rlike("_dx$") &
                                   F.col("event_code").isin(out_keys)))
        logging.info("%sDropped defining codes: %d exposure ingredients, %d outcome concepts",
                     INDENT2, len(drop_ing), len(out_keys))

        long_df = (long_df.withColumnRenamed("ENROLID", "person_id")
                   .filter(F.col("event_code").isNotNull() & (F.col("event_code") != "")))

        os.makedirs(args.out_root, exist_ok=True)
        out_parquet = os.path.join(args.out_root, f"{args.cell or args.source}.parquet")
        logging.info("%sDomain occurrence counts:", INDENT1)
        for r in long_df.groupBy("domain").count().orderBy("domain").collect():
            logging.info("%s%-16s %s", INDENT2, r["domain"], f"{r['count']:,}")
        long_df.write.mode("overwrite").parquet(out_parquet)
        logging.info("Wrote %s. Log: %s", out_parquet, log_file)
        logging.info("BUILD HDPS LONG (OMOP) COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
