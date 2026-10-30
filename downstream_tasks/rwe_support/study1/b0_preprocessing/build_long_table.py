#!/usr/bin/env python3
"""
build_long_table.py  --  the input adapter for autoCovariateSelection.

Day-level source claims -> a LONG table with three required columns (others ignored by the engine):

    person_id | event_code (native code string) | domain (dimension label)

BUILD RULE: one row per code OCCURRENCE, not per distinct (patient, code). The step-3 recurrence
tiers depend on occurrence counts within the 180-d baseline; a pre-deduplicated table collapses
every code to "once" and the tiers vanish. So there is NO dropDuplicates on the long table.

Vocabulary + setting come from the data-source adapter (datasources/):
  MarketScan: dx = 3-digit ICD-9 category; proc = CPT/HCPCS/ICD-9-vol.3 as-is; drug = NDC->generic.
              setting = i/o table lineage -> {inpatient,outpatient}.

Five domains (setting-aware): inpatient_dx, inpatient_proc, outpatient_dx, outpatient_proc, drug.
3-domain fallback (dx, proc, drug) via --domains 3.

Index-day rule: diagnoses ON t0 included; procedures/drugs ON t0 excluded.
Exposure-defining codes are dropped before writing.

Output: intermediate/mdcr/hdps_long/{cohort}_{cell}.parquet   (+ a patient-id vector csv).
Usage: python build_long_table.py --config config/<cohort>.yaml --stratum MDCR [--cell MDCR_s2011]
           --raw_root <root> --restrict_ids <glob> --vocab_ndc <csv> --domains 5
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging

from pyspark.sql import functions as F

import codesets
from utils import setup_logging, time_execution
from build_cohort import create_spark_session, load_config, marketscan_era_years
from datasources import make_adapter, get_source, default_stratum
from resolvers import load_resolved

INDENT1, INDENT2 = "  ", "    "

# setting label (from the adapter: {inpatient,outpatient}) -> hdPS 5-domain label
DX_DOMAIN_MAP = {"inpatient": "inpatient_dx", "outpatient": "outpatient_dx"}
PROC_DOMAIN_MAP = {"inpatient": "inpatient_proc", "outpatient": "outpatient_proc"}


def parse_args():
    p = argparse.ArgumentParser(description="Build the hdPS occurrence-level long table")
    p.add_argument("--config", required=True)
    p.add_argument("--source", default="marketscan",
                   help="Data source name (datasources/registry.py).")
    p.add_argument("--stratum", default=None,
                   help="Analysis cell base (MarketScan payer). Defaults per source.")
    p.add_argument("--cell", default=None,
                   help="I/O cell token (default = stratum). negative_control single-season: '{stratum}_s{season}'.")
    p.add_argument("--raw_root", default=None, help="MarketScan only: the full claims tree")
    # Mirror build_cohort: these must match the values build_cohort used, or the long table would be built from a
    # different patient universe than the cohort it is joined to.
    p.add_argument("--restrict_ids", default=None,
                   help="Tokenizability gate (parquet glob of the tokenized population). MUST "
                        "match build_cohort. Required for the marketscan source.")
    p.add_argument("--vocab_ndc", default=None,
                   help="MarketScan only: NDC->generic mapping csv (drug-domain token map)")
    p.add_argument("--codes_resolved", default="intermediate/codes_resolved",
                   help="Root of resolve_codes's resolved codes. Run resolve_codes first.")
    p.add_argument("--cohort_csv", default=None, help="Defaults to {cohort_root}/.../cohort.csv")
    p.add_argument("--cohort_root", default="intermediate/mdcr/cohorts",
                   help="Root of the cohort.csvs (intermediate/mdcr/cohorts)")
    p.add_argument("--out_root", default="intermediate/mdcr/hdps_long")
    p.add_argument("--results_root", default="results/mdcr")
    p.add_argument("--baseline_days", type=int, default=None)
    p.add_argument("--domains", type=int, default=5, choices=[5, 3],
                   help="5 (setting-aware) or 3 (fallback dx/proc/drug)")
    return p.parse_args()


@time_execution
def build_long(adapter, resolved, cell, cfg, args):
    baseline_days = args.baseline_days or cfg["windows"]["baseline_days"]
    cohort_csv = args.cohort_csv or os.path.join(args.cohort_root, cfg["cohort_id"], cell, "cohort.csv")
    logging.info("%sReading cohort: %s", INDENT1, cohort_csv)
    cohort = (adapter.spark.read.csv(cohort_csv, header=True, inferSchema=True)
              .select(F.col("person_id").cast("long").alias("ENROLID"), "index_date")
              .withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days)))
    ids = cohort.select("ENROLID").distinct()

    use_setting = args.domains == 5

    # ---- diagnosis rows (index-day INCLUDED) ----
    dx = adapter.dx_long_with_setting(person_ids=ids)
    dx = (dx.join(cohort, "ENROLID", "inner")
          .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date"))))
    dx = dx.withColumn("event_code", adapter.dx_long_event_code(F.col("code")))
    if use_setting:
        dx = dx.withColumn(
            "domain", F.create_map([F.lit(x) for kv in DX_DOMAIN_MAP.items() for x in kv])[F.col("setting")])
        dx = dx.withColumn("domain", F.coalesce(F.col("domain"), F.lit("outpatient_dx")))
    else:
        dx = dx.withColumn("domain", F.lit("dx"))
    dx_rows = dx.select("ENROLID", "event_code", "domain")

    # ---- procedure rows (index-day EXCLUDED); code as-is ----
    pr = adapter.proc_long_with_setting()
    pr = (pr.join(cohort, "ENROLID", "inner")
          .filter((F.col("proc_date") >= F.col("baseline_start")) & (F.col("proc_date") < F.col("index_date"))))
    pr = pr.withColumn("event_code", F.col("code"))
    if use_setting:
        pr = pr.withColumn("domain", F.create_map([F.lit(x) for kv in PROC_DOMAIN_MAP.items() for x in kv])[F.col("setting")])
        pr = pr.withColumn("domain", F.coalesce(F.col("domain"), F.lit("outpatient_proc")))
    else:
        pr = pr.withColumn("domain", F.lit("proc"))
    pr_rows = pr.select("ENROLID", "event_code", "domain")

    # ---- drug rows (index-day EXCLUDED); ingredient-level token ----
    dr = adapter.drug_long(ids)
    dr = (dr.join(cohort, "ENROLID", "inner")
          .filter((F.col("svc_date") >= F.col("baseline_start")) & (F.col("svc_date") < F.col("index_date")))
          .withColumn("domain", F.lit("drug")))
    dr_rows = dr.select("ENROLID", "event_code", "domain")

    long_df = dx_rows.unionByName(pr_rows).unionByName(dr_rows)

    # ---- drop exposure-defining codes ----
    long_df = drop_defining_codes(adapter, resolved, long_df, cfg)

    # No dropDuplicates: occurrence multiplicity is load-bearing for the recurrence tiers.
    long_df = long_df.withColumnRenamed("ENROLID", "person_id").filter(
        F.col("event_code").isNotNull() & (F.col("event_code") != ""))
    return long_df, ids


def drop_defining_codes(adapter, resolved, long_df, cfg):
    """Remove exposure-defining codes from the candidate pool (the paper's "exclude selected codes":
    exposure surrogates only). Outcome-history dx codes are NOT removed: the paper keeps prior
    PUD / GI-bleed history as adjustable covariates (its Table 1), and with the prevalent-outcome
    exclusion off they are genuine baseline confounders."""
    drop_gen = [str(g).lower() for g in codesets.exposure_generics(cfg["cohort_id"])]
    if drop_gen:
        long_df = long_df.filter(~((F.col("domain") == "drug") & (F.lower(F.col("event_code")).isin(drop_gen))))

    # Vaccine-exposure codes: V04.81 is itself qualifying exposure evidence, so leaving these in the
    # covariate pool would leak the exposure into the PS. Drop the proc codes exactly, and the dx
    # codes at the 3-digit rollup (V04/V06: over-broad by necessity at that granularity, accepted
    # under the "exclude all flu exposure codes" rule at the cost of their preventive-care-proxy value).
    vac = codesets.vaccine(cfg["cohort_id"]) or {}
    vac_proc = [str(c) for c in vac.get("proc_codes", {})]
    if vac_proc:
        long_df = long_df.filter(~(F.col("domain").rlike("_proc$|^proc$") & F.col("event_code").isin(vac_proc)))
    vac_dx3 = sorted({str(c).replace(".", "")[:3] for c in vac.get("dx_codes", {})})
    if vac_dx3:
        long_df = long_df.filter(~(F.col("domain").rlike("_dx$|^dx$") & F.col("event_code").isin(vac_dx3)))
    return long_df


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("H2 BUILD HDPS LONG STARTING...")
    args = parse_args()

    src = get_source(args.source)
    model = src["data_model"]
    cfg = load_config(args.config, args.source)
    if args.stratum is None:
        args.stratum = default_stratum(args.source)
    cell = args.cell or args.stratum

    era_years = None
    raw_roots = None
    if model == "marketscan":
        if not args.raw_root:
            raise ValueError("--raw_root is required for the marketscan source")
        era_years = marketscan_era_years(cfg)
        raw_roots = [args.raw_root]
        if not args.restrict_ids:
            raise ValueError("--restrict_ids is required (must match build_cohort)")
        logging.info("Reading the full claims tree at %s", args.raw_root)

    spark = create_spark_session(f"h2_{cfg['cohort_id']}_{args.source}_{cell}",
                                 profile=src.get("spark_profile", "bigmem"))
    try:
        adapter = make_adapter(args.source, spark, stratum=args.stratum, era_years=era_years,
                               raw_roots=raw_roots, codes_resolved_root=args.codes_resolved,
                               cfg=cfg, vocab_ndc=args.vocab_ndc, restrict_ids=args.restrict_ids)
        resolved = load_resolved(args.codes_resolved, args.source, cfg["cohort_id"])

        long_df, ids = build_long(adapter, resolved, cell, cfg, args)
        os.makedirs(args.out_root, exist_ok=True)
        tag = f"{cfg['cohort_id']}_{cell}"
        out_parquet = os.path.join(args.out_root, f"{tag}.parquet")
        out_ids = os.path.join(args.out_root, f"{tag}_ids.csv")
        logging.info("%sDomain occurrence counts:", INDENT1)
        for r in long_df.groupBy("domain").count().orderBy("domain").collect():
            logging.info("%s%-16s %s", INDENT2, r["domain"], f"{r['count']:,}")
        long_df.write.mode("overwrite").parquet(out_parquet)
        ids.withColumnRenamed("ENROLID", "person_id").toPandas().to_csv(out_ids, index=False)
        logging.info("%sWrote %s and %s", INDENT1, out_parquet, out_ids)
        logging.info("H2 COMPLETED. Log: %s", log_file)
    except Exception as e:
        logging.error("build_long_table failed: %s", str(e))
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
