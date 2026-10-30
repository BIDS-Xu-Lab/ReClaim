#!/usr/bin/env python3
"""
build_cohort.py -- TTE cohort assembly on OMOP (YNHH), following the reference's own
construction (Tang et al. 2026, Supplement Fig. S1 / Table S1).

  1. entry = first study-class exposure inside the index window; prior use is handled by
     the washout (step 6), not by excluding prevalent users outright.
  2. same-day multi-class entries dropped; age >= 18 (and <= 110, the export's plausibility cap).
  3. eligibility over all available history (dx_date <= entry): T2D required; T1D, ESRD/
     dialysis and any prior primary psychiatric outcome excluded.
  4. prior antidepressant or antipsychotic use (any exposure before entry) excluded.
  5. >= 1 medical visit in the 365d before entry (Table S1), implemented as >= 1 recorded
     diagnosis date in that window.
  6. two separate emulations, each with its own 12-month washout over the two compared
     classes only (prior third-class use is allowed):
       vs_sglt2: entry class in {glp1ra, sglt2i}, no glp1ra/sglt2i exposure in [entry-365d, entry)
       vs_dpp4 : entry class in {glp1ra, dpp4i},  no glp1ra/dpp4i  exposure in [entry-365d, entry)
  7. follow-up ends at the outcome, death or the study end (no observation-period
     censoring), the reference's rule, which an EHR can honour.

OMOP adaptations: diagnoses matched on ICD-10 source values (--dx_route source, the
reference's claims semantics) or on resolved standard concepts (standard); exposures and
medication classes from drug_exposure (drug_concept_id as the dispensing analogue);
insurance from payer_plan_period; the tokenized-population gate (--restrict_ids) keeps only
persons with a ReClaim token sequence. Baseline covariates over 365d: condition clusters,
diabetes-severity flags, medication classes, utilization, insurance.

Outputs (out_root/<cell>):
  cohort.csv                 union of both emulations + elig_vs_sglt2 / elig_vs_dpp4 flags
                             (feeds the hdPS long table and the embedding pass)
  ../<cell>_vs_sglt2/cohort.csv, ../<cell>_vs_dpp4/cohort.csv   per-emulation two-arm cohorts
                             (feed hdPS selection and estimation)

Usage (from the module root): python ty0_preprocessing/build_cohort.py --source ynhh \
    --dx_route source --cell ynhh_src --restrict_ids "<token store>/trajectory/*.parquet"
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging
import re
import sys

from pyspark.sql import functions as F

# shared helpers beside this file; root-level codesets/datasources/utils via PYTHONPATH
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import codesets_tte as cs                                                  # noqa: E402
from datasources import make_adapter, get_source                          # noqa: E402
from utils import setup_logging, time_execution                           # noqa: E402
from cohort_helpers import (load_config, create_spark_session, match_dx_prefix_sets,   # noqa: E402
                          build_source_sets, match_sets, DM_SEVERITY_IDS, sanitize)

INDENT1, INDENT2 = "  ", "    "
EMULATIONS = [  # tag suffix, flag column, the two compared classes (washout scope = exactly these)
    ("vs_sglt2", "elig_vs_sglt2", ("glp1ra", "sglt2i")),
    ("vs_dpp4",  "elig_vs_dpp4",  ("glp1ra", "dpp4i")),
]


def parse_args():
    p = argparse.ArgumentParser(description="TTE cohort assembly on OMOP (two emulations)")
    p.add_argument("--config", default="config/tte_glp1_psych.yaml")
    p.add_argument("--source", default="ynhh")
    p.add_argument("--codes_resolved", default="intermediate/codes_resolved")
    p.add_argument("--out_root", default="intermediate/cohorts")
    p.add_argument("--restrict_ids", default=None, help="token-store trajectory/ parquet glob")
    p.add_argument("--dx_route", default="source", choices=["standard", "source"])
    p.add_argument("--cell", required=True, help="output cell, e.g. ynhh_src")
    return p.parse_args()


def log_gate(cohort, label):
    n = cohort.count()
    logging.info("%sATTRITION | %-58s n=%s", INDENT1, label, f"{n:,}")
    return n


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("BUILD COHORT (OMOP) STARTING (two-emulation design)...")
    args = parse_args()
    cfg = load_config(args.config, args.source)
    baseline_days = int(cfg["windows"]["baseline_days"])
    washout_days = int(cfg["windows"]["washout_days"])
    study_end = str(cfg["era"]["study_end"])
    idx_start, idx_end = str(cfg["era"]["index_start"])[:10], str(cfg["era"]["index_end"])[:10]

    src = get_source(args.source)
    spark = create_spark_session(f"ty0_{cfg['cohort_id']}_{args.source}",
                                 profile=src.get("spark_profile", "bigmem"))
    try:
        import pandas as pd
        adapter = make_adapter(args.source, spark, stratum=args.source,
                               codes_resolved_root=args.codes_resolved, cfg=cfg,
                               restrict_ids=args.restrict_ids)
        cr = args.codes_resolved
        expo = pd.read_csv(os.path.join(cr, "exposure_concept_map.csv"), dtype=str)
        dual = pd.read_csv(os.path.join(cr, "dual_class_concept.csv"), dtype=str)
        medmap = pd.read_csv(os.path.join(cr, "med_class_concept_map.csv"), dtype=str)
        dxmap = pd.read_csv(os.path.join(cr, "dx_concept_map.csv"), dtype=str)
        clmap = pd.read_csv(os.path.join(cr, "cluster_concept_map.csv"), dtype=str)
        # per-class washout concepts: single-class exposure concepts + dual-class combos exploded
        # onto each component class (a GLP-1/DPP4i combo counts as prior use of both).
        wash_rows = [(r.drug_concept_id, r.drug_class) for r in expo.itertuples()]
        for r in dual.itertuples():
            for c in str(r.classes).split("|"):
                wash_rows.append((r.drug_concept_id, c))
        classes = sorted(expo.drug_class.unique())
        logging.info("Resolved: %d exposure concepts (%d classes), %d class-labelled washout rows",
                     expo.drug_concept_id.nunique(), len(classes), len(wash_rows))
        adap_classes = [c for c in ("antidepressants", "antipsychotics") if c in set(medmap.med_class)]
        if len(adap_classes) != 2:
            sys.exit(f"ERROR: med_class_concept_map lacks antidepressants/antipsychotics; classes = "
                     f"{sorted(medmap.med_class.unique())}")

        pharm = adapter.load_pharmacy()   # (ENROLID, svc_date, NDCNUM = drug_concept_id string)
        expo_df = spark.createDataFrame(
            expo.rename(columns={"drug_concept_id": "NDCNUM"})[["NDCNUM", "drug_class"]].drop_duplicates())

        # ---- entry = first IN-WINDOW study-class exposure; drop same-day multi-class ----
        # The reference indexes on the first use inside the window and handles prior use
        # only through the 12-month washout.
        disp = (pharm.join(F.broadcast(expo_df), "NDCNUM", "inner")
                .filter(F.col("svc_date").isNotNull()))
        inwin = disp.filter((F.col("svc_date") >= F.lit(idx_start)) & (F.col("svc_date") <= F.lit(idx_end)))
        first_day = inwin.groupBy("ENROLID").agg(F.min("svc_date").alias("index_date"))
        idx = inwin.join(first_day, "ENROLID").filter(F.col("svc_date") == F.col("index_date"))
        cohort = idx.groupBy("ENROLID").agg(
            F.countDistinct("drug_class").alias("n_classes"),
            F.first("index_date").alias("index_date"),
            F.min("drug_class").alias("drug_class")).cache()
        log_gate(cohort, f"study-class use in window {idx_start}..{idx_end} (entry)")
        cohort = cohort.filter(F.col("n_classes") == 1).drop("n_classes")
        log_gate(cohort, "unambiguous class at entry (same-day multi-class dropped)")

        # ---- demographics (+ tokenized-population gate inside the adapter) + age ----
        demo = adapter.demographics(cfg)
        cohort = cohort.join(demo, "ENROLID", "inner")
        log_gate(cohort, "demographics resolvable (+restrict_ids gate)" if args.restrict_ids
                 else "demographics resolvable")
        cohort = (cohort.withColumn("age_at_index", F.year("index_date") - F.col("DOBYR"))
                  .filter((F.col("age_at_index") >= int(cfg["age_floor"])) & (F.col("age_at_index") <= 110)))
        log_gate(cohort, f"age >= {cfg['age_floor']} and <= 110 at entry")

        # ---- baseline observability gate (config enrollment.gate; "off" in this study) ----
        cohort = cohort.withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days))
        cohort = adapter.apply_continuous_gate(cfg, cohort, "baseline_start", "index_date", "tte")
        log_gate(cohort, "observable over baseline (config gate)")
        cohort = cohort.localCheckpoint()
        cohort_ids = cohort.select("ENROLID").distinct()

        # ---- dx streams + matching route ----
        dx_long = adapter.dx_long_with_setting(person_ids=cohort_ids).localCheckpoint()
        if args.dx_route == "source":
            src_sets = build_source_sets()
            dx_src = adapter.dx_source_long(person_ids=cohort_ids).localCheckpoint()
            def match_named(ids):
                return match_dx_prefix_sets(dx_src, {i: src_sets[i] for i in ids}, spark)
        else:
            dx_events = dx_long.select("ENROLID", "dx_date", "code")
            obs_events = adapter.obs_long(person_ids=cohort_ids)
            if obs_events is not None:
                dx_events = dx_events.unionByName(obs_events.select("ENROLID", "dx_date", "code"))
            dx_events = dx_events.localCheckpoint()
            allmap = pd.concat([dxmap[["set_id", "concept_id"]], clmap[["set_id", "concept_id"]]],
                               ignore_index=True)
            def match_named(ids):
                return match_sets(dx_events, allmap[allmap.set_id.isin(ids)], spark)

        # ---- >=1 medical visit in the 365d before entry (Table S1), as >=1 recorded dx date ----
        visits = (dx_long.join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                  .filter((F.col("dx_date") >= F.date_sub(F.col("index_date"), 365)) &
                          (F.col("dx_date") < F.col("index_date")))
                  .select("ENROLID").distinct())
        cohort = cohort.join(visits, "ENROLID", "inner")
        log_gate(cohort, ">=1 recorded diagnosis date in the 365d before entry")

        # ---- eligibility over ALL history (dx_date <= entry): T2D req; T1D/ESRD/prior outcome excl ----
        out_ids = [f"OUT_{sanitize(k)}" for k in cs.PRIMARY_OUTCOMES]
        elig_ids = ["T2D", "T1D", "ESRD"] + out_ids
        elig_hits = (match_named(elig_ids)
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter(F.col("dx_date") <= F.col("index_date"))
                     .select("ENROLID", "set_id").distinct()
                     .groupBy("ENROLID").pivot("set_id", elig_ids).agg(F.min(F.lit(1))))
        cohort = cohort.join(elig_hits, "ENROLID", "left")
        cohort = cohort.filter(F.col("T2D") == 1)
        log_gate(cohort, "T2D dx ever (<= entry)")
        cohort = cohort.filter(F.col("T1D").isNull())
        log_gate(cohort, "no T1D ever")
        cohort = cohort.filter(F.col("ESRD").isNull())
        log_gate(cohort, "no ESRD/dialysis ever")
        no_prior = None
        for c in out_ids:
            cond = F.col(c).isNull()
            no_prior = cond if no_prior is None else (no_prior & cond)
        cohort = cohort.filter(no_prior).drop(*elig_ids)
        log_gate(cohort, "no primary psychiatric outcome ever (new-onset)")

        # ---- prior antidepressant / antipsychotic use (any exposure < entry) excluded ----
        adap_ids = list(medmap[medmap.med_class.isin(adap_classes)].drug_concept_id.unique())
        adap_hits = (adapter.drug_dispensings(pharm, adap_ids)
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter(F.col("svc_date") < F.col("index_date"))
                     .select("ENROLID").distinct().withColumn("_adap", F.lit(1)))
        cohort = cohort.join(adap_hits, "ENROLID", "left").filter(F.col("_adap").isNull()).drop("_adap")
        log_gate(cohort, "no prior antidepressant/antipsychotic use")

        # ---- per-emulation 12-month washout on the two compared classes only ----
        wash_df = spark.createDataFrame(wash_rows, ["NDCNUM", "wash_class"]).dropDuplicates()
        prior_cls = (pharm.join(F.broadcast(wash_df), "NDCNUM", "inner")
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter((F.col("svc_date") >= F.date_sub(F.col("index_date"), washout_days)) &
                             (F.col("svc_date") < F.col("index_date")))
                     .select("ENROLID", "wash_class").distinct()
                     .groupBy("ENROLID").pivot("wash_class", classes).agg(F.min(F.lit(1))))
        cohort = cohort.join(prior_cls, "ENROLID", "left")
        for _tag, flag, (c1, c2) in EMULATIONS:
            cohort = cohort.withColumn(flag, F.when(
                F.col("drug_class").isin(c1, c2) & F.col(c1).isNull() & F.col(c2).isNull(),
                F.lit(1)).otherwise(F.lit(0)))
        cohort = cohort.drop(*classes)
        cohort = cohort.filter((F.col("elig_vs_sglt2") == 1) | (F.col("elig_vs_dpp4") == 1))
        log_gate(cohort, "eligible for at least one emulation (post-washout)")
        for tag, flag, _ in EMULATIONS:
            logging.info("%semulation %-9s n=%s", INDENT2, tag, f"{cohort.filter(F.col(flag) == 1).count():,}")

        # ---- censoring: min(death, study_end), the reference's rule ----
        death = adapter._death_dates()
        if death is not None:
            cohort = cohort.join(death, "ENROLID", "left")
        else:
            cohort = cohort.withColumn("death_date", F.lit(None).cast("date"))
        cohort = (cohort.withColumn("censor_date",
                                    F.least(F.coalesce(F.col("death_date"), F.lit(study_end).cast("date")),
                                            F.lit(study_end).cast("date")))
                  .drop("death_date"))
        cohort = cohort.filter(F.col("censor_date") > F.col("index_date"))
        log_gate(cohort, "at least 1 day of follow-up")
        cohort = cohort.localCheckpoint()

        # ---- outcomes + NCOs over follow-up (index, censor] ----
        fu_ids = sorted(s for s in dxmap.set_id.unique() if s.startswith(("OUT_", "NCO_")))
        fu_hits = (match_named(fu_ids)
                   .join(cohort.select("ENROLID", "index_date", "censor_date"), "ENROLID", "inner")
                   .filter((F.col("dx_date") > F.col("index_date")) & (F.col("dx_date") <= F.col("censor_date")))
                   .groupBy("ENROLID").pivot("set_id", fu_ids).agg(F.min("dx_date")))
        cohort = cohort.join(fu_hits, "ENROLID", "left")

        # ---- baseline condition-cluster covariates (365d) ----
        cl_ids = sorted(clmap.set_id.unique())
        cl_hits = (match_named(cl_ids)
                   .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                   .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date")))
                   .select("ENROLID", "set_id").distinct()
                   .groupBy("ENROLID").pivot("set_id", cl_ids).agg(F.min(F.lit(1))))
        cohort = cohort.join(cl_hits, "ENROLID", "left")

        # ---- diabetes-severity covariates (Table S5; E11 tree, matched on source values in
        #      both routes so the definition is route-invariant) ----
        dm_sets = {k: v for k, v in build_source_sets().items() if k in DM_SEVERITY_IDS}
        dx_src_dm = dx_src if args.dx_route == "source" else \
            adapter.dx_source_long(person_ids=cohort_ids).localCheckpoint()
        dm_hits = (match_dx_prefix_sets(dx_src_dm, dm_sets, spark)
                   .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                   .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date")))
                   .select("ENROLID", "set_id").distinct()
                   .groupBy("ENROLID").pivot("set_id", DM_SEVERITY_IDS).agg(F.min(F.lit(1))))
        cohort = cohort.join(dm_hits, "ENROLID", "left")
        for _c in DM_SEVERITY_IDS:              # pivot omits a set with zero hits; add it back
            if _c not in cohort.columns:
                cohort = cohort.withColumn(_c, F.lit(None).cast("int"))

        # ---- baseline medication-class covariates ----
        med_df = spark.createDataFrame(
            medmap.rename(columns={"drug_concept_id": "NDCNUM"})
            .assign(set_id=lambda d: "MED_" + d.med_class)[["NDCNUM", "set_id"]].drop_duplicates())
        med_ids = sorted("MED_" + m for m in medmap.med_class.unique())
        med_hits = (pharm.join(F.broadcast(med_df), "NDCNUM", "inner")
                    .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                    .filter((F.col("svc_date") >= F.col("baseline_start")) & (F.col("svc_date") < F.col("index_date")))
                    .select("ENROLID", "set_id").distinct()
                    .groupBy("ENROLID").pivot("set_id", med_ids).agg(F.min(F.lit(1))))
        cohort = cohort.join(med_hits, "ENROLID", "left")

        # ---- baseline utilization (dx-day counts by setting + drug rows) ----
        base_dx = (dx_long.join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                   .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date"))))
        ut = base_dx.groupBy("ENROLID").agg(
            F.countDistinct(F.when(F.col("setting") == "inpatient", F.col("dx_date"))).alias("ut_n_ip_admits"),
            F.countDistinct(F.when(F.col("setting") == "outpatient", F.col("dx_date"))).alias("ut_n_op_days"),
            F.countDistinct("dx_date").alias("ut_n_dx_days"))
        rx = (pharm.join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
              .filter((F.col("svc_date") >= F.col("baseline_start")) & (F.col("svc_date") < F.col("index_date")))
              .groupBy("ENROLID").agg(F.count("*").alias("ut_n_rx_disp")))
        cohort = cohort.join(ut, "ENROLID", "left").join(rx, "ENROLID", "left")

        # ---- insurance covariate (Table S5; payer_plan_period via the payer map) ----
        cohort = cohort.join(adapter.insurance_at_index(cohort), "ENROLID", "left")
        cohort = cohort.withColumn("ins_category", F.coalesce(F.col("ins_category"), F.lit("other_unknown")))

        # ---- finalize: one projection + the two eligibility flags ----
        sel = [F.col("ENROLID").alias("person_id"),
               F.lit(args.source).alias("payer"),
               F.col("drug_class"), F.col("index_date"),
               F.year("index_date").alias("cal_year"),
               F.col("age_at_index"), F.col("SEX").alias("sex"),
               F.col("censor_date"), F.col("ins_category"),
               F.datediff("censor_date", "index_date").alias("followup_days"),
               F.col("elig_vs_sglt2"), F.col("elig_vs_dpp4")]
        for col in fu_ids:
            tgt = col.lower()
            sel.append(F.when(F.col(col).isNotNull(), 1).otherwise(0).alias(f"{tgt}_event"))
            sel.append(F.greatest(F.lit(1), F.datediff(F.coalesce(F.col(col), F.col("censor_date")),
                                                      F.col("index_date"))).alias(f"{tgt}_time"))
        for col in cl_ids + DM_SEVERITY_IDS + med_ids:
            sel.append(F.coalesce(F.col(col), F.lit(0)).alias(col.lower()))
        for col in ["ut_n_ip_admits", "ut_n_op_days", "ut_n_dx_days", "ut_n_rx_disp"]:
            sel.append(F.coalesce(F.col(col), F.lit(0)).alias(col))
        cohort = cohort.select(*sel).dropDuplicates(["person_id"])

        out_dir = os.path.join(args.out_root, args.cell)
        os.makedirs(out_dir, exist_ok=True)
        pdf = cohort.toPandas()
        logging.info("%sUnion cohort: %s rows x %d cols", INDENT1, f"{len(pdf):,}", pdf.shape[1])
        pdf.to_csv(os.path.join(out_dir, "cohort.csv"), index=False)   # person-level: intermediate/ only
        logging.info("Wrote %s", os.path.join(out_dir, "cohort.csv"))
        for tag, flag, cls2 in EMULATIONS:
            sub = pdf[(pdf[flag] == 1) & pdf.drug_class.isin(cls2)].drop(columns=["elig_vs_sglt2", "elig_vs_dpp4"])
            sub_dir = os.path.join(args.out_root, f"{args.cell}_{tag}")
            os.makedirs(sub_dir, exist_ok=True)
            sub.to_csv(os.path.join(sub_dir, "cohort.csv"), index=False)
            logging.info("%semulation %-9s -> %s (%s rows)", INDENT1, tag, os.path.join(sub_dir, "cohort.csv"), f"{len(sub):,}")
            for cls, g in sub.groupby("drug_class"):
                evc = {k: int(g[c].sum()) for k in cs.PRIMARY_OUTCOMES
                       for c in [f"out_{sanitize(k).lower()}_event"] if c in g.columns}
                logging.info("%s%-8s n=%s  events=%s", INDENT2, cls, f"{len(g):,}", evc)
        logging.info("Log: %s", log_file)
        logging.info("BUILD COHORT (OMOP) COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
