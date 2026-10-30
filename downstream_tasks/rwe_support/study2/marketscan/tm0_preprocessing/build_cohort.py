#!/usr/bin/env python3
"""
build_cohort.py  --  cohort assembly following the reference construction (Tang et al. 2026,
Supplement Fig. S1 and Table S1), mapped to claims.

  1. entry = FIRST study-class dispensing inside the index window; prior use acts only
     through the washout (step 6), as in the reference. Raw files are loaded from
     --history_start_year so steps 3-4 see all available history.
  2. same-day multi-class entries dropped (ambiguous assignment); age >= 18 at entry.
  3. eligibility over ALL history (dx_date <= entry), the claims analog of "ever in the EHR":
     T2D (E11) required; T1D (E10), ESRD/dialysis and any prior primary psychiatric outcome
     excluded (new-onset).
  4. prior antidepressant or antipsychotic use (any dispensing before entry) excluded.
  5. >=1 medical claim in the 365d before entry (Table S1's visit criterion).
  6. TWO SEPARATE EMULATIONS, each with its own GLP-1RA arm and a 12-month washout over the
     TWO compared classes only (prior use of the third class is allowed):
       vs_sglt2: entry class in {glp1ra, sglt2i}, no glp1ra/sglt2i fill in [entry-365d, entry)
       vs_dpp4 : entry class in {glp1ra, dpp4i}, no glp1ra/dpp4i  fill in [entry-365d, entry)

Claims adaptations: drug-coverage gate (RX==1 on all plan records; exposure is unobservable
without it); follow-up censored at disenrollment (claims carry no reliable death date; the
reference censors at death/study end); covariates over the 365d before entry.

Outputs (out_root):
  cohort.csv            union of both emulations + elig_vs_sglt2 / elig_vs_dpp4 flags; entry
                        and history are emulation-invariant, so the long-table, sequence and
                        embedding passes run once on this file
  <emulation>/cohort.csv  two-arm emulation rows (flags dropped) -> selection + estimation

Usage: python tm0_preprocessing/build_cohort.py --config config/tte_glp1_psych.yaml
         --raw_root <MS_RAW_SUBSET> --codes_resolved intermediate/codes_resolved
         --out_root intermediate/cohorts
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import datetime as _dt
import logging
import re

import yaml
from pyspark.sql import functions as F

import codesets_tte as cs
from marketscan_io import MarketScanIO, create_spark_session, match_dx_prefix_sets
from utils import setup_logging, time_execution

INDENT1, INDENT2 = "  ", "    "

EMULATIONS = [  # tag, flag column, the two compared classes (washout scope = exactly these)
    ("vs_sglt2", "elig_vs_sglt2", ("glp1ra", "sglt2i")),
    ("vs_dpp4",  "elig_vs_dpp4",  ("glp1ra", "dpp4i")),
]


def parse_args():
    p = argparse.ArgumentParser(description="TTE cohort assembly (two emulations, reference construction)")
    p.add_argument("--config", default="config/tte_glp1_psych.yaml")
    p.add_argument("--raw_root", default=os.environ.get("MS_RAW_SUBSET", ""))
    p.add_argument("--stratum", default="POOL")
    p.add_argument("--codes_resolved", default="intermediate/codes_resolved")
    p.add_argument("--history_start_year", type=int, default=2008,
                   help="first raw-file year to load; 'ever' lookback = all data from here")
    p.add_argument("--out_root", default="intermediate/cohorts")
    return p.parse_args()


def sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    return s


def era_years(cfg):
    """Raw-file years spanning [index_start - lookback, index_end + 1]; used by
    build_long_table, whose 365d baseline window never reaches before the index era."""
    y0 = int(str(cfg["era"]["index_start"])[:4])
    y1 = int(str(cfg["era"]["index_end"])[:4])
    lookback = max(int(cfg["windows"]["washout_days"]), int(cfg["windows"]["baseline_days"]))
    start = _dt.date.fromisoformat(str(cfg["era"]["index_start"])[:10])
    y_lo = (start - _dt.timedelta(days=lookback)).year
    return set(range(min(y_lo, y0), y1 + 2))   # +1yr for follow-up files; nonexistent years glob empty


def log_gate(cohort, label):
    n = cohort.count()
    logging.info("%sATTRITION | %-58s n=%s", INDENT1, label, f"{n:,}")
    return n


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("BUILD COHORT STARTING (two-emulation design)...")
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    baseline_days = int(cfg["windows"]["baseline_days"])
    washout_days = int(cfg["windows"]["washout_days"])
    study_end = str(cfg["era"]["study_end"])
    idx_start = str(cfg["era"]["index_start"])[:10]
    idx_end = str(cfg["era"]["index_end"])[:10]

    roots = [os.path.join(args.raw_root, "MarketScan_val"),
             os.path.join(args.raw_root, "MarketScan_test")]   # held-out split = val+test
    # full-history file filter: eligibility lookback = "ever in the data"
    y_end = int(idx_end[:4])
    years = set(range(args.history_start_year, y_end + 2))
    logging.info("Full-history file filter: keeping raw files for years %d-%d", min(years), max(years))

    spark = create_spark_session(f"tm0_{cfg['cohort_id']}")
    try:
        io = MarketScanIO(spark, raw_roots=roots, payer=args.stratum, era_years=years)

        # ---- resolved NDC sets (resolve_codes) ----
        import pandas as pd
        expo = pd.read_csv(os.path.join(args.codes_resolved, "exposure_ndc_map.csv"), dtype=str)
        dual = pd.read_csv(os.path.join(args.codes_resolved, "dual_class_ndc.csv"), dtype=str)
        medmap = pd.read_csv(os.path.join(args.codes_resolved, "med_class_ndc_map.csv"), dtype=str)
        # per-class washout sets: dual-class combos count as prior use of each component class
        wash_rows = [(r.ndc_drug_code, r.drug_class) for r in expo.itertuples()]
        for r in dual.itertuples():
            for c in str(r.classes).split("|"):
                wash_rows.append((r.ndc_drug_code, c))
        logging.info("Resolved: %d exposure NDCs (%d classes), %d class-labelled washout rows",
                     expo.ndc_drug_code.nunique(), expo.drug_class.nunique(), len(wash_rows))

        pharm = io.load_pharmacy()
        has_src = "source_db" in pharm.columns

        expo_df = spark.createDataFrame(
            [(r.ndc_drug_code, r.drug_class) for r in expo.itertuples()],
            ["NDCNUM", "drug_class"]).dropDuplicates()

        # ---- entry = first IN-WINDOW study-class dispensing (as the reference; pre-window use
        # acts only through the washout); drop same-day multi-class ----
        disp = (pharm.join(F.broadcast(expo_df), "NDCNUM", "inner")
                .filter(F.col("svc_date").isNotNull()))
        inwin = disp.filter((F.col("svc_date") >= F.lit(idx_start)) &
                            (F.col("svc_date") <= F.lit(idx_end)))
        first_day = inwin.groupBy("ENROLID").agg(F.min("svc_date").alias("index_date"))
        idx = inwin.join(first_day, "ENROLID").filter(F.col("svc_date") == F.col("index_date"))
        aggs = [F.countDistinct("drug_class").alias("n_classes"),
                F.first("index_date").alias("index_date"),
                F.min("drug_class").alias("drug_class")]
        if has_src:
            aggs.append(F.min("source_db").alias("payer_src"))
        cohort = idx.groupBy("ENROLID").agg(*aggs).cache()
        log_gate(cohort, f"study-class use in window {idx_start}..{idx_end} (entry)")
        cohort = cohort.filter(F.col("n_classes") == 1).drop("n_classes")
        log_gate(cohort, "unambiguous class at entry (same-day multi-class dropped)")

        # ---- demographics (drug-coverage gate, claims adaptation) + age >= 18 ----
        demo = io.demographics()
        cohort = cohort.join(demo, "ENROLID", "inner")
        log_gate(cohort, "drug-coverage gate (RX==1 all plan records)")
        cohort = (cohort.withColumn("age_at_index", F.year("index_date") - F.col("DOBYR"))
                  .filter(F.col("age_at_index") >= int(cfg["age_floor"])))
        log_gate(cohort, f"age >= {cfg['age_floor']} at entry")
        cohort = cohort.withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days))
        cohort = cohort.localCheckpoint()
        cohort_ids = cohort.select("ENROLID").distinct()

        # ---- full-history diagnosis stream for the candidates ----
        dx_long = io.dx_long_with_setting(person_ids=cohort_ids).localCheckpoint()

        # ---- >=1 medical claim in the 365d before entry (Table S1 visit criterion) ----
        visits = (dx_long.join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                  .filter((F.col("dx_date") >= F.date_sub(F.col("index_date"), 365)) &
                          (F.col("dx_date") < F.col("index_date")))
                  .select("ENROLID").distinct())
        cohort = cohort.join(visits, "ENROLID", "inner")
        log_gate(cohort, ">=1 medical claim in the 365d before entry")

        # ---- eligibility over ALL history (dx_date <= entry): T2D req; T1D/ESRD/prior outcome excl ----
        cov_ex = cs.covariate_exclusion_sets()
        out_sets = cs.outcome_sets()
        elig_sets = {"T2D": cov_ex["T2D"], "T1D": cov_ex["T1D"], "ESRD": cov_ex["ESRD"]}
        elig_sets.update({f"PRIOR_{k}": v for k, v in out_sets.items()})
        elig_hits = (match_dx_prefix_sets(dx_long, elig_sets, spark)
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter(F.col("dx_date") <= F.col("index_date"))
                     .select("ENROLID", "set_id").distinct()
                     .groupBy("ENROLID").pivot("set_id", list(elig_sets)).agg(F.min(F.lit(1))))
        cohort = cohort.join(elig_hits, "ENROLID", "left")
        cohort = cohort.filter(F.col("T2D") == 1)
        log_gate(cohort, "T2D dx ever (E11.*, <= entry)")
        cohort = cohort.filter(F.col("T1D").isNull())
        log_gate(cohort, "no T1D ever (E10.*)")
        cohort = cohort.filter(F.col("ESRD").isNull())
        log_gate(cohort, "no ESRD/dialysis ever")
        no_prior = None
        for k in out_sets:
            cond = F.col(f"PRIOR_{k}").isNull()
            no_prior = cond if no_prior is None else (no_prior & cond)
        cohort = cohort.filter(no_prior)
        log_gate(cohort, "no primary psychiatric outcome ever (new-onset)")
        cohort = cohort.drop(*elig_sets.keys())

        # ---- prior antidepressant / antipsychotic use (any dispensing < entry) excluded (Fig. S1) ----
        adap = medmap[medmap.med_class.isin(["antidepressants", "antipsychotics"])]
        adap_df = spark.createDataFrame([(c,) for c in adap.ndc_drug_code.unique()], ["NDCNUM"])
        adap_hits = (pharm.join(F.broadcast(adap_df), "NDCNUM", "inner")
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter(F.col("svc_date") < F.col("index_date"))
                     .select("ENROLID").distinct().withColumn("_adap", F.lit(1)))
        cohort = cohort.join(adap_hits, "ENROLID", "left").filter(F.col("_adap").isNull()).drop("_adap")
        log_gate(cohort, "no prior antidepressant/antipsychotic use")

        # ---- per-emulation 12-month washout over the two compared classes only; third-class
        # use does not exclude, so a person can be eligible for both emulations ----
        wash_df = spark.createDataFrame(wash_rows, ["NDCNUM", "wash_class"]).dropDuplicates()
        prior_cls = (pharm.join(F.broadcast(wash_df), "NDCNUM", "inner")
                     .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                     .filter((F.col("svc_date") >= F.date_sub(F.col("index_date"), washout_days)) &
                             (F.col("svc_date") < F.col("index_date")))
                     .select("ENROLID", "wash_class").distinct()
                     .groupBy("ENROLID").pivot("wash_class", sorted(expo.drug_class.unique()))
                     .agg(F.min(F.lit(1))))
        cohort = cohort.join(prior_cls, "ENROLID", "left")
        for tag, flag, (c1, c2) in EMULATIONS:
            cohort = cohort.withColumn(flag, F.when(
                F.col("drug_class").isin(c1, c2) & F.col(c1).isNull() & F.col(c2).isNull(),
                F.lit(1)).otherwise(F.lit(0)))
        cohort = cohort.drop(*sorted(expo.drug_class.unique()))
        cohort = cohort.filter((F.col("elig_vs_sglt2") == 1) | (F.col("elig_vs_dpp4") == 1))
        log_gate(cohort, "eligible for at least one emulation (post-washout)")
        for tag, flag, _ in EMULATIONS:
            n = cohort.filter(F.col(flag) == 1).count()
            logging.info("%semulation %-9s n=%s", INDENT2, tag, f"{n:,}")

        # ---- censoring anchor: disenrollment or study end (claims carry no death date) ----
        disen = io.disenrollment()
        cohort = (cohort.join(disen, "ENROLID", "left")
                  .withColumn("censor_date",
                              F.least(F.coalesce(F.col("disenroll_date"), F.lit(study_end).cast("date")),
                                      F.lit(study_end).cast("date")))
                  .drop("disenroll_date"))
        cohort = cohort.filter(F.col("censor_date") > F.col("index_date"))
        log_gate(cohort, "at least 1 day of follow-up")
        cohort = cohort.localCheckpoint()

        # ---- outcome + NCO first dates over follow-up (index, censor] ----
        nco = cs.nco_sets()
        fu_sets = {f"OUT_{sanitize(k)}": v for k, v in out_sets.items()}
        fu_sets.update({f"NCO_{sanitize(k)}": v for k, v in nco.items()})
        fu_hits = (match_dx_prefix_sets(dx_long, fu_sets, spark)
                   .join(cohort.select("ENROLID", "index_date", "censor_date"), "ENROLID", "inner")
                   .filter((F.col("dx_date") > F.col("index_date")) &
                           (F.col("dx_date") <= F.col("censor_date")))
                   .groupBy("ENROLID").pivot("set_id", list(fu_sets)).agg(F.min("dx_date")))
        cohort = cohort.join(fu_hits, "ENROLID", "left")

        # ---- baseline condition clusters (365d) + the five E11 diabetes-severity covariates of Table S5 ----
        cluster_pairs = cs.cluster_master()
        cl_sets = {}
        for code, cl in cluster_pairs:
            cl_sets.setdefault(f"COV_{sanitize(cl)}", []).append(code)
        cl_sets.update({
            "COV_dm_nephropathy":                    ["E11.2"],
            "COV_dm_retinopathy":                    ["E11.3"],
            "COV_dm_neuropathy":                     ["E11.4"],
            "COV_dm_other_specified_complications":  ["E11.6"],
            "COV_dm_unspecified_complications":      ["E11.8"],
        })
        cl_hits = (match_dx_prefix_sets(dx_long, cl_sets, spark)
                   .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                   .filter((F.col("dx_date") >= F.col("baseline_start")) &
                           (F.col("dx_date") <= F.col("index_date")))
                   .select("ENROLID", "set_id").distinct()
                   .groupBy("ENROLID").pivot("set_id", list(cl_sets)).agg(F.min(F.lit(1))))
        cohort = cohort.join(cl_hits, "ENROLID", "left")

        # ---- baseline medication-class covariates ----
        med_df = spark.createDataFrame(
            [(r.ndc_drug_code, f"MED_{sanitize(r.med_class)}") for r in medmap.itertuples()],
            ["NDCNUM", "set_id"]).dropDuplicates()
        med_names = sorted({f"MED_{sanitize(m)}" for m in medmap.med_class.unique()})
        med_hits = (pharm.join(F.broadcast(med_df), "NDCNUM", "inner")
                    .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                    .filter((F.col("svc_date") >= F.col("baseline_start")) &
                            (F.col("svc_date") < F.col("index_date")))
                    .select("ENROLID", "set_id").distinct()
                    .groupBy("ENROLID").pivot("set_id", med_names).agg(F.min(F.lit(1))))
        cohort = cohort.join(med_hits, "ENROLID", "left")

        # ---- baseline utilization ----
        base_dx = (dx_long.join(cohort.select("ENROLID", "index_date", "baseline_start"),
                                "ENROLID", "inner")
                   .filter((F.col("dx_date") >= F.col("baseline_start")) &
                           (F.col("dx_date") <= F.col("index_date"))))
        ut = base_dx.groupBy("ENROLID").agg(
            F.countDistinct(F.when(F.col("setting") == "inpatient", F.col("dx_date"))).alias("ut_n_ip_admits"),
            F.countDistinct(F.when(F.col("setting") == "outpatient", F.col("dx_date"))).alias("ut_n_op_days"),
            F.countDistinct("dx_date").alias("ut_n_dx_days"))
        rx = (pharm.join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
              .filter((F.col("svc_date") >= F.col("baseline_start")) &
                      (F.col("svc_date") < F.col("index_date")))
              .groupBy("ENROLID").agg(F.count("*").alias("ut_n_rx_disp")))
        cohort = cohort.join(ut, "ENROLID", "left").join(rx, "ENROLID", "left")

        # ---- finalize: one projection ----
        sel = [
            F.col("ENROLID").alias("person_id"),
            F.when(F.col("payer_src") == "CCAE", F.lit("commercial"))
             .when(F.col("payer_src") == "MDCR", F.lit("medicare"))
             .when(F.col("payer_src") == "MDCD", F.lit("medicaid"))
             .otherwise(F.lit("other_unknown")).alias("ins_category"),
            (F.col("payer_src") if has_src else F.lit(args.stratum)).alias("payer"),
            F.col("drug_class"), F.col("index_date"),
            F.year("index_date").alias("cal_year"),
            F.col("age_at_index"), F.col("SEX").alias("sex"),
            F.col("censor_date"),
            F.datediff("censor_date", "index_date").alias("followup_days"),
            F.col("elig_vs_sglt2"), F.col("elig_vs_dpp4"),
        ]
        if "REGION" in cohort.columns:
            sel.append(F.col("REGION").alias("region"))
        for col in fu_sets:
            tgt = col.lower()
            sel.append(F.when(F.col(col).isNotNull(), 1).otherwise(0).alias(f"{tgt}_event"))
            sel.append(F.greatest(F.lit(1),
                                  F.datediff(F.coalesce(F.col(col), F.col("censor_date")),
                                             F.col("index_date"))).alias(f"{tgt}_time"))
        for col in list(cl_sets) + med_names:
            sel.append(F.coalesce(F.col(col), F.lit(0)).alias(col.lower()))
        for col in ["ut_n_ip_admits", "ut_n_op_days", "ut_n_dx_days", "ut_n_rx_disp"]:
            sel.append(F.coalesce(F.col(col), F.lit(0)).alias(col))
        cohort = cohort.select(*sel).dropDuplicates(["person_id"])

        out_dir = args.out_root
        os.makedirs(out_dir, exist_ok=True)
        pdf = cohort.toPandas()
        logging.info("%sUnion cohort: %s rows x %d cols", INDENT1, f"{len(pdf):,}", pdf.shape[1])
        pdf.to_csv(os.path.join(out_dir, "cohort.csv"), index=False)
        logging.info("Wrote %s", os.path.join(out_dir, "cohort.csv"))

        for tag, flag, classes in EMULATIONS:
            sub = (pdf[(pdf[flag] == 1) & (pdf.drug_class.isin(classes))]
                   .drop(columns=["elig_vs_sglt2", "elig_vs_dpp4"]))
            sub_dir = os.path.join(out_dir, tag)
            os.makedirs(sub_dir, exist_ok=True)
            sub.to_csv(os.path.join(sub_dir, "cohort.csv"), index=False)
            logging.info("%semulation %-9s -> %s (%s rows)", INDENT1, tag,
                         os.path.join(sub_dir, "cohort.csv"), f"{len(sub):,}")
            for cls, g in sub.groupby("drug_class"):
                ev = {k: int(g[f"out_{sanitize(k).lower()}_event"].sum()) for k in cs.PRIMARY_OUTCOMES}
                logging.info("%s%-8s n=%s  events=%s", INDENT2, cls, f"{len(g):,}", ev)
        logging.info("Log: %s", log_file)
        logging.info("BUILD COHORT COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
