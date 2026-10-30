#!/usr/bin/env python3
"""
build_cohort.py  --  new-user cohort assembly for the hdPS benchmark.

Assembles one row per patient:
    person_id, payer, arm(0/1), index_date, entry_date_embed, outcome(0/1),
    followup_days, person_time, age_at_index, sex, + baseline clinical (l_*) flags.

Raw-data reading is delegated to the data-source adapter (datasources/); the resolver
(resolvers/) turns codesets.py into the source's match keys. The epidemiologic logic below is
written against the adapter interface only.

Output: {out_root}/{cohort_id}/{cell}/cohort.csv, out_root = intermediate/mdcr/cohorts.
Frozen design decisions live in config/<cohort>.yaml.

Usage: python build_cohort.py --config config/<cohort>.yaml --stratum MDCR [--season t]
           --raw_root <root> --restrict_ids <parquet glob> --out_root <dir>
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging
import datetime as _dt

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

import codesets
from utils import setup_logging, time_execution
from datasources import make_adapter, get_source, default_stratum, enrollment_gate_mode
from resolvers import load_resolved

INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

# Warn at most once per run if a source without dx positions meets a cohort's position requirement.
_WARNED_NO_POSITION = False


# =============================================================================
# args + spark
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="hdPS benchmark cohort assembly")
    p.add_argument("--config", required=True, help="Path to config/<cohort>.yaml")
    p.add_argument("--source", default="marketscan",
                   help="Data source name (datasources/registry.py).")
    p.add_argument("--raw_root", default=None,
                   help="MarketScan only: the full claims tree (payer dirs directly under it)")
    p.add_argument("--stratum", default="MDCR",
                   help="Analysis cell (MarketScan payer stratum; MDCR is the design). "
                        "Validated against the source's strata.")
    p.add_argument("--age_floor_override", type=int, default=None,
                   help="Override the config age floor.")
    p.add_argument("--restrict_ids", default=None,
                   help="Parquet glob of the ReClaim-tokenized population (column enrollee_id or "
                        "ENROLID); restricts the patient universe to people who HAVE a token sequence. "
                        "REQUIRED for the marketscan source.")
    p.add_argument("--codes_resolved", default="intermediate/codes_resolved",
                   help="Root of resolve_codes's resolved-code artifacts. Run resolve_codes.py first.")
    p.add_argument("--out_root", default="results/mdcr",
                   help="Output root for cohort.csv (intermediate/mdcr/cohorts)")
    p.add_argument("--season", type=int, default=None,
                   help="negative_control only: build ONE season's two-arm LANDMARK sub-cohort into "
                        "{out_root}/{cohort}/{cell}_s{season}/cohort.csv.")
    return p.parse_args()


def create_spark_session(app_name: str, profile: str = "bigmem") -> SparkSession:
    """Local-mode Spark; 'bigmem' = the MarketScan sizing for one bigmem node, 'small' = a modest driver."""
    b = SparkSession.builder.appName(app_name).master("local[*]")
    if profile == "small":
        b = (b.config("spark.driver.memory", "16g")
             .config("spark.driver.maxResultSize", "4g")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer"))
        return b.getOrCreate()
    return (b.config("spark.driver.memory", "640g")
            .config("spark.driver.maxResultSize", "32g")
            .config("spark.driver.extraJavaOptions",
                    "-XX:+UseG1GC -XX:MaxGCPauseMillis=20000 -XX:InitiatingHeapOccupancyPercent=35 "
                    "-XX:+UseStringDeduplication")
            .config("spark.network.timeout", "1800s")
            .config("spark.executor.heartbeatInterval", "60s")
            .config("spark.executor.memory", "350g")
            .config("spark.executor.cores", "10")
            .config("spark.sql.shuffle.partitions", "240")
            .config("spark.sql.files.maxPartitionBytes", "32m")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())


# =============================================================================
# outcome ascertainment
# =============================================================================
def build_outcome_flag(cohort: DataFrame, dx_long: DataFrame, cfg: dict, resolved, adapter) -> DataFrame:
    """
    Attach outcome(0/1) + first_outcome_date over the follow-up window, as a union of per-setting arms
    (the paper's two-arm definition):
      INPATIENT arm : `outcome_inpatient` codes, restricted to the required dx positions if set
                      (positive_control: 1st/2nd position, as in the reference).
      OUTPATIENT arm: `outcome_outpatient` codes (physician-service arm); no position requirement.
    """
    global _WARNED_NO_POSITION
    codes = resolved.dx("outcome_inpatient")
    settings = cfg["outcome"].get("settings", ["inpatient"])
    positions = cfg["outcome"].get("position_requirement")  # e.g. [1,2] for positive_control inpatient arm
    op_codes = resolved.dx("outcome_outpatient") or codes

    arms = []
    if "inpatient" in settings:
        inp = adapter.match_dx(dx_long.filter(F.col("setting") == "inpatient"), codes)
        if positions:
            if adapter.dx_position_available:
                inp = inp.filter(F.col("position").isin([int(p) for p in positions]))
                logging.info("%sinpatient outcome arm restricted to dx position/status %s (source=%s)",
                             INDENT2, positions, adapter.source_name)
            elif not _WARNED_NO_POSITION:
                logging.warning("%sprimary/secondary-position rule set but source '%s' carries no dx "
                                "position/status -> restriction skipped (per-source deviation)",
                                INDENT2, adapter.source_name)
                _WARNED_NO_POSITION = True
        arms.append(inp)
    if "outpatient" in settings:
        arms.append(adapter.match_dx(dx_long.filter(F.col("setting") == "outpatient"), op_codes))
    dx = arms[0]
    for a in arms[1:]:
        dx = dx.unionByName(a)

    # Follow-up window. positive_control: (index_date, followup_end]. negative_control (landmark): fixed [followup_start, followup_end].
    has_fu_start = "followup_start" in cohort.columns
    keys = ["ENROLID", "index_date", "followup_end"] + (["followup_start"] if has_fu_start else [])
    j = dx.join(cohort.select(*keys), "ENROLID", "inner")
    if has_fu_start:
        j = j.filter((F.col("dx_date") >= F.col("followup_start")) & (F.col("dx_date") <= F.col("followup_end")))
    else:
        j = j.filter((F.col("dx_date") > F.col("index_date")) & (F.col("dx_date") <= F.col("followup_end")))
    first = j.groupBy("ENROLID").agg(F.min("dx_date").alias("first_outcome_date"))
    out = (cohort.join(first, "ENROLID", "left")
           .withColumn("outcome", F.when(F.col("first_outcome_date").isNotNull(), 1).otherwise(0)))
    return out


# =============================================================================
# positive_control assembly  (celecoxib vs ns-NSAID)
# =============================================================================
@time_execution
def build_v1(adapter, resolved, cfg, args) -> DataFrame:
    baseline_days = cfg["windows"]["baseline_days"]
    washout_days = cfg["windows"]["washout_days"]
    followup_days = cfg["windows"]["followup_days"]
    age_floor = adapter.resolve_age_floor(cfg, args)

    cid = cfg["cohort_id"]
    treat_gen = list(codesets.drug_generics(cid, "treatment"))
    comp_gen = list(codesets.drug_generics(cid, "comparator"))
    wash_gen = list(codesets.drug_generics(cid, "washout"))

    # Exposure arms exclude fixed-dose combinations (config default true); washout stays inclusive.
    single_ing = bool(cfg.get("exposure", {}).get("single_ingredient_only", True))
    treat_codes = resolved.drug(treat_gen, single_ingredient_only=single_ing)
    comp_codes = resolved.drug(comp_gen, single_ingredient_only=single_ing)
    wash_codes = resolved.drug(wash_gen)

    pharm = adapter.load_pharmacy()   # one cached scan, reused by all drug filters
    treat_disp = adapter.drug_dispensings(pharm, treat_codes).withColumn("arm", F.lit(1))
    comp_disp = adapter.drug_dispensings(pharm, comp_codes).withColumn("arm", F.lit(0))
    disp = treat_disp.unionByName(comp_disp)

    # --- index = first qualifying dispensing per person; drop same-day both classes ---
    first_day = disp.groupBy("ENROLID").agg(F.min("svc_date").alias("index_date"))
    idx = disp.join(first_day, "ENROLID").filter(F.col("svc_date") == F.col("index_date"))
    aggs = [F.countDistinct("arm").alias("n_arms"), F.max("arm").alias("arm"),
            F.first("index_date").alias("index_date")]
    arm_counts = idx.groupBy("ENROLID").agg(*aggs)
    cohort = arm_counts.filter(F.col("n_arms") == 1).drop("n_arms")  # drop same-day both classes

    # --- era restriction on index date ---
    cohort = cohort.filter((F.col("index_date") >= F.lit(cfg["era"]["index_start"])) &
                           (F.col("index_date") <= F.lit(cfg["era"]["index_end"])))

    # --- demographics (source eligibility applied) + age >= floor ---
    demo = adapter.demographics(cfg)
    cohort = cohort.join(demo, "ENROLID", "inner")
    cohort = cohort.withColumn("age_at_index", F.year("index_date") - F.col("DOBYR")) \
                   .filter(F.col("age_at_index") >= age_floor)

    # --- new-user washout: no NSAID/COX-2 dispensing in [index-548d, index) ---
    wash_disp = adapter.drug_dispensings(pharm, wash_codes)
    prior = (wash_disp.join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
             .filter((F.col("svc_date") >= F.date_sub(F.col("index_date"), washout_days)) &
                     (F.col("svc_date") < F.col("index_date")))
             .select("ENROLID").distinct().withColumn("_prior_use", F.lit(1)))
    cohort = cohort.join(prior, "ENROLID", "left").filter(F.col("_prior_use").isNull()).drop("_prior_use")

    # --- follow-up window + windows for outcome ---
    cohort = (cohort.withColumn("followup_end", F.date_add(F.col("index_date"), followup_days))
              .withColumn("followup_days", F.lit(followup_days))
              .withColumn("entry_date_embed", F.trunc(F.col("index_date"), "month"))  # M0 blackout
              .withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days)))

    # --- observability gate (config; off in the design) and person-time censoring at disenrollment ---
    cohort = adapter.apply_continuous_gate(cfg, cohort, "baseline_start", "followup_end", "positive_control")
    cohort = adapter.add_person_time(cohort)

    cohort = cohort.localCheckpoint()
    cohort_ids = cohort.select("ENROLID").distinct()

    # --- prevalent-outcome exclusion: config exclusions.prior_outcome_baseline. Default OFF as in the
    #     reference: Schneeweiss 2009 retains patients with prior PUD / GI haemorrhage (its Table 1
    #     carries them as covariates). ---
    dx_long = adapter.dx_long_with_setting(person_ids=cohort_ids).localCheckpoint()
    if bool(cfg.get("exclusions", {}).get("prior_outcome_baseline", False)):
        codes = resolved.dx("outcome_inpatient")
        prevalent = (adapter.match_dx(dx_long, codes)
                     .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
                     .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date")))
                     .select("ENROLID").distinct().withColumn("_prev", F.lit(1)))
        cohort = cohort.join(prevalent, "ENROLID", "left").filter(F.col("_prev").isNull()).drop("_prev")
        logging.info("%sprevalent-outcome exclusion APPLIED (not the reference's rule)", INDENT2)
    else:
        logging.info("%sprevalent-outcome exclusion OFF (as in the reference: prior GI history retained)", INDENT2)

    # --- outcome ascertainment over follow-up ---
    cohort = build_outcome_flag(cohort, dx_long, cfg, resolved, adapter)

    # --- baseline clinical (l_*) flags ---
    cohort = attach_predefined_clinical(adapter, resolved, cohort, cfg, dx_long, pharm)

    dx_long.unpersist()
    return finalize_cohort(cohort, adapter.payer_label, cfg)


# =============================================================================
# negative_control assembly  (flu vaccine vs unvaccinated -> hip fracture) -- LANDMARK design
# =============================================================================
def _landmark_oct1(season_col):
    """Shared per-season landmark index date = Oct 1 of season t (the exposure-window start = baseline
    cutoff). Both arms take this same index date; there are no person-level index dates."""
    return F.to_date(F.concat_ws("-", season_col.cast("string"), F.lit("10"), F.lit("01")))


def _v2_finalize_arms(adapter, resolved, cfg, args, cohort, demo, pharm,
                      age_floor, baseline_days, with_predef) -> DataFrame:
    """Shared negative_control tail for a cohort carrying [ENROLID, season, arm, index_date(=Oct-1 landmark)]:
    attach demo + age, set the landmark windows, apply exclusions (prior hip fx, baseline
    bisphosphonate), ascertain the 820.x outcome over Jan-Apr, and (optionally) the l_* flags."""
    if "DOBYR" not in cohort.columns:
        cohort = cohort.join(demo, "ENROLID", "inner")
    cohort = cohort.withColumn("age_at_index", F.col("season") - F.col("DOBYR")) \
                   .filter(F.col("age_at_index") >= age_floor)
    cohort = (cohort
              .withColumn("followup_start", F.to_date(F.concat_ws("-", (F.col("season") + 1), F.lit("01"), F.lit("01"))))
              .withColumn("followup_end", F.to_date(F.concat_ws("-", (F.col("season") + 1), F.lit("04"), F.lit("30"))))
              .withColumn("followup_days", F.datediff("followup_end", "followup_start"))
              .withColumn("entry_date_embed", F.trunc(F.col("index_date"), "month"))
              .withColumn("baseline_start", F.date_sub(F.col("index_date"), baseline_days)))

    cohort = cohort.localCheckpoint()
    cohort_ids = cohort.select("ENROLID").distinct()
    dx_long = adapter.dx_long_with_setting(person_ids=cohort_ids).localCheckpoint()
    # prior hip fracture (any lookback before the landmark) -> exclude
    hip_codes = resolved.dx("prior_exclusion_dx")
    prior_hip = (adapter.match_dx(dx_long, hip_codes)
                 .join(cohort.select("ENROLID", "index_date"), "ENROLID", "inner")
                 .filter(F.col("dx_date") < F.col("index_date"))
                 .select("ENROLID").distinct().withColumn("_ph", F.lit(1)))
    cohort = cohort.join(prior_hip, "ENROLID", "left").filter(F.col("_ph").isNull()).drop("_ph")

    if cfg["exclusions"]["bisphosphonate_baseline"]["apply"]:
        bis_gen = list(codesets.drug_generics(cfg["cohort_id"], "bisphosphonate"))
        bis_codes = resolved.drug(bis_gen)
        bis = (adapter.drug_dispensings(pharm, bis_codes)
               .join(cohort.select("ENROLID", "index_date", "baseline_start"), "ENROLID", "inner")
               .filter((F.col("svc_date") >= F.col("baseline_start")) & (F.col("svc_date") < F.col("index_date")))
               .select("ENROLID").distinct().withColumn("_bis", F.lit(1)))
        cohort = cohort.join(bis, "ENROLID", "left").filter(F.col("_bis").isNull()).drop("_bis")

    cohort = build_outcome_flag(cohort, dx_long, cfg, resolved, adapter)  # 820.x over Jan-Apr
    if with_predef:
        cohort = attach_predefined_clinical(adapter, resolved, cohort, cfg, dx_long, pharm)
    dx_long.unpersist()
    return cohort


@time_execution
def build_v2(adapter, resolved, cfg, args) -> DataFrame:
    """LANDMARK single-season design (the paper's Example Cohort 3). Per season t both arms share ONE
    landmark index date = Oct 1 of t. arm=1 = flu-vaccine claim in Oct-Dec t; arm=0 = eligible
    enrollee with no such claim. Follow-up = fixed Jan-Apr t+1 for everyone."""
    age_floor = adapter.resolve_age_floor(cfg, args)
    baseline_days = cfg["windows"]["baseline_days"]
    y0 = int(str(cfg["era"]["index_start"])[:4])
    y1 = int(str(cfg["era"]["index_end"])[:4])

    # --- flu-vaccine claims -> (ENROLID, season) vaccinated flags (proc + drug + dx, per adapter) ---
    vax = (adapter.vaccine_events(cfg, resolved)
           .filter(F.col("svc_date").isNotNull() & F.month("svc_date").isin([10, 11, 12]) &
                   (F.year("svc_date") >= y0) & (F.year("svc_date") <= y1)))
    vax_season = vax.select("ENROLID", F.year("svc_date").alias("season")).distinct()

    demo = adapter.demographics(cfg)
    pharm = adapter.load_pharmacy()

    if args.season is None:
        raise ValueError("negative_control is a LANDMARK per-season design: pass --season t.")
    season = int(args.season)

    vax_ids = vax_season.filter(F.col("season") == season).select("ENROLID").distinct()
    _gate = enrollment_gate_mode(cfg)
    if _gate != "off":
        # Every gate mode tests the look-back only (window ends at the Oct-1 landmark); season
        # follow-up loss is handled by person_time censoring, never by exclusion.
        ws = F.date_sub(F.to_date(F.lit(f"{season}-10-01")), baseline_days)
        we = F.to_date(F.lit(f"{season}-10-01"))
        windows = demo.select("ENROLID").withColumn("win_start", ws).withColumn("win_end", we)
        enrolled_ids = adapter.continuous_enrolled(windows, mode=_gate)
        _we_lbl = f"{season}-10-01" if _gate == "baseline" else f"{season + 1}-04-30"
        logging.info("%sV2 eligible pool = enrollment gate (mode=%s) over [%s-%dd, %s]",
                     INDENT2, _gate, f"{season}-10-01", baseline_days, _we_lbl)
    else:
        enrolled_ids = adapter.enrolled_at_landmark([season]).select("ENROLID").distinct()
    elig = enrolled_ids.join(demo, "ENROLID", "inner")
    vaccinated = elig.join(vax_ids, "ENROLID", "inner").withColumn("arm", F.lit(1))
    comparator = elig.join(vax_ids, "ENROLID", "left_anti").withColumn("arm", F.lit(0))
    cohort = (vaccinated.unionByName(comparator)
              .withColumn("season", F.lit(season))
              .withColumn("index_date", F.to_date(F.lit(f"{season}-10-01"))))

    cohort = _v2_finalize_arms(adapter, resolved, cfg, args, cohort, demo, pharm,
                               age_floor, baseline_days, with_predef=True)
    cohort = adapter.add_person_time(cohort)
    return finalize_cohort(cohort, adapter.payer_label, cfg, keep_season=True)


# =============================================================================
# baseline clinical (l_*) flags + finalize
# =============================================================================
def attach_predefined_clinical(adapter, resolved, cohort, cfg, dx_long, pharm) -> DataFrame:
    """Attach the baseline clinical covariate flags (native dx + generic-drug + derived HCRU) used
    by the Table 1 summaries. dx flags via adapter.match_dx in the baseline window; drug flags via
    adapter.drug_dispensings; hcru_intensity via baseline visit count. Columns named l_<flag_name>."""
    pc = codesets.predefined(cfg["cohort_id"])

    cohort = cohort.localCheckpoint()  # fixed leaf so the flag-join loop grows the plan linearly
    win = cohort.select("ENROLID", "index_date", "baseline_start")

    dx_flags = {f: spec["codes"] for f, spec in pc.items() if spec["domain"] == "dx_icd9"}
    for flag in sorted(dx_flags):
        codes = resolved.predefined_dx(dx_flags[flag])
        hits = (adapter.match_dx(dx_long, codes)
                .join(win, "ENROLID", "inner")
                .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date")))
                .select("ENROLID").distinct().withColumn(f"l_{flag}", F.lit(1)))
        cohort = cohort.join(hits, "ENROLID", "left").fillna({f"l_{flag}": 0})

    drug_flags = {f: spec["codes"] for f, spec in pc.items() if spec["domain"] == "drug_generic"}
    for flag in sorted(drug_flags):
        codes = resolved.drug(drug_flags[flag])
        hits = (adapter.drug_dispensings(pharm, codes)
                .join(win, "ENROLID", "inner")
                .filter((F.col("svc_date") >= F.col("baseline_start")) & (F.col("svc_date") < F.col("index_date")))
                .select("ENROLID").distinct().withColumn(f"l_{flag}", F.lit(1)))
        cohort = cohort.join(hits, "ENROLID", "left").fillna({f"l_{flag}": 0})

    hcru = (dx_long.join(win, "ENROLID", "inner")
            .filter((F.col("dx_date") >= F.col("baseline_start")) & (F.col("dx_date") <= F.col("index_date")))
            .groupBy("ENROLID").agg(F.countDistinct("dx_date").alias("l_hcru_intensity")))
    cohort = cohort.join(hcru, "ENROLID", "left").fillna({"l_hcru_intensity": 0})
    return cohort


def cell_label(cfg: dict, args) -> str:
    """The single I/O token keying the downstream tree. negative_control single-season -> '{cell}_s{season}';
    else the cell (payer stratum)."""
    if cfg["cohort_id"] == "negative_control_flu_hipfx" and getattr(args, "season", None) is not None:
        return f"{args.stratum}_s{args.season}"
    return args.stratum


def finalize_cohort(cohort: DataFrame, payer: str, cfg: dict, keep_season: bool = False) -> DataFrame:
    l_cols = [c for c in cohort.columns if c.startswith("l_")]
    # Event timing: for cases, person_time ends at the first outcome date, not at the censoring
    # anchor -- the paper censors "after the occurrence of the study end point, death, or
    # disenrollment".
    if "first_outcome_date" in cohort.columns and "person_time" in cohort.columns:
        cohort = cohort.withColumn(
            "person_time",
            F.when(F.col("outcome") == 1,
                   F.greatest(F.lit(1), F.datediff(F.col("first_outcome_date"), F.col("index_date"))))
             .otherwise(F.col("person_time")))
    keep = (["ENROLID", "arm", "index_date", "entry_date_embed", "outcome", "first_outcome_date",
             "followup_days", "person_time", "age_at_index", "SEX"]
            + (["season"] if keep_season else []) + l_cols)
    keep = [c for c in keep if c in cohort.columns]
    dedup_keys = ["person_id", "season"] if keep_season else ["person_id"]
    out = (cohort.select(*keep)
           .withColumnRenamed("ENROLID", "person_id")
           .withColumnRenamed("SEX", "sex")
           .withColumn("payer", F.lit(payer))
           .withColumn("cohort_id", F.lit(cfg["cohort_id"]))
           .dropDuplicates(dedup_keys))
    return out


# =============================================================================
# per-arm count log (build sanity check)
# =============================================================================
def log_arm_counts(cohort: DataFrame, cell: str):
    logging.info("%s===== FEASIBILITY (%s) =====", INDENT1, cell)
    cohort = cohort.cache()
    tab = cohort.groupBy("arm").agg(
        F.count("*").alias("n"),
        F.sum("outcome").alias("events")).orderBy("arm").collect()
    for r in tab:
        arm = "treatment(1)" if r["arm"] == 1 else "comparator(0)"
        flag = "  <-- UNSTABLE (<10 events)" if (r["events"] or 0) < 10 else ""
        logging.info("%s%-14s n=%s  exposed_events=%s%s", INDENT2, arm, f"{r['n']:,}", r["events"], flag)


def load_config(config_path: str, source: str) -> dict:
    """Load the cohort config."""
    cfg = yaml.safe_load(open(config_path))
    return cfg


def marketscan_era_years(cfg: dict) -> set:
    """The year-file glob filter for the MarketScan raw tables (index era + lookback + follow-up).

    The lower bound must cover the lookback windows, not just the index era: claims from unloaded
    year-files do not exist, so an uncovered washout would pass silently and admit prevalent users
    as new users (index 2011 with a 548d washout consults claims from mid-2009).
    """
    era = cfg.get("era", {})
    y0 = int(str(era.get("index_start", "2008"))[:4])
    y1 = int(str(era.get("index_end", "2014"))[:4])
    win = cfg.get("windows", {}) or {}
    lookback = max(int(win.get("washout_days", 0) or 0), int(win.get("baseline_days", 0) or 0))
    y_lo = y0
    if lookback:
        start = _dt.date.fromisoformat(str(era.get("index_start", "2008-01-01"))[:10])
        y_lo = (start - _dt.timedelta(days=lookback)).year
    years = set(range(min(y_lo, y0), y1 + 1))
    gate_on = enrollment_gate_mode(cfg) != "off"
    if cfg["cohort_id"] == "negative_control_flu_hipfx" or gate_on:
        years.add(y1 + 1)
    if gate_on:
        years.add(y0 - 1)
    return years


# =============================================================================
# main
# =============================================================================
@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("H1 BUILD COHORT STARTING...")
    args = parse_args()

    src = get_source(args.source)
    model = src["data_model"]
    cfg = load_config(args.config, args.source)
    cohort_id = cfg["cohort_id"]
    # default + validate the analysis cell (stratum)
    if args.stratum is None:
        args.stratum = default_stratum(args.source)
    if args.stratum not in src["strata"]:
        logging.warning("stratum '%s' not in source '%s' strata %s -- proceeding anyway",
                        args.stratum, args.source, src["strata"])

    if args.season is not None:
        if cohort_id != "negative_control_flu_hipfx":
            raise ValueError("--season only applies to the negative_control_flu_hipfx cohort")
        seasons = cfg.get("season", {}).get("seasons")
        if seasons and args.season not in seasons:
            logging.warning("season %s not in config season.seasons %s -- building it anyway",
                            args.season, seasons)
    logging.info("Config: %s (cohort=%s, source=%s, model=%s, cell=%s%s)",
                 args.config, cohort_id, args.source, model, args.stratum,
                 f", season={args.season}" if args.season is not None else "")

    # ---- MarketScan raw roots + era-year file filter ----
    era_years = None
    raw_roots = None
    if model == "marketscan":
        if not args.raw_root:
            raise ValueError("--raw_root is required for the marketscan source")
        era_years = marketscan_era_years(cfg)
        logging.info("Era-year file filter: keeping raw files for years %d-%d",
                     min(era_years), max(era_years))
        # Full claims tree: payer dirs sit directly under raw_root; the tokenizability gate is mandatory.
        raw_roots = [args.raw_root]
        if not args.restrict_ids:
            raise ValueError(
                "--restrict_ids is required: the full claims tree contains patients "
                "the tokenizer excluded (~31% of MDCR), and without the gate the embedding arm "
                "would silently back-fill rep_*=0 for them. Point it at the tokenized population, "
                "e.g. <MS_PROCESSED_ROOT>/<DATA_VERSION>/<MS_RUN_NAME>/processed_data/trajectory/*.parquet")
        logging.info("Reading the full claims tree at %s", args.raw_root)

    spark = create_spark_session(f"h1_{cohort_id}_{args.source}_{args.stratum}",
                                 profile=src.get("spark_profile", "bigmem"))
    try:
        adapter = make_adapter(args.source, spark, stratum=args.stratum, era_years=era_years,
                               raw_roots=raw_roots, codes_resolved_root=args.codes_resolved, cfg=cfg,
                               restrict_ids=args.restrict_ids)
        resolved = load_resolved(args.codes_resolved, args.source, cohort_id)

        if cohort_id == "positive_control_celecoxib_gi":
            cohort = build_v1(adapter, resolved, cfg, args)
        elif cohort_id == "negative_control_flu_hipfx":
            cohort = build_v2(adapter, resolved, cfg, args)
        else:
            raise ValueError(f"Unknown cohort_id {cohort_id}")

        cohort = cohort.cache()
        log_arm_counts(cohort, args.stratum)

        out_dir = os.path.join(args.out_root, cohort_id, cell_label(cfg, args))
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "cohort.csv")
        logging.info("%sWriting cohort (%s rows) -> %s", INDENT1, f"{cohort.count():,}", out_csv)
        cohort.toPandas().to_csv(out_csv, index=False)
        logging.info("H1 BUILD COHORT COMPLETED. Log: %s", log_file)
    except Exception as e:
        logging.error("build_cohort failed: %s", str(e))
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
