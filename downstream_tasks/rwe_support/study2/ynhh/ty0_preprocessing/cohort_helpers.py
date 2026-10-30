#!/usr/bin/env python3
"""
cohort_helpers.py -- helpers shared by the ty0 preprocessing scripts (build_cohort.py,
build_long_table.py): config overlay, Spark session, the two dx matching routes and the
source-value code sets. Only the names those scripts import live here.
"""

import os
import logging
import re

import yaml
from pyspark.sql import SparkSession, functions as F

import codesets_tte as cs
from datasources import overlay_path


def sanitize(name):
    # MUST stay identical to resolve_codes.sanitize: set ids from both routes must agree.
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")


def match_dx_prefix_sets(dx_long, sets, spark):
    """Match dx rows against many named ICD-10 prefix code-sets at once (the dx_route=source
    matcher). `sets` = {set_id: [match_key, ...]} with decimal-stripped UPPERCASE prefixes.
    Returns df[ENROLID, dx_date, set_id]. Length-bucketed equijoins (substring(code, 1, L) ==
    key per distinct key length L) scale to the 66k-code cluster dictionary where a regex
    cannot. Source values are normalized here: uppercase, dots/whitespace stripped (EHR
    exports carry dotted codes like 'E11.9')."""
    rows = []
    for sid, keys in sets.items():
        for k in keys:
            k = str(k).strip().replace(".", "").upper()
            if k:
                rows.append((sid, k, len(k)))
    if not rows:
        raise ValueError("match_dx_prefix_sets: empty code sets")
    key_df = spark.createDataFrame(rows, ["set_id", "match_key", "klen"]).dropDuplicates()
    lengths = sorted({r[2] for r in rows})
    dx = dx_long.withColumn("code_norm",
                            F.upper(F.regexp_replace(F.col("code"), r"[.\s]", "")))
    out = None
    for L in lengths:
        kd = key_df.filter(F.col("klen") == L).select("set_id", F.col("match_key"))
        part = (dx.withColumn("match_key", F.substring("code_norm", 1, L))
                .join(F.broadcast(kd), "match_key", "inner")
                .select("ENROLID", "dx_date", "set_id"))
        out = part if out is None else out.unionByName(part)
    return out


DM_SEVERITY_IDS = ["COV_dm_nephropathy", "COV_dm_retinopathy", "COV_dm_neuropathy",
                   "COV_dm_other_specified_complications", "COV_dm_unspecified_complications"]


def build_source_sets():
    """All named dx sets as raw ICD-10 match-key lists straight from codesets_tte (the
    dx_route=source universe). Set ids are IDENTICAL to resolve_codes's resolved-map ids
    (same sanitize), so the cohort column contract is route-invariant."""
    src = {}
    for name, keys in cs.outcome_sets().items():
        src[f"OUT_{sanitize(name)}"] = keys
    for name, keys in cs.nco_sets().items():
        src[f"NCO_{sanitize(name)}"] = keys
    for name, keys in cs.covariate_exclusion_sets().items():
        src[name] = keys                                  # T2D / T1D / ESRD
    for code, cl in cs.cluster_master():
        src.setdefault(f"COV_{sanitize(cl)}", []).append(code)
    # Diabetes-severity covariates: reference Table S5 covariates with no cluster
    # counterpart, defined on the ICD-10 E11 tree.
    src.update({
        "COV_dm_nephropathy":                    ["E11.2"],
        "COV_dm_retinopathy":                    ["E11.3"],
        "COV_dm_neuropathy":                     ["E11.4"],
        "COV_dm_other_specified_complications":  ["E11.6"],
        "COV_dm_unspecified_complications":      ["E11.8"],
    })
    return src


def load_config(path, source):
    cfg = yaml.safe_load(open(path))
    ov = overlay_path(source)
    if ov and os.path.exists(ov):
        over = yaml.safe_load(open(ov)) or {}
        def merge(base, o):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        merge(cfg, over.get(cfg.get("cohort_id"), over) if isinstance(over, dict) else over)
        logging.info("Applied config overlay for '%s': %s", source, ov)
    elif ov:
        # A silently-missing overlay would revert era/study_end to the base yaml with no trace.
        logging.warning("CONFIG OVERLAY MISSING for source '%s': %s -- running on the BASE "
                        "config. If this source needs its overlay (ynhh does), STOP and "
                        "restore the file.", source, ov)
    return cfg


def create_spark_session(app_name, profile="bigmem"):
    b = SparkSession.builder.appName(app_name).master("local[*]")
    if profile == "small":
        return (b.config("spark.driver.memory", "16g")
                .config("spark.sql.shuffle.partitions", "64")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .getOrCreate())
    return (b.config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEM", "300g"))
            .config("spark.driver.maxResultSize", "32g")
            .config("spark.driver.extraJavaOptions",
                    "-XX:+UseG1GC -XX:MaxGCPauseMillis=20000 -XX:InitiatingHeapOccupancyPercent=35")
            .config("spark.network.timeout", "1800s")
            .config("spark.sql.shuffle.partitions", "240")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())


def match_sets(dx_long, map_pdf, spark):
    """dx rows joined to a (set_id, concept_id) pandas map -> df[ENROLID, dx_date, set_id]
    (the dx_route=standard matcher: concept membership instead of ICD prefixes)."""
    m = spark.createDataFrame(map_pdf.rename(columns={"concept_id": "code"})[["code", "set_id"]]
                              .drop_duplicates())
    return dx_long.join(F.broadcast(m), "code", "inner").select("ENROLID", "dx_date", "set_id")
