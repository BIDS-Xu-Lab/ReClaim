#!/usr/bin/env python3
"""
marketscan_io.py  --  MarketScan raw-claims reader for the TTE study.

One payer stratum (CCAE / MDCR / MDCD, or their POOL union tagged by source_db) as
normalized event streams: demographics behind the drug-coverage gate, disenrollment date,
pharmacy dispensings (read once, persisted DISK_ONLY), and diagnosis / procedure long
extracts with position and setting. Stray ICD-9 rows (DXVER=='9') are dropped.
"""

import os
import re
import logging
from glob import glob
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.window import Window

INDENT2 = "    "
INDENT3 = "      "

# POOL = CCAE + MDCR + MDCD (the reference is an all-payer study). ENROLID spaces are
# disjoint per database, so ids are NAMESPACED at read time (ID_PREFIX, as the token store
# does); otherwise a Medicaid ENROLID could collide with a commercial one.
COMPONENT_PAYERS = {"POOL": ["CCAE", "MDCR", "MDCD"]}
ID_PREFIX = {"CCAE": "CR", "MDCR": "CR", "MDCD": "MD"}


def _file_year(path):
    m = re.search(r"_(\d{4})[_.]", os.path.basename(path))
    return int(m.group(1)) if m else None


def strip_decimal(col):
    return F.regexp_replace(col, r"\.", "")


def create_spark_session(app_name: str) -> SparkSession:
    """Local-mode Spark sized for the held-out claims subset on a large-memory node."""
    mem = os.environ.get("SPARK_DRIVER_MEM", "360g")
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.driver.memory", mem)
            .config("spark.driver.maxResultSize", "32g")
            .config("spark.driver.extraJavaOptions",
                    "-XX:+UseG1GC -XX:MaxGCPauseMillis=20000 -XX:InitiatingHeapOccupancyPercent=35 "
                    "-XX:+UseStringDeduplication")
            .config("spark.network.timeout", "1800s")
            .config("spark.executor.heartbeatInterval", "60s")
            .config("spark.sql.shuffle.partitions", "240")
            .config("spark.sql.files.maxPartitionBytes", "64m")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())


class MarketScanIO:
    """One MarketScan payer stratum (CCAE/MDCR/MDCD/POOL) as normalized event streams."""

    def __init__(self, spark: SparkSession, *, raw_roots: List[str], payer: str = "POOL",
                 era_years=None):
        self.spark = spark
        self.raw_roots = raw_roots
        self.payer = payer
        self.era_years = era_years
        self._pharm = None

    # ---- raw-file discovery ----
    def get_file_paths(self, payer: str, table: str) -> List[str]:
        files = []
        for base in self.raw_roots:
            if payer == "MDCD":
                pattern = f"Medicaid_{table}*.snappy.parquet"
            else:
                pattern = f"{payer}_{table.upper()}*.snappy.parquet"
            files += glob(os.path.join(base, payer, pattern))
        if self.era_years is not None:
            files = [f for f in files if (_file_year(f) is None or _file_year(f) in self.era_years)]
        files = sorted(files)
        logging.info("%s[%s-%s] %d files", INDENT2, payer, table, len(files))
        return files

    def read_table(self, payer: str, table: str, cols: List[str]) -> DataFrame:
        if payer in COMPONENT_PAYERS:
            parts = []
            for comp in COMPONENT_PAYERS[payer]:
                part = self.read_table(comp, table, cols).withColumn("source_db", F.lit(comp))
                # namespace the person key at read time: collision-safe joins, and the key IS
                # the token-store key
                if "ENROLID" in part.columns:
                    part = part.withColumn("ENROLID", F.concat(F.lit(ID_PREFIX[comp]),
                                                               F.col("ENROLID").cast("string")))
                parts.append(part)
            out = parts[0]
            for p in parts[1:]:
                out = out.unionByName(p, allowMissingColumns=True)
            return out
        paths = self.get_file_paths(payer, table)
        if not paths:
            raise FileNotFoundError(f"No files for {payer}/{table} under {self.raw_roots}")
        df = self.spark.read.parquet(*paths)
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logging.info("%s[%s-%s] missing cols (skipped): %s", INDENT3, payer, table, missing)
        return df.select(*present)

    # ---- demographics + drug-coverage gate ----
    def demographics(self) -> DataFrame:
        """First-record DOBYR/SEX/REGION per ENROLID, restricted to people whose EVERY plan
        record carries drug coverage."""
        cols = ["ENROLID", "DTSTART", "DOBYR", "SEX", "REGION", "DRUGCOVG", "RX", "MEDICARE"]
        t = self.read_table(self.payer, "t", cols)
        # Drug coverage is a different column per database (RX on CCAE/MDCR, DRUGCOVG on
        # Medicaid), so the gate is evaluated PER SOURCE; a single RX rule would silently drop
        # every Medicaid enrollee.
        has_src = "source_db" in t.columns
        rx_ok = (F.col("RX").cast("int") == 1) if "RX" in t.columns else F.lit(None).cast("boolean")
        dc_ok = (F.col("DRUGCOVG").cast("int") == 1) if "DRUGCOVG" in t.columns else F.lit(None).cast("boolean")
        if has_src:
            covered = F.when(F.col("source_db") == "MDCD", dc_ok).otherwise(rx_ok)
            gate_desc = "RX==1 (CCAE/MDCR) / DRUGCOVG==1 (MDCD)"
        else:
            covered = dc_ok if self.payer == "MDCD" else rx_ok
            gate_desc = "DRUGCOVG==1" if self.payer == "MDCD" else "RX==1"
        no_rx_ids = t.filter(~F.coalesce(covered, F.lit(False))).select("ENROLID").distinct()
        t = t.join(no_rx_ids, "ENROLID", "left_anti")
        logging.info("%sdrug-coverage gate: %s on ALL plan records", INDENT2, gate_desc)
        # Medicaid/Medicare dual eligibles are excluded from the Medicaid arm, as in the token
        # store: their claims are split across MDCD and MDCR, and they have no token sequence.
        if "MEDICARE" in t.columns:
            dual_src = (F.col("source_db") == "MDCD") if has_src else F.lit(self.payer == "MDCD")
            dual_ids = (t.filter(dual_src & (F.col("MEDICARE").cast("int") == 1))
                        .select("ENROLID").distinct())
            t = t.join(dual_ids, "ENROLID", "left_anti")
            logging.info("%sMedicaid dual-eligible (MEDICARE==1) exclusion applied", INDENT2)
        w = Window.partitionBy("ENROLID").orderBy("DTSTART", "DOBYR")
        keep = ["ENROLID", "DOBYR", "SEX"] + (["REGION"] if "REGION" in t.columns else [])
        return (t.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
                .select(*keep))

    # ---- censoring anchor ----
    def disenrollment(self) -> DataFrame:
        """df[ENROLID, disenroll_date] = max(DTEND), the censoring anchor (claims carry no death
        date, so disenrollment absorbs death)."""
        t = self.read_table(self.payer, "t", ["ENROLID", "DTEND"])
        return t.groupBy("ENROLID").agg(F.max("DTEND").alias("disenroll_date"))

    # ---- pharmacy ----
    def load_pharmacy(self) -> DataFrame:
        """Pharmacy dispensings read ONCE, persisted DISK_ONLY (ENROLID, svc_date, NDCNUM)."""
        if self._pharm is None:
            df = (self.read_table(self.payer, "d", ["ENROLID", "SVCDATE", "NDCNUM"])
                  .withColumn("NDCNUM", F.col("NDCNUM").cast("string")))
            keep = ["ENROLID", F.col("SVCDATE").alias("svc_date"), "NDCNUM"] + \
                   (["source_db"] if "source_db" in df.columns else [])
            self._pharm = df.select(*keep).persist(StorageLevel.DISK_ONLY)
        return self._pharm

    def drug_dispensings(self, codeset) -> DataFrame:
        """Dispensings whose NDC is in `codeset` -> df[ENROLID, svc_date(, source_db)]."""
        pharm = self.load_pharmacy()
        keep = ["ENROLID", "svc_date"] + (["source_db"] if "source_db" in pharm.columns else [])
        ndcs = sorted({str(x) for x in (codeset or []) if str(x).strip()})
        if not ndcs:
            return pharm.select(*keep).limit(0)
        ndc_df = self.spark.createDataFrame([(x,) for x in ndcs], ["NDCNUM"]).dropDuplicates()
        return pharm.join(F.broadcast(ndc_df), "NDCNUM", "inner").select(*keep)

    def drug_dispensings_map(self, map_df: DataFrame) -> DataFrame:
        """Dispensings joined to a (NDCNUM, set_id) map -> df[ENROLID, svc_date, set_id]."""
        pharm = self.load_pharmacy()
        return (pharm.join(F.broadcast(map_df), "NDCNUM", "inner")
                .select("ENROLID", "svc_date", "set_id"))

    # ---- diagnosis / procedure long extracts ----
    def dx_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """All diagnosis occurrences: df[ENROLID, dx_date, code, position, setting].
        DXVER=='9' rows (stray ICD-9 in the 2018+ files) are dropped; null DXVER kept."""
        payer = self.payer
        i_cols = ["ENROLID", "ADMDATE", "DXVER"] + [f"DX{k}" for k in range(1, 16)]
        i = self.read_table(payer, "i", i_cols)
        o_cols = ["ENROLID", "SVCDATE", "DXVER"] + [f"DX{k}" for k in range(1, 5)]
        o = self.read_table(payer, "o", o_cols)
        for name, df in (("i", i), ("o", o)):
            if "DXVER" in df.columns:
                df = df.filter(F.col("DXVER").isNull() | (F.col("DXVER") != "9"))
            if name == "i":
                i = df
            else:
                o = df
        if person_ids is not None:
            i = i.join(F.broadcast(person_ids), "ENROLID", "left_semi").cache()
            o = o.join(F.broadcast(person_ids), "ENROLID", "left_semi").cache()

        dxcols = [c for c in i.columns if c.startswith("DX") and c != "DXVER"]
        i_long = None
        for pos, c in enumerate(dxcols, start=1):
            part = (i.select(F.col("ENROLID"), F.col("ADMDATE").alias("dx_date"),
                             F.col(c).cast("string").alias("code"), F.lit(pos).alias("position"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("inpatient")))
            i_long = part if i_long is None else i_long.unionByName(part)

        odxcols = [c for c in o.columns if c.startswith("DX") and c != "DXVER"]
        o_long = None
        for pos, c in enumerate(odxcols, start=1):
            part = (o.select(F.col("ENROLID"), F.col("SVCDATE").alias("dx_date"),
                             F.col(c).cast("string").alias("code"), F.lit(pos).alias("position"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("outpatient")))
            o_long = part if o_long is None else o_long.unionByName(part)

        return i_long.unionByName(o_long)

    def proc_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """All procedure occurrences with setting: df[ENROLID, proc_date, code, setting]."""
        payer = self.payer
        i_cols = ["ENROLID", "ADMDATE"] + [f"PROC{k}" for k in range(1, 16)]
        i = self.read_table(payer, "i", i_cols)
        o_cols = ["ENROLID", "SVCDATE", "PROC1"]
        o = self.read_table(payer, "o", o_cols)
        if person_ids is not None:
            i = i.join(F.broadcast(person_ids), "ENROLID", "left_semi")
            o = o.join(F.broadcast(person_ids), "ENROLID", "left_semi")
        pcols = [c for c in i.columns if c.startswith("PROC")]
        i_long = None
        for c in pcols:
            part = (i.select(F.col("ENROLID"), F.col("ADMDATE").alias("proc_date"),
                             F.col(c).cast("string").alias("code"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("inpatient")))
            i_long = part if i_long is None else i_long.unionByName(part)
        o_long = (o.select(F.col("ENROLID"), F.col("SVCDATE").alias("proc_date"),
                           F.col("PROC1").cast("string").alias("code"))
                  .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                  .withColumn("setting", F.lit("outpatient")))
        return i_long.unionByName(o_long)

    def drug_long(self, vocab_ndc: str) -> DataFrame:
        """Every pharmacy dispensing -> (ENROLID, svc_date, event_code=ingredient name) via the
        NDC->rxnorm_ingredient_name vocab (hdPS drug domain)."""
        d = self.load_pharmacy()
        m = (self.spark.read.csv(vocab_ndc, header=True, inferSchema=True)
             .select(F.col("ndc_drug_code").cast("string").alias("NDCNUM"),
                     F.lower(F.col("rxnorm_ingredient_name")).alias("event_code"))
             .dropDuplicates())
        return (d.join(m, "NDCNUM", "inner")
                .filter(F.col("event_code").isNotNull() & (F.col("event_code") != ""))
                .select("ENROLID", "svc_date", "event_code"))


def match_dx_prefix_sets(dx_long: DataFrame, sets: dict, spark) -> DataFrame:
    """Match dx rows against many named prefix code-sets at once: `sets` = {set_id: [prefix]}
    -> df[ENROLID, dx_date, set_id], one row per matching occurrence. Implemented as
    length-bucketed equijoins (substring(code, 1, L) == key), which scales to the 66k-code
    cluster dictionary where a regex cannot."""
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
    dx = dx_long.withColumn("code_norm", F.upper(strip_decimal(F.col("code"))))
    out = None
    for L in lengths:
        kd = key_df.filter(F.col("klen") == L).select("set_id", F.col("match_key"))
        part = (dx.withColumn("match_key", F.substring("code_norm", 1, L))
                .join(F.broadcast(kd), "match_key", "inner")
                .select("ENROLID", "dx_date", "set_id"))
        out = part if out is None else out.unionByName(part)
    return out
