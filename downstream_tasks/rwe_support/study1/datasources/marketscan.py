#!/usr/bin/env python3
"""
datasources/marketscan.py  --  MarketScanAdapter.

The MarketScan raw-claims reader layer behind the DataSourceAdapter interface. MarketScan
specifics live here and only here: table codes t/i/o/d, columns ENROLID/DTSTART/DTEND/MEMDAYS/
DOBYR/SEX/MEDICARE/DRUGCOVG/RX/SVCDATE/ADMDATE/NDCNUM/DX*/PROC*, payer strata, the year-file glob
filter (era_years), decimal-stripped ICD-9 PREFIX matching, and HCPCS/CPT vaccine proc matching.
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

import codesets
from datasources.base import DataSourceAdapter

INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

def _file_year(path):
    """Extract the 4-digit year from a raw-file name (MDCR_D_2011_v30... / Medicaid_d_2011...)."""
    m = re.search(r"_(\d{4})[_.]", os.path.basename(path))
    return int(m.group(1)) if m else None


def strip_decimal(col):
    return F.regexp_replace(col, r"\.", "")


class MarketScanAdapter(DataSourceAdapter):
    """One MarketScan payer stratum (MDCR) as normalized event streams."""

    source_name = "marketscan"
    data_model = "marketscan"

    def __init__(self, spark: SparkSession, *, raw_roots: List[str], payer: str,
                 era_years=None, codes_resolved_root: str = "intermediate/codes_resolved",
                 vocab_ndc: str = None, restrict_ids: str = None):
        self.spark = spark
        self.raw_roots = raw_roots
        self.payer = payer
        self.era_years = era_years
        self.codes_resolved_root = codes_resolved_root
        self.vocab_ndc = vocab_ndc   # build_long_table drug-domain token map (NDC -> rxnorm_ingredient_name)
        # Optional tokenizability gate (parquet glob of the tokenized population): restricts the
        # patient universe to people who HAVE a token sequence, so the embedding arm can never
        # silently back-fill rep_*=0 for a patient the tokenizer excluded. Applied in
        # demographics(); the drop is logged.
        self.restrict_ids = restrict_ids
        self._restrict_df = None

    # ---- capabilities ----
    @property
    def dx_position_available(self) -> bool:
        return True

    def resolve_age_floor(self, cfg, args) -> int:
        """Age floor for this run: --age_floor_override if set, else the per-stratum config value,
        else the config default."""
        payer = self.payer
        if getattr(args, "age_floor_override", None):
            return args.age_floor_override
        for s in cfg.get("strata", []):
            if s.get("id") == payer and s.get("age_floor") is not None:
                return int(s["age_floor"])
        return cfg["age_floor"]

    def _exclude_dual(self, cfg) -> bool:
        """True if this stratum keeps only Medicaid-primary enrollees (drop MEDICARE==1 duals),
        per the matching config strata entry's `exclude_dual_eligible` flag."""
        return any(s.get("id") == self.payer and s.get("exclude_dual_eligible")
                   for s in cfg.get("strata", []))

    # ---- raw-file discovery ----
    def get_file_paths(self, payer: str, table: str) -> List[str]:
        """Glob parquet files for one payer/table across all raw roots, restricted to self.era_years
        (cohort era + look-back/follow-up windows) when set."""
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
        """Read one payer/table selecting only the columns that exist (schemas vary by payer)."""
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
    def demographics(self, cfg) -> DataFrame:
        """First-record DOBYR/SEX per ENROLID, restricted to people whose EVERY enrollment plan record
        carries drug coverage (DRUGCOVG/RX == 1), so exposure/covariate drug capture is complete.
        No enrollment-span gate here."""
        exclude_dual = self._exclude_dual(cfg)
        payer = self.payer
        cols = ["ENROLID", "DTSTART", "DOBYR", "SEX", "MEDICARE", "DRUGCOVG", "RX"]
        t = self.read_table(payer, "t", cols)
        cov_cols = [c for c in ("DRUGCOVG", "RX") if c in t.columns]
        if cov_cols:
            not_covered = F.lit(False)
            for c in cov_cols:
                not_covered = not_covered | F.col(c).isNull() | (F.col(c).cast("int") != 1)
            no_rx_ids = t.filter(not_covered).select("ENROLID").distinct()
            t = t.join(no_rx_ids, "ENROLID", "left_anti")
            logging.info("%sdrug-coverage gate: require %s == 1 on ALL plan records", INDENT2, cov_cols)
        else:
            logging.warning("%sno DRUGCOVG/RX column in %s enrollment -- drug-coverage gate SKIPPED", INDENT2, payer)

        if exclude_dual:
            if "MEDICARE" in t.columns:
                dual_ids = t.filter(F.col("MEDICARE").cast("int") == 1).select("ENROLID").distinct()
                t = t.join(dual_ids, "ENROLID", "left_anti")
                logging.info("%sdual-eligible exclusion: dropped MEDICARE==1 (Medicaid-primary only)", INDENT2)
            else:
                logging.warning("%sexclude_dual requested but no MEDICARE column in %s -- SKIPPED", INDENT2, payer)

        t = self._apply_restrict_ids(t)

        w = Window.partitionBy("ENROLID").orderBy("DTSTART", "DOBYR")
        return (t.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
                .select("ENROLID", "DOBYR", "SEX",
                        *(["MEDICARE"] if "MEDICARE" in t.columns else [])))

    def _restrict_id_df(self) -> DataFrame:
        """df[ENROLID] of the tokenized population (cached): strip the alpha prefix from the token
        store's enrollee_id ('CR' + ENROLID) and cast back to the raw ENROLID type."""
        if self._restrict_df is None:
            files = sorted(f for f in glob(self.restrict_ids)
                           if not os.path.basename(f).startswith("._"))
            if not files:
                raise FileNotFoundError(f"--restrict_ids matched no parquet: {self.restrict_ids}")
            df = self.spark.read.parquet(*files)
            idcol = next((c for c in df.columns if c.lower() in ("enrollee_id", "enrolid")), None)
            if idcol is None:
                raise ValueError(f"--restrict_ids parquet has no enrollee_id/ENROLID column: {df.columns}")
            self._restrict_df = (df.select(
                F.regexp_extract(F.col(idcol).cast("string"), r"^[A-Za-z]*0*(\d+)$", 1)
                 .cast("long").alias("ENROLID"))
                .filter(F.col("ENROLID").isNotNull()).distinct().persist())
            logging.info("%sReClaim-compatibility gate: %d tokenized ENROLID(s) from %s",
                         INDENT2, self._restrict_df.count(), self.restrict_ids)
        return self._restrict_df

    def _apply_restrict_ids(self, t: DataFrame) -> DataFrame:
        """Restrict the enrollment table to tokenized people, logging the drop (never silent)."""
        if not self.restrict_ids:
            return t
        before = t.select("ENROLID").distinct().count()
        # NOT broadcast: the tokenized population is ~120M ids, far past a broadcast-join budget.
        out = t.join(self._restrict_id_df(), "ENROLID", "left_semi")
        after = out.select("ENROLID").distinct().count()
        logging.info("%sReClaim-compatibility filter: %s -> %s enrollee(s)  (dropped %s = %.1f%%)",
                     INDENT2, f"{before:,}", f"{after:,}", f"{before - after:,}",
                     100.0 * (before - after) / before if before else 0.0)
        if after == 0:
            raise ValueError("ReClaim-compatibility filter removed every enrollee -- check --restrict_ids "
                             "(ENROLID prefix/type mismatch?)")
        return out

    # ---- enrollment gate helpers (MarketScan t-table: one row per person-month) ----
    def continuous_enrolled(self, windows_df: DataFrame) -> DataFrame:
        """COVERAGE test (gate mode 'span'): ENROLIDs whose enrollment brackets [win_start, win_end].
        min(DTSTART) <= win_start AND max(DTEND) >= win_end; interior gaps allowed. The ACTIVITY
        construct is claims_footprint(), used by gate mode 'baseline'."""
        t = self.read_table(self.payer, "t", ["ENROLID", "DTSTART", "DTEND"])
        agg = (t.join(windows_df.select("ENROLID", "win_start", "win_end"), "ENROLID", "inner")
               .groupBy("ENROLID")
               .agg(F.min("DTSTART").alias("min_start"), F.max("DTEND").alias("max_end"),
                    F.first("win_start").alias("ws"), F.first("win_end").alias("we")))
        return agg.filter((F.col("min_start") <= F.col("ws")) &
                          (F.col("max_end") >= F.col("we"))).select("ENROLID")


    def claims_footprint(self) -> DataFrame:
        """Per-ENROLID [first_event, last_event] = min/max claim date across i (dx), o (dx), d (pharmacy):
        an observability proxy from actual claim activity rather than the enrollment_detail (t) table."""
        dx = self.dx_long_with_setting().select("ENROLID", F.col("dx_date").alias("ev"))
        dph = self.load_pharmacy().select("ENROLID", F.col("svc_date").alias("ev"))
        ev = dx.unionByName(dph).filter(F.col("ev").isNotNull())
        return ev.groupBy("ENROLID").agg(F.min("ev").alias("first_event"), F.max("ev").alias("last_event"))

    def apply_continuous_gate(self, cfg, cohort, win_start_col, win_end_col, label):
        """Filter `cohort` to observable members. Every mode tests the LOOK-BACK [win_start_col,
        index_date]; follow-up disenrollment is censored via person_time, never excluded.
          baseline -> ACTIVITY: claim footprint (i/o/d) brackets the look-back.
          span     -> COVERAGE: enrollment detail (`t`) brackets the look-back.
        """
        from datasources.base import enrollment_gate_mode
        mode = enrollment_gate_mode(cfg)
        if mode == "off":
            return cohort
        end_col = "index_date"          # every mode is look-back only
        if mode == "baseline":          # ACTIVITY footprint
            fp = self.claims_footprint()
            ok = (cohort.select("ENROLID", F.col(win_start_col).alias("ws"), F.col(end_col).alias("we"))
                  .join(fp, "ENROLID", "inner")
                  .filter((F.col("first_event") <= F.col("ws")) & (F.col("last_event") >= F.col("we")))
                  .select("ENROLID"))
            logging.info("%senrollment gate (%s, mode=baseline: ACTIVITY -- claim footprint brackets "
                         "[%s, %s])", INDENT2, label, win_start_col, end_col)
            return cohort.join(ok, "ENROLID", "inner")
        # COVERAGE mode: span, from the enrollment detail table
        windows = cohort.select("ENROLID",
                                F.col(win_start_col).alias("win_start"),
                                F.col(end_col).alias("win_end"))
        ok = self.continuous_enrolled(windows)
        logging.info("%senrollment gate (%s, mode=%s: COVERAGE -- enrollment table brackets [%s, %s])",
                     INDENT2, label, mode, win_start_col, end_col)
        return cohort.join(ok, "ENROLID", "inner")

    def add_person_time(self, cohort: DataFrame) -> DataFrame:
        """Add `person_time` (follow-up days) = index_date -> censoring, where censoring =
        min(followup_end, last covered day). disenroll_date = max(DTEND); floored at 1 day."""
        t = self.read_table(self.payer, "t", ["ENROLID", "DTEND"])
        disenroll = t.groupBy("ENROLID").agg(F.max("DTEND").alias("disenroll_date"))
        cens = F.least(F.col("followup_end"),
                       F.coalesce(F.col("disenroll_date"), F.col("followup_end")))
        return (cohort.join(disenroll, "ENROLID", "left")
                .withColumn("person_time", F.greatest(F.lit(1), F.datediff(cens, F.col("index_date"))))
                .drop("disenroll_date"))

    def enrolled_at_landmark(self, seasons) -> DataFrame:
        """df[ENROLID, season] for enrollees whose enrollment record covers that season's Oct-1 landmark.
        Presence = a `t`-table record with DTSTART <= Oct1(season) < DTSTART + MEMDAYS."""
        t = self.read_table(self.payer, "t", ["ENROLID", "DTSTART", "MEMDAYS"])
        end = F.date_add(F.col("DTSTART"), F.col("MEMDAYS").cast("int"))
        out = None
        for s in seasons:
            lm = F.to_date(F.lit(f"{s}-10-01"))
            part = (t.filter((F.col("DTSTART") <= lm) & (end > lm))
                    .select("ENROLID").distinct().withColumn("season", F.lit(int(s))))
            out = part if out is None else out.unionByName(part)
        return out

    # ---- pharmacy (drug-event stream) ----
    def load_pharmacy(self) -> DataFrame:
        """Read the pharmacy dispensings table once and persist DISK_ONLY (ENROLID, svc_date, NDCNUM);
        the cohort build filters it against several NDC sets. Off-heap keeps GC pauses short."""
        df = (self.read_table(self.payer, "d", ["ENROLID", "SVCDATE", "NDCNUM"])
              .withColumn("NDCNUM", F.col("NDCNUM").cast("string")))
        keep = ["ENROLID", F.col("SVCDATE").alias("svc_date"), "NDCNUM"] + \
               (["source_db"] if "source_db" in df.columns else [])
        return df.select(*keep).persist(StorageLevel.DISK_ONLY)

    def drug_dispensings(self, pharm: DataFrame, codeset) -> DataFrame:
        """Filter the (cached) pharmacy df to an NDC set -> df[ENROLID, svc_date(, source_db)].
        `codeset` is a list of NDC strings (from the resolver); broadcast join on NDCNUM."""
        keep = ["ENROLID", "svc_date"] + (["source_db"] if "source_db" in pharm.columns else [])
        ndcs = sorted({str(x) for x in (codeset or []) if str(x).strip()})
        if not ndcs:
            # empty NDC set -> no dispensings (schema-preserving)
            return pharm.select(*keep).limit(0)
        ndc_df = self.spark.createDataFrame([(x,) for x in ndcs], ["NDCNUM"]).dropDuplicates()
        return pharm.join(F.broadcast(ndc_df), "NDCNUM", "inner").select(*keep)

    # ---- diagnosis / procedure long extract with SETTING ----
    def dx_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """All diagnosis occurrences: df[ENROLID, dx_date, code, position, setting], setting from the
        i (inpatient) vs o (outpatient) table lineage. `person_ids` prunes the raw tables before the
        DX-column explode."""
        payer = self.payer
        i_cols = ["ENROLID", "ADMDATE", "DXVER"] + [f"DX{k}" for k in range(1, 16)]
        i = self.read_table(payer, "i", i_cols)
        o_cols = ["ENROLID", "SVCDATE", "DXVER"] + [f"DX{k}" for k in range(1, 5)]
        o = self.read_table(payer, "o", o_cols)
        if person_ids is not None:
            i = i.join(F.broadcast(person_ids), "ENROLID", "left_semi").cache()
            o = o.join(F.broadcast(person_ids), "ENROLID", "left_semi").cache()

        dxcols = [c for c in i.columns if c.startswith("DX")]
        i_long = None
        for pos, c in enumerate(dxcols, start=1):
            part = (i.select(F.col("ENROLID"), F.col("ADMDATE").alias("dx_date"),
                             F.col(c).cast("string").alias("code"), F.lit(pos).alias("position"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("inpatient")))
            i_long = part if i_long is None else i_long.unionByName(part)

        odxcols = [c for c in o.columns if c.startswith("DX")]
        o_long = None
        for pos, c in enumerate(odxcols, start=1):
            part = (o.select(F.col("ENROLID"), F.col("SVCDATE").alias("dx_date"),
                             F.col(c).cast("string").alias("code"), F.lit(pos).alias("position"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("outpatient")))
            o_long = part if o_long is None else o_long.unionByName(part)

        return i_long.unionByName(o_long)

    def proc_long_with_setting(self) -> DataFrame:
        """All procedure occurrences with setting. Inpatient PROC1..PROC15 (ICD-9 vol.3) -> inpatient;
        outpatient PROC1 (CPT/HCPCS) -> outpatient. df[ENROLID, proc_date, code, setting]."""
        payer = self.payer
        i_cols = ["ENROLID", "ADMDATE"] + [f"PROC{k}" for k in range(1, 16)]
        i = self.read_table(payer, "i", i_cols)
        pcols = [c for c in i.columns if c.startswith("PROC")]
        i_long = None
        for c in pcols:
            part = (i.select(F.col("ENROLID"), F.col("ADMDATE").alias("proc_date"),
                             F.col(c).cast("string").alias("code"))
                    .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                    .withColumn("setting", F.lit("inpatient")))
            i_long = part if i_long is None else i_long.unionByName(part)

        o_cols = ["ENROLID", "SVCDATE", "PROC1", "PROCTYP"]
        o = self.read_table(payer, "o", o_cols)
        o_long = (o.select(F.col("ENROLID"), F.col("SVCDATE").alias("proc_date"),
                           F.col("PROC1").cast("string").alias("code"))
                  .filter(F.col("code").isNotNull() & (F.col("code") != ""))
                  .withColumn("setting", F.lit("outpatient")))
        return i_long.unionByName(o_long)

    # ---- code-matching primitives (decimal-stripped ICD-9 prefix; proc exact) ----
    def match_dx(self, dx_long: DataFrame, codeset: List[str]) -> DataFrame:
        """Rows of dx_long whose decimal-stripped ICD-9 `code` PREFIX-matches any key in codeset."""
        codes = [str(c) for c in (codeset or []) if str(c).strip()]
        if not codes:
            return dx_long.limit(0)
        pat = "^(" + "|".join(codes) + ")"
        return dx_long.filter(strip_decimal(F.col("code")).rlike(pat))

    def match_proc(self, proc_long: DataFrame, codeset: List[str]) -> DataFrame:
        """Rows of proc_long whose `code` is exactly in codeset (CPT/HCPCS/ICD-9-proc)."""
        codes = [str(c) for c in (codeset or []) if str(c).strip()]
        if not codes:
            return proc_long.limit(0)
        return proc_long.filter(F.col("code").isin(codes))

    # ---- hdPS long-table producers (build_long_table) ----
    def dx_long_event_code(self, code_col):
        """3-digit ICD-9 category: strip the decimal then take the first 3 chars."""
        return F.substring(strip_decimal(code_col), 1, 3)

    def drug_long(self, ids: DataFrame) -> DataFrame:
        """Every pharmacy dispensing -> (ENROLID, svc_date, event_code=generic ingredient name),
        via the NDC->rxnorm_ingredient_name vocab. `ids` is accepted for interface parity."""
        if not self.vocab_ndc:
            raise ValueError("MarketScanAdapter.drug_long needs vocab_ndc (pass --vocab_ndc to build_long_table)")
        d = (self.read_table(self.payer, "d", ["ENROLID", "SVCDATE", "NDCNUM"])
             .withColumn("NDCNUM", F.col("NDCNUM").cast("string")))
        m = (self.spark.read.csv(self.vocab_ndc, header=True, inferSchema=True)
             .select(F.col("ndc_drug_code").cast("string").alias("NDCNUM"),
                     F.lower(F.col("rxnorm_ingredient_name")).alias("event_code"))
             .dropDuplicates())
        return (d.join(m, "NDCNUM", "inner")
                .filter(F.col("event_code").isNotNull() & (F.col("event_code") != ""))
                .select("ENROLID", F.col("SVCDATE").alias("svc_date"), "event_code"))

    def outcome_covariate_drop_keys(self, resolved) -> List[str]:
        """3-digit truncations of the primary outcome-defining ICD-9 match_keys (covariates are 3-digit)."""
        okeys = resolved.dx("outcome_inpatient")
        return sorted({k[:3] for k in okeys})

    # ---- vaccine exposure (HCPCS/CPT proc primary + NDC supplement) ----
    def vaccine_events(self, cfg, resolved) -> DataFrame:
        """df[ENROLID, svc_date] of flu-vaccine claim dates (the caller applies the Oct-Dec season
        filter). Three qualifying streams (the definition lives in codesets.py): proc codes (exact,
        incl. Medicare Q-codes), Redbook-resolved pharmacy NDCs, and V04.81/V06.6 dx."""
        vac = codesets.vaccine(cfg["cohort_id"]) or {}
        proc_codes = [str(c) for c in vac.get("proc_codes", {})]
        proc_long = self.proc_long_with_setting()
        vax_proc = (self.match_proc(proc_long, proc_codes)
                    .select("ENROLID", F.col("proc_date").alias("svc_date")))
        vax_ndc = resolved.drug(vac.get("ndc_generics", []))
        pharm = self.load_pharmacy()
        vax_disp = self.drug_dispensings(pharm, vax_ndc).select("ENROLID", "svc_date")
        out = vax_proc.unionByName(vax_disp)
        dx_keys = [str(c).replace(".", "") for c in vac.get("dx_codes", {})]
        if dx_keys:
            vax_dx = (self.dx_long_with_setting()
                      .filter(strip_decimal(F.col("code")).isin(dx_keys))
                      .select("ENROLID", F.col("dx_date").alias("svc_date")))
            out = out.unionByName(vax_dx)
        logging.info("%svaccine capture streams: %d proc codes, %d NDC, dx keys %s",
                     INDENT2, len(proc_codes), len(vax_ndc), dx_keys or "none")
        return out
