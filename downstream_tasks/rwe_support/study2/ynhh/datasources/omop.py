#!/usr/bin/env python3
"""
datasources/omop.py -- OMOPAdapter (OMOP CDM v5 on parquet dirs).

Reads person / observation_period / drug_exposure / condition_occurrence / procedure_occurrence /
visit_occurrence / payer_plan_period / death from {root}/{table}/*.parquet, lowercases all
column names, and presents them through the DataSourceAdapter interface with the normalized
column names the cohort logic expects:

  ENROLID  = person_id          DOBYR = year_of_birth, else year(date_of_birth/birth_datetime)
  SEX = 8507->1 (M) / 8532->2 (F)   NDCNUM = drug_concept_id (string; drug_exposure is the
                                    dispensing analogue)
  dx/proc code = condition/procedure_concept_id (string); dx_source_long gives the raw
                 ICD-10 source values for the dx_route=source matcher
  setting  = inpatient when visit_concept_id in the inpatient set, else outpatient
  position = synthetic 1/2 from condition_status_concept_id (Primary/Admitting -> 1,
             Secondary -> 2, else NULL)
  observability from observation_period spans, else a visit_occurrence footprint;
  death from the death table and/or person.death_*; insurance from payer_plan_period.

Every *_date / *_datetime read coalesces both forms (_date_expr): exports differ in which
they fill (the YNHH export leaves *_date NULL and fills only *_datetime).
"""

import os
import logging
from glob import glob
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark import StorageLevel
from pyspark.sql import functions as F

from datasources.base import DataSourceAdapter, enrollment_gate_mode

INDENT2 = "    "


class OMOPAdapter(DataSourceAdapter):
    data_model = "omop"

    def __init__(self, spark, *, source_name: str, root: str,
                 inpatient_visit_concept_ids=(9201, 262),
                 dx_status_concept_ids=(32902, 32908, 4203942), cfg=None,
                 codes_resolved_root: str = "intermediate/codes_resolved",
                 restrict_ids: str = None):
        self.spark = spark
        self.source_name = source_name
        self.payer = source_name
        self.root = root
        self.inpatient_visit_concept_ids = [int(x) for x in inpatient_visit_concept_ids]
        # condition_status_concept_ids that count as a principal/secondary diagnosis (the OMOP
        # analogue of a claims dx position 1/2). Default Primary(32902) + Secondary(32908) +
        # Admitting(4203942); per-source in the registry because sites use the Condition Status
        # vocabulary differently.
        self.dx_status_concept_ids = [int(x) for x in dx_status_concept_ids]
        self.cfg = cfg
        self.codes_resolved_root = codes_resolved_root
        self._warned_no_obs_period = False   # one-shot log when we fall back to a visit footprint
        # Tokenized-population gate: parquet glob of the token store's trajectory/ (keyed on
        # person_id). When set, the person universe is restricted to people who HAVE a token
        # sequence, so the embedding specs never silently receive a header-only sequence.
        # Applied in demographics(); the drop is logged.
        self.restrict_ids = restrict_ids
        self._restrict_df = None

    def _restrict_id_df(self):
        """df[ENROLID] of the tokenized population (cached). OMOP token stores key on a bare
        person_id; a stray alpha prefix is tolerated."""
        if self._restrict_df is None:
            files = sorted(f for f in glob(self.restrict_ids)
                           if not os.path.basename(f).startswith("._"))
            if not files:
                raise FileNotFoundError(f"--restrict_ids matched no parquet: {self.restrict_ids}")
            df = self.spark.read.parquet(*files)
            idcol = next((c for c in df.columns if c.lower() in ("person_id", "enrollee_id")), None)
            if idcol is None:
                raise ValueError(f"--restrict_ids parquet has no person_id/enrollee_id column: {df.columns}")
            self._restrict_df = (df.select(
                F.regexp_extract(F.col(idcol).cast("string"), r"^[A-Za-z]*0*(\d+)$", 1)
                 .cast("long").alias("ENROLID"))
                .filter(F.col("ENROLID").isNotNull()).distinct().persist())
            logging.info("    ReClaim-compatibility gate: %d tokenized person id(s) from %s",
                         self._restrict_df.count(), self.restrict_ids)
        return self._restrict_df

    def _apply_restrict_ids(self, persons):
        """Restrict the person universe to tokenized people, logging the drop (never silent)."""
        if not self.restrict_ids:
            return persons
        before = persons.select("ENROLID").distinct().count()
        out = persons.join(self._restrict_id_df(), "ENROLID", "left_semi")
        after = out.select("ENROLID").distinct().count()
        logging.info("    ReClaim-compatibility filter: %s -> %s person(s)  (dropped %s = %.1f%%)",
                     f"{before:,}", f"{after:,}", f"{before - after:,}",
                     100.0 * (before - after) / before if before else 0.0)
        if after == 0:
            raise ValueError("ReClaim-compatibility filter removed every person -- check --restrict_ids")
        return out

    # ---- parquet read + column lowercasing ----
    def _read(self, table: str, cols: List[str] = None) -> DataFrame:
        d = os.path.join(self.root, table)
        files = sorted(f for f in glob(os.path.join(d, "*.parquet"))
                       if not os.path.basename(f).startswith("._"))
        if not files:
            raise FileNotFoundError(f"No parquet for OMOP table '{table}' under {d}")
        df = self.spark.read.parquet(*files)
        df = df.toDF(*[c.lower() for c in df.columns])
        if cols:
            df = df.select(*[c for c in cols if c in df.columns])
        return df

    @staticmethod
    def _date_expr(df: DataFrame, *bases: str):
        """A `date` column coalescing, in order, the `_date` then `_datetime` form of each base name
        present in `df` (populated `_date` wins; else the `_datetime` sibling). Returns a NULL
        date column if no candidate exists."""
        cands = []
        for b in bases:
            for name in (b, f"{b}_date", f"{b}_datetime"):
                if name in df.columns:
                    cands.append(F.to_date(F.col(name)))
        if not cands:
            return F.lit(None).cast("date")
        return F.coalesce(*cands) if len(cands) > 1 else cands[0]

    def _setting_col(self):
        return F.when(F.col("visit_concept_id").isin(self.inpatient_visit_concept_ids), F.lit("inpatient")) \
                .otherwise(F.lit("outpatient"))

    def _status_position_col(self, df: DataFrame):
        """Synthetic dx `position` from condition_status_concept_id: 'Secondary diagnosis' (32908)
        -> 2; any other status in the keep-set -> 1; everything else (status 0/NULL) -> NULL.
        Returns NULL if the column is absent."""
        if "condition_status_concept_id" not in df.columns:
            return F.lit(None).cast("int")
        s = F.col("condition_status_concept_id")
        ids = self.dx_status_concept_ids
        ones = [c for c in ids if c != 32908]      # primary/admitting-type -> position 1
        expr = F.lit(None).cast("int")
        if 32908 in ids:
            expr = F.when(s == 32908, F.lit(2)).otherwise(expr)
        if ones:
            expr = F.when(s.isin(ones), F.lit(1)).otherwise(expr)
        return expr

    # ---- demographics ----
    def demographics(self, cfg) -> DataFrame:
        p = self._read("person", ["person_id", "year_of_birth", "date_of_birth", "birth_datetime",
                                   "gender_concept_id"])
        # DOBYR from year_of_birth when populated; else the birth date/datetime year (the YNHH
        # export leaves year_of_birth NULL but fills date_of_birth/birth_datetime).
        yob_direct = (F.when(F.col("year_of_birth") > 0, F.col("year_of_birth").cast("int"))
                      if "year_of_birth" in p.columns else F.lit(None).cast("int"))
        dobyr = F.coalesce(yob_direct, F.year(self._date_expr(p, "date_of_birth", "birth_datetime")).cast("int"))
        sex = F.when(F.col("gender_concept_id") == 8507, F.lit(1)) \
               .when(F.col("gender_concept_id") == 8532, F.lit(2)).otherwise(F.lit(None))
        out = p.select(F.col("person_id").alias("ENROLID"),
                        dobyr.alias("DOBYR"),
                        sex.alias("SEX"))
        # Drop persons with no resolvable birth year or a non-M/F gender concept: the demographic
        # covariates cannot carry NULLs.
        out = out.filter(F.col("DOBYR").isNotNull() & F.col("SEX").isNotNull())
        # ONE ROW PER PERSON is this method's contract: callers join it as a dimension table, so
        # a surplus row fans the cohort out, and the YNHH export emits one `person` row per MRN.
        # min() per attribute is deterministic under any partitioning (dropDuplicates is not);
        # conflicting values are reported rather than silently resolved.
        dedup = out.groupBy("ENROLID").agg(
            F.min("DOBYR").alias("DOBYR"),
            F.min("SEX").alias("SEX"),
            F.countDistinct("DOBYR").alias("_n_dobyr"),
            F.countDistinct("SEX").alias("_n_sex"))
        n_conflict = dedup.filter((F.col("_n_dobyr") > 1) | (F.col("_n_sex") > 1)).count()
        if n_conflict:
            logging.warning("[omop] %d persons carry conflicting birth year or sex across duplicate "
                            "`person` rows -- taking min() per attribute; verify the export's "
                            "person-level ETL", n_conflict)
        out = dedup.drop("_n_dobyr", "_n_sex")
        return self._apply_restrict_ids(out)

    # ---- observability sources, kept separate ----
    # _obs_spans()      : best-available span (observation_period if present, else the visit footprint)
    # _coverage_spans() : observation_period only; raises where absent (gate mode 'span' = COVERAGE)
    # _visit_footprint(): visit_occurrence only (gate mode 'baseline' = ACTIVITY)
    def _has_observation_period(self) -> bool:
        try:
            self._read("observation_period", ["person_id"])
            return True
        except FileNotFoundError:
            return False

    def _coverage_spans(self) -> DataFrame:
        op = self._read("observation_period",
                        ["person_id", "observation_period_start_date", "observation_period_start_datetime",
                         "observation_period_end_date", "observation_period_end_datetime"])
        return op.select(F.col("person_id").alias("ENROLID"),
                         self._date_expr(op, "observation_period_start").alias("obs_start"),
                         self._date_expr(op, "observation_period_end").alias("obs_end"))

    def _visit_footprint(self) -> DataFrame:
        vo = self._read("visit_occurrence",
                        ["person_id", "visit_start_date", "visit_start_datetime",
                         "visit_end_date", "visit_end_datetime"])
        vs = self._date_expr(vo, "visit_start")
        ve = self._date_expr(vo, "visit_end")
        return (vo.groupBy("person_id")
                .agg(F.min(vs).alias("obs_start"), F.max(F.coalesce(ve, vs)).alias("obs_end"))
                .select(F.col("person_id").alias("ENROLID"), "obs_start", "obs_end"))

    def _obs_spans(self) -> DataFrame:
        """One [obs_start, obs_end] span per person. Prefer observation_period; if that table is
        absent (the YNHH export has none), derive a footprint from visit_occurrence as each
        person's [min visit_start, max visit_end]. payer_plan_period is NOT used: its end_date
        is almost entirely NULL in that export, so it cannot bracket a window."""
        try:
            op = self._read("observation_period",
                            ["person_id", "observation_period_start_date", "observation_period_start_datetime",
                             "observation_period_end_date", "observation_period_end_datetime"])
            return op.select(F.col("person_id").alias("ENROLID"),
                             self._date_expr(op, "observation_period_start").alias("obs_start"),
                             self._date_expr(op, "observation_period_end").alias("obs_end"))
        except FileNotFoundError:
            if not self._warned_no_obs_period:
                logging.warning("%s[omop] no observation_period table -> deriving observability spans "
                                "from visit_occurrence [min start, max end] per person", INDENT2)
                self._warned_no_obs_period = True
            vo = self._read("visit_occurrence",
                            ["person_id", "visit_start_date", "visit_start_datetime",
                             "visit_end_date", "visit_end_datetime"])
            vs = self._date_expr(vo, "visit_start")
            ve = self._date_expr(vo, "visit_end")
            return (vo.groupBy("person_id")
                    .agg(F.min(vs).alias("obs_start"), F.max(F.coalesce(ve, vs)).alias("obs_end"))
                    .select(F.col("person_id").alias("ENROLID"), "obs_start", "obs_end"))

    def apply_continuous_gate(self, cfg, cohort, win_start_col, win_end_col, label):
        """Apply the configured observability gate over the look-back [win_start_col, index_date]
        (no-op when 'off', as in this study).
          baseline -> ACTIVITY: visit_occurrence footprint brackets the look-back.
          span     -> COVERAGE: observation_period brackets the look-back; raises where the
                      export has no observation_period rather than silently substituting
                      activity, since coverage and activity select different people.
        """
        from datasources.base import enrollment_gate_mode
        mode = enrollment_gate_mode(cfg)
        if mode == "off":
            return cohort
        end_col = "index_date"
        if mode == "span":
            if not self._has_observation_period():
                raise ValueError(
                    "enrollment.gate='span' asks for COVERAGE-based observability, but this OMOP "
                    "export has no observation_period table, so coverage is unmeasurable here. "
                    "Use 'baseline' (visit-activity footprint) and report it as an activity gate, "
                    "or 'off'.")
            spans = self._coverage_spans()
            kind = "COVERAGE -- observation_period"
        else:
            spans = self._visit_footprint()
            kind = "ACTIVITY -- visit_occurrence footprint"
        windows = cohort.select("ENROLID", F.col(win_start_col).alias("win_start"),
                                F.col(end_col).alias("win_end"))
        agg = (spans.join(windows.select("ENROLID", "win_start", "win_end"), "ENROLID", "inner")
               .groupBy("ENROLID")
               .agg(F.min("obs_start").alias("min_start"), F.max("obs_end").alias("max_end"),
                    F.first("win_start").alias("ws"), F.first("win_end").alias("we")))
        ok = agg.filter((F.col("min_start") <= F.col("ws")) &
                        (F.col("max_end") >= F.col("we"))).select("ENROLID")
        logging.info("%senrollment gate (%s, mode=%s: %s) over [%s, %s]",
                     INDENT2, label, mode, kind, win_start_col, end_col)
        return cohort.join(ok, "ENROLID", "inner")

    def insurance_at_index(self, cohort: DataFrame) -> DataFrame:
        """df[ENROLID, ins_category] -- the reference's Table S5 insurance covariate, built from
        payer_plan_period.payer_concept_id via the payer map (datasources/payer_map.py).

        Index-anchored rule, START DATES ONLY: the payer period with the latest start on or
        before index_date (ties broken by the larger concept_id for determinism); no period ->
        'other_unknown'. End dates are deliberately not used: they are almost entirely NULL in
        the YNHH export, so an end-date predicate would silently empty the table.

        A missing/empty payer_plan_period is not an error: every member falls back to
        'other_unknown' and the distribution is logged, so a degenerate covariate is visible."""
        from datasources.payer_map import PAYER_CONCEPT_CLASS, PAYER_CLASS_LABEL
        ids = cohort.select("ENROLID", "index_date")
        try:
            ppp = self._read("payer_plan_period",
                             ["person_id", "payer_concept_id",
                              "payer_plan_period_start_date", "payer_plan_period_start_datetime"])
        except FileNotFoundError:
            logging.warning("%s[omop] no payer_plan_period table -> ins_category = other_unknown "
                            "for every cohort member", INDENT2)
            return ids.select("ENROLID", F.lit("other_unknown").alias("ins_category"))
        start = self._date_expr(ppp, "payer_plan_period_start")
        ppp = (ppp.select(F.col("person_id").alias("ENROLID"),
                          F.col("payer_concept_id").cast("long").alias("cid"),
                          start.alias("start_date"))
               .filter(F.col("start_date").isNotNull()))
        mapping = F.create_map(*[x for cid, cls in PAYER_CONCEPT_CLASS.items()
                                 for x in (F.lit(int(cid)), F.lit(PAYER_CLASS_LABEL[cls]))])
        w = Window.partitionBy("ENROLID").orderBy(F.col("start_date").desc(), F.col("cid").desc())
        picked = (ppp.join(ids, "ENROLID", "inner")
                  .filter(F.col("start_date") <= F.col("index_date"))
                  .withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
                  .select("ENROLID",
                          F.coalesce(mapping[F.col("cid")], F.lit("other_unknown")).alias("ins_category")))
        out = (ids.select("ENROLID").join(picked, "ENROLID", "left")
               .withColumn("ins_category", F.coalesce(F.col("ins_category"), F.lit("other_unknown"))))
        dist = out.groupBy("ins_category").count().collect()
        logging.info("%sinsurance covariate (payer_plan_period, start-date rule): %s", INDENT2,
                     ", ".join(f"{r['ins_category']}={r['count']:,}" for r in sorted(dist, key=lambda r: -r["count"])))
        return out

    def _death_dates(self) -> Optional[DataFrame]:
        """df[ENROLID, death_date] from whichever death source(s) exist: person.death_datetime/
        death_date (where YNHH stores death; it has no death table) and/or a death table.
        Union + earliest-per-person; None if neither carries a date."""
        frames = []
        p = self._read("person", ["person_id", "death_date", "death_datetime"])
        if any(c in p.columns for c in ("death_date", "death_datetime")):
            frames.append(p.select(F.col("person_id").alias("ENROLID"),
                                   self._date_expr(p, "death").alias("death_date")))
        try:
            dt = self._read("death", ["person_id", "death_date", "death_datetime"])
            frames.append(dt.select(F.col("person_id").alias("ENROLID"),
                                    self._date_expr(dt, "death").alias("death_date")))
        except FileNotFoundError:
            pass
        if not frames:
            return None
        out = frames[0]
        for f in frames[1:]:
            out = out.unionByName(f)
        return (out.filter(F.col("death_date").isNotNull())
                .groupBy("ENROLID").agg(F.min("death_date").alias("death_date")))

    # ---- drug-event stream ----
    def load_pharmacy(self) -> DataFrame:
        d = self._read("drug_exposure", ["person_id", "drug_exposure_start_date",
                                         "drug_exposure_start_datetime", "drug_concept_id"])
        return (d.select(F.col("person_id").alias("ENROLID"),
                         self._date_expr(d, "drug_exposure_start").alias("svc_date"),
                         F.col("drug_concept_id").cast("string").alias("NDCNUM"))
                .persist(StorageLevel.MEMORY_AND_DISK))

    def drug_dispensings(self, pharm: DataFrame, codeset) -> DataFrame:
        ids = [str(x) for x in (codeset or []) if str(x).strip()]
        if not ids:
            return pharm.select("ENROLID", "svc_date").limit(0)
        return pharm.filter(F.col("NDCNUM").isin(ids)).select("ENROLID", "svc_date")

    # ---- dx / proc long ----
    def dx_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        co = self._read("condition_occurrence",
                        ["person_id", "condition_concept_id", "condition_status_concept_id",
                         "condition_start_date", "condition_start_datetime", "visit_occurrence_id"])
        dx_date = self._date_expr(co, "condition_start")
        position = self._status_position_col(co)
        if person_ids is not None:
            co = co.join(F.broadcast(person_ids.withColumnRenamed("ENROLID", "person_id")),
                         "person_id", "left_semi")
        vo = self._read("visit_occurrence", ["visit_occurrence_id", "visit_concept_id"])
        co = co.join(vo, "visit_occurrence_id", "left")
        return co.select(F.col("person_id").alias("ENROLID"),
                         dx_date.alias("dx_date"),
                         F.col("condition_concept_id").cast("string").alias("code"),
                         position.alias("position"),
                         self._setting_col().alias("setting"))

    def obs_long(self, person_ids: Optional[DataFrame] = None) -> Optional[DataFrame]:
        """Companion dx stream from the `observation` table (Observation-domain standard concepts
        land here, never in condition_occurrence). Same code semantics as dx_long (standard
        concept_id as string), no setting/position."""
        try:
            ob = self._read("observation",
                            ["person_id", "observation_concept_id", "observation_date", "observation_datetime"])
        except Exception:
            return None    # export without an observation table -> no companion stream
        if person_ids is not None:
            ob = ob.join(F.broadcast(person_ids.withColumnRenamed("ENROLID", "person_id")),
                         "person_id", "left_semi")
        return ob.select(F.col("person_id").alias("ENROLID"),
                         self._date_expr(ob, "observation").alias("dx_date"),
                         F.col("observation_concept_id").cast("string").alias("code"))

    def dx_source_long(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """SOURCE-CODE dx stream: raw ICD-10 codes as recorded, from condition_source_value UNION
        observation_source_value -> df[ENROLID, dx_date, code(raw source value)]. For the
        dx_route=source matcher (the default route): prefix-matching these replicates the
        reference study's claims semantics -- no 'Maps to' translation (whose closure can blur
        sibling codes, e.g. T1D/T2D combination concepts), no domain routing (the source value
        survives wherever OMOP sent the row), and rows whose standard concept_id is NULL are
        recovered. Source values are era-mixed (ICD-9 before Oct 2015); the study window keeps
        the matcher in ICD-10 territory."""
        co = self._read("condition_occurrence",
                        ["person_id", "condition_source_value",
                         "condition_start_date", "condition_start_datetime"])
        out = co.select(F.col("person_id").alias("ENROLID"),
                        self._date_expr(co, "condition_start").alias("dx_date"),
                        F.col("condition_source_value").cast("string").alias("code"))
        try:
            ob = self._read("observation",
                            ["person_id", "observation_source_value",
                             "observation_date", "observation_datetime"])
            out = out.unionByName(
                ob.select(F.col("person_id").alias("ENROLID"),
                          self._date_expr(ob, "observation").alias("dx_date"),
                          F.col("observation_source_value").cast("string").alias("code")))
        except Exception:
            pass    # export without an observation table -> condition stream only
        out = out.filter(F.col("code").isNotNull())
        if person_ids is not None:
            out = out.join(F.broadcast(person_ids), "ENROLID", "left_semi")
        return out

    def proc_long_with_setting(self) -> DataFrame:
        pr = self._read("procedure_occurrence",
                        ["person_id", "procedure_concept_id", "procedure_date",
                         "procedure_datetime", "visit_occurrence_id"])
        proc_date = self._date_expr(pr, "procedure")
        vo = self._read("visit_occurrence", ["visit_occurrence_id", "visit_concept_id"])
        pr = pr.join(vo, "visit_occurrence_id", "left")
        return pr.select(F.col("person_id").alias("ENROLID"),
                         proc_date.alias("proc_date"),
                         F.col("procedure_concept_id").cast("string").alias("code"),
                         self._setting_col().alias("setting"))

    # ---- hdPS long-table producers (build_long_table) ----
    def dx_long_event_code(self, code_col):
        """OMOP covariate event_code = the standard concept_id string (as-is)."""
        return code_col

    def drug_long(self, ids: DataFrame) -> DataFrame:
        """Every drug_exposure -> (ENROLID, svc_date, event_code = lower ingredient concept name),
        mapping drug_concept_id -> ingredient via drug_strength (a multi-ingredient product
        yields one row per ingredient)."""
        draw = self._read("drug_exposure", ["person_id", "drug_exposure_start_date",
                                            "drug_exposure_start_datetime", "drug_concept_id"])
        d = draw.select(F.col("person_id").alias("ENROLID"),
                        self._date_expr(draw, "drug_exposure_start").alias("svc_date"),
                        F.col("drug_concept_id").cast("long").alias("drug_concept_id"))
        ds = self._read("drug_strength", ["drug_concept_id", "ingredient_concept_id"]).select(
            F.col("drug_concept_id").cast("long").alias("drug_concept_id"),
            F.col("ingredient_concept_id").cast("long").alias("ingredient_concept_id")).dropDuplicates()
        ing = self._read("concept", ["concept_id", "concept_name", "concept_class_id"]) \
            .filter(F.col("concept_class_id") == "Ingredient") \
            .select(F.col("concept_id").cast("long").alias("ingredient_concept_id"),
                    F.lower(F.col("concept_name")).alias("event_code"))
        return (d.join(ds, "drug_concept_id", "inner")
                .join(F.broadcast(ing), "ingredient_concept_id", "inner")
                .filter(F.col("event_code").isNotNull() & (F.col("event_code") != ""))
                .select("ENROLID", "svc_date", "event_code"))
