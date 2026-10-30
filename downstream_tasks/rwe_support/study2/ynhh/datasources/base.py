#!/usr/bin/env python3
"""
datasources/base.py -- the data-source adapter interface.

The adapter turns a data source into normalized event streams + code-matching primitives;
the cohort/covariate logic is written against that interface once. Column names follow the
claims convention (ENROLID, NDCNUM, svc_date) so the epidemiology code is source-agnostic:

    demographics()          ENROLID, DOBYR, SEX
    load_pharmacy()         ENROLID, svc_date, NDCNUM                    (drug-event stream)
    dx_long_with_setting()  ENROLID, dx_date, code, position, setting    (setting in {inpatient,outpatient})
    proc_long_with_setting()ENROLID, proc_date, code, setting
    (observability spans are adapter-internal; the gate consumes them)

The two OUTPUT contracts (cohort.csv, hdps_long.parquet) are produced downstream of the
adapter, so select_covariates / embed_reclaim / estimate_ladder never change with the source.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from pyspark.sql import DataFrame


# =============================================================================
# model-agnostic config parsing
# =============================================================================
def enrollment_gate_mode(cfg) -> str:
    """The configured observability-gate mode (config `enrollment.gate`). Both gated modes
    test the look-back [baseline_start, index_date].

       'off'      -- no observability gate (this study: the reference applies none).
       'baseline' -- ACTIVITY footprint (any visit_occurrence) brackets the look-back.
       'span'     -- COVERAGE (observation_period) brackets the look-back; RAISES where the
                     source has no coverage table rather than silently substituting activity.

    Any other value RAISES, so a stale or mistyped config fails loudly instead of silently
    degrading to 'off' (which would run ungated while looking gated).
    """
    _VALID = ("off", "baseline", "span")

    def _check(m, origin):
        if m in _VALID:
            return m
        raise ValueError(
            f"unknown observability gate mode {m!r} (from {origin}); expected one of {_VALID} "
            f"-- 'baseline' = ACTIVITY footprint, 'span' = COVERAGE, 'off' = no gate (the default). "
            f'Bare `off` in YAML parses as the boolean False -- quote it as "off".')

    # ENROLLMENT_GATE env override lets the same config run gated and ungated.
    _env = os.environ.get("ENROLLMENT_GATE", "").strip().lower()
    if _env:
        return _check(_env, "ENROLLMENT_GATE env")
    e = cfg.get("enrollment", {}) or {}
    if "gate" in e:
        raw = e["gate"]
        if raw is False or raw is None or raw == "":
            return "off"          # bare `off:` in YAML is the boolean False
        return _check(str(raw).strip().lower(), "config enrollment.gate")
    return "off"


# =============================================================================
# the adapter interface
# =============================================================================
class DataSourceAdapter(ABC):
    """One data source presented as normalized event streams + code-matching primitives.
    Concrete subclass: OMOPAdapter (datasources/omop.py).

    Construction state (set by subclass __init__): spark session, the cohort config `cfg`, and the
    `payer`/cell label. `source_name`/`data_model` identify the source for logging + the results tree.
    """

    source_name: str = "?"      # "ynhh" | ...
    data_model: str = "?"       # "omop"

    # ---- normalized event streams ----
    @abstractmethod
    def demographics(self, cfg) -> DataFrame:
        """df[ENROLID, DOBYR, SEX] -- one row per person, source eligibility applied."""

    @abstractmethod
    def load_pharmacy(self) -> DataFrame:
        """The drug-event stream df[ENROLID, svc_date, NDCNUM], persisted for reuse.
        OMOP: drug_exposure with drug_concept_id as NDCNUM (string)."""

    @abstractmethod
    def drug_dispensings(self, pharm: DataFrame, codeset) -> DataFrame:
        """Filter the drug-event stream to a resolved drug code set -> df[ENROLID, svc_date]."""

    @abstractmethod
    def dx_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """df[ENROLID, dx_date, code, position, setting]. `person_ids` (optional df[ENROLID]) prunes the
        raw scan to cohort persons first."""

    @abstractmethod
    def proc_long_with_setting(self) -> DataFrame:
        """df[ENROLID, proc_date, code, setting]."""

    def obs_long(self, person_ids: Optional[DataFrame] = None) -> Optional[DataFrame]:
        """OPTIONAL companion dx stream from the OMOP `observation` table -> df[ENROLID, dx_date, code],
        or None when the source has no such stream. OMOP routes Observation-domain standard
        concepts to `observation`, so a condition_occurrence-only scan finds none of them; the
        dx_route=standard matcher consults this stream in addition to dx_long."""
        return None

    # ---- hdPS long-table producers (build_long_table) ----
    @abstractmethod
    def dx_long_event_code(self, code_col):
        """Column expr turning a raw dx `code` into the hdPS covariate event_code
        (OMOP: concept_id string as-is)."""

    @abstractmethod
    def drug_long(self, ids: DataFrame) -> DataFrame:
        """df[ENROLID, svc_date, event_code] -- every drug exposure as an ingredient-level token.
        `ids` prunes to the cohort persons."""

    @abstractmethod
    def apply_continuous_gate(self, cfg, cohort: DataFrame, win_start_col: str,
                              win_end_col: str, label: str) -> DataFrame:
        """Filter `cohort` to observable members per the config gate mode (no-op when 'off')."""
