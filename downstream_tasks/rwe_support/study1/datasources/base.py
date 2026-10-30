import os
#!/usr/bin/env python3
"""
datasources/base.py  --  the data-source adapter interface.

The adapter turns a raw data source into normalized event streams + code-matching primitives;
build_cohort/build_long_table's epidemiologic logic is written against this interface only.

NORMALIZED COLUMN CONTRACT (MarketScan-native names):
    demographics()          ENROLID, DOBYR, SEX [, MEDICARE]
    load_pharmacy()         ENROLID, svc_date, NDCNUM [, source_db]      (drug-event stream)
    dx_long_with_setting()  ENROLID, dx_date, code, position, setting    (setting in {inpatient,outpatient})
    proc_long_with_setting()ENROLID, proc_date, code, setting
    (enrollment spans are adapter-internal; the gate helpers consume them)

The two OUTPUT contracts (cohort.csv, hdps_long.parquet) are produced downstream of the adapter,
so select_covariates/embed_reclaim/estimate_ladder never see the raw source.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from pyspark.sql import DataFrame


# =============================================================================
# config parsing (reads cfg only)
# =============================================================================
def enrollment_gate_mode(cfg) -> str:
    """The configured observability-gate mode (config `enrollment.gate`; ENROLLMENT_GATE env overrides).

    Every mode tests the LOOK-BACK window [baseline_start, index_date]; follow-up loss is censored
    via person_time, never excluded.
       'baseline' -- ACTIVITY footprint (any claim) brackets the look-back.
       'span'     -- COVERAGE (enrollment detail) brackets the look-back; interior gaps allowed.
       'off'      -- no gate (the design: the reference specifies no continuous-enrollment rule,
                     and unlike censoring a gate excludes people).
    Any other value RAISES, so a mistyped config cannot silently degrade to 'off'.
    """
    _VALID = ("off", "baseline", "span")

    def _check(m, origin):
        if m in _VALID:
            return m
        raise ValueError(
            f"unknown observability gate mode {m!r} (from {origin}); expected one of {_VALID} "
            f"-- 'baseline' = ACTIVITY footprint, 'span' = COVERAGE, 'off' = no gate (the default). "
            f'NB bare `off` in YAML parses as the boolean False -- quote it as "off". ')

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
    """One data source (a MarketScan payer stratum) presented as normalized event streams +
    code-matching primitives. Concrete subclass: MarketScanAdapter.

    Construction state (set by subclass __init__): spark session, the cohort config `cfg`, and the
    `payer`/cell label. `source_name`/`data_model` identify the source for logging + the results tree.
    """

    source_name: str = "?"      # "marketscan"
    data_model: str = "?"       # "marketscan"

    # ---- capabilities: the epi logic branches on these instead of on payer names / model strings ----
    @property
    def dx_position_available(self) -> bool:
        """True if diagnoses carry a position (1st/2nd/...). When False, a cohort's
        position_requirement (positive_control inpatient 1st/2nd) is skipped with a logged warning."""
        return True

    @property
    def payer_label(self) -> str:
        """Value written to the cohort `payer` column."""
        return getattr(self, "payer", self.source_name)

    @abstractmethod
    def resolve_age_floor(self, cfg, args) -> int:
        """Age floor for this run/stratum."""

    # ---- normalized event streams ----
    @abstractmethod
    def demographics(self, cfg) -> DataFrame:
        """df[ENROLID, DOBYR, SEX (, MEDICARE)] -- first-record demographics, source eligibility applied
        (drug-coverage gate + optional dual exclusion per cfg strata). Dual-exclusion is decided
        inside the adapter from cfg, not by the caller."""

    @abstractmethod
    def load_pharmacy(self) -> DataFrame:
        """The drug-event stream df[ENROLID, svc_date, NDCNUM (, source_db)], persisted for reuse."""

    @abstractmethod
    def drug_dispensings(self, pharm: DataFrame, codeset) -> DataFrame:
        """Filter the drug-event stream to a resolved drug code set -> df[ENROLID, svc_date (, source_db)]."""

    @abstractmethod
    def dx_long_with_setting(self, person_ids: Optional[DataFrame] = None) -> DataFrame:
        """df[ENROLID, dx_date, code, position, setting]. `person_ids` (optional df[ENROLID]) prunes the
        raw scan to cohort persons first."""

    @abstractmethod
    def proc_long_with_setting(self) -> DataFrame:
        """df[ENROLID, proc_date, code, setting]."""

    # ---- code-matching primitives ----
    @abstractmethod
    def match_dx(self, dx_long: DataFrame, codeset: List[str]) -> DataFrame:
        """Return the rows of dx_long whose `code` matches the resolved dx codeset, columns preserved."""

    @abstractmethod
    def match_proc(self, proc_long: DataFrame, codeset: List[str]) -> DataFrame:
        """Return the rows of proc_long whose `code` matches the resolved proc codeset."""

    # ---- exposure-location abstraction ----
    @abstractmethod
    def vaccine_events(self, cfg, resolved) -> DataFrame:
        """df[ENROLID, svc_date] of vaccine-claim dates (the epi logic applies the Oct-Dec season
        filter)."""

    # ---- hdPS long-table producers (build_long_table) ----
    @abstractmethod
    def dx_long_event_code(self, code_col):
        """Column expr turning a raw dx `code` into the hdPS covariate event_code."""

    @abstractmethod
    def drug_long(self, ids: DataFrame) -> DataFrame:
        """df[ENROLID, svc_date, event_code] -- every drug dispensing as an ingredient-level token.
        `ids` prunes to the cohort persons."""

    @abstractmethod
    def outcome_covariate_drop_keys(self, resolved) -> List[str]:
        """dx event_codes to drop from the covariate pool (the outcome-defining codes, matched at the
        SAME granularity as dx_long_event_code)."""

    # ---- enrollment / observability gate + follow-up ----
    @abstractmethod
    def continuous_enrolled(self, windows_df: DataFrame, mode: str = "span") -> DataFrame:
        """df[ENROLID] enrolled/observable across each person's [win_start, win_end] per `mode`
        (gate mode 'span' = COVERAGE). windows_df: df[ENROLID, win_start, win_end]."""

    @abstractmethod
    def apply_continuous_gate(self, cfg, cohort: DataFrame, win_start_col: str,
                              win_end_col: str, label: str) -> DataFrame:
        """Filter `cohort` to observable members per the config gate mode (no-op when 'off')."""

    @abstractmethod
    def add_person_time(self, cohort: DataFrame) -> DataFrame:
        """Add `person_time` = index_date -> min(followup_end, disenroll/last-observed)."""

    @abstractmethod
    def enrolled_at_landmark(self, seasons) -> DataFrame:
        """df[ENROLID, season] for enrollees observable at each season's Oct-1 landmark."""
