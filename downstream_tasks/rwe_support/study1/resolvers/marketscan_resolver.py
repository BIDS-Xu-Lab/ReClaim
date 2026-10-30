#!/usr/bin/env python3
"""
resolvers/marketscan_resolver.py  --  MarketScan code resolution.

  * run(): the resolve_codes body -- resolve drug generics -> NDC (via the NDC->RxNorm vocab) and
    materialize the dx/proc DEFINING codes into intermediate/codes_resolved/{cohort_id}/
    {drug_ndc_map,dx_proc_codes,resolution_summary}.csv. A light pandas step (no Spark).
  * MarketScanResolved: the build_cohort-side lookup that reads those artifacts back, returning
    NDC lists (drugs) and decimal-stripped ICD-9 match_keys (dx).
"""

import os
import logging
from typing import List

import pandas as pd

import codesets
from resolvers.common import ResolvedCodes

INDENT1, INDENT2 = "  ", "    "

# Only these two vocab columns are needed; a narrow, string-typed frame keeps the map cheap and
# preserves NDC leading zeros (they are identifiers, not numbers).
_MAP_USECOLS = ["ndc_drug_code", "rxnorm_ingredient_name"]


# =============================================================================
# resolve_codes body: collect generics, resolve to NDC, materialize dx/proc artifact
# =============================================================================
def collect_drug_generics(cohort_id: str) -> pd.DataFrame:
    """Pull every drug generic the cohort references from codesets.py, as a frame:
        generic_name | source | match | ingredient_name."""
    rows = [{"generic_name": generic, "source": source, "ingredient_name": alias}
            for generic, alias, source in codesets.all_drug_generic_rows(cohort_id)]
    df = pd.DataFrame(rows, columns=["generic_name", "source", "ingredient_name"])
    return df.drop_duplicates(subset=["generic_name", "source"])


def resolve(generics: pd.DataFrame, vocab_ndc: str):
    """(map_df[generic_name, ndc_drug_code, n_ingredients], summary_df). EXACT match on
    lower-cased rxnorm_ingredient_name; no substring mode."""
    logging.info("%sLoading NDC->RxNorm map: %s", INDENT1, vocab_ndc)
    vocab = pd.read_csv(vocab_ndc, usecols=_MAP_USECOLS, dtype=str).dropna(subset=["ndc_drug_code"])
    vocab["ing_lower"] = vocab["rxnorm_ingredient_name"].fillna("").str.lower().str.strip()
    logging.info("%s%s NDC rows, %s distinct ingredient names",
                 INDENT2, f"{len(vocab):,}", f"{vocab['ing_lower'].nunique():,}")

    ndc_ning = (vocab[vocab["ing_lower"] != ""]
                .groupby("ndc_drug_code")["ing_lower"].nunique())

    gen_targets, gen_sources = {}, {}
    for _, r in generics.iterrows():
        g = r["generic_name"]
        alias = r.get("ingredient_name")
        targets = ([t.strip().lower() for t in str(alias).split("|") if t.strip()]
                   if (pd.notna(alias) and str(alias).strip()) else [g.strip().lower()])
        cur = gen_targets.setdefault(g, [])
        for t in targets:
            if t not in cur:
                cur.append(t)
        gen_sources.setdefault(g, []).append(r["source"])

    map_rows, summary_rows = [], []
    for gen in sorted(gen_targets):
        targets = gen_targets[gen]
        hit = vocab[vocab["ing_lower"].isin(targets)]
        ndcs = sorted(hit["ndc_drug_code"].dropna().unique().tolist())
        status = "matched" if ndcs else "UNMATCHED"
        n_combo = 0
        for ndc in ndcs:
            n_ing = int(ndc_ning.get(ndc, 1))
            n_combo += (n_ing > 1)
            map_rows.append({"generic_name": gen, "ndc_drug_code": ndc, "n_ingredients": n_ing})
        n_single = len(ndcs) - n_combo
        via = "" if targets == [gen.strip().lower()] else f" (via {'|'.join(targets)})"
        summary_rows.append({"generic_name": gen,
                             "source": "|".join(sorted(set(gen_sources[gen]))),
                             "match_mode": "exact", "ingredient_targets": "|".join(targets),
                             "n_ndc": len(ndcs), "n_ndc_single": n_single, "n_ndc_combo": n_combo,
                             "status": status})
        level = logging.INFO if ndcs else logging.WARNING
        combo_note = f" ({n_combo} combo)" if n_combo else ""
        logging.log(level, "%s%-24s%s %s -> %d NDC%s%s",
                    INDENT2, gen, via, status, len(ndcs), combo_note,
                    "" if ndcs else "  <-- no ingredient match (review / adjust codeset)")

    map_df = pd.DataFrame(map_rows,
                          columns=["generic_name", "ndc_drug_code", "n_ingredients"]).drop_duplicates()
    summary_df = pd.DataFrame(summary_rows,
                              columns=["generic_name", "source", "match_mode",
                                       "ingredient_targets", "n_ndc", "n_ndc_single",
                                       "n_ndc_combo", "status"])
    return map_df, summary_df


def _norm_key(code) -> str:
    """Decimal-stripped prefix key -- exactly what build_cohort/build_long_table prefix-match on (531.0 -> 5310)."""
    return str(code).replace(".", "").strip()


def build_dx_proc_artifact(cfg: dict) -> pd.DataFrame:
    """Materialize the cohort's DEFINING dx/proc code sets (from codesets.py) into rows tagged by
    `role`, each carrying the normalized `match_key` plus position/setting metadata."""
    rows = []
    cid = cfg["cohort_id"]
    oc = cfg["outcome"]

    def add_set(code_meaning: dict, role, name="", settings="", position="", code_type="ICD9CM"):
        for code, meaning in code_meaning.items():
            if code is None or not str(code).strip():
                continue
            rows.append({"role": role, "name": name, "code": str(code).strip(),
                         "match_key": _norm_key(code), "code_type": code_type,
                         "settings": settings, "position": position, "meaning": meaning or ""})

    oc_settings = oc.get("settings", ["inpatient"])
    positions = ",".join(str(p) for p in (oc.get("position_requirement") or []))
    primary_set = codesets.outcome_set(cid, oc["primary"])
    if "inpatient" in oc_settings:
        add_set(primary_set, "outcome_inpatient", settings="inpatient", position=positions)
    if "outpatient" in oc_settings:
        add_set(codesets.outcome_outpatient_set(cid) or primary_set, "outcome_outpatient",
                settings="outpatient", position="")

    prior = codesets.prior_exclusion_set(cid)
    if prior:
        add_set(prior, "prior_exclusion_dx", name="prior_hip_fracture")

    for flag, spec in codesets.predefined(cid).items():
        if spec["domain"] == "dx_icd9":
            add_set({c: "" for c in spec["codes"]}, "predefined_dx", name=flag)

    return pd.DataFrame(rows, columns=["role", "name", "code", "match_key", "code_type",
                                       "settings", "position", "meaning"])


def run(cfg: dict, vocab_ndc: str, out_dir: str):
    """The resolve_codes MarketScan resolution: write drug_ndc_map.csv + resolution_summary.csv + dx_proc_codes.csv
    under out_dir (= {out_root}/{cohort_id}). Returns nothing; logs a summary."""
    cohort_id = cfg["cohort_id"]
    os.makedirs(out_dir, exist_ok=True)

    generics = collect_drug_generics(cohort_id)
    if generics.empty:
        logging.warning("%sNo drug generics referenced by %s -- nothing to resolve.", INDENT1, cohort_id)
    logging.info("%s%d distinct drug generic(s) to resolve", INDENT1, generics["generic_name"].nunique())
    map_df, summary_df = resolve(generics, vocab_ndc)

    # ---- Redbook route for vaccine NDCs -----------------------------------------------------------
    # The RxNorm rollup has no clean flu-vaccine ingredient row, so the vaccine NDC list is resolved
    # against the MarketScan Redbook dictionary instead: GENNME contains-match (the codeset specifies
    # 'INFLUENZA VIR', deliberately NOT bare 'INFLUENZA' which would sweep in the H1N1/H5N1 pandemic
    # products). Matched NDCs are appended to drug_ndc_map.csv under the codeset's sentinel generic
    # name, so drug()/vaccine_events pick them up with no further plumbing. The distinct matched
    # GENNME values are logged for review.
    vac = codesets.vaccine(cohort_id) or {}
    rb_pats = [p.upper() for p in vac.get("redbook_gennme_contains", [])]
    if rb_pats:
        rb_csv = os.environ.get("REDBOOK_CSV", "")   # RED BOOK dictionary csv; set in env.sh
        if not os.path.exists(rb_csv):
            raise FileNotFoundError(f"Redbook dictionary not found: {rb_csv} (set REDBOOK_CSV)")
        rb = pd.read_csv(rb_csv, usecols=["NDCNUM", "GENNME"], dtype=str)
        g = rb["GENNME"].str.upper().fillna("")
        hit = rb[g.str.contains("|".join(rb_pats))]
        sentinel = (vac.get("ndc_generics") or ["influenza vir (redbook)"])[0]
        add = pd.DataFrame({"generic_name": sentinel, "ndc_drug_code": hit["NDCNUM"].dropna().unique()})
        map_df = pd.concat([map_df, add], ignore_index=True)
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "generic_name": sentinel, "source": "redbook_gennme",
            "match_mode": f"contains:{'|'.join(rb_pats)}", "ingredient_targets": "",
            "n_ndc": len(add), "status": "MATCHED" if len(add) else "UNMATCHED"}])], ignore_index=True)
        logging.info("%sRedbook vaccine route: %d NDCs under '%s'; matched GENNME values:",
                     INDENT1, len(add), sentinel)
        for name, n in hit["GENNME"].str.upper().value_counts().items():
            logging.info("%s  %5d  %s", INDENT1, n, name)

    map_csv = os.path.join(out_dir, "drug_ndc_map.csv")
    sum_csv = os.path.join(out_dir, "resolution_summary.csv")
    map_df.to_csv(map_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)
    n_unmatched = int((summary_df["status"] == "UNMATCHED").sum())
    logging.info("%sWrote %s (%s rows) and %s", INDENT1, map_csv, f"{len(map_df):,}", sum_csv)
    if n_unmatched:
        logging.warning("%s%d generic(s) UNMATCHED -- see resolution_summary.csv before the real run",
                        INDENT1, n_unmatched)

    exp_gen = {g.strip().lower() for g in codesets.exposure_generics(cohort_id)}
    exp_map = map_df[map_df["generic_name"].str.strip().str.lower().isin(exp_gen)] if exp_gen \
        else map_df.iloc[0:0]
    if not exp_map.empty:
        # Redbook-resolved vaccine rows carry no n_ingredients (not ingredient-resolved) -- treat as 1,
        # same as the single_ingredient_only reader below.
        is_combo = pd.to_numeric(exp_map["n_ingredients"], errors="coerce").fillna(1).astype(int) > 1
        logging.info("%sexposure arms: %d of %d resolved NDC are fixed-dose COMBINATIONS "
                     "(dropped when exposure.single_ingredient_only=true)",
                     INDENT1, int(is_combo.sum()), len(exp_map))
        for gen, grp in exp_map.groupby("generic_name"):
            nc = int((pd.to_numeric(grp["n_ingredients"], errors="coerce").fillna(1).astype(int) > 1).sum())
            if nc:
                logging.info("%s%-24s %d combination NDC of %d", INDENT2, gen, nc, len(grp))

    dx_df = build_dx_proc_artifact(cfg)
    dx_csv = os.path.join(out_dir, "dx_proc_codes.csv")
    dx_df.to_csv(dx_csv, index=False)
    logging.info("%sdx/proc defining codes by role:", INDENT1)
    for role, grp in dx_df.groupby("role"):
        meta = ""
        if role.startswith("outcome_"):
            p = grp["position"].iloc[0] or "any"
            meta = f"  (setting={grp['settings'].iloc[0] or '-'}, position={p})"
        logging.info("%s%-20s %d code(s)%s", INDENT2, role, len(grp), meta)
    logging.info("%sWrote %s (%s rows)", INDENT1, dx_csv, f"{len(dx_df):,}")


# =============================================================================
# build_cohort-side lookup: read the artifacts back
# =============================================================================
def _load_csv(path: str) -> pd.DataFrame:
    # dtype=str: codes are string identifiers, not numbers (531.0 must not become 5310 via float).
    return pd.read_csv(path, dtype=str)


class MarketScanResolved(ResolvedCodes):
    """Reads resolve_codes's drug_ndc_map.csv + dx_proc_codes.csv under {codes_resolved_root}/{cohort_id}."""

    def __init__(self, codes_resolved_root: str, cohort_id: str):
        self.cohort_id = cohort_id
        self.dir = os.path.join(codes_resolved_root, cohort_id)
        self._ndc_map = None
        self._dx = None

    def _ndc(self) -> pd.DataFrame:
        if self._ndc_map is None:
            p = os.path.join(self.dir, "drug_ndc_map.csv")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Resolved NDC map not found: {p}. Run resolve_codes.py before build_cohort.")
            self._ndc_map = _load_csv(p)
        return self._ndc_map

    def _dxproc(self) -> pd.DataFrame:
        if self._dx is None:
            p = os.path.join(self.dir, "dx_proc_codes.csv")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Resolved dx/proc codes not found: {p}. Run resolve_codes.py before build_cohort/build_long_table.")
            self._dx = _load_csv(p)
        return self._dx

    def dx(self, role: str) -> List[str]:
        d = self._dxproc()
        keys = d.loc[d["role"] == role, "match_key"].dropna()
        return sorted({str(k).strip() for k in keys if str(k).strip()})

    def predefined_dx(self, codes: List[str]) -> List[str]:
        # decimal-stripped ICD-9 prefixes (matches attach_predefined_clinical's inline strip)
        return [str(c).replace(".", "") for c in codes]

    def drug(self, generics: List[str], single_ingredient_only: bool = False) -> List[str]:
        want = {g.strip().lower() for g in (generics or []) if g and g.strip()}
        if not want:
            return []
        m = self._ndc()
        m = m.assign(_gl=m["generic_name"].str.strip().str.lower())
        sub = m[m["_gl"].isin(want)]
        missing = sorted(want - set(sub["_gl"]))
        if missing:
            logging.warning("%sgenerics with NO resolved NDC in %s (run/refresh resolve_codes?): %s",
                            INDENT2, os.path.join(self.dir, "drug_ndc_map.csv"), missing)
        if single_ingredient_only:
            if "n_ingredients" in sub.columns:
                n_ing = pd.to_numeric(sub["n_ingredients"], errors="coerce").fillna(1)
                n_before = len(sub)
                sub = sub[n_ing <= 1]
                dropped = n_before - len(sub)
                if dropped:
                    logging.info("%ssingle_ingredient_only: dropped %d combination NDC (kept %d) for %s",
                                 INDENT2, dropped, len(sub), sorted(want))
            else:
                logging.warning("%ssingle_ingredient_only requested but 'n_ingredients' absent in %s "
                                "-- re-run resolve_codes to annotate. NOT filtering combinations.",
                                INDENT2, os.path.join(self.dir, "drug_ndc_map.csv"))
        return sorted(sub["ndc_drug_code"].dropna().unique().tolist())
