#!/usr/bin/env python3
"""
codesets_tte.py  --  code-set reader for the GLP-1 TTE replication.

Single reader for the PennCIL lookup CSVs (the reference study's group) under lookup_table/:

  outcome_cov_icd.csv   cluster,cat,cat_sub,icd-10   6 primary psychiatric outcomes (cat=Outcome)
                                                     + T1D / T2D / ESRD definitions (cat=Covariate)
  glp_nco_icd_0225.csv  name,ICD-10                  90 curated negative control outcomes
  cluster_master.csv    concept_code,vocabulary_id,cluster   441 condition clusters (66k codes)
  huilin_med.csv        cluster,name_list            26 medication classes (name substrings)

Codes are normalized to decimal-stripped UPPERCASE prefixes (`match_key`), the convention
the claims matcher uses (raw DX columns carry no decimal).
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUP = os.path.join(HERE, "lookup_table")

# The three study drug classes (RxNorm ingredient names, lowercase); resolve_codes maps
# these to NDC sets via the NDC->RxNorm ingredient map.
EXPOSURE_INGREDIENTS = {
    "glp1ra": ["albiglutide", "dulaglutide", "exenatide", "liraglutide",
               "lixisenatide", "semaglutide", "tirzepatide"],
    "sglt2i": ["canagliflozin", "dapagliflozin", "empagliflozin", "ertugliflozin",
               "sotagliflozin", "bexagliflozin"],
    "dpp4i":  ["sitagliptin", "saxagliptin", "alogliptin", "linagliptin"],
}

# The reference's 6 primary psychiatric outcomes (cluster names in outcome_cov_icd.csv).
PRIMARY_OUTCOMES = [
    "Depression",
    "Anxiety_disorders",
    "Bipolar_disorder",
    "Schizophrenia_and_other_psychotic_disorders",
    "Other_mood_disorders",
    "Suicide_ideation",
]


def _norm(code: str) -> str:
    """ICD-10 code -> decimal-stripped uppercase match key (prefix semantics)."""
    return code.strip().replace(".", "").upper()


def _split_codes(cell: str):
    return [c for c in (x.strip() for x in cell.split(",")) if c]


def outcome_sets():
    """{outcome_name: [match_key, ...]} for the 6 primary psychiatric outcomes."""
    sets = {name: [] for name in PRIMARY_OUTCOMES}
    with open(os.path.join(LOOKUP, "outcome_cov_icd.csv")) as f:
        for row in csv.DictReader(f):
            if row["cat"] == "Outcome" and row["cluster"] in sets:
                sets[row["cluster"]].append(_norm(row["icd-10"]))
    missing = [k for k, v in sets.items() if not v]
    if missing:
        raise ValueError(f"outcome_cov_icd.csv missing codes for: {missing}")
    return sets


def covariate_exclusion_sets():
    """{T1D: [...], T2D: [...], ESRD: [...]} match keys (eligibility/exclusion definitions)."""
    sets = {}
    with open(os.path.join(LOOKUP, "outcome_cov_icd.csv")) as f:
        for row in csv.DictReader(f):
            if row["cat"] == "Covariate":
                sets.setdefault(row["cluster"], []).append(_norm(row["icd-10"]))
    for need in ("T1D", "T2D", "ESRD"):
        if need not in sets:
            raise ValueError(f"outcome_cov_icd.csv missing Covariate cluster {need}")
    return sets


def nco_sets():
    """{nco_name: [match_key, ...]} for the 90 curated negative control outcomes."""
    sets = {}
    with open(os.path.join(LOOKUP, "glp_nco_icd_0225.csv")) as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            keys = [_norm(c) for c in _split_codes(row["ICD-10"])]
            if name and keys:
                sets[name] = keys
    return sets


def cluster_master():
    """[(match_key, cluster_name), ...] for the 441 baseline condition clusters (66k codes)."""
    out = []
    with open(os.path.join(LOOKUP, "cluster_master.csv")) as f:
        for row in csv.DictReader(f):
            code = _norm(row["concept_code"])
            cl = row["cluster"].strip()
            if code and cl:
                out.append((code, cl))
    return out


def med_classes():
    """{class_name: [name_term, ...]} -- lowercase substring terms matched against
    rxnorm_ingredient_name (resolve_codes). Truncated terms in the source CSV are
    intentional (substring semantics)."""
    out = {}
    with open(os.path.join(LOOKUP, "huilin_med.csv")) as f:
        for row in csv.DictReader(f):
            cl = row["cluster"].strip().lower().replace(" ", "_")
            # split on commas and slashes; the one hyphenated combo entry maps to its UNIQUE
            # component (sacubitril) -- splitting it would sweep plain valsartan into ARNIs
            raw = row["name_list"].lower().replace("/", ",")
            raw = raw.replace("sacubitril-valsartan", "sacubitril")
            terms = [t.strip() for t in raw.split(",") if t.strip()]
            if cl and terms:
                # a class listed twice (Thiazolidinediones appears 2x) merges its terms
                out.setdefault(cl, [])
                out[cl] = sorted(set(out[cl]) | set(terms))
    return out


def outcome_drop_keys_3char():
    """3-char truncations of every primary-outcome code, dropped from the hdPS candidate pool
    (outcome-defining codes must not become PS covariates). NCO codes stay: they are
    baseline-morbidity proxies and outcomes only post-index."""
    keys = set()
    for codes in outcome_sets().values():
        keys.update(c[:3] for c in codes)
    return sorted(keys)
