#!/usr/bin/env python3
"""
codesets.py  --  all study code lists as Python dicts.

The single place that defines the exposures / comparators / washout / exclusions / outcomes /
baseline clinical flags for each cohort. resolve_codes/build_cohort/build_long_table import from
here; there are no codeset CSVs.

Conventions
-----------
* DRUG sets are `{generic_name: ingredient_alias}` where the value is the exact
  `rxnorm_ingredient_name` to match when the RxNorm rollup is spelled differently
  (e.g. mefenamic acid -> "mefenamate", dabigatran -> "dabigatran etexilate"); None matches the
  generic name itself. resolve_codes resolves these to NDC codes.
* DIAGNOSIS / PROCEDURE sets are `{code: meaning}` in NATIVE codes (ICD-9-CM / CPT / HCPCS /
  ICD-9 vol.3). Codes are matched by decimal-stripped PREFIX against claims (531.0 -> 5310).
* BASELINE clinical flags (l_*) are `{flag_name: {"domain": ..., "codes": [...]}}` with
  domain in {dx_icd9, drug_generic, derived}. `derived` flags (hcru_intensity) carry no
  codes; build_cohort computes them. They feed the Table 1 summaries, not a PS spec.
"""

# =============================================================================
# positive_control -- celecoxib vs non-selective NSAID -> severe GI complication
# =============================================================================

# ---- drug sets: generic_name -> rxnorm ingredient alias (None = match the generic name) ----
_V1_TREATMENT = {"celecoxib": None}                    # the only US COX-2 across the index era
_V1_TREATMENT_SENSITIVITY = {"etoricoxib": None}       # rare in the US; counted in the washout and code drop
_V1_COMPARATOR = {                                     # 14 ns-NSAIDs, as in the reference
    "ibuprofen": None, "naproxen": None, "diclofenac": None,
    "meloxicam": None,               # COX-2-preferential; pre-specified NON-SELECTIVE
    "ketoprofen": None, "ketorolac": None, "indomethacin": None, "piroxicam": None,
    "sulindac": None, "etodolac": None, "nabumetone": None, "oxaprozin": None,
    "diflunisal": None,
    "mefenamic acid": "mefenamate",  # RxNorm ingredient rollup is 'mefenamate'
}
_V1_WASHOUT = {                                        # any NSAID or COX-2 (new-user gate)
    **_V1_TREATMENT, **_V1_TREATMENT_SENSITIVITY, **_V1_COMPARATOR,
    "aspirin_analgesic_dose": None,  # placeholder: analgesic-dose aspirin (no clean NDC signal)
    "fenoprofen": None, "flurbiprofen": None, "meclofenamate": None,
    "tolmetin": None, "salsalate": None,
}
_V1_ANTICOAGULANT = {                                  # co-medication set (resolved; not an exclusion)
    "warfarin": None, "dabigatran": "dabigatran etexilate", "rivaroxaban": None,
    "apixaban": None, "edoxaban": None, "enoxaparin": None,
}

# ---- outcome dx sets (native ICD-9-CM): code -> meaning ----
_V1_OUTCOME_BROAD = {                                  # PRIMARY (the reference's validated definition)
    "531": "Gastric ulcer (531.x)",
    "532": "Duodenal ulcer (532.x)",
    "533": "Peptic ulcer unspec (533.x)",
    "534": "Gastrojejunal ulcer (534.x)",
    "535": "Gastritis & duodenitis (535.x)",
    "578": "GI hemorrhage (578.x)",
}
_V1_OUTCOME_HEMORRHAGE = {                             # outpatient (physician-service) arm
    "578":   "GI hemorrhage (hematemesis/melena/unspecified) 578.x",
    "531.0": "Gastric ulcer - acute w hemorrhage",
    "531.2": "Gastric ulcer - acute w hemorrhage and perforation",
    "531.4": "Gastric ulcer - chronic/unspec w hemorrhage",
    "531.6": "Gastric ulcer - chronic/unspec w hemorrhage and perforation",
    "532.0": "Duodenal ulcer - acute w hemorrhage",
    "532.2": "Duodenal ulcer - acute w hemorrhage and perforation",
    "532.4": "Duodenal ulcer - chronic/unspec w hemorrhage",
    "532.6": "Duodenal ulcer - chronic/unspec w hemorrhage and perforation",
    "533.0": "Peptic ulcer - acute w hemorrhage",
    "533.2": "Peptic ulcer - acute w hemorrhage and perforation",
    "533.4": "Peptic ulcer - chronic/unspec w hemorrhage",
    "533.6": "Peptic ulcer - chronic/unspec w hemorrhage and perforation",
    "534.0": "Gastrojejunal ulcer - acute w hemorrhage",
    "534.2": "Gastrojejunal ulcer - acute w hemorrhage and perforation",
    "534.4": "Gastrojejunal ulcer - chronic/unspec w hemorrhage",
    "534.6": "Gastrojejunal ulcer - chronic/unspec w hemorrhage and perforation",
}
_V1_OUTCOME_NARROW = {                                 # alternative outcome set (not the paper's)
    **{f"531.{d}": f"Gastric ulcer (531.{d})" for d in range(7)},
    **{f"532.{d}": f"Duodenal ulcer (532.{d})" for d in range(7)},
    **{f"533.{d}": f"Peptic ulcer unspecified site (533.{d})" for d in range(7)},
    **{f"534.{d}": f"Gastrojejunal ulcer (534.{d})" for d in range(7)},
    "578": "GI hemorrhage (hematemesis/melena/unspecified) 578.x",
}

# ---- baseline clinical flags (l_*): flag -> {domain, codes} ----
_V1_PREDEFINED = {
    "prior_gi_ulcer_hx":  {"domain": "dx_icd9",      "codes": ["531", "532", "533", "534", "578"]},
    "gerd_gastritis":     {"domain": "dx_icd9",      "codes": ["530", "535"]},
    "h_pylori":           {"domain": "dx_icd9",      "codes": ["041.86"]},
    "anticoagulant_use":  {"domain": "drug_generic", "codes": ["warfarin", "apixaban", "rivaroxaban"]},
    "antiplatelet_use":   {"domain": "drug_generic", "codes": ["clopidogrel", "prasugrel", "ticagrelor"]},
    "corticosteroid_use": {"domain": "drug_generic", "codes": ["prednisone", "prednisolone", "methylprednisolone"]},
    "ppi_use":            {"domain": "drug_generic", "codes": ["omeprazole", "pantoprazole", "esomeprazole"]},
    "charlson_diabetes":  {"domain": "dx_icd9",      "codes": ["250"]},
    "charlson_chf":       {"domain": "dx_icd9",      "codes": ["428"]},
    "charlson_renal":     {"domain": "dx_icd9",      "codes": ["585"]},
    "charlson_liver":     {"domain": "dx_icd9",      "codes": ["571"]},
    "charlson_cancer":    {"domain": "dx_icd9",      "codes": ["140"]},  # malignancy 140-208 (prefix sample)
    "hcru_intensity":     {"domain": "derived",      "codes": []},       # baseline visit-count, computed in build_cohort
}

# =============================================================================
# negative_control -- influenza vaccination -> hip fracture
# =============================================================================

_V2_BISPHOSPHONATE = {                                 # baseline exclusion
    "alendronate": None, "risedronate": None, "ibandronate": None,
    "zoledronic acid": None, "pamidronate": None,
}

# Flu vaccine: the reference gives no code list, so this dict is the complete, authoritative
# definition. A person is VACCINATED if, in the Oct-Dec exposure window, they have ANY ONE of:
#   (a) a flu-vaccine product/administration procedure code below (HCPCS G0008, the era-valid
#       CPT 9065x/90662 products, and the Medicare product Q-codes Q2035-Q2039);
#   (b) a flu-vaccination diagnosis code (V04.81 / V06.6); or
#   (c) a pharmacy claim for an influenza-virus vaccine NDC (Redbook-resolved by resolve_codes).
# Every code is EXACT-matched. Generic vaccine-administration codes (90471-74/90460-61)
# deliberately do NOT count: they identify "some vaccine", not flu, and since product codes
# qualify alone they would add nobody. Realised uptake under this definition is well below the
# reference's 42% in a PACE population; the gap is structural to employer-supplemental claims,
# which never see Part-B-paid shots.
_V2_VACCINE_PROC = {                                   # HCPCS G0008 + era-valid CPT/HCPCS/ICD9proc
    "G0008": "Administration of influenza virus vaccine (HCPCS; flu-specific, counts alone)",
    "90654": "IIV3 preservative-free intradermal (CPT, 2011+)",
    "90655": "Flu vaccine split preservative-free 6-35mo (CPT)",
    "90656": "Flu vaccine split preservative-free >=3yr (CPT)",
    "90657": "Flu vaccine split 6-35mo (CPT)",
    "90658": "Flu vaccine split >=3yr (CPT; Medicare pays via Q-codes since 2010-11 but code persists on COB claims)",
    "90660": "Flu vaccine intranasal live (CPT)",
    "90661": "ccIIV3 Flucelvax (CPT, 2013+ seasons only; inert for s2011)",
    "90662": "Flu vaccine high-dose preservative-free (CPT; dominant product in 65+)",
    "90672": "LAIV4 FluMist Quadrivalent (CPT, 2013+; inert for s2011)",
    "90673": "RIV3 Flublok (CPT, payable 2014+; safety net)",
    "90685": "IIV4 split PF 6-35mo (CPT, 2013+; inert for s2011)",
    "90686": "IIV4 split PF >=3yr (CPT, 2013+; inert for s2011)",
    "90687": "IIV4 split 6-35mo (CPT, 2013+; inert for s2011)",
    "90688": "IIV4 split >=3yr (CPT, 2013+; inert for s2011)",
    "Q2033": "Flublok 2013 billing code (HCPCS; 2013 season only)",
    "Q2034": "Agriflu (HCPCS; 2013 season only)",
    "Q2035": "Afluria (HCPCS Medicare brand code, 2010-11+)",
    "Q2036": "FluLaval (HCPCS Medicare brand code, 2010-11+)",
    "Q2037": "Fluvirin (HCPCS Medicare brand code, 2010-11+)",
    "Q2038": "Fluzone (HCPCS Medicare brand code, 2010-11+; largest product share)",
    "Q2039": "Influenza vaccine NOS (HCPCS Medicare, 2010-11+)",
    "9952":  "Prophylactic vaccination against influenza (ICD-9 proc 99.52, facility-coded; exact match, no CPT collision)",
}
# Dx evidence of vaccination (always counts).
_V2_VACCINE_DX = {
    "V04.81": "Need for prophylactic vaccination against influenza (ICD-9 dx)",
    "V06.6":  "Need for vaccination against pneumococcus + influenza (ICD-9 dx)",
}
# Pharmacy route: resolved by resolve_codes against the MarketScan Redbook dictionary (GENNME contains
# 'INFLUENZA VIR', NOT bare 'INFLUENZA', which would add the H1N1/H5N1 pandemic products).
# resolve_codes writes the matched NDCs into drug_ndc_map.csv under this sentinel generic name.
_V2_VACCINE_NDC_GENERICS = ["influenza vir (redbook)"]
# Deliberately never counted (recorded for the record): PQRS attestations G8482-84 (G8483/84 mean
# NOT vaccinated), the 2009-10 H1N1 pandemic module (90663-68/90470/G914x), legacy 90724/90659,
# post-window 90630/90653.

_V2_OUTCOME_BASE = {                                   # 820.x hip fracture
    "820.0": "Fracture neck of femur - transcervical closed",
    "820.1": "Fracture neck of femur - transcervical open",
    "820.2": "Fracture neck of femur - pertrochanteric closed",
    "820.3": "Fracture neck of femur - pertrochanteric open",
    "820.8": "Fracture neck of femur - unspecified closed",
    "820.9": "Fracture neck of femur - unspecified open",
}
_V2_PREDEFINED = {
    "osteoporosis":         {"domain": "dx_icd9",      "codes": ["733.0"]},
    "pathologic_fracture":  {"domain": "dx_icd9",      "codes": ["733.1"]},
    "prior_fracture_hx":    {"domain": "dx_icd9",      "codes": ["800", "820"]},  # fractures 800-829 (sample)
    "fall_hx":              {"domain": "dx_icd9",      "codes": ["E888"]},
    "dementia":             {"domain": "dx_icd9",      "codes": ["290"]},
    "parkinson":            {"domain": "dx_icd9",      "codes": ["332"]},
    "vertigo_dizziness":    {"domain": "dx_icd9",      "codes": ["780.4"]},
    "visual_impairment":    {"domain": "dx_icd9",      "codes": ["369"]},
    "rheumatoid_arthritis": {"domain": "dx_icd9",      "codes": ["714"]},
    "bisphosphonate_hx":    {"domain": "drug_generic", "codes": ["alendronate"]},   # also an exclusion
    "vitamin_d":            {"domain": "drug_generic", "codes": ["ergocalciferol"]},
    "glucocorticoid_use":   {"domain": "drug_generic", "codes": ["prednisone"]},
    "hcru_intensity":       {"domain": "derived",      "codes": []},
}

# =============================================================================
# per-cohort registry + accessors (the API resolve_codes/build_cohort/build_long_table use)
# =============================================================================
_COHORTS = {
    "positive_control_celecoxib_gi": {
        "drugs": {
            "treatment": _V1_TREATMENT,
            "treatment_sensitivity": _V1_TREATMENT_SENSITIVITY,
            "comparator": _V1_COMPARATOR,
            "washout": _V1_WASHOUT,
            "anticoagulant": _V1_ANTICOAGULANT,
        },
        "exposure_roles": ["treatment", "treatment_sensitivity", "comparator"],  # dropped from covariates
        "outcome_sets": {"broad": _V1_OUTCOME_BROAD,
                         "hemorrhage": _V1_OUTCOME_HEMORRHAGE,
                         "narrow": _V1_OUTCOME_NARROW},
        "outcome_outpatient_set": "hemorrhage",   # physician-service arm
        "prior_exclusion_set": None,
        "predefined": _V1_PREDEFINED,
        "vaccine": None,
    },
    "negative_control_flu_hipfx": {
        "drugs": {
            "bisphosphonate": _V2_BISPHOSPHONATE,
        },
        "exposure_roles": [],                     # exposure is a vaccine, not a drug covariate
        "outcome_sets": {"base": _V2_OUTCOME_BASE},
        "outcome_outpatient_set": None,           # reuse primary list for the outpatient arm
        "prior_exclusion_set": "base",            # prior hip fx reuses the 820.x outcome codes
        "predefined": _V2_PREDEFINED,
        "vaccine": {"proc_codes": _V2_VACCINE_PROC, "dx_codes": _V2_VACCINE_DX,
                    "ndc_generics": _V2_VACCINE_NDC_GENERICS,
                    "redbook_gennme_contains": ["INFLUENZA VIR"]},
    },
}


def _cohort(cohort_id):
    if cohort_id not in _COHORTS:
        raise KeyError(f"Unknown cohort_id '{cohort_id}'. Known: {list(_COHORTS)}")
    return _COHORTS[cohort_id]


def drug_generics(cohort_id, role):
    """{generic_name: ingredient_alias} for one drug role (treatment/comparator/washout/
    anticoagulant/bisphosphonate/treatment_sensitivity). {} if the role is absent."""
    return _cohort(cohort_id)["drugs"].get(role, {})


def all_drug_generic_rows(cohort_id):
    """Yield (generic_name, ingredient_alias, source) over EVERY drug the cohort references
    -- drug roles + baseline drug_generic flags + vaccine ndc_generics. resolve_codes collects these to
    resolve NDC. A generic used by several sources is yielded once per source (resolve_codes dedups)."""
    c = _cohort(cohort_id)
    for role, d in c["drugs"].items():
        for generic, alias in d.items():
            yield generic, alias, role
    for flag, spec in c["predefined"].items():
        if spec["domain"] == "drug_generic":
            for generic in spec["codes"]:
                yield generic, None, "predefined_clinical"
    vac = c.get("vaccine")
    if vac:
        for generic in vac.get("ndc_generics", []):
            yield generic, None, "vaccinated_ndc"


def exposure_generics(cohort_id):
    """All exposure-defining drug generics (dropped from the hdPS covariate pool in build_long_table)."""
    c = _cohort(cohort_id)
    out = []
    for role in c["exposure_roles"]:
        out += list(c["drugs"].get(role, {}))
    vac = c.get("vaccine")
    if vac:
        out += list(vac.get("ndc_generics", []))
    return out


def outcome_set(cohort_id, name):
    """{code: meaning} for a named outcome set (e.g. 'broad', 'base', 'hemorrhage', 'narrow')."""
    return _cohort(cohort_id)["outcome_sets"][name]


def outcome_outpatient_set(cohort_id):
    """{code: meaning} for the outpatient (physician-service) arm, or None to reuse the primary."""
    c = _cohort(cohort_id)
    k = c.get("outcome_outpatient_set")
    return c["outcome_sets"][k] if k else None


def prior_exclusion_set(cohort_id):
    """{code: meaning} for a prior-condition dx exclusion (negative_control prior hip fx), or None."""
    c = _cohort(cohort_id)
    k = c.get("prior_exclusion_set")
    return c["outcome_sets"][k] if k else None


def predefined(cohort_id):
    """{flag_name: {'domain': ..., 'codes': [...]}} baseline clinical flags (l_*)."""
    return _cohort(cohort_id)["predefined"]


def vaccine(cohort_id):
    """{'proc_codes': {code: meaning}, 'dx_codes': {code: meaning}, 'ndc_generics': [...],
    'redbook_gennme_contains': [...]} or None (non-vaccine cohorts)."""
    return _cohort(cohort_id).get("vaccine")
