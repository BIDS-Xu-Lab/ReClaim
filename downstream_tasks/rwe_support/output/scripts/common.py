"""common.py -- shared paths, labels, formatting and published reference values for the RWE output scripts.

Every table and figure in output/ is computed from the two pipelines' own results:
  study1/results/mdcr/...                          Study 1, the known-effect validation (MarketScan MDCR)
  study2/marketscan/results/...                    Study 2, the target trial emulation on MarketScan claims
  study2/ynhh/results/ynhh_src/...                 Study 2 on the YNHH OMOP EHR
The only hand-entered numbers are the published estimates of the two reference studies, defined below with their
source (table or figure) so they can be checked against the original papers."""
import csv, json, os

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BENCH = os.path.join(REPO, "study1")
TTE = {"MS": os.path.join(REPO, "study2", "marketscan"), "YNHH": os.path.join(REPO, "study2", "ynhh")}
TTE_RESULTS = {"MS": os.path.join(TTE["MS"], "results"), "YNHH": os.path.join(TTE["YNHH"], "results", "ynhh_src")}
TTE_SUMMARY = {"MS": os.path.join(TTE["MS"], "results", "_summary"), "YNHH": os.path.join(TTE["YNHH"], "results", "_summary_ynhh_src")}
OUT_TABLES = os.path.join(REPO, "output", "tables")
OUT_FIGURES = os.path.join(REPO, "output", "figures")

# ---- Study 1: cohorts and specifications --------------------------------------------------------------------------
POS, NEG = "positive_control_celecoxib_gi", "negative_control_flu_hipfx"
CELL = {POS: "MDCR", NEG: "MDCR_s2011"}
S1_RESULTS = os.path.join(BENCH, "results", "mdcr")        # Study 1 (MarketScan MDCR) results and
S1_INTER = os.path.join(BENCH, "intermediate", "mdcr")     # person-level cohort files

# ---- Study 2: comparisons, outcomes, specifications ---------------------------------------------------------------
COMPARISONS = [("glp1_vs_sglt2", "vs SGLT2i"), ("glp1_vs_dpp4", "vs DPP4i")]
OUTCOMES = [("anxiety_disorders", "Anxiety"), ("depression", "Depression"), ("bipolar_disorder", "Bipolar"),
            ("other_mood_disorders", "Other mood"), ("schizophrenia_and_other_psychotic_disorders", "Schizophrenia+"),
            ("suicide_ideation", "Suicidal ideation")]
TTE_SPECS = ["crude", "demo", "paper", "hdps", "reclaim-s-only", "reclaim-m-only", "reclaim-l-only"]

# manuscript labels; "paper" is the pipeline's id for the reference study's own covariate set (Expert)
LABEL = {"crude": "Unadjusted", "demo": "Demographics", "paper": "Expert", "hdps": "hdPS", "delphi-only": "Delphi",
         "reclaim-s-only": "ReClaim-S", "reclaim-m-only": "ReClaim-M", "reclaim-l-only": "ReClaim-L"}

# ---- Reference study 1: Schneeweiss et al., Epidemiology 2009 (PACE claims) ---------------------------------------------
# Table 3 (positive control) and the influenza-vaccine example (negative control): relative risk (lower, upper);
# hdPS = model 5b (demographics + 500 empirical covariates, no investigator-specified covariates), the specification our
# hdPS rung reproduces. c-statistics as reported (demographics 0.61, hdPS 0.71; positive control only).
SCHNEEWEISS_RR = {POS: {"crude": (1.09, 0.91, 1.30), "demo": (1.01, 0.84, 1.21), "hdps": (0.87, 0.72, 1.05)},
                  NEG: {"crude": (0.93, 0.80, 1.08), "hdps": (1.02, 0.85, 1.21)}}
SCHNEEWEISS_C = {POS: {"demo": "0.61", "hdps": "0.71"}, NEG: {}}
# Cohort description (Table 1 and Results text): pooled over both exposure arms; nr = not reported.
# (a) age is reported only as the share aged 75 or older (71.6%); (b) any gastroprotective drug (ours: proton-pump
# inhibitor use); (c) peptic ulcer disease / GI hemorrhage reported separately (ours: one flag over the 180-day baseline).
SCHNEEWEISS_COHORT = {
    POS: {"era": "1999-2002", "patients": "49,653", "exposed": "32,042 (64.5)", "age": "nr (a)", "female": "84.2",
          "covariate_1": "24.9", "covariate_2": "3.3 / 1.5", "events": "552 (1.1)", "exposed_events": "367 (1.2)"},
    NEG: {"era": "Oct-Dec 1996", "patients": "147,583", "exposed": "nr (42.0)", "age": "nr", "female": "nr",
          "covariate_1": "nr", "covariate_2": "nr", "events": "710 (0.48)", "exposed_events": "nr"}}

# ---- Reference study 2: Tang et al., Diabetes Res Clin Pract (Penn Medicine EHR), the emulated target trial ------------
# Fig. 1: hazard ratio (lower, upper), raw then calibrated, per outcome and comparison
TANG_HR = {("anxiety_disorders", "glp1_vs_sglt2"): (1.15, 1.04, 1.26, 1.06, 0.84, 1.35), ("anxiety_disorders", "glp1_vs_dpp4"): (1.17, 1.07, 1.29, 1.12, 1.02, 1.23),
           ("depression", "glp1_vs_sglt2"): (0.99, 0.88, 1.12, 0.92, 0.72, 1.18), ("depression", "glp1_vs_dpp4"): (0.97, 0.87, 1.09, 0.93, 0.83, 1.04),
           ("bipolar_disorder", "glp1_vs_sglt2"): (1.55, 1.00, 2.41, 1.44, 0.88, 2.35), ("bipolar_disorder", "glp1_vs_dpp4"): (0.93, 0.61, 1.43, 0.89, 0.58, 1.37),
           ("other_mood_disorders", "glp1_vs_sglt2"): (0.93, 0.64, 1.35, 0.86, 0.56, 1.33), ("other_mood_disorders", "glp1_vs_dpp4"): (0.84, 0.57, 1.23, 0.80, 0.55, 1.18),
           ("schizophrenia_and_other_psychotic_disorders", "glp1_vs_sglt2"): (1.48, 0.84, 2.59, 1.37, 0.75, 2.49), ("schizophrenia_and_other_psychotic_disorders", "glp1_vs_dpp4"): (0.74, 0.45, 1.19, 0.70, 0.43, 1.14),
           ("suicide_ideation", "glp1_vs_sglt2"): (1.08, 0.47, 2.47, 1.00, 0.43, 2.36), ("suicide_ideation", "glp1_vs_dpp4"): (0.63, 0.31, 1.30, 0.60, 0.29, 1.24)}
# Fig. 1: GLP-1RA events (vs SGLT2i, vs DPP4i)
TANG_EVENTS = {"depression": (492, 568), "anxiety_disorders": (799, 932), "bipolar_disorder": (43, 40),
               "schizophrenia_and_other_psychotic_disorders": (25, 28), "other_mood_disorders": (47, 48), "suicide_ideation": (10, 12)}
# Supplement, 1:1 propensity-score matching sensitivity analysis: GLP-1RA events, HR, lower, upper
TANG_PSM = {("depression", "glp1_vs_sglt2"): (401, 1.07, 0.93, 1.23), ("depression", "glp1_vs_dpp4"): (345, 1.01, 0.87, 1.18),
            ("anxiety_disorders", "glp1_vs_sglt2"): (644, 1.17, 1.05, 1.31), ("anxiety_disorders", "glp1_vs_dpp4"): (549, 1.18, 1.05, 1.34),
            ("bipolar_disorder", "glp1_vs_sglt2"): (28, 1.55, 0.86, 2.80), ("bipolar_disorder", "glp1_vs_dpp4"): (33, 1.44, 0.85, 2.46),
            ("schizophrenia_and_other_psychotic_disorders", "glp1_vs_sglt2"): (25, 1.56, 0.84, 2.93), ("schizophrenia_and_other_psychotic_disorders", "glp1_vs_dpp4"): (21, 1.25, 0.66, 2.36),
            ("other_mood_disorders", "glp1_vs_sglt2"): (36, 1.16, 0.71, 1.87), ("other_mood_disorders", "glp1_vs_dpp4"): (35, 1.01, 0.63, 1.61),
            ("suicide_ideation", "glp1_vs_sglt2"): (9, 1.13, 0.43, 2.92), ("suicide_ideation", "glp1_vs_dpp4"): (7, 0.64, 0.25, 1.65)}
TANG_PSM_PAIRS = {"glp1_vs_sglt2": 5506, "glp1_vs_dpp4": 4349}

# ---- I/O and formatting ---------------------------------------------------------------------------------------------
def read_csv(path): return list(csv.DictReader(open(path)))
def read_json(path): return json.load(open(path))
def ladder_estimate(cohort, spec):
    """decile-adjusted estimate.json of one Study 1 ladder cell, or None if that specification was not run"""
    p = os.path.join(S1_RESULTS, cohort, CELL[cohort], spec, "decile_adjusted", "estimate.json")
    return read_json(p) if os.path.exists(p) else None
def tte_outcomes(src):
    return {(r["comparison"], r["spec"], r["estimator"], r["outcome"]): r for r in read_csv(os.path.join(TTE_SUMMARY[src], "summary_outcomes.csv"))}
def tte_diagnostics(src):
    return {(r["comparison"], r["spec"], r["estimator"]): r for r in read_csv(os.path.join(TTE_SUMMARY[src], "summary_diagnostics.csv"))}
def num(x, nd): return f"{float(x):.{nd}f}"
def ci(est, lb, ub, nd=3, nd_ci=None):
    nd_ci = nd if nd_ci is None else nd_ci
    return f"{float(est):.{nd}f} ({float(lb):.{nd_ci}f}-{float(ub):.{nd_ci}f})"
def missing(x): return x is None or x in ("", "NA", "nan")

def write_table(name, rows, note=None):
    """rows[0] is the header. Writes tables/<name>.csv and a Markdown copy tables/<name>.md."""
    os.makedirs(OUT_TABLES, exist_ok=True)
    with open(os.path.join(OUT_TABLES, f"{name}.csv"), "w", newline="") as f: csv.writer(f).writerows(rows)
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    with open(os.path.join(OUT_TABLES, f"{name}.md"), "w") as f: f.write("\n".join(md) + ("\n\n" + note if note else "") + "\n")
    print(f"  tables/{name}.csv  ({len(rows) - 1} rows)")

def write_optional(name, build):
    """Like write_table, for the cohort tables that summarise the person-level intermediate/ cohort files, which are not
    distributed: when such a file is absent, skip the table (the released copy in output/tables/ is kept). Any other
    missing input is an error."""
    try: rows = build()
    except FileNotFoundError as e:
        if "/intermediate/" not in str(e.filename): raise
        print(f"  tables/{name}.csv  skipped: needs {os.path.relpath(e.filename, REPO)} (person-level, not distributed)")
        return
    write_table(name, rows)

# Tang et al. cohort description (Supplement Tables S5-S6, before weighting), per emulation: GLP-1RA then comparator.
# Medicaid is not applicable to an EHR source; median follow-up is not reported.
TANG_COHORT = {
    "glp1_vs_sglt2": {"n": ("7,308", "10,119"), "age": ("58.3 (12.7)", "64.7 (12.0)"), "female": ("49.3%", "33.6%"),
                      "anxiety_disorders": ("10.9%", "9.6%"), "depression": ("6.7%", "6.7%"), "bipolar_disorder": ("0.59%", "0.38%"),
                      "schizophrenia_and_other_psychotic_disorders": ("0.34%", "0.23%"), "other_mood_disorders": ("0.64%", "0.68%"),
                      "suicide_ideation": ("0.14%", "0.13%")},
    "glp1_vs_dpp4": {"n": ("7,300", "7,394"), "age": ("58.2 (12.7)", "67.0 (13.1)"), "female": ("49.3%", "43.6%"),
                     "anxiety_disorders": ("12.8%", "11.1%"), "depression": ("7.8%", "8.1%"), "bipolar_disorder": ("0.55%", "0.60%"),
                     "schizophrenia_and_other_psychotic_disorders": ("0.38%", "0.54%"), "other_mood_disorders": ("0.66%", "0.78%"),
                     "suicide_ideation": ("0.16%", "0.26%")}}

# The schizophrenia cell (GLP-1RA vs DPP4i, calibrated HR) outside this pipeline, kept for the response's comparison
# with the design of record: (version, data source, cohort, events GLP-1RA vs DPP4i, Expert, ReClaim-L). The first submission (1:1 matched,
# 1M held-out split) had no Expert specification; its events are those of the matched ReClaim-L cohort. The second row
# reruns this pipeline unchanged on the same 1M split (the val half of the held-out cohort). The reference row is the
# published estimate (Tang et al., Fig. 1B: 28/7,300 vs 40/7,394 events; the figure's comparator header reads "SGLT2is").
SCHIZOPHRENIA_EXTERNAL = [
    ("Reference study (published)", "Penn Medicine (EHR)", "Tang et al., 2026", "28 vs 40", "0.70 (0.43-1.14)", "--"),
    ("First submission (original implementation)", "MarketScan (claims)", "1M held-out split", "5 vs 19", "--", "0.25 (0.09-0.67)"),
    ("Revision (reference study's protocol)", "MarketScan (claims)", "Same 1M held-out split", "1 vs 4", "0.09 (0.01-0.98)", "0.10 (0.01-0.92)")]
