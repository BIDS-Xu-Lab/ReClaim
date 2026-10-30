#!/usr/bin/env python3
"""study1_tables.py -- the tables of Study 1, the known-effect validation (Schneeweiss et al. 2009 replicated on
MarketScan Medicare-supplemental claims): a positive control (celecoxib vs nonselective NSAIDs, gastrointestinal complications;
a protective effect expected from randomized trials) and a negative control (influenza vaccination, hip fracture;
expected RR = 1).

Writes to output/tables/:
  s1_cohort_positive, s1_cohort_negative      cohort description (MarketScan MDCR), beside the reference
  s1_cohort_by_exposure_positive, _negative   per-exposure-arm characteristics with standardized mean differences
  s1_ladder_positive, s1_ladder_negative      specification ladder (decile-adjusted RR) and PS c-statistic, beside the reference
  s1_pca_positive, s1_pca_negative            dimension-matched PCA compressions of the ReClaim representations
  s1_doubly_robust_positive, _negative        doubly robust panel: cross-fitted AIPW, effect in the treated"""
import csv, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (POS, NEG, CELL, S1_RESULTS, S1_INTER, LABEL, SCHNEEWEISS_RR, SCHNEEWEISS_C, SCHNEEWEISS_COHORT,
                    read_csv, ladder_estimate, ci, num, missing, write_table, write_optional)

MS = "MarketScan MDCR"
LADDER = ["crude", "demo", "hdps", "delphi-only", "reclaim-s-only", "reclaim-m-only", "reclaim-l-only"]

def summary():
    return {(r["cohort"], r["spec"]): r for r in read_csv(os.path.join(S1_RESULTS, "_summary", "supplementary_table.csv"))}

# ---- cohort description ---------------------------------------------------------------------------------------------
def cohort_stats(cohort):
    """cohort-level description computed from the cohort file (aggregates only)"""
    path = os.path.join(S1_INTER, "cohorts", cohort, CELL[cohort], "cohort.csv")
    cov1, cov2 = ("l_ppi_use", "l_prior_gi_ulcer_hx") if cohort == POS else ("l_osteoporosis", "l_glucocorticoid_use")
    n = n1 = fem = c1 = c2 = ev = ev1 = 0; s = ss = 0.0
    for r in csv.DictReader(open(path)):
        a = r["arm"] == "1"; y = r["outcome"] == "1"; age = float(r["age_at_index"])
        n += 1; n1 += a; fem += r["sex"] == "2"; c1 += r[cov1] == "1"; c2 += r[cov2] == "1"; ev += y; ev1 += a and y
        s += age; ss += age * age
    mean = s / n; sd = math.sqrt((ss - n * mean * mean) / (n - 1))
    return dict(n=n, n1=n1, female=100 * fem / n, age=(mean, sd), c1=100 * c1 / n, c2=100 * c2 / n, ev=ev, ev1=ev1)

def cohort_table(cohort):
    ref = SCHNEEWEISS_COHORT[cohort]; x = cohort_stats(cohort)
    era = "2011-2014" if cohort == POS else "Oct-Dec 2011"
    rows = [["Characteristic", f"Reference (PACE, {ref['era']})", f"{MS}, {era}"]]
    evd = 1 if cohort == POS else 2           # the rarer outcome is shown to two decimals
    if cohort == POS:
        labels = ["Patients, n", "Celecoxib initiators (reference: any COX-2 inhibitor), n (%)", "Age at index, years, mean (SD)",
                  "Female sex, %", "Gastroprotective drug use, % (b)", "Prior peptic ulcer disease or GI hemorrhage, % (c)",
                  "GI complication events, n (%)", "Events in the exposed arm, n (%)"]
    else:
        labels = ["Patients, n", "Vaccinated, n (%)", "Age at index, years, mean (SD)", "Female sex, %", "Osteoporosis, %",
                  "Glucocorticoid use, %", "Hip fracture events, n (%)", "Events in the vaccinated arm, n (%)"]
    keys = ["patients", "exposed", "age", "female", "covariate_1", "covariate_2", "events", "exposed_events"]
    for lab, k in zip(labels, keys):
        rows.append([lab, ref[k], {"patients": f"{x['n']:,}", "exposed": f"{x['n1']:,} ({100 * x['n1'] / x['n']:.1f})",
                                   "age": f"{x['age'][0]:.1f} ({x['age'][1]:.1f})", "female": f"{x['female']:.1f}",
                                   "covariate_1": f"{x['c1']:.1f}", "covariate_2": f"{x['c2']:.1f}",
                                   "events": f"{x['ev']:,} ({100 * x['ev'] / x['n']:.{evd}f})",
                                   "exposed_events": f"{x['ev1']:,} ({100 * x['ev1'] / x['n1']:.{evd}f})"}[k]])
    return rows

def by_exposure(cohort, exp, comp):
    """the pipeline's per-exposure-arm characteristics (results/mdcr/_summary/cohort_characteristics.csv)"""
    rows = [["Characteristic"] + [f"{MS}: {h}" for h in (exp, comp, "SMD")]]
    for r in read_csv(os.path.join(S1_RESULTS, "_summary", "cohort_characteristics.csv")):
        if r["cohort"] == cohort: rows.append([r["characteristic"], r[exp], r[comp], r["SMD"]])
    return rows

# ---- specification ladder ---------------------------------------------------------------------------------------------
def ladder_table(cohort):
    S = summary()
    rows = [["PS specification", MS, "Reference", f"c-statistic, {MS}", "c-statistic, reference"]]
    for spec in LADDER:
        r = S.get((cohort, spec))
        est, cst = ("-", "-") if r is None else (ci(r["rr"], r["lb"], r["ub"], 3, 3),
                                               "none (no PS)" if missing(r["c_stat"]) else num(r["c_stat"], 3))
        ref = SCHNEEWEISS_RR[cohort].get(spec)
        refc = SCHNEEWEISS_C[cohort].get(spec, "nr" if ref else "-")
        rows.append([LABEL[spec], est, ci(*ref, 2) if ref else "-", cst, refc])
    return rows

# ---- PCA compressions -------------------------------------------------------------------------------------------------
def pca_table(cohort):
    def est(spec):
        j = ladder_estimate(cohort, spec); return ci(j["rr"], j["rr_lb"], j["rr_ub"], 3)
    base = {"60": "(application reference expert count)", "120": f"Delphi: {est('delphi-only')}", "500": f"hdPS: {est('hdps')}",
            "full": "(native width 1024/2048/2048)"}
    rows = [["k", "Dimension-matched baseline", "PCA ReClaim-S", "PCA ReClaim-M", "PCA ReClaim-L"]]
    for k in ("60", "120", "500", "full"):
        rows.append([k, base[k]] + [est(f"reclaim-{m}-only" if k == "full" else f"reclaim-{m}-pca{k}-only") for m in "sml"])
    return rows

# ---- doubly robust panel ----------------------------------------------------------------------------------------------
def dr_table(cohort):
    """effect in the treated; reported rows: estimator aipw_att and iptw_att_hajek, weights truncated at their 1st/99th
    percentiles (truncation wpctl_1_99)"""
    G = read_csv(os.path.join(S1_RESULTS, cohort, CELL[cohort], "dr_aipw", "dr_estimators_grid.csv"))
    def cell(ps, om, est):
        r = [x for x in G if x["ps_spec"] == ps and x["om"] == om and x["estimator"] == est and x["truncation"] == "wpctl_1_99"][0]
        return ci(r["rr"], r["rr_lb"], r["rr_ub"], 3)
    OMS = [("om-delphi", "Delphi"), ("om-reclaim-s", "ReClaim-S"), ("om-reclaim-m", "ReClaim-M"), ("om-reclaim-l", "ReClaim-L")]
    rows = [["PS model \\ outcome model", "None (PS only)"] + [f"{lab} outcome model" for _, lab in OMS]]
    for ps in ("hdps", "delphi-only", "reclaim-s-only"):
        rows.append([LABEL[ps], cell(ps, "NA", "iptw_att_hajek")] + [cell(ps, om, "aipw_att") for om, _ in OMS])
    return rows

if __name__ == "__main__":
    print("Study 1 tables")
    write_optional("s1_cohort_positive", lambda: cohort_table(POS))
    write_optional("s1_cohort_negative", lambda: cohort_table(NEG))
    write_table("s1_cohort_by_exposure_positive", by_exposure(POS, "celecoxib", "ns-NSAID"))
    write_table("s1_cohort_by_exposure_negative", by_exposure(NEG, "vaccinated", "unvaccinated"))
    write_table("s1_ladder_positive", ladder_table(POS))
    write_table("s1_ladder_negative", ladder_table(NEG))
    write_table("s1_pca_positive", pca_table(POS))
    write_table("s1_pca_negative", pca_table(NEG))
    write_table("s1_doubly_robust_positive", dr_table(POS))
    write_table("s1_doubly_robust_negative", dr_table(NEG))
