#!/usr/bin/env python3
"""study2_tables.py -- the tables of Study 2, the target trial emulation of GLP-1 receptor agonists vs SGLT2 inhibitors
and vs DPP4 inhibitors for six psychiatric outcomes (Tang et al.), replicated under the reference's own construction on
MarketScan held-out claims (MS) and the YNHH OMOP EHR, varying only the propensity-score specification.

Writes to output/tables/:
  s2_cohorts                     cohort description per emulation: reference, MarketScan, YNHH
  s2_replication_siptw           replication under stabilized IPTW (primary): reference, Expert and ReClaim-L (raw and calibrated), both sources
  s2_replication_psm             the same under 1:1 propensity-score matching
  s2_grid_<source>_<comparison>  every specification x outcome, stabilized IPTW (* raw CI excludes 1, † calibrated CI excludes 1)
  s2_anxiety_by_specification, s2_depression_by_specification   raw / calibrated HR under every specification
  s2_anchor_anxiety              the reference's headline signal (anxiety, Expert) on the three sources
  s2_report_card                 EASE, calibration shift, CI inflation and equipoise per specification
  s2_ps_metrics                  outcome-agnostic metrics per specification: PS c-statistic, equipoise, EASE
  s2_diagnostics                 estimator-panel diagnostics per specification
  s2_caliper_sweep               EASE and retention under 1:1 matching at five calipers
  s2_balance                     covariate balance of every specification on the reference covariates and on its own covariates, weighted and matched
  s2_schizophrenia_cell          the schizophrenia cell across designs and sources"""
import csv, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TTE, TTE_RESULTS, TTE_SUMMARY, COMPARISONS, OUTCOMES, TTE_SPECS, LABEL, TANG_HR, TANG_EVENTS, TANG_PSM, TANG_COHORT,
                    SCHIZOPHRENIA_EXTERNAL, read_csv, tte_outcomes, tte_diagnostics, ci, num, missing, write_table,
                    write_optional)

SRC_LABEL = {"MS": "MarketScan", "YNHH": "YNHH"}
REPORTED = ["paper", "hdps", "reclaim-s-only", "reclaim-m-only", "reclaim-l-only"]   # specifications compared head to head
O = {s: tte_outcomes(s) for s in ("MS", "YNHH")}
G = {s: tte_diagnostics(s) for s in ("MS", "YNHH")}

def hr(r, cal=False, nd=2):
    k = ("hr_cal", "lb_cal", "ub_cal") if cal else ("hr", "lb", "ub")
    return "nr" if r is None or missing(r[k[0]]) else ci(r[k[0]], r[k[1]], r[k[2]], nd)

# ---- cohort description -------------------------------------------------------------------------------------------------
def ms_cohort(comp):
    """MarketScan emulation cohort (intermediate/cohorts/vs_<comparator>/cohort.csv), per arm"""
    path = os.path.join(TTE["MS"], "intermediate", "cohorts", "vs_" + comp.split("_vs_")[1], "cohort.csv")
    arms = {}
    for r in csv.DictReader(open(path)): arms.setdefault(r["drug_class"] == "glp1ra", []).append(r)
    out = {}
    for is_glp, rs in arms.items():
        n = len(rs); age = [float(r["age_at_index"]) for r in rs]
        d = {"n": f"{n:,}", "age": f"{statistics.mean(age):.1f} ({statistics.stdev(age):.1f})",
             "female": f"{100 * sum(r['sex'] == '2' for r in rs) / n:.1f}%", "medicaid": f"{100 * sum(r['payer'] == 'MDCD' for r in rs) / n:.1f}%",
             "followup": f"{statistics.median(float(r['followup_days']) for r in rs):,.0f}"}
        for o, _ in OUTCOMES: d[o] = f"{100 * sum(r[f'out_{o}_event'] == '1' for r in rs) / n:.2f}%"
        out["glp1" if is_glp else "comp"] = d
    return out

def ynhh_cohort(comp):
    """YNHH emulation cohort description, as written by the YNHH module (results/ynhh_src_vs_<comparator>/table1.csv)"""
    other = comp.split("_vs_")[1] + "i"
    T = {r["characteristic"]: r for r in read_csv(os.path.join(TTE["YNHH"], "results", f"ynhh_src_vs_{comp.split('_vs_')[1]}", "table1.csv"))}
    out = {}
    for arm, col in (("glp1", "glp1ra"), ("comp", other)):
        d = {"n": T["n"][col], "age": T["age at index, mean (SD)"][col], "female": T["female"][col], "medicaid": "n/a (EHR)",
             "followup": f"{float(T['follow-up days, median [IQR]'][col].split()[0]):,.0f}"}
        for o, _ in OUTCOMES: d[o] = T[f"{o} events (risk %)"][col].split("(")[1].rstrip(")")
        out[arm] = d
    return out

def cohort_table():
    cols, data = [], []
    for comp, cl in COMPARISONS:
        ref = TANG_COHORT[comp]; ms = ms_cohort(comp); yn = ynhh_cohort(comp); comp_lab = cl.replace("vs ", "")
        for src, d in (("Reference", None), ("MS", ms), ("YNHH", yn)):
            for arm in ("glp1", "comp"):
                cols.append(f"{src} {'GLP-1RA (' + cl + ')' if arm == 'glp1' else comp_lab}")
                if d is None:
                    i = 0 if arm == "glp1" else 1
                    data.append({"n": ref["n"][i], "age": ref["age"][i], "female": ref["female"][i], "medicaid": "n/a (EHR)",
                                 "followup": "nr", **{o: ref[o][i] for o, _ in OUTCOMES}})
                else: data.append(d[arm])
    # column order: reference, MS, YNHH; within each, vs SGLT2i then vs DPP4i
    order = sorted(range(len(cols)), key=lambda j: (["Reference", "MS", "YNHH"].index(cols[j].split()[0]), j))
    cols = [cols[j] for j in order]; data = [data[j] for j in order]
    rows = [["Characteristic"] + cols, ["Index period"] + ["2019-2024"] * len(cols)]
    for k, lab in [("n", "n"), ("age", "Age at index, mean (SD)"), ("female", "Female"), ("medicaid", "Medicaid"),
                   ("followup", "Follow-up, median days")] + [(o, f"{ol} incidence") for o, ol in OUTCOMES]:
        rows.append([lab] + [d[k] for d in data])
    return rows

# ---- replication ----------------------------------------------------------------------------------------------------------
def replication_siptw():
    rows = [["Outcome", "Comparison", "Reference treated events", "Reference HR", "Reference HR, calibrated"]]
    for s in ("MS", "YNHH"):
        rows[0] += [f"{s} treated events", f"{s} Expert", f"{s} Expert, calibrated", f"{s} ReClaim-L", f"{s} ReClaim-L, calibrated"]
    for o, ol in OUTCOMES:
        for c, cl in COMPARISONS:
            ref = TANG_HR[(o, c)]
            row = [ol, cl, str(TANG_EVENTS[o][0 if c == "glp1_vs_sglt2" else 1]), ci(*ref[:3], 2), ci(*ref[3:], 2)]
            for s in ("MS", "YNHH"):
                pr = O[s][(c, "paper", "siptw_cox", o)]; rl = O[s][(c, "reclaim-l-only", "siptw_cox", o)]
                row += [str(int(float(pr["events_treat"]))), hr(pr), hr(pr, True), hr(rl), hr(rl, True)]
            rows.append(row)
    return rows

def replication_psm():
    rows = [["Outcome", "Comparison", "Reference treated events", "Reference HR, PSM"]]
    for s in ("MS", "YNHH"): rows[0] += [f"{s} treated events (Expert)", f"{s} treated events (ReClaim-L)", f"{s} Expert, PSM", f"{s} ReClaim-L, PSM"]
    for o, ol in OUTCOMES:
        for c, cl in COMPARISONS:
            rp = TANG_PSM[(o, c)]; row = [ol, cl, str(rp[0]), ci(*rp[1:], 2)]
            for s in ("MS", "YNHH"):
                pr = O[s].get((c, "paper", "psm_cox", o)); rl = O[s].get((c, "reclaim-l-only", "psm_cox", o))
                # each specification is matched on its own propensity score, so its matched cohort
                # (and treated event count) is its own
                ev = lambda r: "nr" if r is None or missing(r["events_treat"]) else str(int(float(r["events_treat"])))
                row += [ev(pr), ev(rl), hr(pr), hr(rl)]
            rows.append(row)
    return rows

def grid(src, comp):
    rows = [["Outcome"] + [LABEL[s] for s in TTE_SPECS]]
    for o, ol in OUTCOMES:
        row = [ol]
        for s in TTE_SPECS:
            r = O[src].get((comp, s, "siptw_cox", o))
            if r is None or missing(r["hr"]): row.append("NE"); continue
            mark = ("*" if not (float(r["lb"]) <= 1 <= float(r["ub"])) else "") + \
                   ("†" if not missing(r["lb_cal"]) and not (float(r["lb_cal"]) <= 1 <= float(r["ub_cal"])) else "")
            row.append(num(r["hr"], 2) + mark)
        rows.append(row)
    return rows

def by_specification(outcome):
    def c(r, cal): k = ("hr_cal", "lb_cal", "ub_cal") if cal else ("hr", "lb", "ub"); return f"{float(r[k[0]]):.2f} ({float(r[k[1]]):.2f}–{float(r[k[2]]):.2f})"
    rows = [["Specification"] + [f"{SRC_LABEL[s]} {cl}" for s in ("MS", "YNHH") for _, cl in COMPARISONS]]
    for sp in TTE_SPECS:
        rows.append([LABEL[sp]] + [f"{c(O[s][(cm, sp, 'siptw_cox', outcome)], False)} / {c(O[s][(cm, sp, 'siptw_cox', outcome)], True)}"
                                    for s in ("MS", "YNHH") for cm, _ in COMPARISONS])
    return rows

def anchor_anxiety():
    rows = [["Source", "vs SGLT2i (95% CI)", "vs SGLT2i, calibrated", "vs DPP4i (95% CI)", "vs DPP4i, calibrated"]]
    row = ["Reference (Penn Medicine EHR)"]
    for c, _ in COMPARISONS: r = TANG_HR[("anxiety_disorders", c)]; row += [ci(*r[:3], 2), ci(*r[3:], 2)]
    rows.append(row)
    for s, lab in (("MS", "MarketScan (claims, held-out)"), ("YNHH", "YNHH (EHR)")):
        row = [lab]
        for c, _ in COMPARISONS: r = O[s][(c, "paper", "siptw_cox", "anxiety_disorders")]; row += [hr(r), hr(r, True)]
        rows.append(row)
    return rows

# ---- specification diagnostics ------------------------------------------------------------------------------------------
def shift_and_inflation(h, l, u, hc, lc, uc):
    """calibration shift |log HR_cal - log HR| and CI inflation (calibrated / raw interval width), anxiety"""
    return abs(math.log(hc) - math.log(h)), (uc - lc) / (u - l)

def report_card():
    rows = [["Comparison", "Source", "Specification", "EASE", "Calibration shift", "CI inflation", "Equipoise"]]
    for c, cl in COMPARISONS:
        r = TANG_HR[("anxiety_disorders", c)]; sh, inf = shift_and_inflation(*r)
        rows.append([cl, "Reference (Penn Medicine EHR)", "Expert", "nr", f"{sh:.3f}", f"{inf:.2f}", "nr"])
        for s in ("MS", "YNHH"):
            for sp in REPORTED:
                d = G[s][(c, sp, "siptw_cox")]; a = O[s][(c, sp, "siptw_cox", "anxiety_disorders")]
                sh, inf = shift_and_inflation(*(float(a[k]) for k in ("hr", "lb", "ub", "hr_cal", "lb_cal", "ub_cal")))
                rows.append([cl, SRC_LABEL[s], LABEL[sp], num(d["ease"], 3), f"{sh:.3f}", f"{inf:.2f}", num(d["equipoise"], 2)])
    return rows

def ps_metrics():
    rows = [["Source", "Comparison", "Specification", "PS c-statistic", "Equipoise", "EASE (sIPTW)"]]
    for s in ("MS", "YNHH"):
        for c, cl in COMPARISONS:
            for sp in TTE_SPECS:
                w = G[s].get((c, sp, "siptw_cox"))
                if w: rows.append([SRC_LABEL[s], cl, LABEL[sp], "none (no PS)" if sp == "crude" else num(w["c_statistic"], 3),
                                   num(w["equipoise"], 2), num(w["ease"], 3)])
    return rows

def diagnostics():
    f3 = lambda x, nd=3: "nr" if missing(x) else num(x, nd)
    f0 = lambda x: "nr" if missing(x) else f"{float(x):,.0f}"
    rows = [["Source", "Comparison", "Specification", "Covariates", "PS c-statistic", "Equipoise", "EASE, sIPTW", "EASE, 1:1 PSM",
             "EASE, stratified", "ESS treated / comparator (sIPTW)", "Max weight", "Matched pairs"]]
    for s in ("MS", "YNHH"):
        for c, cl in COMPARISONS:
            for sp in TTE_SPECS:
                w = G[s].get((c, sp, "siptw_cox")); m = G[s].get((c, sp, "psm_cox")); st = G[s].get((c, sp, "strat_cox"))
                if not w: continue
                rows.append([SRC_LABEL[s], cl, LABEL[sp], w["n_covariates"], f3(w["c_statistic"]), f3(w["equipoise"], 2), f3(w["ease"]),
                             f3(m["ease"]) if m else "nr", f3(st["ease"]) if st else "nr", f"{f0(w['ess_treat'])} / {f0(w['ess_comp'])}",
                             f3(w["max_weight"], 2), f0(m["n_matched_treat"]) if m else "nr"])
    return rows

def caliper_sweep():
    rows = [["Source", "Comparison", "Caliper (SD)"] + [f"EASE {LABEL[s]}" for s in REPORTED] + [f"Retention {LABEL[s]}" for s in REPORTED]]
    for s in ("MS", "YNHH"):
        for c, cl in COMPARISONS:
            sw = {}
            for sp in REPORTED:
                for r in read_csv(os.path.join(TTE_RESULTS[s], c, sp, "psm_cox", "caliper_sweep.csv")): sw[(sp, r["caliper"])] = r
            for i, cal in enumerate(sorted({k[1] for k in sw}, key=float)):
                rows.append([SRC_LABEL[s] if i == 0 else "", cl if i == 0 else "", cal if "." in cal else cal + ".0"] +
                            [f"{float(sw[(sp, cal)]['ease']):.3f}" for sp in REPORTED] + [f"{float(sw[(sp, cal)]['retention']):.0%}" for sp in REPORTED])
    return rows

def balance():
    """covariates with |SMD| below 0.1 over those evaluated, (a) on the reference covariates (the Expert set) under every
    specification's weights and matched set (expert_balance.R summary) and (b) on each specification's own covariates"""
    def cell(path):
        if not os.path.exists(path): return None
        v = [abs(float(r["smd"])) for r in read_csv(path)]
        return len(v) - sum(x > 0.1 for x in v), len(v)
    # truncated, not rounded, to one decimal: a share shows 100% only when every covariate is below 0.1
    share = lambda k, n: "100%" if k == n else f"{math.floor(1000 * k / n) / 10:.1f}%"
    fm = lambda x: "nr" if x is None else f"{x[0]:,}/{x[1]:,} ({share(*x)})"
    rows = [["Comparison", "Data source", "Specification", "Reference covariates: weighted", "Reference covariates: matched",
             "Own covariates: weighted", "Own covariates: matched"]]
    for c, cl in COMPARISONS:
        for s, sl in (("MS", "MarketScan held-out claims"), ("YNHH", "YNHH EHR")):
            ref = {(r["spec"], r["estimator"]): r for r in read_csv(os.path.join(TTE_SUMMARY[s], "summary_expert_balance.csv"))
                   if r["comparison"] == c}
            def rc(sp, est):
                r = ref.get((sp, est))
                return None if r is None else (int(r["n_terms"]) - int(r["n_gt_0.1_after"]), int(r["n_terms"]))
            r0 = ref[("crude", "siptw_cox")]; n0 = int(r0["n_terms"])
            rows.append([cl.replace("vs", "GLP-1RA vs"), sl, LABEL["crude"], fm((round(float(r0["share_lt_0.1_before"]) * n0), n0)),
                         "-", "-", "-"])
            for sp in ["demo"] + REPORTED:
                w = cell(os.path.join(TTE_RESULTS[s], c, sp, "siptw_cox", "balance_smd.csv"))
                m = cell(os.path.join(TTE_RESULTS[s], c, sp, "psm_cox", "balance_smd.csv"))
                rows.append([cl.replace("vs", "GLP-1RA vs"), sl, LABEL[sp], fm(rc(sp, "siptw_cox")), fm(rc(sp, "psm_cox")), fm(w), fm(m)])
    return rows

def schizophrenia_cell():
    o = "schizophrenia_and_other_psychotic_disorders"
    rows = [["Version", "Data source", "Cohort", "Events (GLP-1RA vs DPP4i)",
             "Expert (reference study's covariates), calibrated HR (95% CI)", "ReClaim-L (representation), calibrated HR (95% CI)"]]
    rows += [list(r) for r in SCHIZOPHRENIA_EXTERNAL]
    for s, src, coh in (("MS", "MarketScan (claims)", "2M held-out cohort (contains the 1M split)"), ("YNHH", "YNHH (EHR)", "All eligible patients")):
        e = O[s][("glp1_vs_dpp4", "paper", "siptw_cox", o)]
        l = O[s][("glp1_vs_dpp4", "reclaim-l-only", "siptw_cox", o)]
        rows.append(["Revision (reference study's protocol)", src, coh,
                     f"{int(float(e['events_treat']))} vs {int(float(e['events_comp']))}", hr(e, True), hr(l, True)])
    return rows

if __name__ == "__main__":
    print("Study 2 tables")
    write_optional("s2_cohorts", cohort_table)
    write_table("s2_replication_siptw", replication_siptw())
    write_table("s2_replication_psm", replication_psm())
    for s in ("MS", "YNHH"):
        for c, _ in COMPARISONS:
            write_table(f"s2_grid_{s.lower()}_{c}", grid(s, c), note="* raw 95% CI excludes 1; † calibrated 95% CI excludes 1; NE = not estimable.")
    write_table("s2_anxiety_by_specification", by_specification("anxiety_disorders"))
    write_table("s2_depression_by_specification", by_specification("depression"))
    write_table("s2_anchor_anxiety", anchor_anxiety())
    write_table("s2_report_card", report_card())
    write_table("s2_ps_metrics", ps_metrics())
    write_table("s2_diagnostics", diagnostics())
    write_table("s2_caliper_sweep", caliper_sweep())
    write_table("s2_balance", balance())
    write_table("s2_schizophrenia_cell", schizophrenia_cell())
