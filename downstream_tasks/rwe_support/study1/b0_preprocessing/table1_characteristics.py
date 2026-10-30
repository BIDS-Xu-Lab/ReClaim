#!/usr/bin/env python3
"""
table1_characteristics.py -- Table-1 style cohort summaries, per built cell.

Run after build_cohort: auto-discovers every {cohort_root}/{cohort}/{cell}/cohort.csv,
computes arm-stratified characteristics + standardized mean differences (SMD), and writes:

    {results_root}/{cohort}/{cell}/table1.csv          one tidy table per cell
    {results_root}/_summary/cohort_characteristics.csv all cells stacked (long format)
    {results_root}/_summary/cohort_characteristics.md  readable per-cell tables

Rows: n, female %, age mean(SD), follow-up days mean, person-time mean, outcome events (+rate),
plus every baseline l_* flag -- flags as %, l_hcru_intensity as mean(SD) encounter count.
SMD (|d| > 0.10 conventionally flags imbalance): binary  (p1-p0)/sqrt((p1q1+p0q0)/2),
continuous (m1-m0)/sqrt((s1^2+s0^2)/2). Pure pandas -- no Spark, safe on a login node.

Usage:
    python table1_characteristics.py [--cohort_root intermediate/mdcr/cohorts --results_root results/mdcr]
"""

import argparse
import glob
import math
import os

import pandas as pd

ARM_LABELS = {
    "positive_control_celecoxib_gi": ("celecoxib", "ns-NSAID"),
    "negative_control_flu_hipfx": ("vaccinated", "unvaccinated"),
}
FEMALE_TOKENS = {"F", "FEMALE"}   # string codings; numeric coding (MarketScan SEX=2) matched numerically


def _female_frac(s: pd.Series) -> float:
    """Female fraction among NON-MISSING sex values. Numeric-aware: a single NaN makes pandas
    read the column as float, so naive str-matching sees "2.0" not "2" and silently returns 0%
    (null-sex rows would otherwise zero the female row)."""
    s = s.dropna()
    if not len(s):
        return float("nan")
    sn = pd.to_numeric(s, errors="coerce")
    return ((sn == 2) | s.astype(str).str.strip().str.upper().isin(FEMALE_TOKENS)).mean()


def _smd_binary(p1, p0):
    pooled = (p1 * (1 - p1) + p0 * (1 - p0)) / 2
    return (p1 - p0) / math.sqrt(pooled) if pooled > 0 else float("nan")


def _smd_continuous(m1, s1, m0, s0):
    pooled = (s1 ** 2 + s0 ** 2) / 2
    return (m1 - m0) / math.sqrt(pooled) if pooled > 0 else float("nan")


def table1(df: pd.DataFrame, cohort_id: str) -> pd.DataFrame:
    lbl1, lbl0 = ARM_LABELS.get(cohort_id, ("exposed", "comparator"))
    a1, a0 = df[df.arm == 1], df[df.arm == 0]
    sex_col = "sex" if "sex" in df.columns else "SEX"
    rows = []

    def add(characteristic, v1, v0, smd, fmt=lambda x: x):
        rows.append({"characteristic": characteristic, lbl1: fmt(v1), lbl0: fmt(v0),
                     "SMD": f"{smd:.3f}" if smd == smd else ""})

    add("n", len(a1), len(a0), float("nan"), lambda x: f"{x:,}")
    p1 = _female_frac(a1[sex_col])
    p0 = _female_frac(a0[sex_col])
    add("female", p1, p0, _smd_binary(p1, p0), lambda p: f"{p:.1%}")
    m1, s1 = a1.age_at_index.mean(), a1.age_at_index.std()
    m0, s0 = a0.age_at_index.mean(), a0.age_at_index.std()
    add("age at index, mean (SD)", (m1, s1), (m0, s0), _smd_continuous(m1, s1, m0, s0),
        lambda t: f"{t[0]:.1f} ({t[1]:.1f})")
    add("follow-up days, mean", a1.followup_days.mean(), a0.followup_days.mean(), float("nan"),
        lambda x: f"{x:.0f}")
    if "person_time" in df.columns:
        add("person-time days, mean", a1.person_time.mean(), a0.person_time.mean(), float("nan"),
            lambda x: f"{x:.0f}")
    e1, e0 = int(a1.outcome.sum()), int(a0.outcome.sum())
    add("outcome events (risk %)", (e1, len(a1)), (e0, len(a0)), float("nan"),
        lambda t: f"{t[0]} ({t[0] / t[1]:.2%})" if t[1] else "0")
    if "person_time" in df.columns:
        # incidence per 1,000 person-years -- the person-time-correct rate (follow-up is censored
        # at disenrollment/death, so raw risk understates whenever person_time < followup_days)
        py1 = a1.person_time.sum() / 365.25
        py0 = a0.person_time.sum() / 365.25
        add("incidence /1,000 person-years", (e1, py1), (e0, py0), float("nan"),
            lambda t: f"{1000 * t[0] / t[1]:.2f}" if t[1] > 0 else "--")

    for c in sorted(c for c in df.columns if c.startswith("l_")):
        name = c[2:]
        if c == "l_hcru_intensity":   # encounter COUNT, not a flag
            m1, s1 = a1[c].mean(), a1[c].std()
            m0, s0 = a0[c].mean(), a0[c].std()
            add("hcru encounters, mean (SD)", (m1, s1), (m0, s0), _smd_continuous(m1, s1, m0, s0),
                lambda t: f"{t[0]:.1f} ({t[1]:.1f})")
        else:
            p1 = a1[c].mean() if len(a1) else float("nan")
            p0 = a0[c].mean() if len(a0) else float("nan")
            add(name, p1, p0, _smd_binary(p1, p0), lambda p: f"{p:.1%}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results_root", default="results/mdcr")
    ap.add_argument("--cohort_root", default="intermediate/mdcr/cohorts",
                   help="root of the cohort.csvs")
    args = ap.parse_args()

    stacked = []
    md = ["# Cohort characteristics (auto-generated by table1_characteristics.py)\n"]
    cohort_csvs = sorted(glob.glob(os.path.join(args.cohort_root, "*", "*", "cohort.csv")))
    if not cohort_csvs:
        raise SystemExit(f"No cohort.csv found under {args.results_root} -- run build_cohort first.")
    for path in cohort_csvs:
        cell_dir = os.path.dirname(path)
        cohort_id, cell = path.split(os.sep)[-3], path.split(os.sep)[-2]
        df = pd.read_csv(path)
        t1 = table1(df, cohort_id)
        t1.to_csv(os.path.join(cell_dir, "table1.csv"), index=False)
        long = t1.copy()
        long.insert(0, "cell", cell)
        long.insert(0, "cohort", cohort_id)
        stacked.append(long)
        md.append(f"\n## {cohort_id} / {cell}  (n={len(df):,})\n")
        md.append(t1.to_markdown(index=False))
        md.append("")
        print(f"  table1: {cohort_id}/{cell}  (n={len(df):,}) -> {cell_dir}/table1.csv")

    out_dir = os.path.join(args.results_root, "_summary")
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(stacked, ignore_index=True).to_csv(
        os.path.join(out_dir, "cohort_characteristics.csv"), index=False)
    with open(os.path.join(out_dir, "cohort_characteristics.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"  combined -> {out_dir}/cohort_characteristics.{{csv,md}}")


if __name__ == "__main__":
    main()
