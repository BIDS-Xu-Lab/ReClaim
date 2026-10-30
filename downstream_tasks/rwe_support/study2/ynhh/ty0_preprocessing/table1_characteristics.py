#!/usr/bin/env python3
"""table1_characteristics.py -- Table 1 (aggregate cohort characteristics) for a TTE cohort cell.

cohort.csv is person-level and stays under intermediate/; this script distills it into
per-arm aggregates so results/ still describes the cohort: n, female %, age mean (SD),
follow-up median [IQR], and per-outcome event counts with risk %. Writes
{out_dir}/table1.csv + a readable table1.md.

    python table1_characteristics.py --cohort_csv intermediate/cohorts/ynhh_src/cohort.csv \
                         --out_dir results/ynhh_src
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--cohort_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cohort_csv)
    arms = sorted(df["drug_class"].unique())
    ev_cols = sorted(c for c in df.columns if c.startswith("out_") and c.endswith("_event"))

    rows = []
    def add(name, fmt):
        rows.append({"characteristic": name, **{a: fmt(df[df.drug_class == a]) for a in arms}})

    add("n", lambda g: f"{len(g):,}")
    add("female", lambda g: f"{(pd.to_numeric(g['sex'], errors='coerce') == 2).mean():.1%}" if len(g) else "")
    add("age at index, mean (SD)", lambda g: f"{g.age_at_index.mean():.1f} ({g.age_at_index.std():.1f})" if len(g) else "")
    add("follow-up days, median [IQR]",
        lambda g: f"{g.followup_days.median():.0f} [{g.followup_days.quantile(.25):.0f}-{g.followup_days.quantile(.75):.0f}]" if len(g) else "")
    if "ins_category" in df.columns:   # Table S5 insurance covariate, when the cohort carries it
        for lvl in ("commercial", "medicare", "medicaid", "other_unknown"):
            add(f"insurance {lvl}", lambda g, lvl=lvl: f"{(g['ins_category'] == lvl).mean():.1%}" if len(g) else "")
    for c in ev_cols:
        name = c[len("out_"):-len("_event")]
        add(f"{name} events (risk %)",
            lambda g, c=c: f"{int(g[c].sum())} ({g[c].mean():.2%})" if len(g) else "0")

    t1 = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    t1.to_csv(os.path.join(args.out_dir, "table1.csv"), index=False)
    with open(os.path.join(args.out_dir, "table1.md"), "w") as fh:
        fh.write(f"# Cohort characteristics (aggregate; source: {os.path.basename(os.path.dirname(args.cohort_csv))})\n\n")
        fh.write(t1.to_markdown(index=False))
        fh.write("\n")
    print(t1.to_string(index=False))
    print(f"-> {args.out_dir}/table1.csv (+ .md)")


if __name__ == "__main__":
    main()
