#!/usr/bin/env python3
"""
Compile the cross-model comparison table from the e4 metric tables.

For each section (retrospective / prospective) and model, gathers:
  - regression on total cost: R2, MAE
  - HNHC ($30K): F1, precision, recall
  - HNHC probability quality: ROC-AUC, PR-AUC (models with scores)

Writes results/<split>/summary_table.csv and prints a markdown table.
"""

import os
import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compile cross-model summary table")
    p.add_argument("--result_root", type=str, required=True)
    p.add_argument("--sections", type=str, nargs="+",
                   default=["retrospective", "prospective"])
    p.add_argument("--cutoff_hnhc", type=str, default="30000")
    return p.parse_args()


def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def pick(df, model, col):
    if df is None:
        return np.nan
    row = df[(df["model"] == model) & (df["source"] == "total")]
    if "cost_type" in df.columns:
        row = row[row["cost_type"] == "total"]
    return row[col].values[0] if len(row) else np.nan


def main():
    args = parse_args()
    for section in args.sections:
        base = os.path.join(args.result_root, section)
        reg = safe_read(os.path.join(base, "regression", "regression_metrics.csv"))
        hn = safe_read(os.path.join(base, "classification", args.cutoff_hnhc,
                                    "classification_metrics.csv"))
        auc = safe_read(os.path.join(base, "binary_auc", args.cutoff_hnhc,
                                     "auc_metrics.csv"))
        if reg is None and hn is None:
            print(f"[{section}] no results yet, skipping")
            continue

        models = []
        for df in (reg, hn, auc):
            if df is not None:
                models.extend(df.loc[df["source"] == "total", "model"].unique())
        models = list(dict.fromkeys(models))  # order-preserving unique

        rows = []
        for m in models:
            # AUC rows use the base model name for MC models (e.g. reclaim-s-pretrain)
            # while reg/cls use the -<N> suffixed id; try both
            m_base = m.rsplit("-", 1)[0] if m.split("-")[-1].isdigit() else m
            rows.append({
                "model": m,
                "R2": pick(reg, m, "R2"),
                "MAE": pick(reg, m, "MAE"),
                "F1_hnhc": pick(hn, m, "f1-binary"),
                "prec_hnhc": pick(hn, m, "precision-binary"),
                "rec_hnhc": pick(hn, m, "recall-binary"),
                "ROC_AUC": pick(auc, m, "roc_auc") if not np.isnan(pick(auc, m, "roc_auc")) else pick(auc, m_base, "roc_auc"),
                "PR_AUC": pick(auc, m, "pr_auc") if not np.isnan(pick(auc, m, "pr_auc")) else pick(auc, m_base, "pr_auc"),
            })
        out = pd.DataFrame(rows)
        # drop duplicate rows created by suffixed/base aliases of the same model
        out = out.drop_duplicates(subset=["model"]).reset_index(drop=True)

        path = os.path.join(base, "summary_table.csv")
        out.to_csv(path, index=False)
        print(f"\n### {section} (saved {path})\n")
        fmt = out.copy()
        for c in fmt.columns:
            if c != "model":
                fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else
                                    (f"{v:,.0f}" if c == "MAE" else f"{v:.3f}"))
        print(fmt.to_markdown(index=False))


if __name__ == "__main__":
    main()
