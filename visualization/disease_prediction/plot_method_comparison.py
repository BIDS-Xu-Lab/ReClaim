#!/usr/bin/env python3
"""ReClaim vs Delphi vs LightGBM on one evaluation dataset.

Paper panels: Fig. 2a-d (MarketScan retrospective), Fig. 2e-h (prospective),
Fig. 3a-c (EHRSHOT), Fig. 3e-g (YNHH).

Outputs in --out_dir:
  {stem}_box.{png,pdf}     box plots on the shared diseases (+ rare subset)
  {stem}_paired.{png,pdf}  per-disease ReClaim vs LightGBM / Delphi scatter
  {stem}_summary.csv       means and paired deltas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_OUT_DIR,
    METHOD_PALETTE, align, draw_box, draw_paired_scatter, paired_summary, rare_codes, read_auc, save, set_style,
)

METHODS = ["LightGBM", "Delphi", "ReClaim"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reclaim", required=True, help="ReClaim auc_results keyed by icd_code")
    parser.add_argument("--delphi", required=True)
    parser.add_argument("--lightgbm", required=True)
    parser.add_argument("--dataset", default="", help="Label prepended to panel titles, e.g. 'EHRSHOT'")
    parser.add_argument("--no_rare", action="store_true", help="Skip the rare-disease panel")
    parser.add_argument("--rare_max_prevalence", type=float, default=0.0005)
    parser.add_argument("--rare_min_cases", type=int, default=10)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="method_comparison")
    args = parser.parse_args()

    frames = {"ReClaim": read_auc(args.reclaim), "Delphi": read_auc(args.delphi), "LightGBM": read_auc(args.lightgbm)}
    wide = align(frames)
    panels = [("All Diseases", wide)]
    if not args.no_rare:
        ref = frames["ReClaim"][frames["ReClaim"]["icd_code"].isin(wide.index)]
        rare = rare_codes(ref, args.rare_max_prevalence, args.rare_min_cases)
        panels.append(("Rare Diseases", wide.loc[sorted(rare)]))
    prefix = f"{args.dataset}: " if args.dataset else ""

    set_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.6), sharey=True, squeeze=False)
    lo = max(0.0, wide.min().min() - 0.03)
    hi = min(1.0, wide.max().max() + 0.02)
    for ax, (title, sub) in zip(axes[0], panels):
        draw_box(ax, [sub[m].to_numpy() for m in METHODS], [METHOD_PALETTE[m] for m in METHODS])
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS)
        ax.set_title(f"{prefix}{title} (N={len(sub)})", pad=10)
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10 + 1e-3, 0.1))
    axes[0][0].set_ylabel("AUC")
    fig.tight_layout(w_pad=2.4, pad=0.8)
    save(fig, args.out_dir / f"{args.stem}_box")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, base in zip(axes, ["LightGBM", "Delphi"]):
        draw_paired_scatter(ax, wide[base], wide["ReClaim"], base, "ReClaim", METHOD_PALETTE[base],
                            title=f"{prefix}ReClaim vs {base}")
    fig.tight_layout(w_pad=2.0)
    save(fig, args.out_dir / f"{args.stem}_paired")

    rows = []
    for title, sub in panels:
        for base in ["LightGBM", "Delphi"]:
            s = paired_summary(sub["ReClaim"], sub[base])
            rows.append({"subset": title, "baseline": base, "n": s["n"],
                         "reclaim_mean": s["mean_a"], "baseline_mean": s["mean_b"],
                         "delta_mean": s["delta_mean"], "delta_ci95": s["delta_ci95"],
                         "frac_reclaim_better": s["frac_a_better"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / f"{args.stem}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
