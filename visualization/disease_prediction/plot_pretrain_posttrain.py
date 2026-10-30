#!/usr/bin/env python3
"""Pre-training vs post-training disease-level AUC for the same backbone.

Paper panels: Fig. 5b (box plot) and Fig. 5c (per-disease paired scatter).
Pre- and post-training checkpoints may use different tokenizers, so both
inputs must be keyed by icd_code.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from common import DEFAULT_OUT_DIR, align, draw_box, draw_paired_scatter, read_auc, save, set_style

COLORS = {"Pretrain": "#5C7DA5", "Posttrain": "#D07C5B", "Delta": "#4C78A8"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrain", required=True, help="icd_code-keyed AUC table of the pre-trained model")
    parser.add_argument("--posttrain", required=True, help="icd_code-keyed AUC table of the post-trained model")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="pretrain_vs_posttrain_auc")
    args = parser.parse_args()

    wide = align({"Pretrain": read_auc(args.pretrain), "Posttrain": read_auc(args.posttrain)})

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    draw_box(ax, [wide["Pretrain"].to_numpy(), wide["Posttrain"].to_numpy()],
             [COLORS["Pretrain"], COLORS["Posttrain"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pretrain", "Posttrain"])
    ax.set_ylabel("AUC")
    ax.set_title(f"Disease-Level AUC (N={len(wide)})")
    s = draw_paired_scatter(axes[1], wide["Pretrain"], wide["Posttrain"], "Pretrain", "Posttrain", COLORS["Delta"])
    for axis in (axes[1].xaxis, axes[1].yaxis):
        axis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout(w_pad=2.0)
    save(fig, args.out_dir / args.stem)

    wide.assign(delta=wide["Posttrain"] - wide["Pretrain"]).to_csv(args.out_dir / f"{args.stem}_paired.csv")
    pd.DataFrame([s]).to_csv(args.out_dir / f"{args.stem}_summary.csv", index=False)
    print(f"N={s['n']} pretrain {s['mean_b']:.4f} posttrain {s['mean_a']:.4f} "
          f"delta {s['delta_mean'] * 100:+.2f} pp, improved {s['frac_a_better'] * 100:.1f}%")


if __name__ == "__main__":
    main()
