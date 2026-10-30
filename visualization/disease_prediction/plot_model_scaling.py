#!/usr/bin/env python3
"""Mean disease-level AUC across model sizes.

Paper panel: Fig. 5a (ReClaim-S/M/L). Means and 95% CIs are computed on the
diseases shared by all inputs; paired deltas compare each model with the
previous one and the last with the first.

    python plot_model_scaling.py --input ReClaim-S=s.csv --input ReClaim-M=m.csv --input ReClaim-L=l.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from common import DEFAULT_OUT_DIR, BOX_FACE, GRID_COLOR, INK_COLOR, align, paired_summary, read_auc, save, set_style

SIZE_COLORS = ["#00A087", "#4DBBD5", "#E64B35", "#3C5488", "#F39B7F"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=CSV",
                        help="Model label and its icd_code-keyed AUC table; repeat from smallest to largest")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="model_scaling_auc")
    args = parser.parse_args()

    specs = [s.split("=", 1) for s in args.input]
    wide = align({label: read_auc(path) for label, path in specs})
    labels = list(wide.columns)
    n = len(wide)
    means = wide.mean()
    ci95 = 1.96 * wide.std(ddof=1) / n ** 0.5
    pairs = [(labels[i], labels[i - 1]) for i in range(1, len(labels))]
    if len(labels) > 2:
        pairs.append((labels[-1], labels[0]))
    deltas = {(a, b): paired_summary(wide[a], wide[b]) for a, b in pairs}

    set_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = range(len(labels))
    ax.plot(x, means.values, color=INK_COLOR, linewidth=2.0, zorder=2)
    for i, label in enumerate(labels):
        ax.errorbar(i, means[label], yerr=ci95[label], color=INK_COLOR, capsize=6, linewidth=1.5, zorder=3)
        ax.scatter(i, means[label], s=70, color=SIZE_COLORS[i % len(SIZE_COLORS)], edgecolor=INK_COLOR,
                   linewidth=1.0, zorder=4)
        ax.annotate(f"{means[label] * 100:.2f}%", (i, means[label] + ci95[label]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=10.5)
    pad = max(0.0035, 2 * ci95.max())
    ax.set_ylim(means.min() - pad, means.max() + pad)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylabel("Mean disease-level AUC")
    ax.set_title(f"Model scaling (N={n} diseases)")
    ax.grid(axis="y", linestyle="--", linewidth=0.9, color=GRID_COLOR)
    ax.grid(axis="x", visible=False)
    ax.text(
        0.03, 0.97,
        "\n".join(f"Paired delta ({a} − {b}): {d['delta_mean'] * 100:+.2f} pp ± {d['delta_ci95'] * 100:.2f} pp"
                  for (a, b), d in deltas.items()),
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BOX_FACE, "edgecolor": GRID_COLOR},
    )
    fig.tight_layout()
    save(fig, args.out_dir / args.stem)

    summary = pd.DataFrame({"model": labels, "n": n, "mean_auc": means.values, "ci95": ci95.values})
    summary.to_csv(args.out_dir / f"{args.stem}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    for (a, b), d in deltas.items():
        print(f"{a} - {b}: {d['delta_mean'] * 100:+.2f} pp ± {d['delta_ci95'] * 100:.2f}, "
              f"{d['frac_a_better'] * 100:.1f}% of diseases improved")


if __name__ == "__main__":
    main()
