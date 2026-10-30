#!/usr/bin/env python3
"""Per-disease AUC change between an external dataset and a reference dataset.

Paper panels: Fig. 3d (EHRSHOT vs MarketScan retrospective) and Fig. 3h
(YNHH vs MarketScan retrospective), for the same model.

Bland-Altman style: x = mean AUC of the two datasets, y = external - reference,
with the mean difference and its 95% limits of agreement.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_OUT_DIR, BOX_FACE, GRID_COLOR, INK_COLOR, align, chapter_table, read_auc, save, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external", required=True, help="AUC table on the external dataset (icd_code keyed)")
    parser.add_argument("--reference", required=True, help="AUC table on the reference dataset (icd_code keyed)")
    parser.add_argument("--external_label", default="EHRSHOT")
    parser.add_argument("--reference_label", default="MarketScan")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="dataset_shift")
    args = parser.parse_args()

    wide = align({"ext": read_auc(args.external), "ref": read_auc(args.reference)})
    wide = wide.join(chapter_table().set_index("icd_code"), how="left")
    mean_auc = (wide["ext"] + wide["ref"]) / 2
    diff = wide["ext"] - wide["ref"]
    md, sd = diff.mean(), diff.std(ddof=1)

    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(mean_auc, diff, s=16, c=wide["color"].fillna("#DDDDDD"), edgecolor=BOX_FACE,
               linewidth=0.3, alpha=0.8, rasterized=True, zorder=3)
    ax.axhline(0, color="#6E6E6E", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(md, color=INK_COLOR, linewidth=1.3, zorder=2)
    for lim in (md - 1.96 * sd, md + 1.96 * sd):
        ax.axhline(lim, color=INK_COLOR, linestyle=":", linewidth=1.0, zorder=2)
    ax.set_xlabel(f"Mean AUC ({args.external_label}, {args.reference_label})")
    ax.set_ylabel(f"AUC difference ({args.external_label} − {args.reference_label})")
    ax.set_title(f"{args.external_label} vs {args.reference_label}")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, color=GRID_COLOR)
    ax.text(
        0.03, 0.97,
        f"N = {len(diff)}\nMean difference: {md * 100:+.2f} pp\n"
        f"95% limits: [{(md - 1.96 * sd) * 100:+.1f}, {(md + 1.96 * sd) * 100:+.1f}] pp\n"
        f"Higher on {args.external_label}: {(diff > 0).mean() * 100:.1f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BOX_FACE, "edgecolor": "#BDBDBD", "alpha": 0.92},
    )
    fig.tight_layout()
    save(fig, args.out_dir / args.stem)
    print(f"N={len(diff)} mean diff={md:+.4f} sd={sd:.4f} corr={np.corrcoef(wide['ext'], wide['ref'])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
