#!/usr/bin/env python3
"""Disease-level AUC broken down by case count, ICD-10 chapter, age and sex.

Paper panels: Fig. 4a-d (ReClaim, MarketScan retrospective).

Inputs are one model's pooled and unpooled AUC tables keyed by icd_code.
Outputs in --out_dir:
  auc_vs_n_diseased.{png,pdf}         AUC vs number of cases, colored by chapter
  auc_vs_n_diseased_legend.{png,pdf}  chapter legend
  auc_chapter_distribution.{png,pdf}  per-chapter box plot
  auc_age_distribution.{png,pdf}      per age bracket: one point per chapter (mean AUC)
  auc_sex_distribution.{png,pdf}      per sex: one point per chapter (mean AUC)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (
    DEFAULT_OUT_DIR,
    BOX_FACE, FALLBACK_COLOR, GRID_COLOR, INK_COLOR, chapter_key, chapter_short, chapter_table, draw_box, read_auc,
    save, set_style,
)

W, H = 4.5, 2.8  # inches, shared by the scatter and chapter panels


def format_chapter(label: str) -> str:
    roman, rest = str(label).split(". ", 1)
    return f"{roman}: {rest[0].upper()}{rest[1:].lower()}"


def chapter_points_by_group(ax, up, group_col, groups, chapters, colors, seed=42):
    """One point per (group, chapter): mean stratum AUC; box over chapters within each group."""
    rng = np.random.default_rng(seed)
    per_group = []
    for g_idx, g in enumerate(groups):
        sub = up[up[group_col] == g]
        means = sub.groupby("chapter")["auc"].mean()
        pts = [(means[c], colors[c]) for c in chapters if c in means.index]
        per_group.append(np.array([v for v, _ in pts]))
        jit = rng.uniform(-0.22, 0.22, size=len(pts))
        for k, (v, col) in enumerate(pts):
            ax.scatter(g_idx + jit[k], v, s=28, facecolor=col, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    ax.boxplot(
        per_group, positions=np.arange(len(groups)), widths=0.42, patch_artist=True, showfliers=False,
        boxprops={"linewidth": 1.0, "edgecolor": INK_COLOR, "facecolor": "none"},
        whiskerprops={"color": INK_COLOR, "linewidth": 0.9}, capprops={"color": INK_COLOR, "linewidth": 0.9},
        medianprops={"color": INK_COLOR, "linewidth": 1.4}, zorder=4,
    )
    ax.scatter(np.arange(len(groups)), [v.mean() for v in per_group], marker="D", s=26,
               facecolor=BOX_FACE, edgecolor=INK_COLOR, linewidth=0.8, zorder=5)
    ax.axhline(0.5, color="#6E6E6E", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_ylim(0.3, 1.02)
    ax.set_yticks(np.linspace(0.3, 1.0, 8))
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("AUC")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, color=GRID_COLOR)
    ax.grid(axis="x", visible=False)
    return per_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pooled", required=True, help="auc_results keyed by icd_code")
    parser.add_argument("--unpooled", required=True, help="auc_results_unpooled keyed by icd_code")
    parser.add_argument("--min_cases", type=int, default=10, help="Keep diseases with more cases than this")
    parser.add_argument("--rare_cutoff", type=int, default=30, help="Case count marking rare vs common")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    chap = chapter_table()
    chapters = sorted(chap["chapter"].unique(), key=chapter_key)
    colors = dict(zip(chap["chapter"], chap["color"]))
    pooled = read_auc(args.pooled).merge(chap, on="icd_code", how="left")
    pooled = pooled[pooled["n_diseased"] > args.min_cases]
    up = read_auc(args.unpooled, unpooled=True).merge(chap, on="icd_code", how="inner")
    present = [c for c in chapters if c in set(pooled["chapter"].dropna())]
    print(f"Diseases with > {args.min_cases} cases: {len(pooled)}; mean AUC {pooled['auc'].mean():.4f}; "
          f"AUC > 0.7: {(pooled['auc'] > 0.7).mean() * 100:.1f}%")

    set_style(font_size=9)

    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.scatter(pooled["n_diseased"], pooled["auc"], c=pooled["color"].fillna(FALLBACK_COLOR),
               s=18, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.75, zorder=2)
    ax.axvline(args.rare_cutoff, color="#555555", linestyle="--", linewidth=0.75, zorder=2)
    ax.text(args.rare_cutoff / 2, 1.03, "Rare", fontsize=8, ha="center", color="#666666")
    ax.text(args.rare_cutoff * 6.7, 1.03, "Common", fontsize=8, ha="center", color="#666666")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs number of cases", fontsize=10)
    fig.tight_layout()
    save(fig, args.out_dir / "auc_vs_n_diseased", dpi=300)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markersize=8,
                      label=format_chapter(c)) for c in present]
    n_rows = -(-len(handles) // 5)
    fig, ax = plt.subplots(figsize=(13, 0.52 * n_rows + 0.3), dpi=150)
    ax.legend(handles=handles, ncol=5, loc="center", frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.2)
    ax.axis("off")
    save(fig, args.out_dir / "auc_vs_n_diseased_legend", dpi=300)

    groups = [pooled.loc[pooled["chapter"] == c, "auc"].to_numpy() for c in present]
    fig, ax = plt.subplots(figsize=(W * 1.1, H), dpi=300)
    draw_box(ax, groups, [colors[c] for c in present], width=0.42)
    ax.axhline(0.5, color="#6E6E6E", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_ylim(0.3, 1.02)
    ax.set_yticks(np.linspace(0.3, 1.0, 8))
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([chapter_short(c) for c in present], fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title("AUC by ICD-10 chapter", fontsize=10)
    fig.tight_layout()
    save(fig, args.out_dir / "auc_chapter_distribution", dpi=300)
    for c, g in zip(present, groups):
        print(f"  {chapter_short(c):>5}  n={len(g):4d}  mean AUC {g.mean():.4f}")

    ages = sorted(up["age"].dropna().unique().astype(int))
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    per_age = chapter_points_by_group(ax, up, "age", ages, present, colors)
    ax.set_xticklabels([str(a) for a in ages])
    ax.set_xlabel("Age group")
    ax.set_title("AUC by age group", fontsize=10)
    fig.tight_layout()
    save(fig, args.out_dir / "auc_age_distribution", dpi=300)
    print("  age:", ", ".join(f"{a}: {v.mean():.4f}" for a, v in zip(ages, per_age)))

    fig, ax = plt.subplots(figsize=(W * 0.45, H), dpi=300)
    per_sex = chapter_points_by_group(ax, up, "sex", ["male", "female"], present, colors)
    ax.set_xticklabels(["Male", "Female"])
    ax.set_xlabel("Sex")
    ax.set_title("AUC by sex", fontsize=10)
    fig.tight_layout()
    save(fig, args.out_dir / "auc_sex_distribution", dpi=300)
    print(f"  sex: male {per_sex[0].mean():.4f}, female {per_sex[1].mean():.4f}")


if __name__ == "__main__":
    main()
