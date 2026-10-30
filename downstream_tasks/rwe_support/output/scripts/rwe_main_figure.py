#!/usr/bin/env python3
"""Render the current manuscript Figure 6 from the released Study 1 tables.

Uses the rounded RR / 95% CI values reported in the manuscript and the
output/tables/s1_ladder_{positive,negative}.csv files. No patient data or
unreleased pipeline outputs are needed.
"""
import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

OUTPUT = Path(__file__).resolve().parents[1]
LABELS = ["Unadjusted", "hdPS", "Delphi", "ReClaim-S", "ReClaim-M", "ReClaim-L"]
COLORS = ["#7a7a7a", "#647ec4", "#4dbbd5", "#f5b09d", "#f07b62", "#e64b35"]


def parse_estimate(value):
    if value in ("-", "--", ""):
        return None
    match = re.fullmatch(r"([\d.]+) \(([\d.]+)-([\d.]+)\)", value)
    if match is None:
        raise ValueError(f"Invalid RR / confidence interval: {value!r}")
    rr, low, high = map(float, match.groups())
    if not 0 < low <= rr <= high:
        raise ValueError(f"Invalid RR / confidence interval bounds: {value!r}")
    return rr, low, high


def load_data():
    groups = []
    for control in ("positive", "negative"):
        source = OUTPUT / "tables" / f"s1_ladder_{control}.csv"
        with source.open(newline="") as handle:
            rows = {row["PS specification"]: row for row in csv.DictReader(handle)}
        group = {}
        for label in LABELS:
            row = rows[label]
            group[label] = [parse_estimate(row[column])
                            for column in ("MarketScan MDCR", "Reference")]
            if group[label][0] is None:
                raise ValueError(f"{source}: missing MarketScan estimate for {label}")
        groups.append(group)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=OUTPUT / "figures",
                        help="Directory for rwe_main_figure.pdf and .png")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_size = 8
    dest = args.out_dir / "rwe_main_figure"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": font_size, "pdf.fonttype": 42, "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.86, wspace=0.48)
    for panel, (ax, group) in enumerate(zip(axes, load_data())):
        ax.set_xscale("log")
        ax.set_xlim(0.7, 1.3)
        ax.set_ylim(-0.55, 5.75)
        ax.xaxis.set_major_locator(FixedLocator([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)
        ax.axvline(1, color="#4d4d4d", linestyle=(0, (3, 3)), linewidth=0.65)
        for row, label in enumerate(LABELS):
            mdcr, pace = group[label]
            y = 5 - row
            for estimate, offset, hollow in [(mdcr, -0.17 if pace else 0, False), (pace, 0.17, True)]:
                if estimate is None:
                    continue
                rr, low, high = estimate
                ax.errorbar(rr, y + offset, xerr=[[rr-low], [high-rr]], fmt="o",
                            markersize=3.2, markerfacecolor="white" if hollow else COLORS[row],
                            markeredgecolor=COLORS[row], markeredgewidth=0.8,
                            ecolor=COLORS[row], elinewidth=0.8, capsize=0, zorder=3)
        ax.set_yticks(range(5, -1, -1), LABELS)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=font_size)
        ax.tick_params(axis="x", length=2, width=0.5, pad=2, labelsize=font_size-0.5)
        ax.set_xlabel("Relative risk (log scale)", fontsize=font_size, labelpad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#4d4d4d")
        ax.text(-0.22, 1.12, "ab"[panel], transform=ax.transAxes, weight="bold", fontsize=font_size+2)
        ax.text(0.5, 1.035, "Expected effect: RR < 1" if panel == 0 else "Expected effect: RR = 1",
                transform=ax.transAxes, ha="center", fontsize=font_size)
        if panel == 0:
            ax.annotate("", xy=(0.84, 5.57), xytext=(1.0, 5.57),
                        arrowprops={"arrowstyle": "->", "lw": 0.65, "mutation_scale": 6})
    handles = [Line2D([], [], marker="o", linestyle="none", color="#4d4d4d", markerfacecolor="white", markersize=3.5, label="PACE"),
               Line2D([], [], marker="o", linestyle="none", color="#4d4d4d", markersize=3.5, label="MDCR")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2,
               frameon=False, fontsize=font_size, handletextpad=0.35, columnspacing=1.3)
    fig.savefig(dest.with_suffix(".pdf"))
    fig.savefig(dest.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {dest.with_suffix('.pdf')}")
    print(f"Saved: {dest.with_suffix('.png')}")


if __name__ == "__main__":
    main()
