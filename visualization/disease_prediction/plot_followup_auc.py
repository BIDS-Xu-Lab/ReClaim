#!/usr/bin/env python3
"""Disease-level AUC across prediction horizons (OFFSET values).

Paper panel: Fig. 4e. Each input is the same model evaluated with a different
`OFFSET` (e.g. 30, 180, 365.25, 1825 days); points are colored by ICD-10
chapter and only diseases evaluated at every horizon are kept.

    python plot_followup_auc.py --input 1=auc_30d.csv --input 6=auc_180d.csv \
        --input 12=auc_365d.csv --input 60=auc_1825d.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_OUT_DIR, FALLBACK_COLOR, align, chapter_table, draw_box, read_auc, save, set_style

HORIZON_COLORS = ["#5C7DA5", "#7B93B4", "#D07C5B", "#C56D53", "#B75B4D", "#A04A40"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=CSV",
                        help="Horizon label (months) and its icd_code-keyed AUC table; repeat in order")
    parser.add_argument("--all_diseases", action="store_true",
                        help="Use every disease per horizon instead of the shared set")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="followup_auc")
    args = parser.parse_args()

    specs = [s.split("=", 1) for s in args.input]
    frames = {label: read_auc(path) for label, path in specs}
    color_of = chapter_table().set_index("icd_code")["color"]
    if args.all_diseases:
        series = {k: df.set_index("icd_code")["auc"] for k, df in frames.items()}
    else:
        wide = align(frames)
        series = {k: wide[k] for k in frames}
    labels = list(series)
    groups = [series[k].to_numpy() for k in labels]
    point_colors = [series[k].index.map(color_of).fillna(FALLBACK_COLOR).to_numpy() for k in labels]

    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    draw_box(ax, groups, HORIZON_COLORS[: len(labels)], point_colors=point_colors)
    ax.axhline(0.5, color="#6E6E6E", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_title("Disease-Level AUC Across Prediction Horizons")
    fig.tight_layout()
    save(fig, args.out_dir / args.stem)

    summary = pd.DataFrame({"months": labels, "n": [len(g) for g in groups],
                            "mean_auc": [g.mean() for g in groups], "median_auc": [np.median(g) for g in groups]})
    summary.to_csv(args.out_dir / f"{args.stem}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
