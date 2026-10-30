#!/bin/bash
#
# Reproduce the EHRSHOT method-comparison panels (paper Fig. 3a-c) from the
# bundled results.
#
# Usage (from the repository root; works from any directory):
#   bash visualization/disease_prediction/run_ehrshot_figures.sh [OUT_DIR]
# Default: downstream_tasks/disease_prediction/output/figures/ehrshot
#
# Fig. 3d additionally needs ReClaim's MarketScan retrospective results:
#   MARKETSCAN_RECLAIM=/path/to/reclaim_retrospective.icd.csv bash run_ehrshot_figures.sh
#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$HERE/../../downstream_tasks/disease_prediction"
OUT_DIR="${1:-$TASK_ROOT/output/figures/ehrshot}"
PYTHON="${PYTHON:-python}"
R="$TASK_ROOT/results/ehrshot"

"$PYTHON" "$HERE/plot_method_comparison.py" \
    --reclaim "$R/reclaim_l.csv" --delphi "$R/delphi.csv" --lightgbm "$R/lightgbm.csv" \
    --dataset EHRSHOT --no_rare --out_dir "$OUT_DIR" --stem ehrshot

if [ -n "${MARKETSCAN_RECLAIM:-}" ]; then
    "$PYTHON" "$HERE/plot_dataset_shift.py" \
        --external "$R/reclaim_l.csv" --reference "$MARKETSCAN_RECLAIM" \
        --external_label EHRSHOT --reference_label MarketScan \
        --out_dir "$OUT_DIR" --stem ehrshot_vs_marketscan
fi
