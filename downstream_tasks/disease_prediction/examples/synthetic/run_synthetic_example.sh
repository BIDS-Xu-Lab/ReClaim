#!/bin/bash
#
# Run the full three-stage evaluation on a synthetic cohort (CPU, a few minutes).
#
# Usage (from downstream_tasks/disease_prediction/):
#   bash examples/synthetic/run_synthetic_example.sh [WORK_DIR]
#
# WORK_DIR defaults to ./outputs/synthetic_example and is safe to delete.
#

set -euo pipefail

WORK_DIR="${1:-./outputs/synthetic_example}"
PYTHON="${PYTHON:-python}"

"$PYTHON" examples/synthetic/make_synthetic_example.py --out_dir "$WORK_DIR"

SOURCE_DATA_DIR="$WORK_DIR/trajectories" \
EVAL_DATA_DIR="$WORK_DIR/eval" \
MODEL_PATH="$WORK_DIR/model" \
ICD_CODE_MAP_FILE="" \
SEQ_LEN=1024 \
NUM_THREADS=4 \
NUM_GPUS=0 \
CUDA_VISIBLE_DEVICES="" \
AUC_WORKERS=4 \
PYTHON="$PYTHON" \
bash scripts/eval_ehr.sh
