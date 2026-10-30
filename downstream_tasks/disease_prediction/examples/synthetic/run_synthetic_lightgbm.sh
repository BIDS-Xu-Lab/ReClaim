#!/bin/bash
#
# Train and evaluate the LightGBM baseline on synthetic data (CPU, a few minutes).
# Run run_synthetic_example.sh first with the same WORK_DIR: its evaluation
# cohort and tokenizer are reused, so the LightGBM AUCs are directly comparable
# with the ReClaim ones.
#
# Usage (from downstream_tasks/disease_prediction/):
#   bash examples/synthetic/run_synthetic_lightgbm.sh [WORK_DIR]
#

set -euo pipefail

WORK_DIR="${1:-./outputs/synthetic_example}"
PYTHON="${PYTHON:-python}"

if [ ! -f "$WORK_DIR/eval/traj_test_pd.parquet" ]; then
    echo "Run examples/synthetic/run_synthetic_example.sh $WORK_DIR first." >&2
    exit 1
fi

# Separate synthetic patients for training and for early stopping.
for split in train:1:6000 val:2:2000; do
    IFS=: read -r name seed n <<< "$split"
    "$PYTHON" examples/synthetic/make_synthetic_example.py --out_dir "$WORK_DIR/lgb_$name" \
        --seed "$seed" --n_eval "$n" --n_train 0 --train_steps 0
done

# Stages 1-2 on the validation split (CPU only).
SOURCE_DATA_DIR="$WORK_DIR/lgb_val/trajectories" \
EVAL_DATA_DIR="$WORK_DIR/lgb_val/eval" \
MODEL_PATH="$WORK_DIR/model" \
ICD_CODE_MAP_FILE="" SEQ_LEN=1024 NUM_THREADS=4 STAGE=gen PYTHON="$PYTHON" \
bash scripts/eval_ehr.sh

MODEL_DIR="$WORK_DIR/lightgbm" \
FEATURE_VOCAB="$WORK_DIR/model" \
TRAIN_DATA_DIR="$WORK_DIR/lgb_train/trajectories" \
VAL_DATA_DIR="$WORK_DIR/lgb_val/eval" \
EVAL_DATA_DIR="$WORK_DIR/eval" \
EVAL_OUTPUT_DIR="$WORK_DIR/eval/lightgbm_eval" \
ICD_CODE_MAP_FILE="" SEQ_LEN=1024 \
NUM_WORKERS=0 COHORT_WORKERS=0 FEATURE_WORKERS=0 EVAL_WORKERS=4 LGB_N_JOBS=4 \
PYTHON="$PYTHON" \
bash scripts/run_lightgbm.sh
