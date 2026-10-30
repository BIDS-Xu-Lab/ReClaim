#!/bin/bash
#
# LightGBM bag-of-codes baseline: train per-disease models, then score the
# same case/control cohort used for ReClaim (scripts/eval_ehr.sh).
#
# STAGE=train  - build training cohorts from TRAIN_DATA_DIR, extract features,
#                fit one LightGBM per disease (early stopping on the validation
#                cohort) and write models to MODEL_DIR.
# STAGE=eval   - load the models in MODEL_DIR and score the evaluation cohort
#                in EVAL_DATA_DIR (traj_test_pd.parquet + case/control ids from
#                stages 1-2 of scripts/eval_ehr.sh).
# STAGE=all    - train, then eval (default).
#
# Required:
#   MODEL_DIR          - where models/vectorizer/cohorts are written (train) or read (eval)
#   FEATURE_VOCAB      - tokenizer/checkpoint directory (or JSON {token: id}) defining
#                        the DX-MAJOR / PROC / RX / ATT feature vocabulary
# Required for STAGE=train:
#   TRAIN_DATA_DIR     - training trajectory parquet shards (same schema as SOURCE_DATA_DIR)
#   VAL_DATA_DIR       - directory with traj_test_pd.parquet and the case/control ids
#                        file of a held-out validation cohort (used for early stopping)
# Required for STAGE=eval:
#   EVAL_DATA_DIR      - directory with traj_test_pd.parquet and the case/control ids file
#
# Optional (defaults shown):
#   EVAL_OUTPUT_DIR=$EVAL_DATA_DIR/lightgbm_eval_${OFFSET}-${AGE_GROUP_RANGE}
#   ICD_CODE_MAP_FILE=./data/icd_code_map_with_category.csv   # "" = every DX-MAJOR in FEATURE_VOCAB
#   TRAIN_SAMPLE_SIZE=          # subsample training patients
#   OFFSET=365.25  AGE_GROUP_RANGE="20,100,10"  ATT_TYPE=month
#   AGE_INDEX=3  DEMO_END_INDEX=3  SEQ_LEN=4096
#   NUM_WORKERS=8  COHORT_WORKERS=8  FEATURE_WORKERS=4  EVAL_WORKERS=8  LGB_N_JOBS=12
#   BASELINE_FEATURE_MODE=raw   # raw | relfreq; must match how the models were trained
#   PYTHON=python
#

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${FEATURE_VOCAB:?FEATURE_VOCAB is required}"

STAGE="${STAGE:-all}"
PYTHON="${PYTHON:-python}"
ICD_CODE_MAP_FILE="${ICD_CODE_MAP_FILE-./data/icd_code_map_with_category.csv}"
OFFSET="${OFFSET:-365.25}"
AGE_GROUP_RANGE="${AGE_GROUP_RANGE:-20,100,10}"
ATT_TYPE="${ATT_TYPE:-month}"
AGE_INDEX="${AGE_INDEX:-3}"
DEMO_END_INDEX="${DEMO_END_INDEX:-3}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_WORKERS="${NUM_WORKERS:-8}"
COHORT_WORKERS="${COHORT_WORKERS:-8}"
FEATURE_WORKERS="${FEATURE_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
LGB_N_JOBS="${LGB_N_JOBS:-12}"
TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-}"
CC_NAME="${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet"

COMMON_ARGS=(
    --output_dir "$MODEL_DIR"
    --tokens_file "$FEATURE_VOCAB"
    --offset "$OFFSET"
    --age_group_range "$AGE_GROUP_RANGE"
    --att_type "$ATT_TYPE"
    --demo_end_index "$DEMO_END_INDEX"
    --age_index "$AGE_INDEX"
    --seq_len "$SEQ_LEN"
    --xgb_n_jobs "$LGB_N_JOBS"
    --num_workers "$NUM_WORKERS"
    --cohort_workers "$COHORT_WORKERS"
    --feature_workers "$FEATURE_WORKERS"
    --resume
    --save_feature_cache
)

if [ -n "$ICD_CODE_MAP_FILE" ]; then
    TARGET_ARGS=(--icd_code_map_file "$ICD_CODE_MAP_FILE")
else
    TARGET_ARGS=(--all_diseases)
fi

if [ "$STAGE" != "eval" ]; then
    : "${TRAIN_DATA_DIR:?TRAIN_DATA_DIR is required for STAGE=train}"
    : "${VAL_DATA_DIR:?VAL_DATA_DIR is required for STAGE=train}"
    mkdir -p "$MODEL_DIR"
    "$PYTHON" run_lightgbm_baseline.py --phase all \
        "${COMMON_ARGS[@]}" \
        --train_data_dir "$TRAIN_DATA_DIR" \
        --case_control_ids_file "$VAL_DATA_DIR/$CC_NAME" \
        --val_data_file "$VAL_DATA_DIR/traj_test_pd.parquet" \
        "${TARGET_ARGS[@]}" \
        ${TRAIN_SAMPLE_SIZE:+--train_sample_size "$TRAIN_SAMPLE_SIZE"}
    echo "Training complete. Models in: $MODEL_DIR/per_disease"
fi

if [ "$STAGE" != "train" ]; then
    : "${EVAL_DATA_DIR:?EVAL_DATA_DIR is required for STAGE=eval}"
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_DATA_DIR}/lightgbm_eval_${OFFSET}-${AGE_GROUP_RANGE}}"
    mkdir -p "$EVAL_OUTPUT_DIR"
    "$PYTHON" run_lightgbm_baseline.py --phase eval --inference_only \
        "${COMMON_ARGS[@]}" \
        --eval_workers "$EVAL_WORKERS" \
        --inference_case_control_ids_file "$EVAL_DATA_DIR/$CC_NAME" \
        --inference_val_data_file "$EVAL_DATA_DIR/traj_test_pd.parquet" \
        --inference_output_dir "$EVAL_OUTPUT_DIR"
    echo "Evaluation complete. Results in: $EVAL_OUTPUT_DIR"
fi
