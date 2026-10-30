#!/bin/bash
#
# End-to-end EHR disease-onset AUC evaluation pipeline.
#
# Three stages:
#   1. prepare_eval_data_new.py        - Build per-patient evaluation sequences.
#   2. prepare_disease_case_controls_fast.py
#                                      - Build case/control IDs per (disease, sex, age) stratum.
#   3. test_case_controls.py           - Run model inference and compute per-disease AUC.
#
# Stages 1-2 are CPU-only and model-independent (the checkpoint is used for its
# tokenizer only); stage 3 needs the model and ideally GPUs. Use STAGE to split
# them across jobs, e.g. STAGE=gen on a large-memory CPU node, then STAGE=eval
# once per checkpoint on a GPU node.
#
# All paths are supplied via environment variables. Required:
#   SOURCE_DATA_DIR   - directory with trajectory parquet shards (searched recursively)
#   EVAL_DATA_DIR     - output directory for prepared evaluation artefacts
#   MODEL_PATH        - Hugging Face causal LM checkpoint to evaluate
#
# Optional overrides (defaults shown):
#   STAGE=all                    # all | gen (stages 1+2) | eval (stage 3)
#   PYTHON=python
#   CC_PREP_MODEL_PATH=$MODEL_PATH   # tokenizer used to build case/control IDs
#   EVAL_OUTPUT_DIR=$EVAL_DATA_DIR/auc_case_control_eval_${OFFSET}-${AGE_GROUP_RANGE}
#   ICD_CODE_MAP_FILE=./data/icd_code_map_with_category.csv   # set to "" to use all <DX-MAJOR_*> tokens
#   ATT_TYPE=month
#   SEQ_LEN=4096
#   BATCH_SIZE=8
#   EVAL_BATCH_SIZE=16
#   OFFSET=365.25
#   AGE_GROUP_RANGE="20,100,10"
#   AGE_INDEX=3
#   DEMO_END_INDEX=3
#   SEX_INDEX=1
#   NUM_THREADS=48
#   NUM_WORKERS=0
#   NUM_GPUS=4
#   AUC_WORKERS=12
#

set -euo pipefail

: "${SOURCE_DATA_DIR:?SOURCE_DATA_DIR is required}"
: "${EVAL_DATA_DIR:?EVAL_DATA_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"

STAGE="${STAGE:-all}"
PYTHON="${PYTHON:-python}"
CC_PREP_MODEL_PATH="${CC_PREP_MODEL_PATH:-$MODEL_PATH}"
ICD_CODE_MAP_FILE="${ICD_CODE_MAP_FILE-./data/icd_code_map_with_category.csv}"
ATT_TYPE="${ATT_TYPE:-month}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
OFFSET="${OFFSET:-365.25}"
AGE_GROUP_RANGE="${AGE_GROUP_RANGE:-20,100,10}"
AGE_INDEX="${AGE_INDEX:-3}"
DEMO_END_INDEX="${DEMO_END_INDEX:-3}"
SEX_INDEX="${SEX_INDEX:-1}"
NUM_THREADS="${NUM_THREADS:-48}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
AUC_WORKERS="${AUC_WORKERS:-12}"

DATA_FILE="${EVAL_DATA_DIR}/traj_test_pd.parquet"
CASE_CONTROL_IDS_FILE="${EVAL_DATA_DIR}/${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_DATA_DIR}/auc_case_control_eval_${OFFSET}-${AGE_GROUP_RANGE}}"

if [ "$STAGE" != "eval" ]; then
# ------------------------------------------------------------------
# Stage 1 - Prepare evaluation data
# ------------------------------------------------------------------
if [ -f "$DATA_FILE" ]; then
    echo "[stage 1] $DATA_FILE already exists - skipping."
else
    mkdir -p "$EVAL_DATA_DIR"
    "$PYTHON" prepare_eval_data_new.py \
        --data_dir "$SOURCE_DATA_DIR" \
        --output_dir "$EVAL_DATA_DIR" \
        --demo_end_index "$DEMO_END_INDEX" \
        --age_index "$AGE_INDEX" \
        --att_type "$ATT_TYPE" \
        --longitudinal
fi

# ------------------------------------------------------------------
# Stage 2 - Prepare case/control IDs (also writes disease_token_str)
# ------------------------------------------------------------------
if [ -f "$CASE_CONTROL_IDS_FILE" ]; then
    echo "[stage 2] $CASE_CONTROL_IDS_FILE already exists - skipping."
else
    "$PYTHON" prepare_disease_case_controls_fast.py \
        --data_dir "$DATA_FILE" \
        --model_path "$CC_PREP_MODEL_PATH" \
        --output_dir "$EVAL_DATA_DIR" \
        --output_file "$(basename "$CASE_CONTROL_IDS_FILE")" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" \
        --age_index "$AGE_INDEX" \
        --sex_index "$SEX_INDEX" \
        --age_group_range "$AGE_GROUP_RANGE" \
        --offset "$OFFSET" \
        --num_threads "$NUM_THREADS" \
        --num_workers "$NUM_WORKERS" \
        ${ICD_CODE_MAP_FILE:+--icd_code_map_file "$ICD_CODE_MAP_FILE"}
fi
fi  # end STAGE != eval

if [ "$STAGE" == "gen" ]; then
    echo "STAGE=gen: data generation done, skipping evaluation."
    exit 0
fi

# ------------------------------------------------------------------
# Stage 3 - Run model evaluation (re-resolves disease tokens per model tokenizer)
# ------------------------------------------------------------------
mkdir -p "$EVAL_OUTPUT_DIR"
"$PYTHON" test_case_controls.py \
    --data_file "$DATA_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --max_length "$SEQ_LEN" \
    --case_control_ids_file "$CASE_CONTROL_IDS_FILE" \
    --num_gpus "$NUM_GPUS" \
    --auc_workers "$AUC_WORKERS"

echo "Pipeline complete. Results in: $EVAL_OUTPUT_DIR"
