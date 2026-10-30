#!/bin/bash
#
# ETHOS baseline on the same EHRSHOT case/control cohort as ReClaim.
#
# Three stages:
#   1. ethos_baseline/preprocess.py                 - OMOP tables -> ETHOS timelines (HDF5).
#   2. ethos_baseline/transform_predefined_cohort.py
#                                                   - Map the cohort onto those timelines.
#   3. ethos_baseline/evaluate.py                   - Score the checkpoint and compute AUC.
#
# Stage 1 uses PySpark and needs Java; stages 2-3 are CPU except the model, which uses a GPU if
# available. Training is not included: ETHOS_MODEL must hold a checkpoint pretrained by the
# upstream ETHOS project.
#
# All paths are supplied via environment variables. Required:
#   EHRSHOT_OMOP_DIR  - EHRSHOT OMOP CDM parquet tables
#   ETHOS_VOCAB       - vocabulary the checkpoint was trained with
#   ETHOS_MODEL       - ETHOS checkpoint
#   ETHOS_REF_DIR     - ethos/data of an ETHOS checkout, holding the reference tables
#   EVAL_DATA_DIR     - directory written by scripts/eval_ehr.sh stages 1-2
#                       (traj_test_pd.parquet + case/control parquet)
#
# Optional overrides (defaults shown):
#   STAGE=all                    # all | gen (stages 1+2) | eval (stage 3)
#   PYTHON=python
#   OFFSET=365.25
#   AGE_GROUP_RANGE="20,100,10"
#   CASE_CONTROL_IDS_FILE=$EVAL_DATA_DIR/${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet
#   ETHOS_WORK_DIR=$EVAL_DATA_DIR/ethos          # timelines, prepared cohort, logs
#   EVAL_OUTPUT_DIR=$ETHOS_WORK_DIR/auc_${OFFSET}-${AGE_GROUP_RANGE}
#   TARGET_TOKENS=ethos_baseline/target_tokens.json
#   BATCH_SIZE=32
#

set -euo pipefail

: "${EVAL_DATA_DIR:?EVAL_DATA_DIR is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETHOS_DIR="$(cd "${SCRIPT_DIR}/../ethos_baseline" && pwd)"

STAGE="${STAGE:-all}"
PYTHON="${PYTHON:-python}"
OFFSET="${OFFSET:-365.25}"
AGE_GROUP_RANGE="${AGE_GROUP_RANGE:-20,100,10}"
CASE_CONTROL_IDS_FILE="${CASE_CONTROL_IDS_FILE:-${EVAL_DATA_DIR}/${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet}"
ETHOS_WORK_DIR="${ETHOS_WORK_DIR:-${EVAL_DATA_DIR}/ethos}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ETHOS_WORK_DIR}/auc_${OFFSET}-${AGE_GROUP_RANGE}}"
TARGET_TOKENS="${TARGET_TOKENS:-${ETHOS_DIR}/target_tokens.json}"
BATCH_SIZE="${BATCH_SIZE:-32}"

TIMELINES="${ETHOS_WORK_DIR}/timelines.hdf5"
PREPARED_COHORT="${ETHOS_WORK_DIR}/predefined_cohort.json"

mkdir -p "${ETHOS_WORK_DIR}"
cd "${ETHOS_WORK_DIR}"   # scripts write their logs/ here

if [ "$STAGE" != "eval" ]; then
: "${ETHOS_VOCAB:?ETHOS_VOCAB is required to build the timelines}"
: "${ETHOS_REF_DIR:?ETHOS_REF_DIR is required: ethos/data of an ETHOS checkout}"

# ------------------------------------------------------------------
# Stage 1 - OMOP tables -> one timeline per patient
# ------------------------------------------------------------------
if [ -f "$TIMELINES" ]; then
    echo "[stage 1] $TIMELINES already exists - skipping."
else
    : "${EHRSHOT_OMOP_DIR:?EHRSHOT_OMOP_DIR is required to build the timelines}"
    "$PYTHON" "${ETHOS_DIR}/preprocess.py" \
        --out_dir "$ETHOS_WORK_DIR" \
        --timelines "$TIMELINES" \
        --ref_dir "$ETHOS_REF_DIR"
fi

# ------------------------------------------------------------------
# Stage 2 - case/control cohort -> ETHOS patients and cut points
# ------------------------------------------------------------------
if [ -f "$PREPARED_COHORT" ]; then
    echo "[stage 2] $PREPARED_COHORT already exists - skipping."
else
    "$PYTHON" "${ETHOS_DIR}/transform_predefined_cohort.py" \
        --timelines "$TIMELINES" \
        --ref_dir "$ETHOS_REF_DIR" \
        --case_control_ids "$CASE_CONTROL_IDS_FILE" \
        --traj_parquet "${EVAL_DATA_DIR}/traj_test_pd.parquet" \
        --target_tokens "$TARGET_TOKENS" \
        --output "$PREPARED_COHORT"
fi
fi  # end STAGE != eval

if [ "$STAGE" == "gen" ]; then
    echo "STAGE=gen: data generation done, skipping evaluation."
    exit 0
fi

# ------------------------------------------------------------------
# Stage 3 - score the checkpoint
# ------------------------------------------------------------------
: "${ETHOS_VOCAB:?ETHOS_VOCAB is required for evaluation}"
: "${ETHOS_MODEL:?ETHOS_MODEL is required for evaluation}"
"$PYTHON" "${ETHOS_DIR}/evaluate.py" \
    --timelines "$TIMELINES" \
    --predefined_cohort "$PREPARED_COHORT" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$EVAL_OUTPUT_DIR"

echo "ETHOS evaluation complete. Results in: $EVAL_OUTPUT_DIR"
