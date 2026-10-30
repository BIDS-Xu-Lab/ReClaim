#!/bin/bash
#
# Delphi baseline on the same EHR case/control cohort as ReClaim.
#
# Three stages:
#   1. delphi_baseline/preprocess.py                - Convert eval_seq to Delphi's binary format.
#   2. delphi_baseline/transform_predefined_cohort.py
#                                                   - Map the cohort to Delphi patients/prefixes.
#   3. delphi_baseline/evaluate.py                  - Score the checkpoint and compute AUC.
#
# Stages 1-2 are CPU-only (stage 1 uses PySpark and needs Java); stage 3 uses a GPU if
# available. Training is not included: DELPHI_CKPT_DIR must hold a trained ckpt.pt.
#
# All paths are supplied via environment variables. Required:
#   EVAL_DATA_DIR     - directory written by scripts/eval_ehr.sh stages 1-2
#                       (traj_test_pd.parquet + case/control parquet)
#   DELPHI_CKPT_DIR   - directory containing a trained Delphi ckpt.pt
#
# Optional overrides (defaults shown):
#   STAGE=all                    # all | gen (stages 1+2) | eval (stage 3)
#   PYTHON=python
#   OFFSET=365.25
#   AGE_GROUP_RANGE="20,100,10"
#   CASE_CONTROL_IDS_FILE=$EVAL_DATA_DIR/${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet
#   DELPHI_WORK_DIR=$EVAL_DATA_DIR/delphi       # Delphi binary, prepared cohort, logs
#   EVAL_OUTPUT_DIR=$DELPHI_WORK_DIR/auc_${OFFSET}-${AGE_GROUP_RANGE}
#   PID_COL=person_id
#   ATT_TIME_UNIT=months
#   SPARK_DRIVER_MEMORY=32g
#   BATCH_SIZE=64
#

set -euo pipefail

: "${EVAL_DATA_DIR:?EVAL_DATA_DIR is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELPHI_DIR="$(cd "${SCRIPT_DIR}/../delphi_baseline" && pwd)"

STAGE="${STAGE:-all}"
PYTHON="${PYTHON:-python}"
OFFSET="${OFFSET:-365.25}"
AGE_GROUP_RANGE="${AGE_GROUP_RANGE:-20,100,10}"
CASE_CONTROL_IDS_FILE="${CASE_CONTROL_IDS_FILE:-${EVAL_DATA_DIR}/${OFFSET}-${AGE_GROUP_RANGE}-disease_case_control_ids.parquet}"
DELPHI_WORK_DIR="${DELPHI_WORK_DIR:-${EVAL_DATA_DIR}/delphi}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${DELPHI_WORK_DIR}/auc_${OFFSET}-${AGE_GROUP_RANGE}}"
PID_COL="${PID_COL:-person_id}"
ATT_TIME_UNIT="${ATT_TIME_UNIT:-months}"
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-32g}"
BATCH_SIZE="${BATCH_SIZE:-64}"

DATA_FILE="${EVAL_DATA_DIR}/traj_test_pd.parquet"
DISEASE_MAP="${DELPHI_DIR}/nature_disease_map.csv"
BIN_ROOT="${DELPHI_WORK_DIR}/data"
PREPARED_COHORT="${DELPHI_WORK_DIR}/predefined_cohort.json"

mkdir -p "${DELPHI_WORK_DIR}"
cd "${DELPHI_WORK_DIR}"   # scripts write their logs/ here

find_bin_dir() {
    ls -dt "${BIN_ROOT}"/ehr_data_test_* 2>/dev/null | head -1
}

if [ "$STAGE" != "eval" ]; then
# ------------------------------------------------------------------
# Stage 1 - eval_seq -> Delphi binary
# ------------------------------------------------------------------
if [ -n "$(find_bin_dir)" ]; then
    echo "[stage 1] $(find_bin_dir) already exists - skipping."
else
    "$PYTHON" "${DELPHI_DIR}/preprocess.py" \
        --input_folder_path "$DATA_FILE" \
        --output_folder_path "$BIN_ROOT" \
        --delphi_disease_map "$DISEASE_MAP" \
        --split test \
        --output_prefix ehr \
        --pid_col "$PID_COL" \
        --seq_col eval_seq \
        --att_time_unit "$ATT_TIME_UNIT" \
        --att_time_mode relative \
        --keep_sex_only_patients \
        --spark_driver_memory "$SPARK_DRIVER_MEMORY"
fi
BIN_DIR="$(find_bin_dir)"

# ------------------------------------------------------------------
# Stage 2 - case/control cohort -> Delphi patient indices and prefixes
# ------------------------------------------------------------------
if [ -f "$PREPARED_COHORT" ]; then
    echo "[stage 2] $PREPARED_COHORT already exists - skipping."
else
    "$PYTHON" "${DELPHI_DIR}/transform_predefined_cohort.py" \
        --case_control_file "$CASE_CONTROL_IDS_FILE" \
        --traj_parquet "$DATA_FILE" \
        --delphi_bin "${BIN_DIR}/test.bin" \
        --pid_mapping "${BIN_DIR}/pid_mapping.tsv" \
        --delphi_disease_map "$DISEASE_MAP" \
        --pid_col "$PID_COL" \
        --seq_col eval_seq \
        --att_time_unit "$ATT_TIME_UNIT" \
        --att_time_mode relative \
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
: "${DELPHI_CKPT_DIR:?DELPHI_CKPT_DIR is required for evaluation}"
BIN_DIR="$(find_bin_dir)"
"$PYTHON" "${DELPHI_DIR}/evaluate.py" \
    --out_dir "$DELPHI_CKPT_DIR" \
    --val_path "${BIN_DIR}/test.bin" \
    --predefined_cohort "$PREPARED_COHORT" \
    --offset "$OFFSET" \
    --batch_size "$BATCH_SIZE" \
    --labels_file "${DELPHI_DIR}/delphi_labels_chapters_colours_icd.csv" \
    --output_dir "$EVAL_OUTPUT_DIR"

echo "Delphi evaluation complete. Results in: $EVAL_OUTPUT_DIR"
