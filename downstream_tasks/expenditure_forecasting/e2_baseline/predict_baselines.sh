#!/bin/bash
#SBATCH --job-name=predict_baselines
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e2: evaluation-cohort predictions of the CPU baselines -- the LightGBM models and the two
# non-learned anchors (persistence = last observed-year cost; trainmean = training mean; both capped at $1M).
#   reads : models/lightgbm-*, data/test/<section>/baseline_features_*.parquet
#   writes: data/test/<section>/model_gen/{lightgbm-*,persistence,trainmean}/cost_predictions.csv
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
for SECTION in retrospective prospective; do
    SECTION_DIR="${DATA_DIR}/${SPLIT}/${SECTION}"
    for H in "${HISTORY_LENGTHS[@]}"; do for T in "${TOKEN_FEATURE_TYPES[@]}"; do
        W="$(hist_wtag "${H}")"
        FEATS="${SECTION_DIR}/baseline_features_${H}_${T}.parquet"
        ${CONDA_PYTHON} e2_baseline/predict_lightgbm.py \
            --model_dir "${COST_HOME}/models/lightgbm-reg-${W}" \
            --data_path "${FEATS}" --output_dir "${SECTION_DIR}/model_gen/lightgbm-reg-${W}"
        ${CONDA_PYTHON} e2_baseline/predict_lightgbm.py --task classification \
            --model_dir "${COST_HOME}/models/lightgbm-cls-${W}" \
            --data_path "${FEATS}" --output_dir "${SECTION_DIR}/model_gen/lightgbm-cls-${W}"
    done; done
    ${CONDA_PYTHON} e2_baseline/predict_anchors.py \
        --baseline_train "${BASELINE_DIR}/train/preprocessed_input.parquet" \
        --section_dirs "${SECTION_DIR}" --history_length "${HISTORY_LENGTH}" \
        --token_feature_type "${TOKEN_FEATURE_TYPES[0]}"
done
echo "predict_baselines completed"
