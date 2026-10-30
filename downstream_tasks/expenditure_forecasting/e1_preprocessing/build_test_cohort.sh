#!/bin/bash
#SBATCH --job-name=build_test_cohort
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=1500G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e1: held-out evaluation cohort. Retrospective and prospective prediction points from
# final_data/test, then the LightGBM feature tables for both sections.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
${CONDA_PYTHON} e1_preprocessing/preprocess_find_prediction_point.py \
    --mode test --reclaim_data_dir "${TEST_DATA_DIR}" --processed_data_dir "${DATA_DIR}" \
    --min_age "${MIN_AGE}" --seed "${SEED}" --num_proc "${SLURM_CPUS_PER_TASK:-8}"
for H in "${HISTORY_LENGTHS[@]}"; do for T in "${TOKEN_FEATURE_TYPES[@]}"; do
    ${CONDA_PYTHON} e1_preprocessing/preprocess_baseline_features.py \
        --section_dirs "${DATA_DIR}/${SPLIT}/retrospective" "${DATA_DIR}/${SPLIT}/prospective" \
        --vocab_path "${VOCAB_DIR}" --history_length "${H}" \
        --token_feature_type "${T}" --num_proc "${SLURM_CPUS_PER_TASK:-8}"
done; done
rm -rf "${HF_DATASETS_CACHE:?}"/* 2>/dev/null || true   # the arrow cache is per-run scratch
echo "build_test_cohort completed"
