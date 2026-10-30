#!/bin/bash
#SBATCH --job-name=build_train_cohort
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=1800G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e1: supervision pool. Sample BASELINE_SAMPLE_N enrollees from final_data/train, draw
# retrospective prediction points, split 90/10 into train/val, build the LightGBM feature tables.
# Sampling and the split are keyed on a salted hash of the enrollee id (utils.py), so they are
# identical on any copy of the corpus; the job logs a layout-independent corpus fingerprint.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
${CONDA_PYTHON} e1_preprocessing/preprocess_find_prediction_point.py \
    --mode train --reclaim_data_dir "${TRAIN_DATA_DIR}" --sample_n "${BASELINE_SAMPLE_N}" \
    --processed_data_dir "${BASELINE_DIR}" --min_age "${MIN_AGE}" --seed "${SEED}" \
    --num_proc "${SLURM_CPUS_PER_TASK:-8}"
for H in "${HISTORY_LENGTHS[@]}"; do for T in "${TOKEN_FEATURE_TYPES[@]}"; do
    ${CONDA_PYTHON} e1_preprocessing/preprocess_baseline_features.py \
        --section_dirs "${BASELINE_DIR}/train" "${BASELINE_DIR}/val" \
        --vocab_path "${VOCAB_DIR}" --history_length "${H}" \
        --token_feature_type "${T}" --num_proc "${SLURM_CPUS_PER_TASK:-8}"
done; done
rm -rf "${HF_DATASETS_CACHE:?}"/* 2>/dev/null || true   # the arrow cache is per-run scratch
echo "build_train_cohort completed"
