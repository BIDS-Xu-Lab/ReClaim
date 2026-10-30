#!/bin/bash
#SBATCH --job-name=train_lightgbm
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e2: LightGBM baseline. One regressor for total expenditure and one classifier for the
# HNHC label per feature window (HISTORY_LENGTHS). deterministic=True and a fixed seed make
# retraining bit-reproducible.
#   reads : data/baseline/{train,val}/baseline_features_*.parquet
#   writes: models/lightgbm-{reg,cls}-{1year,allhist}/
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
for H in "${HISTORY_LENGTHS[@]}"; do for T in "${TOKEN_FEATURE_TYPES[@]}"; do
    W="$(hist_wtag "${H}")"
    ${CONDA_PYTHON} e2_baseline/train_lightgbm.py \
        --processed_data_dir "${BASELINE_DIR}" --history_length "${H}" \
        --token_feature_type "${T}" --output_dir "${COST_HOME}/models/lightgbm-reg-${W}" \
        --random_seed "${SEED}" --n_jobs "${SLURM_CPUS_PER_TASK:-16}"
    ${CONDA_PYTHON} e2_baseline/train_lightgbm.py --task classification --cutoffs "${CUTOFF_HNHC}" \
        --processed_data_dir "${BASELINE_DIR}" --history_length "${H}" \
        --token_feature_type "${T}" --output_dir "${COST_HOME}/models/lightgbm-cls-${W}" \
        --random_seed "${SEED}" --n_jobs "${SLURM_CPUS_PER_TASK:-16}"
done; done
echo "train_lightgbm completed"
