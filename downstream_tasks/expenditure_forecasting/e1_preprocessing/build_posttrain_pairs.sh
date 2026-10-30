#!/bin/bash
#SBATCH --job-name=build_posttrain_pairs
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e1: post-training pairs (N_TRAIN_PAIRS train, N_VAL_PAIRS validation) drawn from the
# supervision pool by the same id-hash rule, with labels censored at $1M and expressed as one
# <COST-me> token. Needs build_train_cohort.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
${CONDA_PYTHON} e1_preprocessing/build_cost_posttrain_data.py --baseline_dir "${BASELINE_DIR}" \
    --out_dir "${PAIRS_DIR}" --n_train_pairs "${N_TRAIN_PAIRS}" --n_val_pairs "${N_VAL_PAIRS}"
echo "build_posttrain_pairs completed"
