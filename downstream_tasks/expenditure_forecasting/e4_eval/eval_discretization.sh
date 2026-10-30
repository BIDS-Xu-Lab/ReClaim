#!/bin/bash
#SBATCH --job-name=eval_discretization
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e4 (run once; independent of models): cost-token discretization audit against the raw
# claims for the evaluation cohorts. Reconstructs exact monthly sums, tokenizes them with the
# pre-training operator, compares annual totals (figure) and reports R2, HNHC prevalence and label
# changes overall and by payer (table); cross-checks the prospective labels. Needs MS_RAW_DIR.
# The reconstruction skips existing results/discretization/monthly/ files: clear that folder
# before re-running on a changed cohort.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
RAW_DIR="${MS_RAW_DIR:?set MS_RAW_DIR in env.sh (raw MarketScan claims)}"
LABELS="${DATA_DIR}/${SPLIT}/prospective/preprocessed_input.parquet"
EXTRA=(); [ -f "${LABELS}" ] && EXTRA=(--prospective_labels "${LABELS}")
${CONDA_PYTHON} e4_eval/quantify_cost_discretization.py --phase all --raw_dir "${RAW_DIR}" \
    --cohort_dir "${DATA_DIR}/${SPLIT}/retrospective" "${DATA_DIR}/${SPLIT}/prospective" \
    --out_dir "${COST_HOME}/results/discretization" --figure_dir "${OUTPUT_DIR}/figures" \
    --years $(seq 2008 2023) --workers 24 "${EXTRA[@]}"
bash output/scripts/make_all.sh || echo "WARN: output/scripts/make_all.sh failed"
echo "eval_discretization completed"
