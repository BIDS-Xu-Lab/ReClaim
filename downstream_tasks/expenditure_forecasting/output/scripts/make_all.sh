#!/bin/bash
#SBATCH --job-name=make_output
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Format the metric tables in results/ into the paper's expenditure tables and figures; nothing is
# re-scored. Needs only results/ (no predictions, gold or GPU). From the repository root:
#     bash output/scripts/make_all.sh        or        sbatch output/scripts/make_all.sh
# PYTHON=<python with pandas + matplotlib> overrides the interpreter of env.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
. output/scripts/roster.sh
load_modules
PY="${PYTHON:-${CONDA_PYTHON}}"; T="${OUTPUT_DIR}/tables"; F="${OUTPUT_DIR}/figures"
# The metric tables this script formats (released with the repository). Check them before writing anything, so a
# missing input stops the run instead of overwriting the released tables with empty cells.
MISSING=()
for f in inference_cost.csv \
         {retrospective,prospective}/regression/regression_metrics.csv \
         {retrospective,prospective}/classification/${CUTOFF_HNHC}/classification_metrics.csv \
         {retrospective,prospective}/binary_auc/${CUTOFF_HNHC}/auc_metrics.csv; do
    [ -f "${RESULT_DIR}/${f}" ] || MISSING+=("${RESULT_DIR}/${f}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    printf 'ERROR: missing metric table: %s\n' "${MISSING[@]}"; echo "Nothing written; output/ is unchanged."; exit 1
fi
mkdir -p "${T}" "${F}"
"${PY}" output/scripts/final_tables.py --result_root "${RESULT_DIR}" --tables_dir "${T}" --models "${FINAL_TABLE_MODELS[@]}" \
    --gen_num "${FINAL_N_GENERATION}" --lightgbm_window "${FINAL_LIGHTGBM_WINDOW}" --cutoff_hnhc "${CUTOFF_HNHC}"
"${PY}" output/scripts/main_figure.py --tables_dir "${T}" --figure_dir "${F}"
"${PY}" output/scripts/mc_ablation.py --result_root "${RESULT_DIR}" --tables_dir "${T}" --figure_dir "${F}" \
    --models "${RECLAIM_MODELS[@]}" --cutoff_hnhc "${CUTOFF_HNHC}"
D="${COST_HOME}/results/discretization"   # aggregates of the discretization audit, when it has been run
if [ -f "${D}/discretization_metrics.csv" ]; then
    cp "${D}/discretization_metrics.csv" "${T}/discretization.csv"; cp "${D}/discretization_table.tex" "${T}/discretization.tex"
fi
echo "make_all completed: ${T} ${F}"
