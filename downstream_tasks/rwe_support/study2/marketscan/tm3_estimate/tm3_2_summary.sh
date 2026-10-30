#!/bin/bash
#SBATCH --job-name=tm3_2_summary
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm3_2_summary: collect results/<comparison>/<spec>/<estimator>/estimate.json into summary
# tables, forest/KM figures and a RESULTS.md digest under results/_summary/.
# Usage: sbatch tm3_estimate/tm3_2_summary.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (codesets_tte, marketscan_io, utils)
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm3_2_summary_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
command -v Rscript >/dev/null || { echo "ERROR: Rscript not on PATH"; exit 1; }

Rscript tm3_estimate/summarize_results.R --results_root results \
  --out_dir results/_summary || { echo "!!!! summarize_results FAILED !!!!"; exit 1; }
echo "==== tm3_2_summary completed OK ===="
