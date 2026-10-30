#!/bin/bash
#SBATCH --job-name=b3_1_estimate
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b3_1: the spec ladder (estimate_ladder.R via ladder_driver.sh) + results-tree assembly.
# Depends on b1_1 (hdps_covars) and, for embedding specs, b2_1. Roots: results/mdcr +
# intermediate/mdcr (RESULTS_ROOT / INTER_ROOT override); ENABLE_EMBEDDINGS=1 (default) adds the embedding specs; SPECS /
# BASELINE_SPECS / EMBED_SPECS override the ladder (see ladder_driver.sh).
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac  # tolerate sbatch from a section dir
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root: cd <root> && sbatch b3_estimate/b3_1_estimate.sh"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b3_1_estimate_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u  # conda activate.d scripts reference unset vars

command -v Rscript >/dev/null || { echo "ERROR: reclaim env not active (Rscript not found)"; exit 1; }

# ---- data roots (as in b0_2_build_cohorts) ----
export RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr} INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
echo "###### b3_1_estimate estimation: ($RESULTS_ROOT, $INTER_ROOT) ######"

bash b3_estimate/ladder_driver.sh || { echo "ERROR: ladder finished with failed cells (see above)"; exit 1; }
echo "Estimation panel complete -> $RESULTS_ROOT/"
