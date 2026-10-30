#!/bin/bash
#SBATCH --job-name=b0_1_resolve_codes
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b0_1: resolve drug generic names -> NDC codes once, up front (pandas, no Spark, no claims).
# Writes intermediate/codes_resolved/{cohort_id}/{drug_ndc_map,resolution_summary,dx_proc_codes}.csv,
# the reviewable code lists that build_cohort then uses to filter MarketScan. Run before b0_2.
# Light enough for a login node:  bash b0_preprocessing/b0_1_resolve_codes.sh
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac  # tolerate sbatch from a section dir
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root: cd <root> && sbatch b0_preprocessing/b0_1_resolve_codes.sh"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b0_1_resolve_codes_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u  # conda activate.d scripts reference unset vars

python -c "import yaml, pandas" || { echo "ERROR: reclaim env not active (python deps missing)"; exit 1; }

VOCAB_NDC=${VOCAB_NDC:?export VOCAB_NDC in env.sh}

for CFG in config/positive_control_celecoxib_gi.yaml config/negative_control_flu_hipfx.yaml; do
  echo "==== resolve codes: $CFG ===="
  python b0_preprocessing/resolve_codes.py --config "$CFG" --vocab_ndc "$VOCAB_NDC" \
    || { echo "!!!! resolve_codes FAILED: $CFG !!!!"; exit 1; }
done
echo "Resolved NDC maps written to intermediate/codes_resolved/. Review resolution_summary.csv (UNMATCHED rows) before b0_2_build_cohorts."
