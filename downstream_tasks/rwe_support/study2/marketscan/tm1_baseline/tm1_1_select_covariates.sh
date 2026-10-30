#!/bin/bash
#SBATCH --job-name=tm1_1_select_covariates
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm1_1_select_covariates: Schneeweiss hdPS covariate selection (n=200 per dimension,
# k=500) on the union long table, one run per emulation cohort.
# Inputs : intermediate/hdps_long/tte_glp1_psych.parquet, intermediate/cohorts/<emulation>/cohort.csv
# Output : intermediate/hdps_covars/<comparison>/hdps_covars.csv
# Usage  : sbatch tm1_baseline/tm1_1_select_covariates.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (codesets_tte, marketscan_io, utils)
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm1_1_select_covariates_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
command -v Rscript >/dev/null || { echo "ERROR: Rscript not on PATH"; exit 1; }

LONG=intermediate/hdps_long/tte_glp1_psych.parquet
EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}   # emulation_dir:comparison:treat:comp
for EM in $EMULATIONS; do
  IFS=: read -r EDIR CID TCLS CCLS <<<"$EM"
  COHORT="intermediate/cohorts/$EDIR/cohort.csv"
  [ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run tm0_1_preprocessing)"; exit 1; }
  echo "==== select_covariates: $CID ($TCLS vs $CCLS, emulation $EDIR) ===="
  Rscript tm1_baseline/select_covariates.R --long_parquet "$LONG" --cohort_csv "$COHORT" \
    --treat_class "$TCLS" --comp_class "$CCLS" \
    --out_dir "intermediate/hdps_covars/$CID" || { echo "!!!! select_covariates $CID FAILED !!!!"; exit 1; }
done
echo "==== tm1_1_select_covariates completed OK ===="
