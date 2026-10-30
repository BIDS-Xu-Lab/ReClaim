#!/bin/bash
#SBATCH --job-name=b1_1_select_covariates
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b1_1: hdPS covariate selection (autoCovariateSelection steps 2-4) per cohort x cell.
# Roots: results/mdcr + intermediate/mdcr (RESULTS_ROOT / INTER_ROOT override).
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac  # tolerate sbatch from a section dir
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root: cd <root> && sbatch b1_baseline/b1_1_select_covariates.sh"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b1_1_select_covariates_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u  # conda activate.d scripts reference unset vars

command -v Rscript >/dev/null || { echo "ERROR: reclaim env not active (Rscript not found)"; exit 1; }
# Fail loudly if the R deps are missing -- otherwise every Rscript dies on `library(...)` and,
# without set -e, the loop still prints "complete".
Rscript -e 'for (p in c("nanoparquet","autoCovariateSelection","dplyr","optparse")) if (!requireNamespace(p, quietly=TRUE)) { cat("MISSING R package:", p, "\n"); quit(status=1) }' \
  || { echo "ERROR: required R packages missing (see above). Install into the reclaim R lib first."; exit 1; }
FAILED=()

# ---- data roots (as in b0_2_build_cohorts) ----
RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
echo "###### b1_1_select_covariates covariate selection: ($RESULTS_ROOT, $INTER_ROOT) ######"

# negative_control per-season sub-cohorts: covariate selection runs per cell ({stratum}_s{season}),
# so each season gets its own hdPS covariate set.
NEGATIVE_SEASONS=(2011)
POSITIVE_STRATA=(${POSITIVE_STRATA:-MDCR})
NEGATIVE_STRATA=(${NEGATIVE_STRATA:-MDCR})

for COHORT in ${COHORTS:-positive_control_celecoxib_gi negative_control_flu_hipfx}; do
  if [ "$COHORT" = "negative_control_flu_hipfx" ]; then STRATA=("${NEGATIVE_STRATA[@]}"); else STRATA=("${POSITIVE_STRATA[@]}"); fi
  for STRATUM in "${STRATA[@]}"; do
    if [ "$COHORT" = "negative_control_flu_hipfx" ]; then
      CELLS=(); for S in "${NEGATIVE_SEASONS[@]}"; do CELLS+=("${STRATUM}_s${S}"); done
    else
      CELLS=("$STRATUM")
    fi
    for CELL in "${CELLS[@]}"; do
      LONG="$INTER_ROOT/hdps_long/${COHORT}_${CELL}.parquet"
      COH="$INTER_ROOT/cohorts/${COHORT}/${CELL}/cohort.csv"
      OUT="$INTER_ROOT/hdps_covars/${COHORT}_${CELL}"
      [ -d "$LONG" ] || { echo "SKIP $COHORT/$CELL: $LONG missing"; continue; }
      echo "==== select covariates: $COHORT / $CELL ===="
      Rscript b1_baseline/select_covariates.R --long_parquet "$LONG" --cohort_csv "$COH" \
        --out_dir "$OUT" --n 200 --min_num_patients 100 --k 500
    done
  done
done
echo "Covariate selection complete (negative_control = per season)."
