#!/bin/bash
#SBATCH --job-name=b3_2b_estimate_dr
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b3_2b: the doubly robust estimates alone (estimate_dr.R), from the nuisances that
# b3_2_dr.sh checkpointed. Nothing is refitted. DR_COHORTS / POPULATION as in b3_2_dr.sh.
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b3_2b_estimate_dr_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"
module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
DR_COHORTS=${DR_COHORTS:-positive negative}
POPULATION=${POPULATION:-at_risk}
run_one() {  # $1 = cohort_id, $2 = cell, $3 = days from index to the opening of the outcome window ("" = none), $4 = the ladder's design
  local CID="$1" CELL="$2" RISK_START="$3" DESIGN="$4"
  local POP="$POPULATION"; [ -n "$RISK_START" ] || POP=all
  local REPORTED_POP=all; [ -n "$RISK_START" ] && REPORTED_POP=at_risk
  # the reported design (negative control at-risk) writes dr_aipw/; the other population gets a
  # suffixed folder so it can never overwrite the reported one
  local DEST=dr_aipw; [ "$POP" = "$REPORTED_POP" ] || DEST="dr_aipw_${POP}"
  local RISK=(); [ "$POP" = at_risk ] && RISK=(--risk_start_days "$RISK_START")
  Rscript b3_estimate/estimate_dr.R --cohort_csv "${INTER_ROOT}/cohorts/${CID}/${CELL}/cohort.csv" \
    --cache_dir "${INTER_ROOT}/nuisance_cache/${CID}_${CELL}" \
    --out_root "${RESULTS_ROOT}/${CID}/${CELL}/${DEST}" \
    --population "$POP" --design "$DESIGN" "${RISK[@]}"
}
FAILED=()
for C in $DR_COHORTS; do
  case "$C" in
    positive) run_one positive_control_celecoxib_gi MDCR       "" cohort_logistic || FAILED+=("positive");;
    negative) run_one negative_control_flu_hipfx    MDCR_s2011 92 landmark_cox    || FAILED+=("negative");;
    *) echo "ERROR: unknown DR_COHORTS entry '$C'"; exit 1;;
  esac
done
[ ${#FAILED[@]} -gt 0 ] && { echo "!!!! estimate_dr FAILED for: ${FAILED[*]} !!!!"; exit 1; }
echo "==== b3_2b_estimate_dr completed OK ===="
