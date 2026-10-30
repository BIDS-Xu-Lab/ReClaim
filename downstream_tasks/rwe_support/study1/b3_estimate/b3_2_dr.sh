#!/bin/bash
#SBATCH --job-name=b3_2_dr
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1500G
# The negative cell holds every covariate block at once (about 2.5M at-risk rows x hdps 500 +
# delphi 120 + reclaim-s 1024 + reclaim-m/l 2048 columns each).
#SBATCH --time=24:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b3_2: doubly robust step on both controls: fit and checkpoint the models
# (fit_dr_models.R), then compute the estimates from them (estimate_dr.R). The ladder is untouched.
# Inputs: cohort.csv, hdps_covars.csv, embedding parquets. Roots: RESULTS_ROOT / INTER_ROOT
# (default results/mdcr, intermediate/mdcr); DR_COHORTS="positive negative" selects the cells.
# POPULATION=at_risk|all  default at_risk, which applies to the landmark (negative) control only:
#                         rows still enrolled when the outcome window opens. The positive control
#                         always runs on all rows.
# OM_SPECS=a,b            optional subset of outcome-model columns (default: all four).
# PS_SPECS=a,b            optional PS rows (default: hdps,delphi-only,reclaim-s-only); any model
#                         with an embedding parquet, e.g. reclaim-m-only, reclaim-s-pca120-only.
# PS_CORES / OM_CORES     parallel workers for the PS rows (default 1) and outcome models (default 5).
# RUN_ESTIMATES=1         then run estimate_dr.R (default). 0 = fit and checkpoint only, e.g. when
#                         the wide outcome models are split across several jobs.
# Outcome models are fitted per arm; the reported effect in the treated uses the comparator-arm one.
# The defaults are the reported design and write <cohort>/<cell>/dr_aipw/; POPULATION=all for the
# negative control writes dr_aipw_all/ instead.
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b3_2_dr_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
command -v Rscript >/dev/null || { echo "ERROR: Rscript not on PATH"; exit 1; }

# Roots: honour RESULTS_ROOT / INTER_ROOT as the other stages do; per-cohort existence checks
# live inside run_dr(), which builds its paths from these roots.
RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
# DR_COHORTS: which cells to run (default both, so a chain can depend on a single b3_2 job).
DR_COHORTS=${DR_COHORTS:-positive negative}
POPULATION=${POPULATION:-at_risk}
OM_SPECS=${OM_SPECS:-}
PS_SPECS=${PS_SPECS:-}
RUN_ESTIMATES=${RUN_ESTIMATES:-1}
case "$POPULATION" in all|at_risk) ;; *) echo "ERROR: POPULATION must be all|at_risk"; exit 1;; esac
run_dr() {  # $1 = cohort_id, $2 = cell, $3 = days from index to the opening of the outcome window ("" = no at_risk option), $4 = the ladder's design
  local CID="$1" CELL="$2" RISK_START="$3" DESIGN="$4"
  local POP="$POPULATION"; [ -n "$RISK_START" ] || POP=all
  local REPORTED_POP=all; [ -n "$RISK_START" ] && REPORTED_POP=at_risk
  # the reported design (negative control at-risk) writes dr_aipw/; the other population gets a
  # suffixed folder so it can never overwrite the reported one
  local DEST=dr_aipw; [ "$POP" = "$REPORTED_POP" ] || DEST="dr_aipw_${POP}"
  local COHORT="${INTER_ROOT}/cohorts/${CID}/${CELL}/cohort.csv"
  local COVARS="${INTER_ROOT}/hdps_covars/${CID}_${CELL}/hdps_covars.csv"
  local EMB_DIR="${INTER_ROOT}/embeddings/${CID}_${CELL}"
  local OUT_ROOT="${RESULTS_ROOT}/${CID}/${CELL}/${DEST}"
  local CACHE_DIR="${INTER_ROOT}/nuisance_cache/${CID}_${CELL}"
  [ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run b0_2_build_cohorts)"; return 1; }
  [ -f "$COVARS" ] || { echo "ERROR: $COVARS missing (run b1_1_select_covariates)"; return 1; }
  ls "$EMB_DIR"/cohort_with_*_embeddings.parquet >/dev/null 2>&1 \
    || echo "NOTE: no embedding parquets for $CID/$CELL -- conventional cells only"
  local EXTRA=(--population "$POP")
  [ "$POP" = at_risk ] && EXTRA+=(--risk_start_days "$RISK_START")
  [ -n "$OM_SPECS" ] && EXTRA+=(--om_specs "$OM_SPECS")
  [ -n "$PS_SPECS" ] && EXTRA+=(--ps_specs "$PS_SPECS")
  local RISK=(); [ "$POP" = at_risk ] && RISK=(--risk_start_days "$RISK_START")
  echo "==== doubly robust step: $CID / $CELL  population=$POP  (roots: $RESULTS_ROOT | $INTER_ROOT) ===="
  Rscript b3_estimate/fit_dr_models.R --cohort_csv "$COHORT" --hdps_covars "$COVARS" \
    --embed_dir "$EMB_DIR" --cache_dir "$CACHE_DIR" "${EXTRA[@]}" || return 1
  [ "$RUN_ESTIMATES" = 1 ] || return 0
  Rscript b3_estimate/estimate_dr.R --cohort_csv "$COHORT" --cache_dir "$CACHE_DIR" \
    --out_root "$OUT_ROOT" --population "$POP" --design "$DESIGN" "${RISK[@]}"
}
FAILED=()
for C in $DR_COHORTS; do
  case "$C" in
    positive) run_dr positive_control_celecoxib_gi MDCR       ""  cohort_logistic || FAILED+=("positive");;
    # Oct-1 landmark; the outcome window opens Jan 1 (config/negative_control_flu_hipfx.yaml), 92 days on
    negative) run_dr negative_control_flu_hipfx    MDCR_s2011 92  landmark_cox    || FAILED+=("negative");;
    *) echo "ERROR: unknown DR_COHORTS entry '$C'"; exit 1;;
  esac
done
[ ${#FAILED[@]} -gt 0 ] && { echo "!!!! doubly robust step FAILED for: ${FAILED[*]} !!!!"; exit 1; }
echo "==== b3_2_dr completed OK ===="
