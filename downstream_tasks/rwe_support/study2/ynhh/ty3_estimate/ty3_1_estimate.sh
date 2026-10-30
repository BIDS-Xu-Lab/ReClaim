#!/bin/bash
#SBATCH --job-name=ty3_1_estimate
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Relative output path: submit from the module root.
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# Stage ty3_1 (estimate_ladder): the spec ladder x estimator grid, run per emulation cohort.
# Needs ty1_1 (hdps spec) and, with ENABLE_EMBEDDINGS=1, ty2_1 (embedding specs).
# Output: results/<cell>/<comparison>/<spec>/<estimator>/ + results/<cell>/RUN_MANIFEST.json.
#   SPECS="crude demo paper hdps"      spec subset      EMB_MODELS="reclaim-s reclaim-m reclaim-l"
#   PS_ENGINE=ridge|lasso              embedding-spec PS engine
#   CALIPER_SWEEP="0.05,0.1,0.2,0.5,1" psm_cox caliper sweep (default on; "" = off)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root"; exit 1; }
[ -d config ] || { echo "ERROR: not the study2/ynhh root (no config/)"; exit 1; }
# env.sh: all site-specific paths + CRAN_REPO (pure exports, sourced by every stage)
. "$MODULE_ROOT/env.sh" || { echo "ERROR: cannot source $MODULE_ROOT/env.sh"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
_LOG="$MODULE_ROOT/slurm_job_logs/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

# Runtime environment: miniconda env by name (RECLAIM_ENV, default hdps) plus a JVM for pyspark.
# The cluster's Spark module is never loaded: its PYTHONPATH would shadow the conda env and its
# SPARK_HOME the pip pyspark.
module purge
module load miniconda
module load Java/17 2>/dev/null || module load Java/17.0.4 2>/dev/null || module load Java 2>/dev/null || true
unset SPARK_HOME
# PYTHONPATH is reset (not unset) to the module root: drops any module-injected path while keeping
# the root-level imports (datasources, codesets_tte, utils) for the section scripts.
export PYTHONPATH="$MODULE_ROOT"
# CONDA_EXE / CONDA_PYTHON_EXE stay set: `conda` may be a shell function dispatching through CONDA_EXE.
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
RECLAIM_ENV=${RECLAIM_ENV:-hdps}
set +u
# conda.sh probe order: EasyBuild module root, `conda info --base`, then the conda binary on PATH
# (`type -P`, since `command -v` can return the bare name of a shell function).
CONDA_BASE="${EBROOTMINICONDA:-}"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  _CBIN="$(type -P conda 2>/dev/null || true)"
  [ -n "$_CBIN" ] && CONDA_BASE="$(dirname "$(dirname "$_CBIN")")"
fi
[ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "ERROR: cannot locate conda.sh"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u
PY="${CONDA_PREFIX:?conda activate failed}/bin/python"; RSCRIPT="$CONDA_PREFIX/bin/Rscript"; export RSCRIPT
export PYSPARK_PYTHON="$PY" PYSPARK_DRIVER_PYTHON="$PY"
# Provenance guard: the imports must come from THIS env, so a shadowed interpreter fails at the
# top of the log rather than deep inside a stage.
"$PY" -c "import sys,yaml,pandas; assert pandas.__file__.startswith(sys.prefix), pandas.__file__; print('env OK:', sys.prefix)" \
  || { echo "ERROR: conda env broken or shadowed (PY=$PY)"; exit 1; }
if [ -z "${SPARK_LOCAL_DIRS:-}" ]; then
  export SPARK_LOCAL_DIRS="/tmp/spark_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "$SPARK_LOCAL_DIRS"' EXIT
fi
mkdir -p "$SPARK_LOCAL_DIRS" || { echo "ERROR: cannot create SPARK_LOCAL_DIRS=$SPARK_LOCAL_DIRS"; exit 1; }
chmod 700 "$SPARK_LOCAL_DIRS" 2>/dev/null || true
echo "SPARK_LOCAL_DIRS=$SPARK_LOCAL_DIRS"

SOURCE=${SOURCE:-ynhh}
# ROUTE: dx matching route (build_cohort --dx_route), must match across stages. source (default)
# = ICD-10 prefixes on *_source_value, the reference's claims semantics, cell ${SOURCE}_src;
# standard = resolved standard concepts, cell ${SOURCE}.
ROUTE=${ROUTE:-source}
case "$ROUTE" in standard) CELL=$SOURCE;; source) CELL=${SOURCE}_src;;
  *) echo "ERROR: unknown ROUTE '$ROUTE' (standard|source)"; exit 1;; esac
COHORT=intermediate/cohorts/$CELL/cohort.csv
EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}   # suffix:comparison:treat:comp

SPECS=${SPECS:-crude demo paper hdps}
if [ "${ENABLE_EMBEDDINGS:-0}" = "1" ]; then
  EMB_MODELS=${EMB_MODELS:-reclaim-s reclaim-m reclaim-l}
  for M in $EMB_MODELS; do SPECS="$SPECS ${M}-only"; done
fi
# PS_ENGINE: penalised engine for the embedding specs (ridge default); conventional specs
# always use plain logistic.
PS_ENGINE=${PS_ENGINE:-ridge}
# CALIPER_SWEEP: comma-separated calipers re-matched on the already-fitted PS (about 1 s per
# cell). Pass it as a plain env prefix (CALIPER_SWEEP=... sbatch ...), never via
# `sbatch --export=ALL,CALIPER_SWEEP=...`: --export splits on commas and would truncate the list.
CALIPER_SWEEP=${CALIPER_SWEEP-0.05,0.1,0.2,0.5,1}
case "$PS_ENGINE" in ridge|lasso) ;; *) echo "ERROR: PS_ENGINE must be ridge|lasso"; exit 1;; esac
# ---- RUN MANIFEST: stamp the design into the results tree ----
write_manifest() {
  local root="$1"; mkdir -p "$root"
  cat > "$root/RUN_MANIFEST.json" <<EOF
{
  "written_utc":     "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id":    "${SLURM_JOB_ID:-manual}",
  "design":          "reference construction (Tang 2026, Supplement Fig. S1): two separate emulations, each with its own GLP-1 arm and a 12-month washout over the two compared classes only; entry = first in-window study-class exposure; same-day multi-class entries dropped; age >= 18; >= 1 recorded diagnosis date in the prior 365d; ever-lookback T2D required and T1D/ESRD/prior psychiatric outcome excluded; prior antidepressant/antipsychotic use excluded; 365-day baseline covariates; censoring at death/study end only",
  "specs":           "${SPECS}",
  "emulations":      "${EMULATIONS}",
  "comparisons":     "${COMPARISONS}",
  "estimators":      "siptw_cox,psm_cox,strat_cox",
  "ps_engine":       "${PS_ENGINE} for embedding specs; plain logistic for reference specs",
  "cv_folds":        5,
  "caliper_sweep":   "${CALIPER_SWEEP:-off}",
  "enrollment_gate": "off (the reference applies none)",
  "covariate_spec":  "paper = reference Table S5 covariates buildable on this source",
  "nco_panel":       "reference-24 (Supplement Table S4) with the >=0.1% risk rule applied on top",
  "population":      "OMOP cell ${CELL} (dx route ${ROUTE:-source})"
}
EOF
  echo "wrote $root/RUN_MANIFEST.json"
}
FAILED=()
# COMPARISONS: space-separated id:treat:comp triplets. Default = the reference's two contrasts
COMPARISONS=${COMPARISONS:-glp1_vs_sglt2:glp1ra:sglt2i glp1_vs_dpp4:glp1ra:dpp4i}
write_manifest "results/$CELL"
for EM in $EMULATIONS; do
  IFS=: read -r ESUF CID TCLS CCLS <<<"$EM"
  ECOH="intermediate/cohorts/${CELL}_${ESUF}/cohort.csv"
  [ -f "$ECOH" ] || { echo "ERROR: $ECOH missing (run ty0_2_build_cohorts)"; exit 1; }
  for SPEC in $SPECS; do
    echo "==== estimate_ladder: $CID / $SPEC (emulation $ESUF) ===="
    # every later addition must use += (a bare EXTRA=(...) would clobber --caliper_sweep)
    EXTRA=()
    [ -n "$CALIPER_SWEEP" ] && EXTRA+=(--caliper_sweep "$CALIPER_SWEEP")
    if [ "$SPEC" = "hdps" ]; then
      HD="intermediate/hdps_covars/$CELL/$CID/hdps_covars.csv"
      [ -f "$HD" ] || { echo "!!!! $CID/hdps skipped: $HD missing (run ty1_1_select_covariates) !!!!"; FAILED+=("$CID/hdps"); continue; }
      EXTRA+=(--hdps_covars "$HD")
    fi
    case "$SPEC" in
      *-only)
        M=${SPEC%-only}
        EP="intermediate/embeddings/${CELL}/cohort_with_${M}_embeddings.parquet"
        [ -f "$EP" ] || { echo "!!!! $CID/$SPEC skipped: $EP missing !!!!"; FAILED+=("$CID/$SPEC"); continue; }
        EXTRA+=(--embed_parquet "$EP");;
    esac
    "$RSCRIPT" ty3_estimate/estimate_ladder.R --cohort_csv "$ECOH" --comparison "$CID" --ps_engine "$PS_ENGINE" \
      --treat_class "$TCLS" --comp_class "$CCLS" --spec "$SPEC" "${EXTRA[@]}" \
      --out_dir "results/$CELL/$CID/$SPEC" \
      || { echo "!!!! estimate_ladder $CID/$SPEC FAILED !!!!"; FAILED+=("$CID/$SPEC"); continue; }
  done
done
[ ${#FAILED[@]} -gt 0 ] && { echo "==== ty3_1_estimate FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; }
echo "==== ty3_1_estimate completed OK ===="
