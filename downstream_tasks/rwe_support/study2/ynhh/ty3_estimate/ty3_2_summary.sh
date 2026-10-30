#!/bin/bash
#SBATCH --job-name=ty3_2_summary
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Relative output path: submit from the module root.
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# Stage ty3_2 (summarize_results): summary tables + figures over results/<cell>/.
# Needs ty3_1. Output: results/_summary_<cell>/.
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

"$RSCRIPT" ty3_estimate/summarize_results.R --results_root "results/$CELL" \
  --out_dir "results/_summary_$CELL" || { echo "!!!! summarize_results FAILED !!!!"; exit 1; }
echo "==== ty3_2_summary completed OK ===="
