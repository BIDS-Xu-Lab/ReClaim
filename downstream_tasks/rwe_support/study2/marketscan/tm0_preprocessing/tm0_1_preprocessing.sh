#!/bin/bash
#SBATCH --job-name=tm0_1_preprocessing
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm0_1_preprocessing: resolve_codes (ingredient -> NDC sets) + build_cohort (union cohort
# + two emulation cohorts) + build_long_table (hdPS long table on the union).
# Population: MarketScan held-out val+test split (unseen in ReClaim pretraining), POOL.
# Usage: sbatch tm0_preprocessing/tm0_1_preprocessing.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (codesets_tte, marketscan_io, utils)
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root."; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm0_1_preprocessing_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda Java/17.0.4
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE
RECLAIM_ENV=${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u
PY="$RECLAIM_ENV/bin/python"
export PYSPARK_PYTHON="$PY"
export PYSPARK_DRIVER_PYTHON="$PY"
[ -n "${JAVA_HOME:-}" ] || { echo "ERROR: JAVA_HOME not set"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: env python not found at $PY"; exit 1; }
"$PY" -c "import yaml, pandas, pyspark" || { echo "ERROR: env deps missing"; exit 1; }

if [ -z "${SPARK_LOCAL_DIRS:-}" ]; then
  export SPARK_LOCAL_DIRS="/tmp/spark_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "$SPARK_LOCAL_DIRS"' EXIT
fi
mkdir -p "$SPARK_LOCAL_DIRS"
export SPARK_DRIVER_MEM=${SPARK_DRIVER_MEM:-180g}

RAW_ROOT=${RAW_ROOT:-${MS_RAW_SUBSET:?export MS_RAW_SUBSET in env.sh}}
VOCAB_NDC=${VOCAB_NDC:?export VOCAB_NDC in env.sh}

echo "==== resolve_codes: resolve NDC code sets ===="
"$PY" tm0_preprocessing/resolve_codes.py --vocab_ndc "$VOCAB_NDC" || { echo "!!!! resolve_codes FAILED !!!!"; exit 1; }

echo "==== build_cohort: union + two emulation cohorts (held-out val+test, POOL) ===="
"$PY" tm0_preprocessing/build_cohort.py --config config/tte_glp1_psych.yaml \
  --raw_root "$RAW_ROOT" --codes_resolved intermediate/codes_resolved --out_root intermediate/cohorts \
  || { echo "!!!! build_cohort FAILED !!!!"; exit 1; }
COHORT=intermediate/cohorts/cohort.csv
[ -f "$COHORT" ] || { echo "!!!! BUILD FAILED: no cohort.csv !!!!"; exit 1; }
echo "     built OK: $COHORT ($(wc -l <"$COHORT") rows, union)"
EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}   # emulation_dir:comparison:treat:comp
for EM in $EMULATIONS; do
  IFS=: read -r EDIR CID TCLS CCLS <<<"$EM"
  [ -f "intermediate/cohorts/$EDIR/cohort.csv" ] || { echo "!!!! BUILD FAILED: no intermediate/cohorts/$EDIR/cohort.csv !!!!"; exit 1; }
  echo "     emulation $EDIR: $(( $(wc -l <"intermediate/cohorts/$EDIR/cohort.csv") - 1 )) rows"
done

echo "==== build_long_table: build hdPS long table (union cohort) ===="
"$PY" tm0_preprocessing/build_long_table.py --config config/tte_glp1_psych.yaml \
  --raw_root "$RAW_ROOT" --stratum POOL --vocab_ndc "$VOCAB_NDC" \
  || { echo "!!!! build_long_table FAILED !!!!"; exit 1; }
echo "==== tm0_1_preprocessing completed OK ===="
