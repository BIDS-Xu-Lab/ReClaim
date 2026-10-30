#!/bin/bash
#SBATCH --job-name=b0_2_build_cohorts
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=1800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# =============================================================================
# Stage b0_2: resolve_codes + build_cohort + build_long_table for BOTH cohorts
# (positive_control = celecoxib -> GI bleed, paper Ex.1; negative_control = flu vaccine -> hip
# fracture, paper Ex.3, season 2011), on the full MarketScan claims tree, with the mandatory
# --restrict_ids tokenizability gate (about 31% of MDCR enrollees have no ReClaim token sequence;
# without the gate the embedding specs would silently receive rep_*=0 for them).
# Writes results/mdcr + intermediate/mdcr/. RAW_ROOT / RESULTS_ROOT / INTER_ROOT are overridable.
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac  # tolerate sbatch from a section dir
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root: cd <root> && sbatch b0_preprocessing/b0_2_build_cohorts.sh"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b0_2_build_cohorts_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda Java/17.0.4

unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE
RECLAIM_ENV=${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}
# conda's init scripts reference unset vars, which abort under `set -u`; relax it just here.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u

PY="$RECLAIM_ENV/bin/python"
export PYSPARK_PYTHON="$PY"
export PYSPARK_DRIVER_PYTHON="$PY"

[ -n "${JAVA_HOME:-}" ] || { echo "ERROR: JAVA_HOME not set (module load Java failed) -- PySpark cannot start"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: reclaim env python not found at $PY"; exit 1; }
"$PY" -c "import yaml, pandas, pyspark" || { echo "ERROR: reclaim env deps missing (python=$PY)"; exit 1; }

if [ -z "${SPARK_LOCAL_DIRS:-}" ]; then
  export SPARK_LOCAL_DIRS="/tmp/spark_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "$SPARK_LOCAL_DIRS"' EXIT
fi
mkdir -p "$SPARK_LOCAL_DIRS" || { echo "ERROR: cannot create SPARK_LOCAL_DIRS=$SPARK_LOCAL_DIRS"; exit 1; }
chmod 700 "$SPARK_LOCAL_DIRS" 2>/dev/null || true
echo "SPARK_LOCAL_DIRS=$SPARK_LOCAL_DIRS  ($(df -h "$SPARK_LOCAL_DIRS" | tail -1 | awk '{print $4}') free)"

# ---- data roots -------------------------------------------------------------------------------
# Tokenizability gate: membership is restricted to persons with a ReClaim token sequence. The
# trajectory store is the ground truth for "this person has a sequence".
TOKENS="${MS_PROCESSED_ROOT:?export MS_PROCESSED_ROOT in env.sh}/${DATA_VERSION:?export DATA_VERSION in env.sh}/${MS_RUN_NAME:?export MS_RUN_NAME in env.sh}/processed_data/trajectory/*.parquet"
RAW_ROOT=${RAW_ROOT:-${MS_RAW_FULL:?export MS_RAW_FULL in env.sh}}
RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
RAW_ARGS=(--restrict_ids "$TOKENS")
VOCAB_NDC=${VOCAB_NDC:?export VOCAB_NDC in env.sh}
echo "###### b0_2_build_cohorts build: raw=$RAW_ROOT -> $RESULTS_ROOT + $INTER_ROOT ######"

# Code resolution first (cheap; build_cohort reads intermediate/codes_resolved/).
# build_long_table still reads VOCAB_NDC directly (it maps ALL dispensed NDCs -> ingredient names).
for CFG in ${COHORT_CONFIGS:-config/positive_control_celecoxib_gi.yaml config/negative_control_flu_hipfx.yaml}; do
  echo "==== resolve codes (resolve_codes): $CFG ===="
  "$PY" b0_preprocessing/resolve_codes.py --config "$CFG" --vocab_ndc "$VOCAB_NDC"
done

# Both controls run the MDCR stratum (age >= 65), the closest match to the paper's elderly
# population. negative_control is a per-season landmark design -> one cell per (stratum, season);
# season 2011 is the single season of the design.
POSITIVE_STRATA=(MDCR)
NEGATIVE_STRATA=(MDCR)
NEGATIVE_SEASONS=(2011)

# build_cohort writes cohort.csv only if the build succeeds. Verify it exists after each build; if
# not, log a loud marker and skip build_long_table for that cell.
FAILED=()
check_built() {  # $1 = cell label; uses $COHORT/$INTER_ROOT
  local cell="$1"
  local f="$INTER_ROOT/cohorts/$COHORT/$cell/cohort.csv"
  if [ ! -f "$f" ]; then echo "!!!! BUILD FAILED: $COHORT / $cell -- no cohort.csv (build_cohort errored; check its log) !!!!"; FAILED+=("$COHORT/$cell"); return 1; fi
  echo "     built OK: $f ($(wc -l <"$f") rows)"
  "$PY" - "$f" <<'PYEOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
g = d.groupby("arm")["outcome"].agg(n="size", ev="sum")
exp = g.loc[1, "n"] / len(d) if 1 in g.index else float("nan")
print(f"     n={len(d):,}  events={int(d.outcome.sum()):,}  exposed fraction={exp:.3f}")
PYEOF
  return 0
}

for CFG in ${COHORT_CONFIGS:-config/positive_control_celecoxib_gi.yaml config/negative_control_flu_hipfx.yaml}; do
  COHORT=$(basename "$CFG" .yaml)
  if [ "$COHORT" = "negative_control_flu_hipfx" ]; then STRATA=("${NEGATIVE_STRATA[@]}"); else STRATA=("${POSITIVE_STRATA[@]}"); fi
  for STRATUM in "${STRATA[@]}"; do
    if [ "$COHORT" = "negative_control_flu_hipfx" ]; then
      # negative_control: one sub-cohort per season. build_cohort --season t writes {stratum}_s{t}/cohort.csv;
      # build_long_table --cell reads/writes the matching per-season long table.
      for S in "${NEGATIVE_SEASONS[@]}"; do
        CELL="${STRATUM}_s${S}"
        echo "==== build cohort: $COHORT / $STRATUM / season $S ===="
        "$PY" b0_preprocessing/build_cohort.py --config "$CFG" --stratum "$STRATUM" --season "$S" \
          --raw_root "$RAW_ROOT" --out_root "$INTER_ROOT/cohorts" "${RAW_ARGS[@]}"
        check_built "$CELL" || continue
        echo "==== build hdps long: $COHORT / $CELL ===="
        "$PY" b0_preprocessing/build_long_table.py --config "$CFG" --stratum "$STRATUM" --cell "$CELL" \
          --raw_root "$RAW_ROOT" --vocab_ndc "$VOCAB_NDC" --domains 5 \
          --cohort_root "$INTER_ROOT/cohorts" --out_root "$INTER_ROOT/hdps_long" "${RAW_ARGS[@]}"
      done
    else
      echo "==== build cohort: $COHORT / $STRATUM ===="
      "$PY" b0_preprocessing/build_cohort.py --config "$CFG" --stratum "$STRATUM" \
        --raw_root "$RAW_ROOT" --out_root "$INTER_ROOT/cohorts" "${RAW_ARGS[@]}"
      check_built "$STRATUM" || continue
      echo "==== build hdps long: $COHORT / $STRATUM ===="
      "$PY" b0_preprocessing/build_long_table.py --config "$CFG" --stratum "$STRATUM" \
        --raw_root "$RAW_ROOT" --vocab_ndc "$VOCAB_NDC" --domains 5 \
        --cohort_root "$INTER_ROOT/cohorts" --out_root "$INTER_ROOT/hdps_long" "${RAW_ARGS[@]}"
    fi
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "==== b0_2_build_cohorts FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; fi
echo "==== b0_2_build_cohorts all cohorts built OK ===="
echo "Cohorts + hdps long tables built (negative_control = one cell per season)."
