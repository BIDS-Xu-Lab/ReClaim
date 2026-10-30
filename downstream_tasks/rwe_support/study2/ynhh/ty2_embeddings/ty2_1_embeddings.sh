#!/bin/bash
#SBATCH --job-name=ty2_1_embeddings
#SBATCH --partition=gpu
#SBATCH --nodes=1
# The gpu partition is heterogeneous: submit with an explicit type, e.g.
#   sbatch --gpus=h200:1 ty2_embeddings/ty2_1_embeddings.sh
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# Relative output path: submit from the module root.
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# Stage ty2_1 (prepare_sequences + embed_reclaim): pre-index token sequences and ReClaim
# embeddings for the UNION cohort (one pass serves both emulations). Needs ty0_2 and the
# token store (PROCESSED_ROOT). Output: intermediate/embeddings/<cell>/cohort_with_<model>_embeddings.parquet.
#   MODELS="reclaim-s reclaim-m"  restricts the models (default: all three; "none" = sequences only)
#   BACKEND=hf                    HuggingFace backend instead of vLLM
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root"; exit 1; }
[ -d config ] || { echo "ERROR: not the study2/ynhh root (no config/)"; exit 1; }
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

[ -f "$MODULE_ROOT/ty2_embeddings/embed_reclaim.py" ] || { echo "ERROR: missing encoder ty2_embeddings/embed_reclaim.py"; exit 1; }
# Checkpoint + token-store paths come from env.sh at the module root.
. "$MODULE_ROOT/env.sh" || { echo "ERROR: cannot source $MODULE_ROOT/env.sh"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
case "$SOURCE" in
  ynhh) PROOT="${PROCESSED_ROOT:?env.sh not sourced}"; STORE=ynhh_full;;
  *) echo "ERROR: unknown SOURCE '$SOURCE' (this repo is ynhh-only)"; exit 1;;
esac
# Fixed model scheme: the key reclaim-<size> resolves to RECLAIM_<SIZE>_CKPT from env.sh and
# names the embeddings parquet and the results spec dir (<key>-only).
MODELS=(${MODELS:-reclaim-s reclaim-m reclaim-l}); [ "${MODELS[0]}" = "none" ] && MODELS=()
declare -A RECLAIM=()
for M in "${MODELS[@]}"; do
  case "$M" in
    reclaim-s) RECLAIM[$M]="${RECLAIM_S_CKPT:?set RECLAIM_S_CKPT in env.sh}";;
    reclaim-m) RECLAIM[$M]="${RECLAIM_M_CKPT:?set RECLAIM_M_CKPT in env.sh}";;
    reclaim-l) RECLAIM[$M]="${RECLAIM_L_CKPT:?set RECLAIM_L_CKPT in env.sh}";;
    *) echo "ERROR: unknown model '$M' (reclaim-s|reclaim-m|reclaim-l)"; exit 1;;
  esac
done
export VLLM_USE_DEEP_GEMM=0
# Offline flags: on a no-egress cluster a missing flag is a multi-minute hang, not an error.
# All checkpoints/tokenizers are local paths.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export SPARK_DRIVER_MEM=${SPARK_DRIVER_MEM:-120g}

# Cell-scoped embeddings, so the two dx routes never share prevseq/parquets (the cache check
# would otherwise silently reuse the other route's person set). ty3_1 reads the same $CELL path.
EMB=intermediate/embeddings/$CELL
mkdir -p "$EMB"
PREV=$EMB/cohort_with_pre_entry_seq.parquet
[ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run ty0_2_build_cohorts)"; exit 1; }
echo "==== prepare_sequences: pre-index sequences ($SOURCE store=$STORE) ===="
[ -f "$PREV" ] && echo "     cached: $PREV" || \
  "$PY" ty2_embeddings/prepare_sequences.py --cohort_csv "$COHORT" --out_dir "$EMB" \
    --processed_root "$PROOT" --data_version "${DATA_VERSION:?export DATA_VERSION in env.sh}" --store "$STORE" || { echo "!!!! prepare_sequences FAILED !!!!"; exit 1; }

FAILED=()
for M in "${MODELS[@]}"; do
  OUT=$EMB/cohort_with_${M}_embeddings.parquet
  [ -f "$OUT" ] && { echo "     cached: $OUT"; continue; }
  [ -d "${RECLAIM[$M]}" ] || { echo "!!!! $M checkpoint missing: ${RECLAIM[$M]} (set RECLAIM_*_CKPT) !!!!"; FAILED+=("$M"); continue; }
  echo "==== embed_reclaim: $M ===="
  "$PY" "$MODULE_ROOT/ty2_embeddings/embed_reclaim.py" \
    --backend "${BACKEND:-vllm}" --model "$M" --model_path "${RECLAIM[$M]}" \
    --input "$PREV" --output "$OUT" --max_length 4096 --gpus auto \
    || { echo "!!!! embed_reclaim $M FAILED !!!!"; FAILED+=("$M"); }
done
[ ${#FAILED[@]} -gt 0 ] && { echo "==== ty2_1_embeddings FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; }
echo "==== ty2_1_embeddings completed OK (union cohort: one pass serves both emulations) ===="
