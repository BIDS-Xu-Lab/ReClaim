#!/bin/bash
#SBATCH --job-name=tm2_1_embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm2_1_embeddings: embedding pass on the UNION cohort (one pass serves both emulations).
#   prepare_sequences  pre-index token sequences from the token store (val+test split)
#   embed_reclaim      ReClaim rep_* embeddings (vLLM or HF backend), one parquet per size
# Models: reclaim-s / reclaim-m / reclaim-l, checkpoints RECLAIM_S/M/L_CKPT from env.sh; the
# key names the output parquet (cohort_with_<key>_embeddings.parquet) and the results spec.
# Usage : sbatch tm2_embeddings/tm2_1_embeddings.sh   (from the module root; GPU)
#         MODELS="reclaim-s" or MODELS=none (sequences only); BACKEND=vllm|hf
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm2_1_embeddings_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
module --force purge; module load miniconda Java/17.0.4
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE
RECLAIM_ENV=${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u
PY="$RECLAIM_ENV/bin/python"
export PYSPARK_PYTHON="$PY"; export PYSPARK_DRIVER_PYTHON="$PY"
[ -n "${JAVA_HOME:-}" ] || { echo "ERROR: JAVA_HOME not set"; exit 1; }
"$PY" -c "import torch, pandas, pyspark" || { echo "ERROR: env deps missing"; exit 1; }

if [ -z "${SPARK_LOCAL_DIRS:-}" ]; then
  export SPARK_LOCAL_DIRS="/tmp/spark_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "$SPARK_LOCAL_DIRS"' EXIT
fi
mkdir -p "$SPARK_LOCAL_DIRS"
export SPARK_DRIVER_MEM=${SPARK_DRIVER_MEM:-120g}
export VLLM_USE_DEEP_GEMM=0

[ -f "$MODULE_ROOT/tm2_embeddings/embed_reclaim.py" ] || { echo "ERROR: missing encoder tm2_embeddings/embed_reclaim.py"; exit 1; }
# Model registry: key reclaim-<size> -> RECLAIM_<SIZE>_CKPT (env.sh); exactly three sizes.
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (codesets_tte, marketscan_io, utils)
MODELS=(${MODELS:-reclaim-s reclaim-m reclaim-l}); [ "${MODELS[0]}" = "none" ] && MODELS=()
declare -A RECLAIM=()
for M in "${MODELS[@]}"; do
  case "$M" in
    reclaim-s) _v=RECLAIM_S_CKPT;;
    reclaim-m) _v=RECLAIM_M_CKPT;;
    reclaim-l) _v=RECLAIM_L_CKPT;;
    *) echo "ERROR: unknown model '$M' (expected reclaim-s, reclaim-m or reclaim-l)"; exit 1;;
  esac
  RECLAIM[$M]="${!_v:?$_v not set -- export it in env.sh}"
done
BACKEND=${BACKEND:-vllm}

for M in "${MODELS[@]}"; do
  [ -d "${RECLAIM[$M]}" ] || { echo "ERROR: checkpoint missing: ${RECLAIM[$M]}"; exit 1; }
done

EMB=intermediate/embeddings
PREV=$EMB/cohort_with_pre_entry_seq.parquet
COHORT=intermediate/cohorts/cohort.csv
[ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run tm0_1_preprocessing)"; exit 1; }

echo "==== prepare_sequences: pre-index sequences (token store val+test) ===="
if [ -f "$PREV" ]; then
  echo "     cached: $PREV"
else
  "$PY" tm2_embeddings/prepare_sequences.py --cohort_csv "$COHORT" --out_dir "$EMB" \
    --data_version "${DATA_VERSION:?export DATA_VERSION in env.sh}" \
    || { echo "!!!! prepare_sequences FAILED !!!!"; exit 1; }
fi

FAILED=()
for M in "${MODELS[@]}"; do
  OUT=$EMB/cohort_with_${M}_embeddings.parquet
  if [ -f "$OUT" ]; then echo "     cached: $OUT"; continue; fi
  echo "==== embed_reclaim: $M ($BACKEND) ===="
  "$PY" "$MODULE_ROOT/tm2_embeddings/embed_reclaim.py" \
    --backend "$BACKEND" --model "$M" --model_path "${RECLAIM[$M]}" \
    --input "$PREV" --output "$OUT" --max_length 4096 --gpus auto \
    || { echo "!!!! embed_reclaim $M FAILED !!!!"; FAILED+=("$M"); }
done

# row-count check: every embedding file must cover the full cohort (estimate_ladder refuses partial files)
"$PY" - <<PYEOF || FAILED+=("rowcheck")
import pandas as pd, glob, sys
n = len(pd.read_csv("$COHORT", usecols=["person_id"]))
bad = []
for f in sorted(glob.glob("$EMB/cohort_with_*_embeddings.parquet")):
    m = len(pd.read_parquet(f, columns=["person_id"]))
    print(f"     {f}: {m:,} rows (cohort {n:,})")
    if m != n: bad.append(f)
sys.exit(1 if bad else 0)
PYEOF

if [ ${#FAILED[@]} -gt 0 ]; then echo "==== tm2_1_embeddings FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; fi
echo "==== tm2_1_embeddings completed OK (union cohort: one pass serves both emulations) ===="
