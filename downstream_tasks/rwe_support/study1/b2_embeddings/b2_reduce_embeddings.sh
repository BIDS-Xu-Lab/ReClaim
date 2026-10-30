#!/bin/bash
#SBATCH --job-name=b2_reduce_embeddings
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the repo root
# =============================================================================
# PCA plug-in: dimension-matched low-dim variants of the ReClaim embeddings. Writes
# cohort_with_<model>-pca<k>_embeddings.parquet beside the source parquets; estimation then
# runs through the unchanged ladder via spec names, e.g.:
#   SPECS="reclaim-m-pca120-only" sbatch b3_estimate/b3_1_estimate.sh
# Knobs: MODELS="reclaim-s reclaim-m reclaim-l" | KS=60,120,500 | INTER_ROOT (default intermediate/mdcr)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the benchmark root (no config/)"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b2_reduce_embeddings_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
PY="${CONDA_PREFIX:?conda activate failed}/bin/python"
"$PY" -c "import sklearn, pyarrow, numpy" || { echo "ERROR: env missing sklearn/pyarrow"; exit 1; }

INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
MODELS=${MODELS:-reclaim-s reclaim-m reclaim-l}
KS=${KS:-60,120,500}
CELLS=${CELLS:-positive_control_celecoxib_gi_MDCR negative_control_flu_hipfx_MDCR_s2011}

FAILED=()
for CELL in $CELLS; do
  for M in $MODELS; do
    echo "==== reduce: $CELL / $M (ks=$KS) ===="
    "$PY" b2_embeddings/reduce_embeddings.py \
      --embed_dir "$INTER_ROOT/embeddings/$CELL" --model "$M" --ks "$KS" \
      || FAILED+=("$CELL/$M")
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "==== b2_reduce_embeddings FAILED: ${FAILED[*]} ===="; exit 1; fi
echo "==== b2_reduce_embeddings OK. Next: SPECS=\"<model>-pca<k>-only ...\" sbatch b3_estimate/b3_1_estimate.sh ===="
