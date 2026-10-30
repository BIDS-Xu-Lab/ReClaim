#!/bin/bash
#SBATCH --job-name=b2_1_embeddings
#SBATCH --partition=gpu
#SBATCH --nodes=1
# For data parallelism submit with more GPUs (e.g. --gres=gpu:h200:4); embed_reclaim auto-shards
# across them (see its --gpus).
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# Stage b2_1 (GPU): ReClaim (+ Delphi) rep_* embeddings on the pre-index token sequences.
# Needs only cohort.csv from b0_2. Flow per cohort x cell:
#   prepare_sequences.py -> prevseq_{tokver}/cohort_with_pre_entry_seq.parquet
#   embed_reclaim.py / embed_delphi.py -> cohort_with_{model}_embeddings.parquet
# Consumed by b3_1_estimate (ladder_driver.sh -> estimate_ladder.R) and b3_2_dr.
#
# Usage:
#   sbatch b2_embeddings/b2_1_embeddings.sh                                    # reclaim-s/m/l + delphi
#   MODELS=reclaim-l ENABLE_DELPHI=0 sbatch b2_embeddings/b2_1_embeddings.sh   # one model only
#   MODELS=none ENABLE_DELPHI=1 sbatch b2_embeddings/b2_1_embeddings.sh        # prevseq + delphi only
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac  # tolerate sbatch from a section dir
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
[ -d config ] || { echo "ERROR: $MODULE_ROOT is not the module root (no config/). Submit from the module root."; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b2_1_embeddings_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"
MODULE_DIR="$(pwd)"

module --force purge; module load miniconda Java/17.0.4
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE
RECLAIM_ENV=${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u
PY="$RECLAIM_ENV/bin/python"
export PYSPARK_PYTHON="$PY"; export PYSPARK_DRIVER_PYTHON="$PY"
[ -n "${JAVA_HOME:-}" ] || { echo "ERROR: JAVA_HOME not set (module load Java failed)"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: env python not found at $PY"; exit 1; }
"$PY" -c "import torch, pandas, pyspark" || { echo "ERROR: env deps missing (python=$PY)"; exit 1; }

# -- ReClaim checkpoints (env.sh) ----------------------------------------------------------------
# The three ReClaim checkpoints are RECLAIM_S_CKPT / RECLAIM_M_CKPT / RECLAIM_L_CKPT, addressed by
# the keys reclaim-s / reclaim-m / reclaim-l. The key names the output parquet
# (cohort_with_{key}_embeddings.parquet) and the ladder spec ({key}-only). A key resolves to the
# env.sh var of the same name (strip 'reclaim-', uppercase, '-'->'_').
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
# MODELS=none -> run NO ReClaim models (delphi-only job). An EMPTY MODELS= would silently fall
# back to the default list (:- substitutes on empty), hence the explicit sentinel.
MODELS=(${MODELS:-reclaim-s reclaim-m reclaim-l}); [ "${MODELS[0]}" = "none" ] && MODELS=()
declare -A RECLAIM=()
for M in "${MODELS[@]}"; do
  _v="RECLAIM_$(echo "${M#reclaim-}" | tr 'a-z-' 'A-Z_')_CKPT"
  RECLAIM[$M]="${!_v:?unknown model '$M' -- export ${_v}=<path> in env.sh}"
done

# -- Delphi (ON by default; ENABLE_DELPHI=0 to skip) ---------------------------------------------
# Delphi shares ReClaim's prevseq (one Spark build per cell, via the prep_prevseq cache); its DX
# vocabulary is fixed by the disease map. Default ON so the delphi rung cannot silently vanish.
ENABLE_DELPHI=${ENABLE_DELPHI:-1}
DELPHI_CKPT=${DELPHI_CKPT:?env.sh not sourced}
DELPHI_MAP=${DELPHI_MAP:?env.sh not sourced}
# DELPHI_TAG names the output parquet (cohort_with_${DELPHI_TAG}_embeddings.parquet) and hence
# the spec ('${DELPHI_TAG}-only'). Tag and checkpoint move together.
DELPHI_TAG=${DELPHI_TAG:-delphi}

# -- data roots + token store --------------------------------------------------------------------
# The prevseq must be built from the token store the cohorts were gated on (b0_2's --restrict_ids):
# $MS_PROCESSED_ROOT/$DATA_VERSION/$MS_RUN_NAME. A mismatch does not crash (the cohort-driven left
# join emits header-only sequences that embed without error), so the store comes from env.sh.
SOURCE=${SOURCE:-marketscan}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
# hf = transformers reference path; vllm = the faster default, interchangeable with hf after
# pre-tokenisation (both see the same 4096-token truncation).
BACKEND=${BACKEND:-vllm}
export VLLM_USE_DEEP_GEMM=0
# Spark sizing for the big cells; the Spark phase never overlaps the GPU phase.
export SPARK_DRIVER_MEM=${SPARK_DRIVER_MEM:-200g}
export SPARK_MAX_RESULT=${SPARK_MAX_RESULT:-32g}
export SPARK_SHUFFLE_PARTITIONS=${SPARK_SHUFFLE_PARTITIONS:-600}

PROCESSED_ROOT=${PROCESSED_ROOT:-${MS_PROCESSED_ROOT:?export MS_PROCESSED_ROOT in env.sh}}
case "$SOURCE" in
  marketscan)
    DATA_VERSION=${DATA_VERSION:?export DATA_VERSION in env.sh}
    RUN_NAME=${MS_RUN_NAME:?export MS_RUN_NAME in env.sh}
    # cohort.csv person_id holds a MarketScan ENROLID; the token store keys on 'CR'+ENROLID.
    ID_PREFIX=${ID_PREFIX:-CR}
    POSITIVE_STRATA=(${POSITIVE_STRATA:-MDCR})
    NEGATIVE_STRATA=(${NEGATIVE_STRATA:-MDCR})
    NEGATIVE_SEASONS=(2011)
    ;;
  *) echo "ERROR: unknown SOURCE '$SOURCE' (this module is MarketScan-only)"; exit 1;;
esac

# Validate checkpoints up-front: a long GPU job must not die an hour in on a mistyped path.
for M in "${MODELS[@]}"; do
  [ -d "${RECLAIM[$M]}" ] || { echo "ERROR: checkpoint missing for '$M': ${RECLAIM[$M]}"; exit 1; }
done
echo "###### b2_1_embeddings: source=$SOURCE tokens=${DATA_VERSION}/${RUN_NAME} models=${MODELS[*]} ######"

FAILED=()

# Build the prevseq for one cell at one data version; echoes the out dir on stdout (logs -> stderr).
prep_prevseq() {  # $1=cohort_csv $2=cell_dir $3=data_version $4=run_name [$5=processed_root]
  local coh="$1" celldir="$2" tokver="$3" runname="$4" root="${5:-$PROCESSED_ROOT}"
  local out="$celldir/prevseq_${tokver}"
  # Cache is per person: an existing prevseq is reused only if it covers every cohort member;
  # otherwise it is rebuilt for the whole cohort.
  if [ -f "$out/cohort_with_pre_entry_seq.parquet" ]; then
    local miss; miss=$("$PY" b2_embeddings/embed_cache.py coverage "$coh" "$out/cohort_with_pre_entry_seq.parquet") || return 1
    if [ "$miss" = "0" ]; then echo "     prevseq cached (covers cohort): $out" >&2; echo "$out"; return 0; fi
    echo "     prevseq stale: $miss cohort ids missing -> rebuilding $out" >&2
  fi
  "$PY" b2_embeddings/prepare_sequences.py --cohort_csv "$coh" --out_dir "$out" \
    --processed_root "$root" --data_version "$tokver" --run_name "$runname" \
    --id_prefix "$ID_PREFIX" >&2 || return 1
  echo "$out"
}

COHORTS=(${COHORTS:-positive_control_celecoxib_gi negative_control_flu_hipfx})
for COHORT in "${COHORTS[@]}"; do
  if [ "$COHORT" = "negative_control_flu_hipfx" ]; then STRATA=("${NEGATIVE_STRATA[@]}"); else STRATA=("${POSITIVE_STRATA[@]}"); fi
  for STRATUM in "${STRATA[@]}"; do
    # positive_control: one cell (the stratum). negative_control: one cell per season.
    if [ "$COHORT" = "negative_control_flu_hipfx" ]; then
      CELLS=(); for S in "${NEGATIVE_SEASONS[@]}"; do CELLS+=("${STRATUM}_s${S}"); done
    else
      CELLS=("$STRATUM")
    fi
    for CELL in "${CELLS[@]}"; do
      COH="$INTER_ROOT/cohorts/${COHORT}/${CELL}/cohort.csv"
      CELLDIR="$INTER_ROOT/embeddings/${COHORT}_${CELL}"
      [ -f "$COH" ] || { echo "SKIP $COHORT/$CELL: cohort.csv missing (run b0_2_build_cohorts first)"; continue; }

      echo "==== prepare_sequences prepare prevseq (${DATA_VERSION}): $COHORT / $CELL ===="
      PREV=$(prep_prevseq "$COH" "$CELLDIR" "$DATA_VERSION" "$RUN_NAME") \
        || { echo "!!!! prepare_sequences FAILED: $COHORT/$CELL"; FAILED+=("prepare_sequences:$COHORT/$CELL"); continue; }

      for M in "${MODELS[@]}"; do
        IN="$MODULE_DIR/$PREV/cohort_with_pre_entry_seq.parquet"
        OUT="$MODULE_DIR/$CELLDIR/cohort_with_${M}_embeddings.parquet"
        [ -f "$IN" ] || { echo "!!!! missing prevseq input: $IN"; FAILED+=("$M:$COHORT/$CELL"); continue; }
        # Cache is per person: an existing parquet that covers the cohort is reused; one that
        # misses members gets ONLY the missing people embedded and appended (everyone already
        # embedded keeps their exact vector). FORCE=1 regenerates from scratch.
        if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
          MISS=$("$PY" b2_embeddings/embed_cache.py coverage "$COH" "$OUT") || { FAILED+=("$M:$COHORT/$CELL"); continue; }
          if [ "$MISS" = "0" ]; then
            echo "---- ReClaim $M on $COHORT/$CELL -- cached (covers cohort), skipping (FORCE=1 to redo) ----"; continue
          fi
          echo "---- ReClaim $M on $COHORT/$CELL -- DELTA: embedding $MISS new cohort members ----"
          DIN="$MODULE_DIR/$CELLDIR/delta_${M}_prevseq.parquet"; DOUTM="$MODULE_DIR/$CELLDIR/delta_${M}_embeddings.parquet"
          "$PY" b2_embeddings/embed_cache.py filter "$IN" "$COH" "$OUT" "$DIN" \
            || { echo "!!!! delta filter FAILED for $M"; FAILED+=("$M:$COHORT/$CELL"); continue; }
          ( "$PY" b2_embeddings/embed_reclaim.py \
              --backend "${BACKEND:-hf}" --model "$M" --model_path "${RECLAIM[$M]}" \
              --input "$DIN" --output "$DOUTM" ) \
            || { echo "!!!! ReClaim $M (delta) FAILED on $COHORT/$CELL"; FAILED+=("$M:$COHORT/$CELL"); continue; }
          "$PY" b2_embeddings/embed_cache.py merge "$OUT" "$DOUTM" "$COH" "$OUT" \
            || { echo "!!!! delta merge FAILED for $M"; FAILED+=("$M:$COHORT/$CELL"); continue; }
          rm -f "$DIN" "$DOUTM"; continue
        fi
        echo "---- ReClaim $M on $COHORT/$CELL ----"
        ( "$PY" b2_embeddings/embed_reclaim.py \
            --backend "${BACKEND:-hf}" --model "$M" --model_path "${RECLAIM[$M]}" \
            --input "$IN" --output "$OUT" ) \
          || { echo "!!!! ReClaim $M FAILED on $COHORT/$CELL"; FAILED+=("$M:$COHORT/$CELL"); continue; }
        [ -f "$OUT" ] || { echo "!!!! $M wrote no output: $OUT"; FAILED+=("$M:$COHORT/$CELL"); }
      done

      if [ "$ENABLE_DELPHI" = "1" ] && [ -f "$DELPHI_CKPT" ]; then
        DOUT="$MODULE_DIR/$CELLDIR/cohort_with_${DELPHI_TAG}_embeddings.parquet"
        DMISS=0
        if [ -f "$DOUT" ] && [ "${FORCE:-0}" != "1" ]; then
          DMISS=$("$PY" b2_embeddings/embed_cache.py coverage "$COH" "$DOUT") || { FAILED+=("delphi:$COHORT/$CELL"); DMISS=0; }
        fi
        if [ -f "$DOUT" ] && [ "${FORCE:-0}" != "1" ] && [ "$DMISS" = "0" ]; then
          echo "---- Delphi on $COHORT/$CELL -- cached (covers cohort), skipping (FORCE=1 to redo) ----"
        elif [ -f "$DOUT" ] && [ "${FORCE:-0}" != "1" ]; then
          echo "---- Delphi [${DELPHI_TAG}] on $COHORT/$CELL -- DELTA: embedding $DMISS new cohort members ----"
          DPREV=$(prep_prevseq "$COH" "$CELLDIR" "$DATA_VERSION" "$RUN_NAME") \
            || { echo "!!!! prepare_sequences(delphi) FAILED: $COHORT/$CELL"; FAILED+=("prepare_sequences-delphi:$COHORT/$CELL"); DPREV=""; }
          if [ -n "$DPREV" ]; then
            DDIN="$MODULE_DIR/$CELLDIR/delta_${DELPHI_TAG}_prevseq.parquet"; DDOUT="$MODULE_DIR/$CELLDIR/delta_${DELPHI_TAG}_embeddings.parquet"
            "$PY" b2_embeddings/embed_cache.py filter "$MODULE_DIR/$DPREV/cohort_with_pre_entry_seq.parquet" "$COH" "$DOUT" "$DDIN" \
              && ( "$PY" b2_embeddings/embed_delphi.py --model_path "$DELPHI_CKPT" \
                     --disease_map "$DELPHI_MAP" --input "$DDIN" --output "$DDOUT" ) \
              && "$PY" b2_embeddings/embed_cache.py merge "$DOUT" "$DDOUT" "$COH" "$DOUT" \
              || { echo "!!!! Delphi (delta) FAILED on $COHORT/$CELL"; FAILED+=("delphi:$COHORT/$CELL"); }
            rm -f "$DDIN" "$DDOUT"
          fi
        else
          echo "---- Delphi [${DELPHI_TAG}] (ckpt ${DELPHI_CKPT}) on $COHORT/$CELL ----"
          DPREV=$(prep_prevseq "$COH" "$CELLDIR" "$DATA_VERSION" "$RUN_NAME") \
            || { echo "!!!! prepare_sequences(delphi) FAILED: $COHORT/$CELL"; FAILED+=("prepare_sequences-delphi:$COHORT/$CELL"); DPREV=""; }
          DIN="$MODULE_DIR/$DPREV/cohort_with_pre_entry_seq.parquet"
          if [ -n "$DPREV" ] && [ -f "$DIN" ]; then
            ( "$PY" b2_embeddings/embed_delphi.py \
                --model_path "$DELPHI_CKPT" \
                --disease_map "$DELPHI_MAP" --input "$DIN" --output "$DOUT" ) \
              || { echo "!!!! Delphi FAILED on $COHORT/$CELL"; FAILED+=("delphi:$COHORT/$CELL"); }
            [ -f "$DOUT" ] || { echo "!!!! Delphi wrote no output: $DOUT"; FAILED+=("delphi:$COHORT/$CELL"); }
          elif [ -n "$DPREV" ]; then
            echo "!!!! missing delphi prevseq input: $DIN"; FAILED+=("delphi:$COHORT/$CELL")
          fi
        fi
      fi
    done
  done
done

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "==== b2_1_embeddings FINISHED WITH FAILURES (${#FAILED[@]}): ${FAILED[*]} ===="; exit 1
fi
echo "==== b2_1_embeddings OK: cohort_with_<model>_embeddings.parquet written under $INTER_ROOT/embeddings/ ===="
