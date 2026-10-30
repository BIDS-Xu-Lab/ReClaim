#!/bin/bash
# config.sh -- pipeline constants and derived paths (tracked). Sourced at the end of env.sh.

# ---- Inputs ---------------------------------------------------------------------------------
export TEST_DATA_DIR="${DATA_ROOT}"
export TRAIN_DATA_DIR="${DATA_ROOT}"
export VOCAB_DIR="${DATA_ROOT}/processed_data/vocab"
export SPLIT="test"                                   # evaluation cohort; outputs under data/<SPLIT>/

# ---- Outputs (all under the repository root; gitignored) ------------------------------------
export PYTHONPATH="${COST_HOME}/e0_common${PYTHONPATH:+:${PYTHONPATH}}"
export DATA_DIR="${COST_HOME}/data"                   # cohorts, features, pairs, model_gen predictions
export BASELINE_DIR="${COST_HOME}/data/baseline"      # supervision split and its LightGBM features
export PAIRS_DIR="${COST_HOME}/data/posttrain_pairs"  # post-training pairs + bin definition
export RESULT_DIR="${COST_HOME}/results/${SPLIT}"     # metric tables (aggregates)
export OUTPUT_DIR="${COST_HOME}/output"               # the paper's tables and figures

# ---- Models ---------------------------------------------------------------------------------
# Backbones are reclaim-s|m|l. Derived ids: reclaim-<size>-pretrain (Monte-Carlo anchor; one file
# set per trajectory count N, scored as reclaim-<size>-pretrain-<N>) and reclaim-<size>-posttrain.
backbone_path() {  # backbone_path <s|m|l> -> checkpoint path
    local v="RECLAIM_$(echo "$1" | tr a-z A-Z)_CKPT"
    echo "${!v:?unknown backbone '$1' -- set ${v} in env.sh}"
}
export RECLAIM_MODELS=("reclaim-s-pretrain" "reclaim-m-pretrain" "reclaim-l-pretrain")
export ANCHOR_MODELS=("persistence" "trainmean")       # non-learned anchors (dollars only)

# ---- Monte-Carlo anchor ---------------------------------------------------------------------
export NUM_GENERATION="${NUM_GENERATION:-64}"   # trajectories per enrollee to generate; every count generated is
                                                # scored as its own model id, output/scripts/roster.sh picks the reported one
export SEQ_LEN=4096                             # model context length
export TEMPERATURE=1.0
export TOP_P=1.0
export SEED=66

# ---- Supervision pool and LightGBM ----------------------------------------------------------
export BASELINE_SAMPLE_N=5000000                # enrollees drawn from the pre-training corpus
export HISTORY_LENGTHS=("1year" "all")          # LightGBM feature windows: last observed year, whole history
export HISTORY_LENGTH="1year"                   # window used by the persistence anchor
hist_wtag() { [ "$1" = "1year" ] && echo "1year" || echo "allhist"; }   # window tag in lightgbm-{reg,cls}-<tag>
export TOKEN_FEATURE_TYPES=("count")            # bag-of-words counts
export MIN_AGE=16                               # gates the first observed year; prediction points are then all >= 18

# ---- Post-training --------------------------------------------------------------------------
export N_TRAIN_PAIRS=100000
export N_VAL_PAIRS=10000
# The reported recipe. A run with any other value must carry NAME_SUFFIX=-<tag> so that
# reclaim-<size>-posttrain always denotes this recipe.
export POSTTRAIN_LR="3e-5"
export POSTTRAIN_EPOCHS="2"
posttrain_global_batch() { case "$1" in l) echo 64 ;; *) echo 32 ;; esac; }   # pairs per optimizer update, by size

# ---- Evaluation -----------------------------------------------------------------------------
export CUTOFF_HNHC="30000"                      # high-need, high-cost threshold (next-year total, dollars)

# ---- Software helpers -----------------------------------------------------------------------
load_modules() {      # CPU jobs
    [ -z "${MODULES:-}" ] || { module --force purge 2>/dev/null; module load ${MODULES}; }
}
load_gpu_stack() {    # GPU jobs: modules + the conda env on PATH
    load_modules
    [ -z "${GPU_MODULES:-}" ] || module load ${GPU_MODULES}
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
}
gres() { echo "gpu:${GPU_TYPE:+${GPU_TYPE}:}$1"; }   # gres <n_gpus> -> gpu:<n> or gpu:<type>:<n>

# ---- Job-local scratch and caches (inside a SLURM job only) ---------------------------------
# Temp files and JIT caches go to node-local storage: multiprocess temp dirs and compile caches on a
# shared filesystem race across data-parallel workers. Removed when the job exits.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export JOB_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}"
    mkdir -p "${JOB_SCRATCH}" || { echo "ERROR: cannot create ${JOB_SCRATCH}"; exit 1; }
    chmod 700 "${JOB_SCRATCH}" 2>/dev/null || true
    export TMPDIR="${JOB_SCRATCH}"
    export TRITON_CACHE_DIR="${JOB_SCRATCH}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH}/inductor"
    export TORCH_EXTENSIONS_DIR="${JOB_SCRATCH}/torch_ext"
    export TQDM_DISABLE=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1   # batch logs are not terminals
    trap 'rm -rf "${JOB_SCRATCH}"' EXIT
fi
export HF_DATASETS_CACHE="${COST_HOME}/data/hf_cache"
export HF_HOME="${COST_HOME}/data/hf_cache"

# ---- Naming of post-trained checkpoints -----------------------------------------------------
posttrain_name() {  # posttrain_name <s|m|l> -> id; refuses a non-reported recipe without NAME_SUFFIX
    local s="$1" lr="${LR:-${POSTTRAIN_LR}}" ep="${EPOCHS:-${POSTTRAIN_EPOCHS}}" gb="${GLOBAL_BATCH:-$(posttrain_global_batch "$1")}"
    if [ "${lr}" != "${POSTTRAIN_LR}" ] || [ "${ep}" != "${POSTTRAIN_EPOCHS}" ] || [ "${gb}" != "$(posttrain_global_batch "$s")" ] || [ -n "${EXTRA_ARGS:-}" ]; then
        [ -n "${NAME_SUFFIX:-}" ] || { echo "ERROR: a non-default recipe (LR / EPOCHS / GLOBAL_BATCH / EXTRA_ARGS) needs NAME_SUFFIX=-<tag>" >&2; return 1; }
    fi
    echo "reclaim-${s}-posttrain${NAME_SUFFIX:-}"
}
