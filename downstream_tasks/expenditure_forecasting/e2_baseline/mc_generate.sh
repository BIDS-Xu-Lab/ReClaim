#!/bin/bash
#SBATCH --job-name=mc_generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e2: Monte-Carlo anchor. NUM_GENERATION sampled one-year trajectories per enrollee from the
# pre-trained backbone (temperature 1, top-p 1), vLLM data-parallel over the allocated GPUs.
#   MODEL_SIZE=s|m|l   [SECTION=both|retrospective|prospective]
#   writes: data/test/<section>/model_gen/reclaim-<size>-pretrain/{generated_data_<N>.parquet,timing_<N>.json}
# Every trajectory count gets its own files, so runs at different NUM_GENERATION coexist.
# Host memory grows with NUM_GENERATION (every request is held until its shard finishes): 20
# trajectories fit in 900G on this cohort, 64 need about 1.5T (sbatch --mem, or MC_MEM in run.sh).
# NUM_GENERATION=1 decodes greedily and is not a Monte-Carlo estimate.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_gpu_stack
export VLLM_DISABLE_COMPILE_CACHE=1   # a shared-filesystem compile cache corrupts under data-parallel workers
MODEL_SIZE="${MODEL_SIZE:?set MODEL_SIZE=s|m|l}"
MODEL_PATH="$(backbone_path "${MODEL_SIZE}")"
echo "mc_generate reclaim-${MODEL_SIZE}-pretrain: N=${NUM_GENERATION} DP=${SLURM_GPUS_ON_NODE:-1} SECTION=${SECTION:-both}"
${CONDA_PYTHON} e2_baseline/generate_vllm.py \
    --processed_data_base_dir "${DATA_DIR}/${SPLIT}" \
    --model "reclaim-${MODEL_SIZE}-pretrain" --model_path "${MODEL_PATH}" \
    --num_generation "${NUM_GENERATION}" --seq_len "${SEQ_LEN}" \
    --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --seed "${SEED}" \
    --dp_size "${SLURM_GPUS_ON_NODE:-1}" --section "${SECTION:-both}"
echo "mc_generate completed"
