#!/bin/bash
#SBATCH --job-name=readout_instruct_vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e3: readout (vLLM) -- evaluation-cohort predictions from a post-trained checkpoint: one
# forward pass and one decoded token per enrollee; the softmax over the cost bins gives the dollar
# forecast (expectation over the bins' dollar values, capped at $1M) and prob_ge_<cutoff>.
#   MODEL=<dir under models/> or MODEL_SIZE=s|m|l (-> reclaim-<size>-posttrain)
#   [TAG=]  suffix of the output folder   [EXTRA_ARGS]  e.g. "--max_rows 64" with TAG=-smoke
#   GPUs of the allocation -> data-parallel replicas (one engine per GPU)
#   writes: data/test/<section>/model_gen/${MODEL}${TAG}/{cost_predictions.csv,timing.json}
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_gpu_stack
export VLLM_DISABLE_COMPILE_CACHE=1   # a shared-filesystem compile cache corrupts under data-parallel workers
MODEL="${MODEL:-reclaim-${MODEL_SIZE:?set MODEL=<dir under models/> or MODEL_SIZE=s|m|l}-posttrain}"; TAG="${TAG:-}"; N="${SLURM_GPUS_ON_NODE:-1}"
[ -d "${COST_HOME}/models/${MODEL}/final" ] || { echo "ERROR: no checkpoint at models/${MODEL}/final -- train it first"; exit 1; }
echo "readout_instruct_vllm ${MODEL} -> model_gen/${MODEL}${TAG}: DP=${N}"
${CONDA_PYTHON} e3_posttrain/readout_instruct_vllm.py --model_dir "${COST_HOME}/models/${MODEL}/final" \
    --section_dirs "${DATA_DIR}/${SPLIT}/retrospective" "${DATA_DIR}/${SPLIT}/prospective" \
    --model_name "${MODEL}${TAG}" --seq_len "${SEQ_LEN}" --dp_size "${N}" ${EXTRA_ARGS:-}
echo "readout_instruct_vllm completed: ${MODEL}${TAG}"
