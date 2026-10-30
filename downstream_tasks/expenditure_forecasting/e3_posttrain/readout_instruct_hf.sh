#!/bin/bash
#SBATCH --job-name=readout_instruct_hf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e3: reference readout (HF transformers) -- the same readout without the inference engine,
# kept for cross-engine verification. Same interface as readout_instruct_vllm.sh; the output
# folder is suffixed -hf by default so it never overwrites the vLLM predictions.
#   MODEL=<dir under models/> or MODEL_SIZE=s|m|l   [TAG=-hf]   [EXTRA_ARGS="--max_rows 64"]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_gpu_stack
MODEL="${MODEL:-reclaim-${MODEL_SIZE:?set MODEL=<dir under models/> or MODEL_SIZE=s|m|l}-posttrain}"; TAG="${TAG:--hf}"; N="${SLURM_GPUS_ON_NODE:-1}"
[ -d "${COST_HOME}/models/${MODEL}/final" ] || { echo "ERROR: no checkpoint at models/${MODEL}/final -- train it first"; exit 1; }
echo "readout_instruct_hf ${MODEL} -> model_gen/${MODEL}${TAG}: DP=${N}"
${CONDA_PYTHON} e3_posttrain/readout_instruct_hf.py --model_dir "${COST_HOME}/models/${MODEL}/final" \
    --section_dirs "${DATA_DIR}/${SPLIT}/retrospective" "${DATA_DIR}/${SPLIT}/prospective" \
    --model_name "${MODEL}${TAG}" --seq_len "${SEQ_LEN}" --dp_size "${N}" --batch_size 32 ${EXTRA_ARGS:-}
echo "readout_instruct_hf completed: ${MODEL}${TAG}"
