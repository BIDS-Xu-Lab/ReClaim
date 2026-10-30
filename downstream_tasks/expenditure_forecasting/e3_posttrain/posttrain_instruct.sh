#!/bin/bash
#SBATCH --job-name=posttrain_instruct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e3: expenditure post-training. The response is one <COST-me> token (next-year total) + <eos>;
# the loss is a soft ordinal cross-entropy over the cost bins plus a Huber term on the expected
# log-dollars (posttrain_instruct.py). Training only: saves models/<name>/final; predictions are a
# separate job (readout_instruct_vllm.sh).
#   MODEL_SIZE=s|m|l   [LR= EPOCHS= GLOBAL_BATCH= PER_DEVICE_BATCH= EXTRA_ARGS= NAME_SUFFIX= SMOKE=1 RESUME=1]
# Reported recipe (config.sh): learning rate POSTTRAIN_LR, POSTTRAIN_EPOCHS epochs, global batch 32
# (S, M) or 64 (L) -> reclaim-<size>-posttrain. Any other value needs NAME_SUFFIX=-<tag>.
# More than one GPU -> torchrun DDP; gradient accumulation keeps the global batch fixed:
# grad_accum = GLOBAL_BATCH / (PER_DEVICE_BATCH x GPUs).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_gpu_stack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_SIZE="${MODEL_SIZE:?set MODEL_SIZE=s|m|l}"
N_GPUS="${SLURM_GPUS_ON_NODE:-1}"
LR="${LR:-${POSTTRAIN_LR}}"; EPOCHS="${EPOCHS:-${POSTTRAIN_EPOCHS}}"
GB="${GLOBAL_BATCH:-$(posttrain_global_batch "${MODEL_SIZE}")}"
case "${EXTRA_ARGS:-}" in *--lr*|*--epochs*) echo "ERROR: pass the learning rate and epochs as LR= / EPOCHS=, not in EXTRA_ARGS"; exit 1;; esac
NAME="$(posttrain_name "${MODEL_SIZE}")"
if [ -z "${PER_DEVICE_BATCH:-}" ]; then   # size default (8; 4 at L), capped so PER_DEVICE_BATCH x GPUs <= GLOBAL_BATCH
    PER_DEVICE_BATCH=$([ "${MODEL_SIZE}" = l ] && echo 4 || echo 8)
    [ $(( PER_DEVICE_BATCH * N_GPUS )) -gt "${GB}" ] && PER_DEVICE_BATCH=$(( GB / N_GPUS ))
fi
[ $(( PER_DEVICE_BATCH * N_GPUS )) -le "${GB}" ] && [ $(( GB % (PER_DEVICE_BATCH * N_GPUS) )) -eq 0 ] || { echo "ERROR: PER_DEVICE_BATCH x GPUs = $(( PER_DEVICE_BATCH * N_GPUS )) must divide GLOBAL_BATCH=${GB}"; exit 1; }
GRAD_ACCUM=$(( GB / (PER_DEVICE_BATCH * N_GPUS) ))
MODEL_PATH="$(backbone_path "${MODEL_SIZE}")"
LAUNCH=(${CONDA_PYTHON}); [ "${N_GPUS}" -gt 1 ] && LAUNCH=(${CONDA_PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${N_GPUS})
EXTRA=(--lr "${LR}" --epochs "${EPOCHS}" ${EXTRA_ARGS:-}); [ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)
[ "${SMOKE:-0}" = 1 ] && { EXTRA+=(--max_train_pairs 256 --eval_steps 4); NAME="${NAME}-smoke"; }
echo "posttrain ${NAME}: ${N_GPUS} GPU(s), lr ${LR}, ${EPOCHS} epochs, global batch ${GB} (per-device ${PER_DEVICE_BATCH}, grad_accum ${GRAD_ACCUM})"
"${LAUNCH[@]}" e3_posttrain/posttrain_instruct.py \
    --model_path "${MODEL_PATH}" --pairs_dir "${PAIRS_DIR}/mantissa" \
    --bin_artifact "${PAIRS_DIR}/bin_artifact_mantissa.json" \
    --output_dir "${COST_HOME}/models/${NAME}" \
    --per_device_batch "${PER_DEVICE_BATCH}" --grad_accum "${GRAD_ACCUM}" \
    --eval_steps 250 --seed "${SEED}" "${EXTRA[@]}"
echo "posttrain completed: ${NAME}   (next: MODEL=${NAME} sbatch e3_posttrain/readout_instruct_vllm.sh)"
