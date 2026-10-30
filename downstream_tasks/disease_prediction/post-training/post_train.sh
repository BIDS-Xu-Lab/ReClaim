#!/bin/bash
#SBATCH --job-name=qwen3-post
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=976G
#SBATCH --output=logs/post-train/slurm-%A_%a.out

module load CUDA/12.8.0

# set model
model_type="qwen3-L"
TRAINING_MODE="post_training"
CKPT_TRAIN="false"

PRE_TRAIN_MODEL="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/vf_marketscan_full_combined_seqlen_4096_prospect/model/qwen_zloss_compute_loss/qwen3-L/final_models/pre_training"

# defaults
DATA_VERSION="vf"
DATA_SOURCE="${DATA_VERSION}_marketscan_full_combined_seqlen_4096_prospect"

SPLIT="random"
POST_TRAIN_SIZE="100000"
POST_TRAIN_DATA_BASE="${DATA_VERSION}_marketscan_full_combined_seqlen_4096_prospect"
POST_TRAIN_DATA_SOURCE="tokens_size_${POST_TRAIN_SIZE}_pos_after_split_${SPLIT}"

OUTPUT_FOLDER_NAME="qwen_zloss_compute_loss"
# Set this to your checkout root, including when submitting with sbatch.
REPO_ROOT="***->[REPLACE WITH YOUR PATH]<-***"
BASE_FOLDER="${REPO_ROOT}/pre-training/qwen_training"
DS_CONFIG="${BASE_FOLDER}/config/ds_config.json"
PRE_TRAIN_CONFIG="${BASE_FOLDER}/config/pre_train_config.json"
POST_TRAIN_CONFIG="${BASE_FOLDER}/config/post_train_config.json"

export MASTER_PORT=29501
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export OMP_NUM_THREADS=1

echo "Nodes: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "WorldL $SLURM_NNODES nodes x $SLURM_NTASKS_PER_NODE = $((SLURM_JOB_NUM_NODES*SLURM_NTASKS_PER_NODE)) ranks"

echo "[$(date)] Node: $(hostname)"
echo "Launching multi-node training..."

torchrun \
    --nproc_per_node=8 \
    "${BASE_FOLDER}/train_zloss.py" \
    --deepspeed "${DS_CONFIG}" \
    --model_type ${model_type} \
    --training_mode ${TRAINING_MODE} \
    --pre_train_config "${PRE_TRAIN_CONFIG}" \
    --post_train_config "${POST_TRAIN_CONFIG}" \
    --data_source "$DATA_SOURCE" \
    --pre_train_model "$PRE_TRAIN_MODEL" \
    --post_train_data_base "$POST_TRAIN_DATA_BASE" \
    --post_train_data_source "$POST_TRAIN_DATA_SOURCE" \
    --output_folder_name "$OUTPUT_FOLDER_NAME" \
    --resume_from_checkpoint ${CKPT_TRAIN}
