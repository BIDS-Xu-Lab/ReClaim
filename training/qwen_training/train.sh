module load CUDA/12.8.0
module load GCC/13.3.0

# set model
model_type="qwen3-L"
TRAINING_MODE="post_training"
CKPT_TRAIN="false"

# defaults
DATA_VERSION="v6"
DATA_SOURCE="${DATA_VERSION}_marketscan_1000k_seqlen_4096"
POST_TRAIN_DATA_SOURCE="tokens_split_size_100000_pos_after_split_random"
PRETRAINED_MODEL="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/Models_Hopper/v6_models_zloss/qwen3-l-v6-zloss/pretrain"

POST_TRAIN_TASK="disease"
OUTPUT_FOLDER_NAME="${POST_TRAIN_TASK}"
DS_CONFIG="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_all_qwen_training/config/ds_config.json"
PRE_TRAIN_CONFIG="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_all_qwen_training/config/pre_train_config.json"
POST_TRAIN_CONFIG="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_all_qwen_training/config/post_train_config.json"

# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=1
# export NCCL_SOCKET_IFNAME=eno12399np0

echo "[$(date)] Node: $(hostname)"

# Wait until no other processes are using the allocated GPUs
while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -eq 0 ]; then
        echo "[$(date)] GPUs are free. Starting training..."
        break
    fi
    echo "[$(date)] GPUs in use ($gpu_procs process(es)). Waiting 30s..."
    sleep 30
done

torchrun \
    --nproc_per_node=1 \
    ***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_all_qwen_training/train.py \
    --deepspeed ${DS_CONFIG} \
    --model_type ${model_type} \
    --training_mode ${TRAINING_MODE} \
    --pre_train_config ${PRE_TRAIN_CONFIG} \
    --post_train_config ${POST_TRAIN_CONFIG} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --data_source "$DATA_SOURCE" \
    --post_train_data_source "$POST_TRAIN_DATA_SOURCE" \
    --post_train_task "$POST_TRAIN_TASK" \
    --output_folder_name "$OUTPUT_FOLDER_NAME" \
    --resume_from_checkpoint ${CKPT_TRAIN}