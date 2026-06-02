#!/bin/bash
#SBATCH --job-name=s2_generate_vllm
#SBATCH --partition=gpu  
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --qos=qos_bids
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=32        # 64 cpu-cores per task
#SBATCH --mem=900G                # 900G memory per node
#SBATCH --time=24:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s2_generate_vllm_%j.out
#SBATCH --error=slurm_job_logs/s2_generate_vllm_%j.err



echo "=================================="
echo "ReClaim Cost Prediction Generation Job Starting"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "Start Time: $(date)"
echo "=================================="
echo ""


# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================
GPU_MEMORY_UTILIZATION=0.6
NUM_GENERATION=20
SPLIT='test_1m'
SECTION='prospective'   # "both", "retrospective", or "prospective"
DATA_PARALLEL_SIZE=${SLURM_GPUS_ON_NODE:-1}   # auto-detected from --gres gpu count
COST_TASK_ROOT="${COST_TASK_ROOT:-/path/to/expenditure_forecasting_root}"
RECLAIM_MODEL_ROOT="${RECLAIM_MODEL_ROOT:-/path/to/reclaim_model_root}"
###########
# Model Configurations: [model] = model_path
# Dictionary-like syntax - fill in the configurations below
###########

declare -A MODEL_CONFIGS=(
    ["v6-s-pretrain"]="${RECLAIM_MODEL_ROOT}/qwen3-s-v6-zloss/pretrain"
    ["v6-m-pretrain"]="${RECLAIM_MODEL_ROOT}/qwen3-m-v6-zloss/pretrain"
    ["v6-l-pretrain"]="${RECLAIM_MODEL_ROOT}/qwen3-l-v6-zloss/pretrain"
)

NUM_MODELS=${#MODEL_CONFIGS[@]}
echo "Number of model configurations: $NUM_MODELS"

# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda
module load CUDA/12.8.0
module load GCC/12.2.0

# Initialize and activate conda BEFORE loading Spark
echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh

echo "Activating conda environment 'reclaim'..."
conda activate reclaim

# Get conda environment path - use CONDA_PREFIX which is set after activation
# or find it from conda env list
if [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_ENV_PATH="$CONDA_PREFIX"
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
    echo "Using CONDA_PREFIX: $CONDA_ENV_PATH"
else
    # Fallback: try to find from conda env list
    CONDA_ENV_PATH=$(conda env list | grep -E "^\s*reclaim\s" | awk '{print $NF}' | head -1)
    if [ -z "$CONDA_ENV_PATH" ]; then
        # Try user directory
        CONDA_ENV_PATH="$HOME/.conda/envs/reclaim"
    fi
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
fi

# Verify conda environment exists
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment 'reclaim' not found at $CONDA_PYTHON"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Trying to find correct path..."
    # Try to get path from conda env list
    ENV_PATH=$(conda env list | grep -E "^\s*reclaim\s" | awk '{print $NF}' | head -1)
    if [ -n "$ENV_PATH" ] && [ -f "${ENV_PATH}/bin/python" ]; then
        CONDA_ENV_PATH="$ENV_PATH"
        CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
        echo "Found environment at: $CONDA_ENV_PATH"
    else
        echo "Could not locate 'reclaim' environment"
        exit 1
    fi
fi

echo "Conda Python: $CONDA_PYTHON"




###########
# Generate Sequences - Loop through all configurations
###########
EXIT_STATUS=0
CONFIG_NUM=0

for MODEL in "${!MODEL_CONFIGS[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    MODEL_PATH="${MODEL_CONFIGS[$MODEL]}"
    
    echo ""
    echo "=================================="
    echo " Generate Sequences - Config ${CONFIG_NUM}/${NUM_MODELS}"
    echo " Model ID: $MODEL"
    echo " Model Path: $MODEL_PATH"
    echo "=================================="
    
    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Model path does not exist: $MODEL_PATH"
        echo "Skipping this configuration..."
        continue
    fi

    # Extract VERSION number from MODEL (e.g., "v6-s-pretrain" -> "6")
    VERSION=$(echo "$MODEL" | sed -n 's/^v\([0-9]\+\).*/\1/p')
    PREPROCESSED_DATA_DIR="${COST_TASK_ROOT}/evaluation/versions/v${VERSION}/data/${SPLIT}"
    echo " Version: $VERSION"
    echo " Data Dir: $PREPROCESSED_DATA_DIR"

    
    srun $CONDA_PYTHON s2_generate_vllm.py \
        --processed_data_base_dir ${PREPROCESSED_DATA_DIR} \
        --model ${MODEL} \
        --model_path ${MODEL_PATH} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --num_generation ${NUM_GENERATION} \
        --dp_size ${DATA_PARALLEL_SIZE} \
        --section ${SECTION}

 
    
    # Capture exit status for this run
    RUN_STATUS=$?
    if [ $RUN_STATUS -ne 0 ]; then
        echo "WARNING: Config ${CONFIG_NUM} failed with exit status $RUN_STATUS"
        EXIT_STATUS=$RUN_STATUS
    else
        echo "Config ${CONFIG_NUM} completed successfully"
    fi
done

echo ""
echo "=================================="
echo "Cost Prediction Generation Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
