#!/bin/bash
#SBATCH --job-name=s2_generate_embeddings
#SBATCH --partition=gpu  
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=qos_nmi
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=32        # 64 cpu-cores per task
#SBATCH --mem=500G                # 900G memory per node
#SBATCH --time=1-00:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s2_generate_embeddings_%j.out
#SBATCH --error=/dev/null



echo "=================================="
echo "ReClaim Embedding Evaluation Job Starting"
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



# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda

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




EXIT_STATUS=0

RECLAIM_MODEL_ROOT="${RECLAIM_MODEL_ROOT:-/path/to/reclaim_model_root}"
DELPHI_MODEL_PATH="${DELPHI_MODEL_PATH:-/path/to/delphi_model_checkpoint.pt}"
DELPHI_DISEASE_MAP="${DELPHI_DISEASE_MAP:-/path/to/delphi_disease_map.csv}"


# =============================================================================
# SECTION 1: ReClaim Embeddings
# =============================================================================

echo ""
echo "###########"
echo "# SECTION 1: ReClaim Embeddings"
echo "###########"

declare -A RECLAIM_CONFIGS=(
    ["v6-no"]=""  # No embeddings mode - copies data without generating embeddings
    ["v6-s"]="${RECLAIM_MODEL_ROOT}/qwen3-s-v6-zloss/pretrain"
    ["v6-m"]="${RECLAIM_MODEL_ROOT}/qwen3-m-v6-zloss/pretrain"
    ["v6-l"]="${RECLAIM_MODEL_ROOT}/qwen3-l-v6-zloss/pretrain"
)

NUM_RECLAIM=${#RECLAIM_CONFIGS[@]}
echo "Number of ReClaim model configurations: $NUM_RECLAIM"

CONFIG_NUM=0
for MODEL in "${!RECLAIM_CONFIGS[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    MODEL_PATH="${RECLAIM_CONFIGS[$MODEL]}"

    echo ""
    echo "=================================="
    echo " ReClaim - Config ${CONFIG_NUM}/${NUM_RECLAIM}"
    echo " Model ID: $MODEL"
    echo " Model Path: $MODEL_PATH"
    echo "=================================="

    MODEL_SIZE="${MODEL#*-}"
    if [ "$MODEL_SIZE" != "no" ] && [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Model path does not exist: $MODEL_PATH"
        echo "Skipping this configuration..."
        continue
    fi

    python p4_generate_reclaim_embeddings.py \
        --model "$MODEL" \
        --model_path "$MODEL_PATH"

    RUN_STATUS=$?
    if [ $RUN_STATUS -ne 0 ]; then
        echo "WARNING: ReClaim config ${CONFIG_NUM} failed with exit status $RUN_STATUS"
        EXIT_STATUS=$RUN_STATUS
    else
        echo "ReClaim config ${CONFIG_NUM} completed successfully"
    fi
done


# =============================================================================
# SECTION 2: Delphi Embeddings
# =============================================================================

echo ""
echo "###########"
echo "# SECTION 2: Delphi Embeddings"
echo "###########"

DELPHI_DATA_VERSIONS=("v6")

NUM_DELPHI=${#DELPHI_DATA_VERSIONS[@]}
echo "Number of Delphi configurations: $NUM_DELPHI"
echo "Delphi model: $DELPHI_MODEL_PATH"
echo "Disease map: $DELPHI_DISEASE_MAP"

CONFIG_NUM=0
for DATA_VERSION in "${DELPHI_DATA_VERSIONS[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))

    echo ""
    echo "=================================="
    echo " Delphi - Config ${CONFIG_NUM}/${NUM_DELPHI}"
    echo " Data Version: $DATA_VERSION"
    echo "=================================="

    if [ ! -f "$DELPHI_MODEL_PATH" ]; then
        echo "WARNING: Delphi model not found: $DELPHI_MODEL_PATH"
        echo "Skipping this configuration..."
        continue
    fi

    python p4_generate_delphi_embeddings.py \
        --data_version "$DATA_VERSION" \
        --model_path "$DELPHI_MODEL_PATH" \
        --disease_map_path "$DELPHI_DISEASE_MAP"

    RUN_STATUS=$?
    if [ $RUN_STATUS -ne 0 ]; then
        echo "WARNING: Delphi config ${CONFIG_NUM} failed with exit status $RUN_STATUS"
        EXIT_STATUS=$RUN_STATUS
    else
        echo "Delphi config ${CONFIG_NUM} completed successfully"
    fi
done


echo ""
echo "=================================="
echo "Embedding Generation Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
