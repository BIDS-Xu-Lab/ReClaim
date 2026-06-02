#!/bin/bash
#SBATCH --job-name=s1_preprocess_test
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=1500G                
#SBATCH --time=06:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s1_preprocess_test_%j.out
#SBATCH --error=slurm_job_logs/s1_preprocess_test_%j.err



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


# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================

VERSION=6
SPLIT="test_1m"
TOKEN_FEATURE_TYPES=("binary" "count")
# Data directories 
PROCESSED_MARKETSCAN_DATA_ROOT="${PROCESSED_MARKETSCAN_DATA_ROOT:-/path/to/processed_marketscan_data_root}"
COST_TASK_ROOT="${COST_TASK_ROOT:-/path/to/expenditure_forecasting_root}"

RECLAIM_DATA_DIR="${PROCESSED_MARKETSCAN_DATA_ROOT}/v${VERSION}/marketscan_test"
PROCESSED_DATA_DIR="${COST_TASK_ROOT}/evaluation/versions/v${VERSION}/data"
VOCAB_PATH="${COST_TASK_ROOT}/evaluation/versions/v${VERSION}/vocab"


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





###########
# STEP 1: Find Prediction Point
###########
echo "=================================="
echo "STEP 1: Find Prediction Point (split: ${SPLIT})"
echo "=================================="

${CONDA_PYTHON} preprocess_find_prediction_point.py \
    --reclaim_data_dir ${RECLAIM_DATA_DIR} \
    --split ${SPLIT} \
    --processed_data_dir ${PROCESSED_DATA_DIR} \
    --num_proc ${SLURM_CPUS_PER_TASK-1}

STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: STEP 1 failed for split ${SPLIT}"
    exit $STEP1_EXIT
fi

###########
# STEP 2: Feature Processing (loop over token_feature_type)
###########
for TOKEN_FEATURE_TYPE in "${TOKEN_FEATURE_TYPES[@]}"; do
    echo ""
    echo "=================================="
    echo "STEP 2: Feature Processing (split: ${SPLIT}, token_feature_type: ${TOKEN_FEATURE_TYPE})"
    echo "=================================="

    ${CONDA_PYTHON} preprocess_baseline_features.py \
        --processed_data_dir ${PROCESSED_DATA_DIR} \
        --split ${SPLIT} \
        --vocab_path ${VOCAB_PATH} \
        --token_feature_type ${TOKEN_FEATURE_TYPE} \
        --num_proc ${SLURM_CPUS_PER_TASK-1}

    STEP2_EXIT=$?
    if [ $STEP2_EXIT -ne 0 ]; then
        echo "ERROR: STEP 2 failed for split ${SPLIT}, token_feature_type ${TOKEN_FEATURE_TYPE}"
        exit $STEP2_EXIT
    fi
done

echo ""
echo "=================================="
echo "All processing completed successfully"
echo "=================================="
