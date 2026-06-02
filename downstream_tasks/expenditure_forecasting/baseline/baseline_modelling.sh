#!/bin/bash
#SBATCH --job-name=baseline_modelling
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=1800G                
#SBATCH --time=1-00:00:00         
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/baseline_modelling_%j.out
#SBATCH --error=slurm_job_logs/baseline_modelling_%j.err



echo "=================================="
echo "Baseline Modeling Job Starting (XGBoost + LightGBM)"
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

HISTORY_LENGTH="1year"  # Options: "1year" or "all"
TOKEN_FEATURE_TYPES=("binary" "count")
# Data directories
COST_TASK_ROOT="${COST_TASK_ROOT:-/path/to/expenditure_forecasting_root}"
PROCESSED_DATA_DIR="${COST_TASK_ROOT}/baseline/versions/v${VERSION}/data"
OUTPUT_DIR="${COST_TASK_ROOT}/baseline/versions/v${VERSION}/model"



# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda

# Optionally pin conda env/package caches to a specific local root.
if [ -n "${CONDA_WORK_ROOT:-}" ]; then
    export CONDA_ENVS_PATH="${CONDA_WORK_ROOT}/envs"
    export CONDA_PKGS_DIRS="${CONDA_WORK_ROOT}/pkgs"
fi

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


for TOKEN_FEATURE_TYPE in "${TOKEN_FEATURE_TYPES[@]}"; do

    ###########
    # XGBoost Training ##
    ###########
    echo "=================================="
    echo "XGBoost Training (token_feature_type: ${TOKEN_FEATURE_TYPE})"
    echo "=================================="
    ${CONDA_PYTHON} regressor_xgboost.py \
        --processed_data_dir ${PROCESSED_DATA_DIR} \
        --history_length ${HISTORY_LENGTH} \
        --token_feature_type ${TOKEN_FEATURE_TYPE} \
        --output_dir ${OUTPUT_DIR}

    EXIT_STATUS=$?
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "ERROR: XGBoost training failed with exit status $EXIT_STATUS (token_feature_type: ${TOKEN_FEATURE_TYPE})"
        exit $EXIT_STATUS
    fi

    echo "=================================="
    echo "XGBoost training completed successfully (token_feature_type: ${TOKEN_FEATURE_TYPE})"
    echo "=================================="
    echo ""

    ###########
    # LightGBM Training ##
    ###########
    echo "=================================="
    echo "LightGBM Training (token_feature_type: ${TOKEN_FEATURE_TYPE})"
    echo "=================================="
    ${CONDA_PYTHON} regressor_lightgbm.py \
        --processed_data_dir ${PROCESSED_DATA_DIR} \
        --history_length ${HISTORY_LENGTH} \
        --token_feature_type ${TOKEN_FEATURE_TYPE} \
        --output_dir ${OUTPUT_DIR}

    EXIT_STATUS=$?
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "ERROR: LightGBM training failed with exit status $EXIT_STATUS (token_feature_type: ${TOKEN_FEATURE_TYPE})"
        exit $EXIT_STATUS
    fi

    echo "=================================="
    echo "LightGBM training completed successfully (token_feature_type: ${TOKEN_FEATURE_TYPE})"
    echo "=================================="

done

echo ""
echo "=================================="
echo "All baseline models completed successfully"
echo "End Time: $(date)"
echo "=================================="
exit 0
