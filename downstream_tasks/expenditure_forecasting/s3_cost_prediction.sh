#!/bin/bash
#SBATCH --job-name=s3_cost_pred
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60    
#SBATCH --mem=400G                # memory per node
#SBATCH --time=24:00:00         # time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s3_cost_pred_%j.out
#SBATCH --error=slurm_job_logs/s3_cost_pred_%j.err



echo "=================================="
echo "Cost Prediction Job Starting"
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





##########################
## Overall Configuration
##########################

VERSION=6
COST_TASK_ROOT="${COST_TASK_ROOT:-/path/to/expenditure_forecasting_root}"
BASE_DIR="${COST_TASK_ROOT}"
SPLIT="test_1m"
BASELINE_TYPES=(binary count)
SECTIONS=(retrospective prospective)

# =============================================================================
# Configuration - RECLAIM model names
# =============================================================================
RECLAIM_MODELS=(
    v6-s-pretrain
    v6-m-pretrain
    v6-l-pretrain
)

EXIT_STATUS=0

for SECTION in "${SECTIONS[@]}"; do

echo ""
echo "######################################################################"
echo "# SECTION: ${SECTION}"
echo "######################################################################"

SECTION_DATA_DIR="${BASE_DIR}/evaluation/versions/v${VERSION}/data/${SPLIT}/${SECTION}"

###########
# XGBoost Cost Prediction
###########

echo ""
echo "=================================="
echo " [${SECTION}] XGBoost Cost Prediction"
echo "=================================="

NUM_BASELINE_MODELS=${#BASELINE_TYPES[@]}
echo "Number of baseline models: $NUM_BASELINE_MODELS (types: ${BASELINE_TYPES[*]})"
echo ""

for i in $(seq 0 $((NUM_BASELINE_MODELS - 1))); do
    TYPE="${BASELINE_TYPES[$i]}"
    MODEL="xgb_${TYPE}"
    MODEL_DIR="${BASE_DIR}/baseline/versions/v${VERSION}/model/1year/${TYPE}/xgboost"
    DATA_PATH="${SECTION_DATA_DIR}/baseline_features_1year_${TYPE}.parquet"
    OUTPUT_DIR="${SECTION_DATA_DIR}/model_gen/${MODEL}"

    echo "----------------------------------------"
    echo "[${SECTION}] Baseline model $((i + 1))/$NUM_BASELINE_MODELS"
    echo "----------------------------------------"
    echo "Model name: $MODEL"
    echo "Model dir: $MODEL_DIR"
    echo "Data path: $DATA_PATH"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    mkdir -p ${OUTPUT_DIR}

    echo "Running prediction for baseline model $((i + 1))..."
    ${CONDA_PYTHON} cost_pred_xgb.py \
        --model ${MODEL} \
        --model_dir ${MODEL_DIR} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}

    CONFIG_EXIT_STATUS=$?
    if [ $CONFIG_EXIT_STATUS -ne 0 ]; then
        echo "WARNING: [${SECTION}] XGBoost prediction failed for ${MODEL} with exit status $CONFIG_EXIT_STATUS"
        EXIT_STATUS=$CONFIG_EXIT_STATUS
    else
        echo "[${SECTION}] XGBoost prediction completed successfully for ${MODEL}"
    fi
    echo ""
done


###########
# LightGBM Cost Prediction
###########

echo ""
echo "=================================="
echo " [${SECTION}] LightGBM Cost Prediction"
echo "=================================="

NUM_BASELINE_MODELS=${#BASELINE_TYPES[@]}
echo "Number of baseline models: $NUM_BASELINE_MODELS (types: ${BASELINE_TYPES[*]})"
echo ""

for i in $(seq 0 $((NUM_BASELINE_MODELS - 1))); do
    TYPE="${BASELINE_TYPES[$i]}"
    MODEL="lgbm_${TYPE}"
    MODEL_DIR="${BASE_DIR}/baseline/versions/v${VERSION}/model/1year/${TYPE}/lightgbm"
    DATA_PATH="${SECTION_DATA_DIR}/baseline_features_1year_${TYPE}.parquet"
    OUTPUT_DIR="${SECTION_DATA_DIR}/model_gen/${MODEL}"

    echo "----------------------------------------"
    echo "[${SECTION}] Baseline model $((i + 1))/$NUM_BASELINE_MODELS"
    echo "----------------------------------------"
    echo "Model name: $MODEL"
    echo "Model dir: $MODEL_DIR"
    echo "Data path: $DATA_PATH"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    mkdir -p ${OUTPUT_DIR}

    echo "Running prediction for baseline model $((i + 1))..."
    ${CONDA_PYTHON} cost_pred_lgbm.py \
        --model ${MODEL} \
        --model_dir ${MODEL_DIR} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}

    CONFIG_EXIT_STATUS=$?
    if [ $CONFIG_EXIT_STATUS -ne 0 ]; then
        echo "WARNING: [${SECTION}] LightGBM prediction failed for ${MODEL} with exit status $CONFIG_EXIT_STATUS"
        EXIT_STATUS=$CONFIG_EXIT_STATUS
    else
        echo "[${SECTION}] LightGBM prediction completed successfully for ${MODEL}"
    fi
    echo ""
done


###########
# RECLAIM Cost Aggregation
###########

echo ""
echo "=================================="
echo " [${SECTION}] RECLAIM Cost Aggregation"
echo "=================================="

NUM_RECLAIM_MODELS=${#RECLAIM_MODELS[@]}
echo "Number of RECLAIM models: $NUM_RECLAIM_MODELS (models: ${RECLAIM_MODELS[*]})"
echo ""

for i in $(seq 0 $((NUM_RECLAIM_MODELS - 1))); do
    MODEL="${RECLAIM_MODELS[$i]}"
    OUTPUT_DIR="${SECTION_DATA_DIR}/model_gen/${MODEL}"

    echo "----------------------------------------"
    echo "[${SECTION}] RECLAIM model $((i + 1))/$NUM_RECLAIM_MODELS: $MODEL"
    echo "----------------------------------------"
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""

    echo "Running cost aggregation for $MODEL..."
    ${CONDA_PYTHON} cost_agg_reclaim.py \
        --model ${MODEL} \
        --output_dir ${OUTPUT_DIR}

    CONFIG_EXIT_STATUS=$?
    if [ $CONFIG_EXIT_STATUS -ne 0 ]; then
        echo "WARNING: [${SECTION}] RECLAIM cost aggregation failed for $MODEL with exit status $CONFIG_EXIT_STATUS"
        EXIT_STATUS=$CONFIG_EXIT_STATUS
    else
        echo "[${SECTION}] RECLAIM cost aggregation completed successfully for $MODEL"
    fi
    echo ""
done

done  # end SECTIONS loop


echo ""
echo "=================================="
echo "Cost Prediction Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
