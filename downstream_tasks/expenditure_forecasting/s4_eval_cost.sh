#!/bin/bash
#SBATCH --job-name=s4_eval_cost
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60         # lightweight evaluation, no heavy parallelism needed
#SBATCH --mem=1600G                 # sufficient for metric computation
#SBATCH --time=06:00:00            # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s4_eval_cost_%j.out
#SBATCH --error=slurm_job_logs/s4_eval_cost_%j.err



echo "=================================="
echo "Cost Outcome Evaluation Job Starting"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"
echo "=================================="
echo ""


# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================

###########
# Paths
###########
SPLIT="test_1m"
COST_TASK_ROOT="${COST_TASK_ROOT:-/path/to/expenditure_forecasting_root}"
BASE_DATA_DIR="${COST_TASK_ROOT}/evaluation/versions/v6/data"
BASE_RESULT_DIR="${COST_TASK_ROOT}/evaluation/versions/v6/results"
SECTIONS=(retrospective prospective)

###########
# Model list - add model names here
###########

MODELS=(
    "v6-s-pretrain"
    "v6-m-pretrain"
    "v6-l-pretrain"
    # "v6-s-posttrain"
    # "v6-m-posttrain"
    # "v6-l-posttrain"
    # "xgb_count"
    # "xgb_binary"
    "lgbm_count"
    "lgbm_binary"
)

NUM_MODELS=${#MODELS[@]}
echo "Number of models: $NUM_MODELS"

###########
# Cutoff configurations - each string is comma-separated cutoffs
# Examples:
#   "15000"              -> binary classification (< $15,000 vs >= $15,000)
#   "1500,15000,30000"   -> 4-way classification
###########
CUTOFF_LIST=(
    "30000"
    "1500,15000"
)

NUM_CUTOFFS=${#CUTOFF_LIST[@]}
echo "Number of cutoff configurations: $NUM_CUTOFFS"
echo "Sections: ${SECTIONS[*]}"


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
# Cost Outcome Evaluation - Loop through sections
###########
EXIT_STATUS=0

for SECTION in "${SECTIONS[@]}"; do

GOLD_PATH="${BASE_DATA_DIR}/${SPLIT}/${SECTION}/preprocessed_input.parquet"
MODEL_GEN_DIR="${BASE_DATA_DIR}/${SPLIT}/${SECTION}/model_gen"
RESULT_DIR="${BASE_RESULT_DIR}/${SPLIT}/${SECTION}"

echo ""
echo "######################################################################"
echo "# SECTION: ${SECTION}"
echo "######################################################################"
echo ""
echo "=================================="
echo " [${SECTION}] Cost Outcome Evaluation"
echo " Models: ${MODELS[@]}"
echo " Cutoff configurations: ${CUTOFF_LIST[@]}"
echo " Total configurations: $((NUM_MODELS * NUM_CUTOFFS))"
echo " Gold path: ${GOLD_PATH}"
echo " Model gen dir: ${MODEL_GEN_DIR}"
echo " Result directory: ${RESULT_DIR}"
echo "=================================="

mkdir -p ${RESULT_DIR}

${CONDA_PYTHON} eval_cost_main.py \
    --model_list ${MODELS[@]} \
    --gold_path ${GOLD_PATH} \
    --model_gen_dir ${MODEL_GEN_DIR} \
    --cutoff_list ${CUTOFF_LIST[@]} \
    --result_dir ${RESULT_DIR}

SECTION_EXIT=$?
if [ $SECTION_EXIT -ne 0 ]; then
    echo "WARNING: [${SECTION}] Outcome evaluation failed with exit status $SECTION_EXIT"
    EXIT_STATUS=$SECTION_EXIT
else
    echo "[${SECTION}] Outcome evaluation completed successfully"
fi

done  # end SECTIONS loop

echo ""
echo "=================================="
echo "Cost Outcome Evaluation Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
