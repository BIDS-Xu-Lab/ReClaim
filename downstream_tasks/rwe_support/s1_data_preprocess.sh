#!/bin/bash
#SBATCH --job-name=s1_data_preprocess
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=150G                # 900G memory per node
#SBATCH --time=1-00:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s1_data_preprocess_%j.out
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


# =============================================================================
# Configuration - Modify these variables as needed
# =============================================================================

# Version, timestamp granularity is set at month level
VERSION=6
# Which split to use: test, val, or both
TEST_SPLIT="${TEST_SPLIT:-val}"

# Data and mapping roots
PROCESSED_MARKETSCAN_DATA_ROOT="${PROCESSED_MARKETSCAN_DATA_ROOT:-/path/to/processed_marketscan_data_root}"
VOCAB_MAPPING_DIR="${VOCAB_MAPPING_DIR:-../../preprocessing/vocab_mapping/mapping}"

if [ "$TEST_SPLIT" = "both" ]; then
    MARKETSCAN_SPLIT_DIR="${MARKETSCAN_SPLIT_DIR:-${PROCESSED_MARKETSCAN_DATA_ROOT}/v${VERSION}/marketscan_test,${PROCESSED_MARKETSCAN_DATA_ROOT}/v${VERSION}/marketscan_val}"
else
    MARKETSCAN_SPLIT_DIR="${MARKETSCAN_SPLIT_DIR:-${PROCESSED_MARKETSCAN_DATA_ROOT}/v${VERSION}/marketscan_${TEST_SPLIT}}"
fi

DRUG_MAPPING_FILE="${DRUG_MAPPING_FILE:-${VOCAB_MAPPING_DIR}/ndc_drug_to_rxnorm_ingredient_mapping.csv}"

echo "Configuration:"
echo "  VERSION: v${VERSION}"
echo "  TEST_SPLIT: ${TEST_SPLIT}"
echo "  PROCESSED_MARKETSCAN_DATA_ROOT: ${PROCESSED_MARKETSCAN_DATA_ROOT}"
echo "  MARKETSCAN_SPLIT_DIR: ${MARKETSCAN_SPLIT_DIR}"
echo "  VOCAB_MAPPING_DIR: ${VOCAB_MAPPING_DIR}"
echo "  DRUG_MAPPING_FILE: ${DRUG_MAPPING_FILE}"
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


# Now load Spark module (this may add its Python to PATH)
echo "Loading Spark module..."
module load Spark/3.5.4-foss-2022b-Scala-2.13



###########
# STEP 1: Cohort Extraction ##
###########
echo "=================================="
echo "STEP 1: Data Preparation"
echo "=================================="
python p1_data_preparation.py \
    --drug_mapping_file "${DRUG_MAPPING_FILE}" \
    --marketscan_split_dir "${MARKETSCAN_SPLIT_DIR}" \
    --version ${VERSION} \
    --test_split ${TEST_SPLIT}

# ##########
# STEP 2: Cohort Extraction ##
# ##########
echo "=================================="
echo "STEP 2: Cohort Extraction"
echo "=================================="
python p2_cohort_extract.py \
    --version ${VERSION}

##########
# STEP 3: Construct Pre Entry Sequence ##
##########
echo "=================================="
echo "STEP 3: Construct Pre Entry Sequence"
echo "=================================="
python p3_construct_prevseq.py \
    --marketscan_split_dir "${MARKETSCAN_SPLIT_DIR}" \
    --version ${VERSION} \
    --test_split ${TEST_SPLIT}

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "=================================="
echo "Embedding Evaluation Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
