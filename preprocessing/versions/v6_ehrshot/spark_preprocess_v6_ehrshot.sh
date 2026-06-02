#!/bin/bash
#SBATCH --job-name=preprocess_v6_ehrshot
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=400G                # 400G memory per node
#SBATCH --time=1-00:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/preprocess_v6_ehrshot_%j.out
#SBATCH --error=slurm_job_logs/preprocess_v6_ehrshot_%j.err

echo "=================================="
echo "Preprocess v6 EHRShot Job Starting"
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

# Keep protected data roots local to the environment. Set these before running.
EHRSHOT_DATA_ROOT="${EHRSHOT_DATA_ROOT:-/path/to/ehrshot_data_root}"
PROCESSED_EHRSHOT_DATA_ROOT="${PROCESSED_EHRSHOT_DATA_ROOT:-/path/to/processed_ehrshot_data_root}"
MARKETSCAN_DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR:-/path/to/marketscan_dictionary_dir}"

OMOP_DIR="${EHRSHOT_DATA_ROOT}/ehrshot_omop"
VOCAB_MAPPING_DIR="../../vocab_mapping/mapping"
DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR}"
EXTERNAL_VAL_DATA_DIR="${PROCESSED_EHRSHOT_DATA_ROOT}/v${VERSION}/ehrshot"

echo "Configuration:"
echo "  OMOP_DIR: $OMOP_DIR"
echo "  VOCAB_MAPPING_DIR: $VOCAB_MAPPING_DIR"
echo "  MARKETSCAN_DICTIONARY_DIR: $MARKETSCAN_DICTIONARY_DIR"
echo "  DICTIONARY_DIR: $DICTIONARY_DIR"
echo "  EXTERNAL_VAL_DATA_DIR: $EXTERNAL_VAL_DATA_DIR"
echo ""

# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda

echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh

echo "Activating conda environment 'reclaim'..."
conda activate reclaim

if [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_ENV_PATH="$CONDA_PREFIX"
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
    echo "Using CONDA_PREFIX: $CONDA_ENV_PATH"
else
    CONDA_ENV_PATH=$(conda env list | grep -E "^\s*reclaim\s" | awk '{print $NF}' | head -1)
    if [ -z "$CONDA_ENV_PATH" ]; then
        CONDA_ENV_PATH="$HOME/.conda/envs/reclaim"
    fi
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
fi

if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment 'reclaim' not found at $CONDA_PYTHON"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Trying to find correct path..."
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

echo "Loading Spark module..."
module load Spark/3.5.4-foss-2022b-Scala-2.13

# Change to the directory the job was submitted from (not the SLURM spool dir)
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

set -e

###########################
# Stage 0: Data Filtering
############################
echo "=============================================="
echo "Stage 0: Data Filtering"
echo "=============================================="
python s0_data_filtering.py \
    --omop_dir ${OMOP_DIR} \
    --mapping_dir ${VOCAB_MAPPING_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 0 completed!"
echo ""

############################
# Stage 1: Construct Tokens
############################
echo "=============================================="
echo "Stage 1: Construct Tokens"
echo "=============================================="
python s1_construct_tokens.py \
    --omop_dir ${OMOP_DIR} \
    --mapping_dir ${VOCAB_MAPPING_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 1 completed!"
echo ""

############################
# Stage 2: Construct Sequences
############################
echo "=============================================="
echo "Stage 2: Construct Sequences"
echo "=============================================="
python s2_construct_seqences.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 2 completed!"
echo ""

############################
# Stage 3: Finalize Vocabulary
############################
echo "=============================================="
echo "Stage 3: Finalize Vocabulary"
echo "=============================================="
python s3_finalize_vocab.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --dictionary_dir ${DICTIONARY_DIR}
echo "Stage 3 completed!"
echo ""

############################
# Stage 5: Sanity Check
############################
echo "=============================================="
echo "Stage 5: Sanity Check"
echo "=============================================="
python s5_sanity_check.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 5 completed!"
echo ""

EXIT_STATUS=$?

echo ""
echo "=================================="
echo "Preprocess v6 EHRShot Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
