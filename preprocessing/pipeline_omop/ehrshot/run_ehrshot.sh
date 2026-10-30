#!/bin/bash
#SBATCH --job-name=reclaim_preprocess_ehrshot
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dev/null              # redirected below into slurm_job_logs/ (created at run time)
#SBATCH --error=/dev/null
# =============================================================================
# EHRSHOT (OMOP CDM) -> ReClaim token sequences (s0, s1, s2, s3, s5), local-mode Spark on one node.
# There is no s4: an OMOP source is an external evaluation set and is not split.
# Paths come from ../../env.sh (copy ../../env.sh.example). Output: $MS_PROCESSED_ROOT/$DATA_VERSION/$EHRSHOT_RUN_NAME
# Submit from this folder:  cd pipeline_omop/ehrshot && sbatch run_ehrshot.sh
# =============================================================================

PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Slurm opens --output before the script starts and does not create folders, so the job log is
# redirected here instead: slurm_job_logs/<job name>_<job id>.out/.err
mkdir -p "$PIPELINE_DIR/slurm_job_logs"
_LOG="$PIPELINE_DIR/slurm_job_logs/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
exec >"$_LOG.out" 2>"$_LOG.err"

echo "=================================="
echo "ReClaim Preprocess EHRSHOT Job Starting"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"
echo "=================================="
echo ""

cd "$PIPELINE_DIR" || { echo "ERROR: cannot cd to $PIPELINE_DIR"; exit 1; }
[ -f s1_construct_tokens.py ] || { echo "ERROR: submit from pipeline_omop/ehrshot/ (cd pipeline_omop/ehrshot && sbatch run_ehrshot.sh)"; exit 1; }
. ../../env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }

# =============================================================================
# Configuration
# =============================================================================

EHRSHOT_OMOP_DIR="${EHRSHOT_OMOP_DIR:?export EHRSHOT_OMOP_DIR in env.sh}"
MAPPING_DIR="${VOCAB_MAPPING_DIR:?export VOCAB_MAPPING_DIR in env.sh}"
DICTIONARY_DIR="${MS_DICTIONARY_DIR:?export MS_DICTIONARY_DIR in env.sh}"
EXTERNAL_VAL_DATA_DIR="${MS_PROCESSED_ROOT:?export MS_PROCESSED_ROOT in env.sh}/${DATA_VERSION:?export DATA_VERSION in env.sh}/${EHRSHOT_RUN_NAME:?export EHRSHOT_RUN_NAME in env.sh}"

# Split point used by s2/s5; must match the MarketScan pipeline (pipeline_marketscan/run_marketscan.sh)
TRAINING_CUTOFF_YEAR=2023

echo "Configuration:"
echo "  EHRSHOT_OMOP_DIR: $EHRSHOT_OMOP_DIR"
echo "  MAPPING_DIR: $MAPPING_DIR"
echo "  DICTIONARY_DIR: $DICTIONARY_DIR"
echo "  EXTERNAL_VAL_DATA_DIR: $EXTERNAL_VAL_DATA_DIR"
echo "  TRAINING_CUTOFF_YEAR: $TRAINING_CUTOFF_YEAR"
echo ""

echo "Loading modules..."
module --force purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Activating conda environment ${RECLAIM_ENV}..."
conda activate "${RECLAIM_ENV:?export RECLAIM_ENV in env.sh}" || { echo "ERROR: cannot activate $RECLAIM_ENV"; exit 1; }
echo "Conda Python: $(which python)"

echo "Loading Spark module..."
module load "$SPARK_MODULE"

# Any failed stage stops the job (later stages read the earlier stages' output)
set -e

############################
# Stage 0: Data Filtering
############################
echo "=============================================="
echo "Stage 0: Data Filtering"
echo "=============================================="
python s0_data_filtering.py \
    --omop_dir ${EHRSHOT_OMOP_DIR} \
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
    --omop_dir ${EHRSHOT_OMOP_DIR} \
    --mapping_dir ${MAPPING_DIR} \
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
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}
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
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}
echo "Stage 5 completed!"
echo ""

echo "=================================="
echo "Preprocess EHRSHOT Job Completed: all stages succeeded"
echo "End Time: $(date)"
echo "=================================="
