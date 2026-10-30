#!/bin/bash
#SBATCH --job-name=reclaim_preprocess_ynhh_full
#SBATCH --partition=day
#SBATCH --nodes=4                 # multi-node standalone Spark cluster; scale --nodes with cohort size
#SBATCH --ntasks-per-node=1       # one task per node; Spark spreads executors over the cores
#SBATCH --cpus-per-task=60
#SBATCH --mem=850G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dev/null              # redirected below into slurm_job_logs/ (created at run time)
#SBATCH --error=/dev/null
# =============================================================================
# YNHH (OMOP CDM) -> ReClaim token sequences for the FULL eligible cohort (s0 sampling disabled),
# stages s0, s1, s2, s3, s5 submitted with spark-submit to a Spark standalone cluster started inside the
# Slurm allocation (spark-start from the cluster's Spark module). The stage scripts detect cluster mode
# from SPARK_MASTER_URL and take master/memory/partitions from the spark-submit flags.
# There is no s4: an OMOP source is an external evaluation set and is not split.
# Paths come from ../../env.sh (copy ../../env.sh.example). Output: $MS_PROCESSED_ROOT/$DATA_VERSION/$YNHH_FULL_RUN_NAME
# Submit from this folder:  cd pipeline_omop/ynhh && sbatch run_ynhh_full.sh
# =============================================================================

PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Slurm opens --output before the script starts and does not create folders, so the job log is
# redirected here instead: slurm_job_logs/<job name>_<job id>.out/.err
mkdir -p "$PIPELINE_DIR/slurm_job_logs"
_LOG="$PIPELINE_DIR/slurm_job_logs/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
exec >"$_LOG.out" 2>"$_LOG.err"

echo "=================================="
echo "ReClaim Preprocess YNHH (FULL) Job Starting"
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

cd "$PIPELINE_DIR" || { echo "ERROR: cannot cd to $PIPELINE_DIR"; exit 1; }
[ -f s1_construct_tokens.py ] || { echo "ERROR: submit from pipeline_omop/ynhh/ (cd pipeline_omop/ynhh && sbatch run_ynhh_full.sh)"; exit 1; }
. ../../env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }

# =============================================================================
# Configuration
# =============================================================================

YNHH_OMOP_DIR="${YNHH_OMOP_DIR:?export YNHH_OMOP_DIR in env.sh}"
MAPPING_DIR="${VOCAB_MAPPING_DIR:?export VOCAB_MAPPING_DIR in env.sh}"
DICTIONARY_DIR="${MS_DICTIONARY_DIR:?export MS_DICTIONARY_DIR in env.sh}"
EXTERNAL_VAL_DATA_DIR="${MS_PROCESSED_ROOT:?export MS_PROCESSED_ROOT in env.sh}/${DATA_VERSION:?export DATA_VERSION in env.sh}/${YNHH_FULL_RUN_NAME:?export YNHH_FULL_RUN_NAME in env.sh}"

# Split point used by s2/s5; must match the MarketScan pipeline (pipeline_marketscan/run_marketscan.sh)
TRAINING_CUTOFF_YEAR=2023

# Full cohort: sample_size <= 0 disables sampling in s0 (all eligible patients are kept)
SAMPLE_SIZE=0
SEED=66

# =============================================================================
# End Configuration
# =============================================================================

echo "Configuration:"
echo "  YNHH_OMOP_DIR: $YNHH_OMOP_DIR"
echo "  MAPPING_DIR: $MAPPING_DIR"
echo "  DICTIONARY_DIR: $DICTIONARY_DIR"
echo "  EXTERNAL_VAL_DATA_DIR: $EXTERNAL_VAL_DATA_DIR"
echo "  TRAINING_CUTOFF_YEAR: $TRAINING_CUTOFF_YEAR"
echo "  SAMPLE_SIZE: $SAMPLE_SIZE (seed $SEED; <= 0 means the full cohort)"
echo ""

echo "Loading modules..."
module --force purge
module load miniconda
module load "$SPARK_MODULE"

echo "Activating conda environment ${RECLAIM_ENV}..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RECLAIM_ENV:?export RECLAIM_ENV in env.sh}" || { echo "ERROR: cannot activate $RECLAIM_ENV"; exit 1; }

# export for both driver & executors
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

echo "Using python: $PYSPARK_PYTHON"
echo "Python version: $(python --version 2>&1)"
echo ""

# Start the Spark instance (multi-node standalone cluster inside the Slurm allocation)
echo "Starting Spark cluster..."
spark-start

# Source spark-env.sh to get useful env variables (SPARK_MASTER_URL etc.)
source ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh
# Export explicitly: the stage scripts detect cluster mode via SPARK_MASTER_URL and then
# skip their local[*] fallback session config (see create_spark_session in s*.py)
export SPARK_MASTER_URL

echo ""
echo "=================================="
echo "Spark Cluster Started"
echo "=================================="
echo "Spark Master URL: ${SPARK_MASTER_URL}"
echo "=================================="
echo ""

# Calculate executor resources
TOTAL_CORES=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))
EXECUTOR_CORES=8                    # 8 cores per executor
EXECUTOR_MEMORY="80G"               # 80G per executor
# Reserve about 10% of cores for driver and overhead
TOTAL_EXECUTOR_CORES=$((TOTAL_CORES * 9 / 10))

echo "Spark Submit Configuration:"
echo "  - Executor cores: ${EXECUTOR_CORES}"
echo "  - Executor memory: ${EXECUTOR_MEMORY}"
echo "  - Total executor cores: ${TOTAL_EXECUTOR_CORES}"
echo "  - Approximate executors: $((TOTAL_EXECUTOR_CORES / EXECUTOR_CORES))"
echo ""

set -e

###########
# STAGE 0: DATA FILTERING (full cohort - no sampling)
###########

echo "=================================="
echo "STAGE 0: DATA FILTERING"
echo "=================================="

spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --driver-memory 100G \
  --py-files utils.py \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.default.parallelism=${TOTAL_EXECUTOR_CORES} \
  --conf spark.sql.shuffle.partitions=$((2 * TOTAL_EXECUTOR_CORES )) \
  --conf spark.network.timeout=1200s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --conf spark.sql.autoBroadcastJoinThreshold=100m \
  s0_data_filtering.py \
    --omop_dir ${YNHH_OMOP_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --sample_size ${SAMPLE_SIZE} \
    --seed ${SEED}

echo "Stage 0 completed!"
echo ""

###########
# STAGE 1: CONSTRUCT TOKENS
###########

echo "=================================="
echo "STAGE 1: Construct tokens"
echo "=================================="

# Higher shuffle partitions for the large token dataset
spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --driver-memory 100G \
  --py-files utils.py \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.default.parallelism=${TOTAL_EXECUTOR_CORES} \
  --conf spark.sql.shuffle.partitions=$((4 * TOTAL_EXECUTOR_CORES )) \
  --conf spark.network.timeout=1200s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --conf spark.sql.autoBroadcastJoinThreshold=100m \
  --conf spark.shuffle.file.buffer=1m \
  --conf spark.reducer.maxSizeInFlight=96m \
  --conf spark.shuffle.io.retryWait=60s \
  --conf spark.shuffle.io.maxRetries=10 \
  s1_construct_tokens.py \
    --omop_dir ${YNHH_OMOP_DIR} \
    --mapping_dir ${MAPPING_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}

echo "Stage 1 completed!"
echo ""

###########
# STAGE 2: CONSTRUCT SEQUENCES
###########

echo "=================================="
echo "STAGE 2: Construct sequences"
echo "=================================="

# Reduced executor heap + explicit overhead to leave room for off-heap/shuffle memory
# during the heavy per-patient window shuffle (same scheme as the MarketScan pipeline)
spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory 60G \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --driver-memory 100G \
  --py-files utils.py \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.executor.memoryOverhead=15G \
  --conf spark.default.parallelism=${TOTAL_EXECUTOR_CORES} \
  --conf spark.sql.shuffle.partitions=$((4 * TOTAL_EXECUTOR_CORES )) \
  --conf spark.network.timeout=1200s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --conf spark.sql.autoBroadcastJoinThreshold=100m \
  --conf spark.shuffle.file.buffer=1m \
  --conf spark.reducer.maxSizeInFlight=96m \
  --conf spark.shuffle.io.retryWait=60s \
  --conf spark.shuffle.io.maxRetries=10 \
  s2_construct_seqences.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}

echo "Stage 2 completed!"
echo ""

###########
# STAGE 3: FINALIZE VOCABULARY
###########

echo "=================================="
echo "STAGE 3: Finalize vocabulary"
echo "=================================="

spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --driver-memory 100G \
  --py-files utils.py \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.default.parallelism=${TOTAL_EXECUTOR_CORES} \
  --conf spark.sql.shuffle.partitions=$((2 * TOTAL_EXECUTOR_CORES )) \
  --conf spark.network.timeout=1200s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --conf spark.sql.autoBroadcastJoinThreshold=100m \
  s3_finalize_vocab.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --dictionary_dir ${DICTIONARY_DIR}

echo "Stage 3 completed!"
echo ""

###########
# STAGE 5: SANITY CHECK
###########

echo "=================================="
echo "STAGE 5: Sanity Check"
echo "=================================="

spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --total-executor-cores ${TOTAL_EXECUTOR_CORES} \
  --driver-memory 100G \
  --py-files utils.py \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.default.parallelism=${TOTAL_EXECUTOR_CORES} \
  --conf spark.sql.shuffle.partitions=$((2 * TOTAL_EXECUTOR_CORES)) \
  --conf spark.network.timeout=1200s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.files.maxPartitionBytes=256m \
  s5_sanity_check.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}

echo "Stage 5 completed!"
echo ""

echo "=================================="
echo "Preprocess YNHH (FULL) Job Completed: all stages succeeded"
echo "End Time: $(date)"
echo "=================================="
