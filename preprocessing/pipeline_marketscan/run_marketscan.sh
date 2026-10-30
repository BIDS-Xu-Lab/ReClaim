#!/bin/bash
#SBATCH --job-name=reclaim_preprocess
#SBATCH --partition=day
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       # one task per node; Spark spreads executors over the cores
#SBATCH --cpus-per-task=60
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null              # redirected below into slurm_job_logs/ (created at run time)
#SBATCH --error=/dev/null
# =============================================================================
# MarketScan claims -> ReClaim token sequences, stages s0-s5 on a Spark standalone cluster started
# inside the Slurm allocation (spark-start from the cluster's Spark module).
# Paths come from ../env.sh (copy ../env.sh.example). Output: $MS_PROCESSED_ROOT/$DATA_VERSION/$MS_RUN_NAME
# Submit from this folder:  cd pipeline_marketscan && sbatch run_marketscan.sh
# The full corpus was processed on 2 nodes x 60 cores x 450G; small subsets run on the same layout.
# =============================================================================

PIPELINE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Slurm opens --output before the script starts and does not create folders, so the job log is
# redirected here instead: slurm_job_logs/<job name>_<job id>.out
# (Spark stderr is very verbose and is discarded; stage logs go to logs/)
mkdir -p "$PIPELINE_DIR/slurm_job_logs"
_LOG="$PIPELINE_DIR/slurm_job_logs/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
exec >"$_LOG.out" 2>/dev/null

echo "=================================="
echo "ReClaim Preprocess Job Starting"
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
[ -f s1_construct_tokens.py ] || { echo "ERROR: submit from pipeline_marketscan/ (cd pipeline_marketscan && sbatch run_marketscan.sh)"; exit 1; }
. ../env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR="${MS_RAW_DIR:?export MS_RAW_DIR in env.sh}"
INPUT_FOLDER_PATH="${MS_PROCESSED_ROOT:?export MS_PROCESSED_ROOT in env.sh}/${DATA_VERSION:?export DATA_VERSION in env.sh}/${MS_RUN_NAME:?export MS_RUN_NAME in env.sh}"
VOCAB_MAPPING_FOLDER_PATH="${VOCAB_MAPPING_DIR:?export VOCAB_MAPPING_DIR in env.sh}"
MARKETSCAN_DICTIONARY_FOLDER_PATH="${MS_DICTIONARY_DIR:?export MS_DICTIONARY_DIR in env.sh}"

# Events with year >= TRAINING_CUTOFF_YEAR are held out from pretraining (s2/s5); the OMOP
# pipeline must use the same value.
TRAINING_CUTOFF_YEAR=2023

# Output partitions of each train/val/test split written by s4
NUM_PARTITIONS=200

# =============================================================================
# End Configuration
# =============================================================================

echo "Configuration:"
echo "  Base Directory: $BASE_DIR"
echo "  Input folder path: $INPUT_FOLDER_PATH"
echo "  Vocab mapping: $VOCAB_MAPPING_FOLDER_PATH"
echo "  Dictionary: $MARKETSCAN_DICTIONARY_FOLDER_PATH"
echo "  Training cutoff year: $TRAINING_CUTOFF_YEAR"
echo ""


# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda

# Initialize and activate conda BEFORE loading Spark
echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh

CONDA_ENV_PATH="${RECLAIM_ENV:?export RECLAIM_ENV in env.sh}"
echo "Activating conda environment ${CONDA_ENV_PATH}..."
conda activate "$CONDA_ENV_PATH" || { echo "ERROR: cannot activate $CONDA_ENV_PATH"; exit 1; }

CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
echo "Conda Python: $CONDA_PYTHON"

# Now load Spark module (this may prepend its own Python to PATH)
echo "Loading Spark module..."
module load "$SPARK_MODULE"

# Force conda Python ahead of Spark's Python, and expose pyspark from SPARK_HOME
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
export PYTHONPATH="${CONDA_ENV_PATH}/lib/python3.11/site-packages:${SPARK_HOME}/python:${SPARK_HOME}/python/lib/pyspark.zip"

# Point PySpark at the conda Python
export PYSPARK_PYTHON="$CONDA_PYTHON"
export PYSPARK_DRIVER_PYTHON="$CONDA_PYTHON"
echo "PYSPARK_PYTHON set to: $PYSPARK_PYTHON ($("$CONDA_PYTHON" --version 2>&1))"
echo ""



# Start the Spark instance (provided by the loaded Spark module)
echo "Starting Spark cluster..."
spark-start

# spark-start writes its environment (incl. SPARK_MASTER_URL) here; source it
source ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh

# From here on any failed stage stops the job (later stages read the earlier stages' output)
set -e



echo ""
echo "=================================="
echo "Spark Cluster Started"
echo "=================================="
echo "Spark Master URL: ${SPARK_MASTER_URL}"
echo "=================================="
echo ""

# Set up variables for SSH tunnel (optional, for web UI access)
node=$(hostname -s)
user=$(whoami)
web_port=8080

# Print tunneling instructions for web UI access
echo "To access Spark Web UI, create SSH tunnel with:"
echo ""
echo "ssh -N -L ${web_port}:${node}:${web_port} ${user}@<cluster-login-host>"
echo ""
echo "Then navigate to: http://localhost:${web_port}"
echo ""
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




###########
# STEP 0: DATA FILTERING ##
###########

echo "=================================="
echo "STEP 0: DATA FILTERING"
echo "=================================="

MIN_MEM_DAYS=180
MIN_CLAIMS_DURATION_DAYS=30

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
    --base_dir ${BASE_DIR} \
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --min_mem_days ${MIN_MEM_DAYS} \
    --min_claims_duration_days ${MIN_CLAIMS_DURATION_DAYS}


    

# ###########
# # STEP 1: Construct tokens ##
# ###########

echo "=================================="
echo "STEP 1: Construct tokens"
echo "=================================="

# Submit the Spark job - use higher shuffle partitions for large token dataset
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
    --base_dir ${BASE_DIR} \
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --mapping_dir ${VOCAB_MAPPING_FOLDER_PATH}


# ###########
# # STEP 2: Construct sequences ##
# ###########

echo "=================================="
echo "STEP 2: Construct sequences"
echo "=================================="

# Submit the Spark job - reduced executor memory (60G heap + 15G overhead = 75G total)
# to leave room for off-heap/shuffle memory and avoid OOM-killed executors
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
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}



# ###########
# # STEP 3: Finalize vocabulary ##
# ###########

echo "=================================="
echo "STEP 3: Finalize vocabulary"
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
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --dictionary_dir ${MARKETSCAN_DICTIONARY_FOLDER_PATH}


# ###########
# # STEP 4: Train/Val/Test Split ##
# ###########

echo "=================================="
echo "STEP 4: Train/Val/Test Split"
echo "=================================="

# Hybrid approach: HuggingFace for splitting (deterministic), Spark for parallel writes
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
  s4_train_test_split.py \
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --num_proc ${SLURM_CPUS_PER_TASK} \
    --num_partitions ${NUM_PARTITIONS}

# ###########
# # STEP 5: Sanity Check ##
# ###########

echo "=================================="
echo "STEP 5: Sanity Check"
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
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --training_cutoff_year ${TRAINING_CUTOFF_YEAR}



echo ""
echo "=================================="
echo "Spark Job Completed: all stages succeeded"
echo "End Time: $(date)"
echo "=================================="
