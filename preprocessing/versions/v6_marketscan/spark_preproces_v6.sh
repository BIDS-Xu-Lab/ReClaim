#!/bin/bash
#SBATCH --job-name=reclaim_preprocess
#SBATCH --partition=day
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=400G                # 400G memory per node
#SBATCH --time=24:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/preprocess_v6_%j.out
#SBATCH --error=/dev/null
##SBATCH --error=slurm_job_logs/preprocess_v6_%j.err



echo "=================================="
echo "ReClaim Preprocess v6 Job Starting"
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

# Data directories
RAW_MARKETSCAN_DATA_ROOT="${RAW_MARKETSCAN_DATA_ROOT:-/path/to/raw_marketscan_data_root}"
PROCESSED_MARKETSCAN_DATA_ROOT="${PROCESSED_MARKETSCAN_DATA_ROOT:-/path/to/processed_marketscan_data_root}"
MARKETSCAN_DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR:-/path/to/marketscan_dictionary_dir}"

BASE_DIR="${RAW_MARKETSCAN_DATA_ROOT}"
INPUT_FOLDER_PATH="${PROCESSED_MARKETSCAN_DATA_ROOT}/v${VERSION}/marketscan_val"

VOCAB_MAPPING_DIR="../../vocab_mapping/mapping"
DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR}"
PAYER_LIST="CCAE MDCR MDCD"
TABLE_LIST="t i o d"

NUM_PARTITIONS=200

# =============================================================================
# End Configuration
# =============================================================================

echo "Configuration:"
echo "  Version: v$VERSION"
echo "  Raw MarketScan data root: $RAW_MARKETSCAN_DATA_ROOT"
echo "  Processed MarketScan data root: $PROCESSED_MARKETSCAN_DATA_ROOT"
echo "  MarketScan dictionary directory: $MARKETSCAN_DICTIONARY_DIR"
echo "  Base Directory: $BASE_DIR"
echo "  Input folder path: $INPUT_FOLDER_PATH"
echo "  Payers: $PAYER_LIST"
echo "  Table Types: $TABLE_LIST"
echo ""


# Load required modules
echo "Loading modules..."

module --force purge
module load miniconda

# Initialize and activate conda BEFORE loading Spark
echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh

CONDA_ENV_NAME="${CONDA_ENV_NAME:-reclaim}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
echo "Activating conda environment: ${CONDA_ENV_PATH:-$CONDA_ENV_NAME}"
if [ -n "$CONDA_ENV_PATH" ]; then
    conda activate "$CONDA_ENV_PATH"
else
    conda activate "$CONDA_ENV_NAME"
fi

# Get conda environment path - use CONDA_PREFIX which is set after activation
# or find it from conda env list
if [ -n "${CONDA_PREFIX:-}" ]; then
    CONDA_ENV_PATH="$CONDA_PREFIX"
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
    echo "Using CONDA_PREFIX: $CONDA_ENV_PATH"
else
    # Fallback: try to find from conda env list (search by name or by path)
    CONDA_ENV_PATH=$(conda env list | grep -E "^[[:space:]]*${CONDA_ENV_NAME}[[:space:]]" | awk '{print $NF}' | head -1)
    if [ -z "$CONDA_ENV_PATH" ]; then
        # Try to find by path ending in the environment name
        CONDA_ENV_PATH=$(conda env list | grep -E "/${CONDA_ENV_NAME}[[:space:]]*$" | awk '{print $NF}' | head -1)
    fi
    if [ -z "$CONDA_ENV_PATH" ]; then
        # Try a common user-local conda environment path
        CONDA_ENV_PATH="$HOME/.conda/envs/${CONDA_ENV_NAME}"
    fi
    CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
fi

# Verify conda environment exists
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment '$CONDA_ENV_NAME' not found at $CONDA_PYTHON"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Trying to find correct path..."
    # Try to get path from conda env list
    ENV_PATH=$(conda env list | grep -E "^[[:space:]]*${CONDA_ENV_NAME}[[:space:]]" | awk '{print $NF}' | head -1)
    if [ -n "$ENV_PATH" ] && [ -f "${ENV_PATH}/bin/python" ]; then
        CONDA_ENV_PATH="$ENV_PATH"
        CONDA_PYTHON="${CONDA_ENV_PATH}/bin/python"
        echo "Found environment at: $CONDA_ENV_PATH"
    else
        echo "Could not locate '$CONDA_ENV_NAME' environment"
        exit 1
    fi
fi

echo "Conda Python: $CONDA_PYTHON"

# Now load Spark module (this may add its Python to PATH)
echo "Loading Spark module..."
module load Spark/3.5.4-foss-2022b-Scala-2.13

# Force conda Python to be first in PATH to ensure it's used after loading Spark
export PATH="${CONDA_ENV_PATH}/bin:$PATH"

# Ensure conda site-packages takes priority
export PYTHONPATH="${CONDA_ENV_PATH}/lib/python3.11/site-packages:${SPARK_HOME}/python:${SPARK_HOME}/python/lib/pyspark.zip"

# Verify we're using the correct Python
CONDA_PYTHON=$(which python)
echo "Python path: $CONDA_PYTHON"

# Check if it's from conda environment
if [[ "$CONDA_PYTHON" == *".conda/envs/reclaim"* ]] || [[ "$CONDA_PYTHON" == *"miniconda"* ]]; then
    echo " Using conda environment Python"
else
    echo " WARNING: Not using conda environment Python!"
    echo "  Expected: .../.conda/envs/reclaim/bin/python"
    echo "  Got: $CONDA_PYTHON"
    # Force use conda Python
    export PATH="${CONDA_ENV_PATH}/bin:$PATH"
    CONDA_PYTHON=$(which python)
    echo "  Updated to: $CONDA_PYTHON"
fi

# Set PySpark to use this Python
export PYSPARK_PYTHON="$CONDA_PYTHON"
export PYSPARK_DRIVER_PYTHON="$CONDA_PYTHON"
echo "PYSPARK_PYTHON set to: $PYSPARK_PYTHON"

# Verify Python and dependencies
echo ""
echo "Checking dependencies..."
echo "Python version:"
"$CONDA_PYTHON" --version
echo "Python location: $CONDA_PYTHON"
echo ""



# Start the Spark instance
echo "Starting Spark cluster..."
if ! command -v spark-start &>/dev/null; then
    echo "ERROR: spark-start not found"
    echo "This script requires spark-start to be available."
    echo "If spark-start is not available, use spark_make_subset_misha_manual.sh instead."
    exit 1
fi

spark-start

# Source spark-env.sh to get useful env variables
# spark-start creates the spark environment in .spark-local
if [ -f "${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh" ]; then
    source ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh
else
    echo "WARNING: spark-env.sh not found at expected location"
    echo "  Expected: ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh"
    echo "  SPARK_MASTER_URL may not be set correctly"
fi



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
cluster=$(hostname -f | awk -F"." '{print $2}')
web_port=8080

# Print tunneling instructions for web UI access
echo "To access Spark Web UI, create SSH tunnel with:"
echo ""
echo "ssh -N -L ${web_port}:${node}:${web_port} ${user}@${cluster}.ycrc.yale.edu"
echo ""
echo "Then navigate to: http://localhost:${web_port}"
echo ""
echo "=================================="
echo ""

# Calculate executor resources
# With 12 nodes x 60 cores = 720 total cores and 900G x 12 = 10.8TB total memory
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

# Submit the Spark job for demo
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
    --payer_list ${PAYER_LIST} \
    --table_list ${TABLE_LIST} \
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
    --payer_list ${PAYER_LIST} \
    --table_list ${TABLE_LIST} \
    --mapping_dir ${VOCAB_MAPPING_DIR}


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
    --input_folder_path ${INPUT_FOLDER_PATH}



# ###########
# # STEP 3: Finalize vocabulary ##
# ###########

echo "=================================="
echo "STEP 3: Finalize vocabulary"
echo "=================================="

# Submit the Spark job for demo
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
    --dictionary_dir ${DICTIONARY_DIR}


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
  s5_sanity_check.py \
    --input_folder_path ${INPUT_FOLDER_PATH}



# Capture exit status
EXIT_STATUS=$?

echo ""
echo "=================================="
echo "Spark Job Completed"
echo "=================================="
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
