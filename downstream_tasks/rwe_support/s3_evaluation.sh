#!/bin/bash
#SBATCH --job-name=s3_ps_ease_eval
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=60        # 64 cpu-cores per task
#SBATCH --mem=900G                # 900G memory per node
#SBATCH --time=1-00:00:00         # 1 day time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_job_logs/s3_ps_ease_eval_%j.out
#SBATCH --error=slurm_job_logs/s3_ps_ease_eval_%j.err



echo "=================================="
echo "ReClaim PS/EASE Evaluation Job Starting"
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
# Load modules and activate conda environment
# =============================================================================

echo "Loading modules..."
module --force purge
module load miniconda

echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh

echo "Activating conda environment 'reclaim'..."
conda activate reclaim

echo "Using R from conda: $(which Rscript)"
echo "R version: $(Rscript --version 2>&1)"




###########
# Model Configurations: [model] = model_path
# Same configurations as s2_generate_embeddings.sh
###########

declare -a MODELS=(
    "delphi"
    "v6-no"
    "v6-s"
    "v6-m"
    "v6-l"
)

NUM_MODELS=${#MODELS[@]}
echo "Number of model configurations: $NUM_MODELS"

###########
# Treatment-Control Comparisons
###########

declare -a COMPARISONS=(
    "glp1:dpp4"    # treatment=glp1, control=dpp4
    "glp1:sglt2"   # treatment=glp1, control=sglt2
    "sglt2:dpp4"   # treatment=sglt2, control=dpp4
)

NUM_COMPARISONS=${#COMPARISONS[@]}
echo "Number of treatment-control comparisons: $NUM_COMPARISONS"

###########
# Low Prevalence Threshold
###########

LOW_PREVALENCE_THRESHOLD=0.005
echo "Low prevalence threshold: $LOW_PREVALENCE_THRESHOLD"
echo ""

###########
# PS/EASE Evaluation - Loop through all configurations
###########
EXIT_STATUS=0
TOTAL_RUNS=$((NUM_MODELS * NUM_COMPARISONS))
RUN_NUM=0

for MODEL in "${MODELS[@]}"; do
    for COMPARISON in "${COMPARISONS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        
        # Parse treatment and control from comparison string
        TREATMENT="${COMPARISON%%:*}"
        CONTROL="${COMPARISON##*:}"
        
        echo ""
        echo "=================================="
        echo " PS/EASE Evaluation - Run ${RUN_NUM}/${TOTAL_RUNS}"
        echo " Model: $MODEL"
        echo " Treatment: $TREATMENT"
        echo " Control: $CONTROL"
        echo "=================================="
        
        Rscript p5_ps_ease_eval.R \
            --model "$MODEL" \
            --treatment "$TREATMENT" \
            --control "$CONTROL" \
            --low_prevalence_threshold "$LOW_PREVALENCE_THRESHOLD"
        
        # Capture exit status for this run
        RUN_STATUS=$?
        if [ $RUN_STATUS -ne 0 ]; then
            echo "WARNING: Run ${RUN_NUM} (Model: $MODEL, Treatment: $TREATMENT, Control: $CONTROL) failed with exit status $RUN_STATUS"
            EXIT_STATUS=$RUN_STATUS
        else
            echo "Run ${RUN_NUM} completed successfully"
        fi
    done
done

echo ""
echo "=================================="
echo "PS/EASE Evaluation Job Completed"
echo "=================================="
echo "Total runs attempted: $RUN_NUM / $TOTAL_RUNS"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=================================="

exit $EXIT_STATUS
