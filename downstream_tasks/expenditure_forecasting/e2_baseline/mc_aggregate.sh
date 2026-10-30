#!/bin/bash
#SBATCH --job-name=mc_aggregate
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e2: aggregate the Monte-Carlo trajectories. For every generated_data_<N>.parquet of an
# anchor: parse each trajectory's first simulated year of cost tokens, cap at $1M per trajectory,
# and write cost_predictions_<N>.csv = mean dollars + prob_ge_<cutoff> (fraction of the N
# trajectories at or above CUTOFF_HNHC). A folder is entered when it holds the NUM_GENERATION file.
#   MC_MODELS="<anchor> [<anchor>]"   restrict to some anchors (one job per size in parallel);
#                                     unset = every anchor in RECLAIM_MODELS
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
load_modules
for SECTION in retrospective prospective; do for M in ${MC_MODELS:-"${RECLAIM_MODELS[@]}"}; do
    D="${DATA_DIR}/${SPLIT}/${SECTION}/model_gen/${M}"
    if [ -f "${D}/generated_data_${NUM_GENERATION}.parquet" ]; then
        ${CONDA_PYTHON} e2_baseline/aggregate_generations.py --model_gen_dir "${D}" --num_proc "${SLURM_CPUS_PER_TASK:-8}"
    else echo "no generations for ${M}/${SECTION} - skipping"; fi
done; done
echo "mc_aggregate completed"
