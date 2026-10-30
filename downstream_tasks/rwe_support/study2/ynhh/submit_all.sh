#!/bin/bash
# =============================================================================
# submit_all.sh -- end-to-end submitter for study2/ynhh. Not an sbatch script: run it on the
# login node from the module root.
#
#     bash submit_all.sh                  # ty0_1_resolve_codes -> ty0_2_build_cohorts ->
#                                         #   ty1_1_select_covariates (per emulation) | ty2_1_embeddings (GPU, union)
#                                         #   -> ty3_1_estimate (per emulation) -> ty3_2_summary, ty3_3_expert_balance
#     ROUTE=standard bash submit_all.sh   # standard-concept dx route (cell ynhh instead of ynhh_src)
#     EMBED=0 bash submit_all.sh          # CPU-only: skip ty2_1_embeddings; ty3_1_estimate runs crude/demo/paper/hdps
#     GPU_TYPE=b200 bash submit_all.sh    # typed GPU override
#
# Stages are chained with --dependency=afterok. A failed stage holds its dependents PENDING
# (DependencyNeverSatisfied): scancel, fix, resubmit; cached outputs (cohort, prevseq,
# per-model parquets) make reruns cheap. ROUTE is exported to every job so the chain stays
# on one cell tree.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
[ -f ty0_preprocessing/build_cohort.py ] || { echo "ERROR: run from the repo root"; exit 1; }
mkdir -p slurm_job_logs
# env.sh preflight: prints a WARN line per missing checkpoint path (ty2_1_embeddings soft-skips those models)
bash -c '. ./env.sh'

export ROUTE=${ROUTE:-source}
EMBED=${EMBED:-1}
GPU_TYPE=${GPU_TYPE:-h200}     # the gpu partition is heterogeneous: typed GPUs required
GPU_COUNT=${GPU_COUNT:-1}

S0=$(sbatch --parsable ty0_preprocessing/ty0_1_resolve_codes.sh)
S2=$(sbatch --parsable --dependency=afterok:$S0 ty0_preprocessing/ty0_2_build_cohorts.sh)
S3=$(sbatch --parsable --dependency=afterok:$S2 ty1_baseline/ty1_1_select_covariates.sh)
if [ "$EMBED" = "1" ]; then
  S4=$(sbatch --parsable --gpus=${GPU_TYPE}:${GPU_COUNT} --dependency=afterok:$S2 ty2_embeddings/ty2_1_embeddings.sh)
  S5=$(ENABLE_EMBEDDINGS=1 sbatch --parsable --dependency=afterok:$S3:$S4 ty3_estimate/ty3_1_estimate.sh)
  EX="ty2_1_embeddings(gpu ${GPU_TYPE}x${GPU_COUNT})=$S4 "
else
  S5=$(sbatch --parsable --dependency=afterok:$S3 ty3_estimate/ty3_1_estimate.sh)
  EX=""
fi
S6=$(sbatch --parsable --dependency=afterok:$S5 ty3_estimate/ty3_2_summary.sh)
S7=$(ENABLE_EMBEDDINGS=$EMBED sbatch --parsable --dependency=afterok:$S5 ty3_estimate/ty3_3_expert_balance.sh)
echo "submitted (ROUTE=$ROUTE): ty0_1_resolve_codes=$S0 ty0_2_build_cohorts=$S2 ty1_1_select_covariates=$S3 ${EX}ty3_1_estimate=$S5 ty3_2_summary=$S6 ty3_3_expert_balance=$S7"
echo "watch:     squeue -u \$USER --format='%.9i %.12j %.8T %.10M %R'"
echo "logs:      slurm_job_logs/"
