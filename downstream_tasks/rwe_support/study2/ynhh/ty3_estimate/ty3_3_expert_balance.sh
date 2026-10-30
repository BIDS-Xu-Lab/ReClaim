#!/bin/bash
#SBATCH --job-name=ty3_3_expert_balance
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Relative output path: submit from the module root.
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# Stage ty3_3 (expert_balance): balance of every PS spec on the reference's covariate set
# (Table S5), before adjustment, after sIPTW and after 1:1 PSM, one run per emulation cohort x spec.
# Reuses ty3_1's PS, weights and matching exactly (expert_balance.R); no outcome model is fitted.
# Inputs : as ty3_1_estimate (same ROUTE, PS_ENGINE)
# Output : results/<cell>/<comparison>/<spec>/{siptw_cox,psm_cox}/balance_expert_smd.csv,
#          results/_summary_<cell>/summary_expert_balance.csv
# Usage  : sbatch ty3_estimate/ty3_3_expert_balance.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root"; exit 1; }
[ -d config ] || { echo "ERROR: not the study2/ynhh root (no config/)"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: cannot source $MODULE_ROOT/env.sh"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

# Runtime environment: miniconda env by name (RECLAIM_ENV, default hdps); R only, no JVM.
module purge
module load miniconda
RECLAIM_ENV=${RECLAIM_ENV:-hdps}
set +u
CONDA_BASE="${EBROOTMINICONDA:-}"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
[ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "ERROR: cannot locate conda.sh"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate "$RECLAIM_ENV"
set -u
RSCRIPT="${CONDA_PREFIX:?conda activate failed}/bin/Rscript"

SOURCE=${SOURCE:-ynhh}
ROUTE=${ROUTE:-source}
case "$ROUTE" in standard) CELL=$SOURCE;; source) CELL=${SOURCE}_src;;
  *) echo "ERROR: unknown ROUTE '$ROUTE' (standard|source)"; exit 1;; esac
EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}
SPECS=${SPECS:-crude demo paper hdps}
if [ "${ENABLE_EMBEDDINGS:-1}" = "1" ]; then
  EMB_MODELS=${EMB_MODELS:-reclaim-s reclaim-m reclaim-l}
  for M in $EMB_MODELS; do SPECS="$SPECS ${M}-only"; done
fi
PS_ENGINE=${PS_ENGINE:-ridge}
FAILED=()
for EM in $EMULATIONS; do
  IFS=: read -r ESUF CID TCLS CCLS <<<"$EM"
  ECOH="intermediate/cohorts/${CELL}_${ESUF}/cohort.csv"
  [ -f "$ECOH" ] || { echo "ERROR: $ECOH missing (run ty0_2_build_cohorts)"; exit 1; }
  for SPEC in $SPECS; do
    echo "==== expert_balance: $CID / $SPEC (emulation $ESUF) ===="
    EXTRA=()
    [ "$SPEC" = "hdps" ] && EXTRA+=(--hdps_covars "intermediate/hdps_covars/$CELL/$CID/hdps_covars.csv")
    case "$SPEC" in *-only) EXTRA+=(--embed_parquet "intermediate/embeddings/${CELL}/cohort_with_${SPEC%-only}_embeddings.parquet");; esac
    "$RSCRIPT" ty3_estimate/expert_balance.R --cohort_csv "$ECOH" --comparison "$CID" --ps_engine "$PS_ENGINE" \
      --treat_class "$TCLS" --comp_class "$CCLS" --spec "$SPEC" "${EXTRA[@]}" \
      --out_dir "results/$CELL/$CID/$SPEC" \
      || { echo "!!!! expert_balance $CID/$SPEC FAILED !!!!"; FAILED+=("$CID/$SPEC"); continue; }
  done
done
"$RSCRIPT" ty3_estimate/summarize_expert_balance.R "results/$CELL" "results/_summary_$CELL/summary_expert_balance.csv" \
  || FAILED+=("summary")
if [ ${#FAILED[@]} -gt 0 ]; then echo "==== ty3_3_expert_balance FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; fi
echo "==== ty3_3_expert_balance completed OK ===="
