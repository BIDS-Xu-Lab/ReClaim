#!/bin/bash
#SBATCH --job-name=tm3_3_expert_balance
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm3_3_expert_balance: balance of every PS spec on the reference's covariate set (Table S5),
# before adjustment, after sIPTW and after 1:1 PSM, one run per emulation cohort x spec.
# Reuses tm3_1's PS, weights and matching exactly (expert_balance.R); no outcome model is fitted.
# Inputs : as tm3_1_estimate
# Output : results/<comparison>/<spec>/{siptw_cox,psm_cox}/balance_expert_smd.csv,
#          results/_summary/summary_expert_balance.csv
# Usage  : sbatch tm3_estimate/tm3_3_expert_balance.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm3_3_expert_balance_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
command -v Rscript >/dev/null || { echo "ERROR: Rscript not on PATH"; exit 1; }

SPECS=${SPECS:-crude demo paper hdps}
if [ "${ENABLE_EMBEDDINGS:-1}" = "1" ]; then
  EMB_MODELS=${EMB_MODELS:-reclaim-s reclaim-m reclaim-l}
  for M in $EMB_MODELS; do SPECS="$SPECS ${M}-only"; done
fi
EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}
FAILED=()
for EM in $EMULATIONS; do
  IFS=: read -r EDIR CID TCLS CCLS <<<"$EM"
  COHORT="intermediate/cohorts/$EDIR/cohort.csv"
  [ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run tm0_1_preprocessing)"; exit 1; }
  for SPEC in $SPECS; do
    echo "==== expert_balance: $CID / $SPEC ===="
    EXTRA=()
    [ "$SPEC" = "hdps" ] && EXTRA+=(--hdps_covars "intermediate/hdps_covars/$CID/hdps_covars.csv")
    case "$SPEC" in *-only) EXTRA+=(--embed_parquet "intermediate/embeddings/cohort_with_${SPEC%-only}_embeddings.parquet");; esac
    Rscript tm3_estimate/expert_balance.R --cohort_csv "$COHORT" --comparison "$CID" \
      --treat_class "$TCLS" --comp_class "$CCLS" --spec "$SPEC" "${EXTRA[@]}" \
      --out_dir "results/$CID/$SPEC" \
      || { echo "!!!! expert_balance $CID/$SPEC FAILED !!!!"; FAILED+=("$CID/$SPEC"); continue; }
  done
done
Rscript tm3_estimate/summarize_expert_balance.R results results/_summary/summary_expert_balance.csv \
  || FAILED+=("summary")
if [ ${#FAILED[@]} -gt 0 ]; then echo "==== tm3_3_expert_balance FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; fi
echo "==== tm3_3_expert_balance completed OK ===="
