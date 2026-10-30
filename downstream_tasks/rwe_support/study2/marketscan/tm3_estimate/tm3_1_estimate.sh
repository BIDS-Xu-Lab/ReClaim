#!/bin/bash
#SBATCH --job-name=tm3_1_estimate
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out
# =============================================================================
# tm3_1_estimate: the reference's estimator panel over the PS-spec ladder, one run per
# emulation cohort x spec.
#   sIPTW weighted Cox (primary) + SMD balance + Schoenfeld + NCO calibration (EASE)
#   + 1:1 PSM Cox (with caliper sweep) + PS-decile-stratified Cox + obesity subgroup.
# Inputs : intermediate/cohorts/<emulation>/cohort.csv, intermediate/hdps_covars/<comparison>/,
#          intermediate/embeddings/cohort_with_<key>_embeddings.parquet
# Output : results/<comparison>/<spec>/<estimator>/estimate.json (+ CSVs), results/RUN_MANIFEST.json
# Usage  : sbatch tm3_estimate/tm3_1_estimate.sh   (from the module root)
# =============================================================================
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cannot cd to module root: $MODULE_ROOT"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (codesets_tte, marketscan_io, utils)
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/tm3_1_estimate_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u
command -v Rscript >/dev/null || { echo "ERROR: Rscript not on PATH"; exit 1; }

SPECS=${SPECS:-crude demo paper hdps}
# ENABLE_EMBEDDINGS=1 (default) adds the <key>-only specs (age+sex+rep_*), one per ReClaim size
if [ "${ENABLE_EMBEDDINGS:-1}" = "1" ]; then
  EMB_MODELS=${EMB_MODELS:-reclaim-s reclaim-m reclaim-l}
  for M in $EMB_MODELS; do SPECS="$SPECS ${M}-only"; done
fi
# CALIPER_SWEEP: calipers re-matched on the already-fitted PS ("" = off). Pass as an env
# prefix, not via `sbatch --export=...`, which splits on commas.
CALIPER_SWEEP=${CALIPER_SWEEP-0.05,0.1,0.2,0.5,1}
# ---- RUN MANIFEST: stamp the design into the results tree ---------------------
write_manifest() {
  local root="$1"; mkdir -p "$root"
  cat > "$root/RUN_MANIFEST.json" <<EOF
{
  "written_utc":     "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id":    "${SLURM_JOB_ID:-manual}",
  "design":          "reference construction (Tang 2026, Supplement Fig. S1): two separate emulations, each with its own GLP-1RA arm; per-emulation 12-month washout over the two compared classes only; entry = first in-window study-class dispensing; ever-lookback eligibility (T2D required; T1D/ESRD/prior psychiatric outcome excluded); prior antidepressant/antipsychotic use excluded; >=1 medical claim in the prior 365d; age >= 18",
  "specs":           "${SPECS}",
  "emulations":      "${EMULATIONS}",
  "estimators":      "siptw_cox,psm_cox,strat_cox",
  "ps_engine":       "ridge for embedding specs; plain logistic for reference specs",
  "cv_folds":        5,
  "caliper_sweep":   "${CALIPER_SWEEP:-off}",
  "enrollment_gate": "none (follow-up censored at disenrollment)",
  "covariate_spec":  "paper = strict Table S5 subset",
  "nco_panel":       "reference-24 (Supplement Table S4) with the >=0.1% risk rule applied on top",
  "population":      "MarketScan held-out val+test, POOL = CCAE + MDCR + MDCD"
}
EOF
  echo "wrote $root/RUN_MANIFEST.json"
}
FAILED=()

EMULATIONS=${EMULATIONS:-vs_sglt2:glp1_vs_sglt2:glp1ra:sglt2i vs_dpp4:glp1_vs_dpp4:glp1ra:dpp4i}   # emulation_dir:comparison:treat:comp
# WRITE_MANIFEST=0 for a partial spec rerun (the manifest must not record a partial SPECS list)
[ "${WRITE_MANIFEST:-1}" = "1" ] && write_manifest "results" || echo "(manifest left untouched: WRITE_MANIFEST=0)"
for EM in $EMULATIONS; do
  IFS=: read -r EDIR CID TCLS CCLS <<<"$EM"
  COHORT="intermediate/cohorts/$EDIR/cohort.csv"
  [ -f "$COHORT" ] || { echo "ERROR: $COHORT missing (run tm0_1_preprocessing)"; exit 1; }
  for SPEC in $SPECS; do
    echo "==== estimate_ladder: $CID / $SPEC (emulation $EDIR) ===="
    EXTRA=()
    [ -n "$CALIPER_SWEEP" ] && EXTRA+=(--caliper_sweep "$CALIPER_SWEEP")
    if [ "$SPEC" = "hdps" ]; then
      HD="intermediate/hdps_covars/$CID/hdps_covars.csv"
      [ -f "$HD" ] || { echo "!!!! $CID/hdps skipped: $HD missing (run tm1_1_select_covariates) !!!!"; FAILED+=("$CID/hdps"); continue; }
      EXTRA+=(--hdps_covars "$HD")
    fi
    case "$SPEC" in
      *-only)
        M=${SPEC%-only}
        EP="intermediate/embeddings/cohort_with_${M}_embeddings.parquet"
        [ -f "$EP" ] || { echo "!!!! $CID/$SPEC skipped: $EP missing (run tm2_1_embeddings) !!!!"; FAILED+=("$CID/$SPEC"); continue; }
        EXTRA+=(--embed_parquet "$EP")
        ;;
    esac
    Rscript tm3_estimate/estimate_ladder.R --cohort_csv "$COHORT" --comparison "$CID" \
      --treat_class "$TCLS" --comp_class "$CCLS" --spec "$SPEC" \
      "${EXTRA[@]}" \
      --out_dir "results/$CID/$SPEC" \
      || { echo "!!!! estimate_ladder $CID/$SPEC FAILED !!!!"; FAILED+=("$CID/$SPEC"); continue; }
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "==== tm3_1_estimate FINISHED WITH FAILURES: ${FAILED[*]} ===="; exit 1; fi
echo "==== tm3_1_estimate completed OK ===="
