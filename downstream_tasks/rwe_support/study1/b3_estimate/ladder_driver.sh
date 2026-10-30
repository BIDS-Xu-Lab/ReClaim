#!/bin/bash
# =============================================================================
# ladder_driver.sh -- driver for the estimation stage (invoked by b3_1_estimate.sh).
# Iterates cohorts x cells x specs and assembles the results tree:
#   results/mdcr/{cohort}/{cell}/{spec}/decile_adjusted/estimate.json
#
# Usage (b3_1_estimate.sh sets RESULTS_ROOT / INTER_ROOT):
#   bash b3_estimate/ladder_driver.sh                                        # all cohorts, full ladder
#   bash b3_estimate/ladder_driver.sh positive_control_celecoxib_gi MDCR     # one cohort/stratum
# Knobs: SPECS (explicit list, wins), BASELINE_SPECS, EMBED_SPECS, ENABLE_EMBEDDINGS, DRY=1.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."   # module root: all roots below are relative to it

RESULTS_ROOT=${RESULTS_ROOT:-results/mdcr}
INTER_ROOT=${INTER_ROOT:-intermediate/mdcr}
RSCRIPT=${RSCRIPT:-Rscript}

COHORTS=(${1:-${COHORTS:-positive_control_celecoxib_gi negative_control_flu_hipfx}})
# Strata: MDCR (elderly) for both controls, the closest match to the reference's elderly
# cohorts. The negative control is one cell per (stratum, season). Pass $2 to override the strata.
POSITIVE_STRATA=(${POSITIVE_STRATA:-MDCR})
NEGATIVE_STRATA=(${NEGATIVE_STRATA:-MDCR})
STRATA_OVERRIDE=()
[ -n "${2:-}" ] && STRATA_OVERRIDE=($2)
NEGATIVE_SEASONS=(2011)

# Spec ladder, in two modules so the embedding arm is a pluggable add-on:
#   BASELINE_SPECS -- need only cohort.csv + hdps_covars.csv (b0_2 / b1_1).
#   EMBED_SPECS    -- need cohort_with_<model>_embeddings.parquet from b2_1.
# crude = the reference's model 1 (unadjusted): zero covariates, constant PS.
# Embedding specs are SUBSTITUTION arms (age+sex + rep_*): each model replaces the hdPS
# empirical covariates, asking whether the embedding carries the confounding signal hdPS
# engineers by hand.
# `${VAR:-default}` treats an EMPTY value as unset -- pass BASELINE_SPECS=" " (a space)
# to deliberately skip the baselines on an incremental embedding run.
BASELINE_SPECS=(${BASELINE_SPECS:-crude demo hdps})
EMBED_SPECS=(${EMBED_SPECS:-delphi-only reclaim-s-only reclaim-m-only reclaim-l-only})
ENABLE_EMBEDDINGS=${ENABLE_EMBEDDINGS:-1}

# ---- RUN MANIFEST -----------------------------------------------------------
# Stamp the design into the results tree so a stale result cannot be mistaken for a current one
# (each estimate.json also carries ps_engine + cv_folds individually).
write_manifest() {
  local root="$1"; mkdir -p "$root"
  cat > "$root/RUN_MANIFEST.json" <<EOF
{
  "written_utc":       "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id":      "${SLURM_JOB_ID:-manual}",
  "enrollment_gate":   "${ENROLLMENT_GATE:-<config default: off>}",
  "baseline_specs":    "${BASELINE_SPECS[*]}",
  "embed_specs":       "${EMBED_SPECS[*]}",
  "enable_embeddings": "${ENABLE_EMBEDDINGS}",
  "ps_engine_rule":    "unpenalised logistic for reference specs; RIDGE for embedding specs",
  "om_engine_rule":    "fit_dr_models AIPW: ridge on embedding outcome models",
  "cv_folds":          5
}
EOF
  echo "wrote $root/RUN_MANIFEST.json"
}
# Final ladder: explicit SPECS override wins (e.g. SPECS="hdps" for a quick check).
if [ -n "${SPECS:-}" ]; then SPECS=($SPECS)
elif [ "$ENABLE_EMBEDDINGS" = "1" ]; then SPECS=("${BASELINE_SPECS[@]}" "${EMBED_SPECS[@]}")
else SPECS=("${BASELINE_SPECS[@]}"); fi
write_manifest "$RESULTS_ROOT"
# Estimator: estimate_ladder implements the reference's primary only (decile_adjusted); results land under
# .../{spec}/decile_adjusted/ so summarize_results/fit_dr_models keep their layout.

FAILED=()
run() { echo "+ $*"; [ "${DRY:-0}" = "1" ] || "$@" || FAILED+=("$*"); }

for cohort in "${COHORTS[@]}"; do
  case "$cohort" in negative_control_*) design=landmark_cox;; *) design=cohort_logistic;; esac
  if [ ${#STRATA_OVERRIDE[@]} -gt 0 ]; then strata=("${STRATA_OVERRIDE[@]}")
  elif [ "$cohort" = "negative_control_flu_hipfx" ]; then strata=("${NEGATIVE_STRATA[@]}")
  else strata=("${POSITIVE_STRATA[@]}"); fi
  for stratum in "${strata[@]}"; do
    # positive control: one cell (the stratum); negative control: one cell per season
    # ({stratum}_s{season}), matching how b0_2 wrote the per-season sub-cohorts.
    if [ "$cohort" = "negative_control_flu_hipfx" ]; then
      cells=(); for s in "${NEGATIVE_SEASONS[@]}"; do cells+=("${stratum}_s${s}"); done
    else
      cells=("$stratum")
    fi
    for cell in "${cells[@]}"; do
      cohort_csv="${INTER_ROOT}/cohorts/${cohort}/${cell}/cohort.csv"
      hdps_covars="${INTER_ROOT}/hdps_covars/${cohort}_${cell}/hdps_covars.csv"
      if [ ! -f "$cohort_csv" ]; then
        echo "ERROR: ${cohort}/${cell}: $cohort_csv missing (run b0_preprocessing/b0_2_build_cohorts.sh + b1_1_select_covariates first)"; exit 1
      fi
      for spec in "${SPECS[@]}"; do
        # Resolve the embedding parquet for embedding specs: the model id IS the spec minus its
        # '-only' suffix, so any model key b2_1 can produce works with no mapping table:
        #   reclaim-l-only           -> cohort_with_reclaim-l_embeddings.parquet
        #   reclaim-m-pca120-only    -> cohort_with_reclaim-m-pca120_embeddings.parquet
        embed_arg=(); embed_parquet=""
        case "$spec" in
          delphi*-only|reclaim-*-only)
            model="${spec%-only}"
            embed_parquet="${INTER_ROOT}/embeddings/${cohort}_${cell}/cohort_with_${model}_embeddings.parquet";;
        esac
        if [ -n "$embed_parquet" ]; then
          if [ ! -f "$embed_parquet" ]; then
            echo "ERROR: ${cohort}/${cell}/${spec}: embedding parquet missing ($embed_parquet) -- run b2_embeddings/b2_1_embeddings.sh first"; exit 1
          fi
          embed_arg=(--embed_parquet "$embed_parquet")
        fi
        hdps_arg=()
        if [[ "$spec" == hdps* ]]; then
          if [ ! -f "$hdps_covars" ]; then
            echo "ERROR: ${cohort}/${cell}/${spec}: hdps_covars missing ($hdps_covars) -- run b1_1_select_covariates first"; exit 1
          fi
          hdps_arg=(--hdps_covars "$hdps_covars")
        fi

        out_dir="${RESULTS_ROOT}/${cohort}/${cell}/${spec}/decile_adjusted"
        run "$RSCRIPT" b3_estimate/estimate_ladder.R \
          --cohort_csv "$cohort_csv" \
          "${hdps_arg[@]}" "${embed_arg[@]}" \
          --spec "$spec" --design "$design" \
          --out_dir "$out_dir"
      done
    done
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "ladder_driver.sh FINISHED WITH ${#FAILED[@]} FAILED cell(s):"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
echo "ladder_driver.sh done. Assemble figure/table with: Rscript b3_estimate/summarize_results.R"
