#!/bin/bash
# Submit stage e2.   TASK=lightgbm|lightgbm_train|lightgbm_predict|mc (default lightgbm)   DRY=1 prints.
#   lightgbm         : LightGBM train -> predictions (incl. the non-learned anchors), chained; CPU
#   lightgbm_train   : training only          lightgbm_predict : predictions only
#   mc               : Monte-Carlo anchor (GPU) for MODEL_SIZE=s|m|l|all at NUM_GENERATION trajectories,
#                      aggregation chained.  [GPUS= WALL= MC_MEM=]
set -euo pipefail; cd "$(dirname "$0")/.."; . ./env.sh
sb() { if [ "${DRY:-0}" = 1 ]; then echo "  sbatch $*" >&2; echo DRY; else sbatch --parsable "$@"; fi; }
SIZES="${MODEL_SIZE:-s}"; SIZES="${SIZES,,}"; [ "${SIZES}" = all ] && SIZES="s m l"
case "${TASK:-lightgbm}" in
  lightgbm_train)   echo "lightgbm_train=$(sb e2_baseline/train_lightgbm.sh)" ;;
  lightgbm_predict) echo "lightgbm_predict=$(sb e2_baseline/predict_baselines.sh)" ;;
  lightgbm) TR=$(sb e2_baseline/train_lightgbm.sh); P=$(sb --dependency=afterok:$TR e2_baseline/predict_baselines.sh)
            echo "lightgbm_train=$TR lightgbm_predict=$P" ;;
  mc) GEN=()
      for S in ${SIZES}; do
        case "${S}" in s) W0=08:00:00;; m) W0=12:00:00;; l) W0=24:00:00;; *) echo "MODEL_SIZE must be s|m|l|all (got '${S}')"; exit 1;; esac
        J=$(sb --gres="$(gres "${GPUS:-8}")" --time="${WALL:-$W0}" --mem="${MC_MEM:-1500G}" --job-name="mc_generate_${S}" \
               --export=ALL,MODEL_SIZE=${S} e2_baseline/mc_generate.sh)
        GEN+=("${J}"); echo "mc generate ${S} (N=${NUM_GENERATION}): ${J}"
      done
      DEP=$(IFS=:; echo "${GEN[*]}")
      echo "mc aggregate (chained): $(sb --dependency=afterok:${DEP} e2_baseline/mc_aggregate.sh)" ;;
  *) echo "TASK must be lightgbm|lightgbm_train|lightgbm_predict|mc"; exit 1 ;;
esac
