#!/bin/bash
# Submit stage e3. Training and readout are separate jobs.   DRY=1 prints the sbatch commands.
#   ARM=posttrain (default)  train reclaim-<size>-posttrain          (posttrain_instruct.sh)
#   ARM=readout              predictions from an existing checkpoint (readout_instruct_vllm.sh; ENGINE=hf -> readout_instruct_hf.sh)
#   MODEL_SIZE=s|m|l|all   [GPUS= WALL= PER_DEVICE_BATCH=]   readout also: [MODEL=<dir under models/> TAG=]
#   A recipe other than the reported one: LR= / EPOCHS= / GLOBAL_BATCH= / EXTRA_ARGS= together with NAME_SUFFIX=-<tag>.
set -euo pipefail; cd "$(dirname "$0")/.."; . ./env.sh
sb() { if [ "${DRY:-0}" = 1 ]; then echo "  sbatch $*" >&2; echo DRY; else sbatch --parsable "$@"; fi; }
ARM="${ARM:-posttrain}"; SIZES="${MODEL_SIZE:-s}"; SIZES="${SIZES,,}"; [ "${SIZES}" = all ] && SIZES="s m l"
case "${ARM}" in
  posttrain)
    for S in ${SIZES}; do
      case "${S}" in s) G0=1; W0=04:00:00;; m) G0=1; W0=08:00:00;; l) G0=2; W0=08:00:00;; *) echo "MODEL_SIZE must be s|m|l|all (got '${S}')"; exit 1;; esac
      NAME="$(posttrain_name "${S}")"
      J=$(sb --gres="$(gres "${GPUS:-$G0}")" --time="${WALL:-$W0}" --job-name="posttrain_${S}" \
             --export=ALL,MODEL_SIZE=${S}${LR:+,LR=${LR}}${EPOCHS:+,EPOCHS=${EPOCHS}}${GLOBAL_BATCH:+,GLOBAL_BATCH=${GLOBAL_BATCH}}${PER_DEVICE_BATCH:+,PER_DEVICE_BATCH=${PER_DEVICE_BATCH}}${NAME_SUFFIX:+,NAME_SUFFIX=${NAME_SUFFIX}} \
             e3_posttrain/posttrain_instruct.sh)
      echo "posttrain ${S}: ${J} -> models/${NAME}/"
    done ;;
  readout)
    ENGINE="${ENGINE:-vllm}"; case "${ENGINE}" in vllm|hf) ;; *) echo "ENGINE must be vllm|hf"; exit 1;; esac
    NAMES=(); if [ -n "${MODEL:-}" ]; then NAMES=("${MODEL}"); else for S in ${SIZES}; do NAMES+=("reclaim-${S}-posttrain${NAME_SUFFIX:-}"); done; fi
    for N in "${NAMES[@]}"; do
      [ -d "models/${N}/final" ] || [ "${DRY:-0}" = 1 ] || { echo "readout ${N}: skipped (no checkpoint at models/${N}/final)"; continue; }
      J=$(sb --gres="$(gres "${GPUS:-1}")" ${WALL:+--time=${WALL}} --job-name="readout_${ENGINE}_${N}" \
             --export=ALL,MODEL=${N}${TAG:+,TAG=${TAG}} e3_posttrain/readout_instruct_${ENGINE}.sh)
      echo "readout ${N} (${ENGINE}): ${J} -> model_gen/${N}${TAG:-$([ "${ENGINE}" = hf ] && echo -hf)}/"
    done ;;
  *) echo "ARM must be posttrain|readout"; exit 1 ;;
esac
