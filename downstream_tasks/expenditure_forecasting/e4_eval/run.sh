#!/bin/bash
# Submit stage e4.   TASK=metrics|discretization|output|all (default metrics)   DRY=1 prints.
#   metrics        : score every model from predictions + gold, then the paper's tables and figures
#   discretization : raw-claims audit of the cost-token encoding (run once; needs MS_RAW_DIR)
#   output         : tables and figures only, from results/ (no predictions, gold or GPU)
#   all            : metrics + discretization
set -euo pipefail; cd "$(dirname "$0")/.."
sb() { if [ "${DRY:-0}" = 1 ]; then echo "  sbatch $*" >&2; echo DRY; else sbatch --parsable "$@"; fi; }
case "${TASK:-metrics}" in
  metrics)        echo "metrics=$(sb e4_eval/eval_metrics.sh)" ;;
  discretization) echo "discretization=$(sb e4_eval/eval_discretization.sh)" ;;
  output)         echo "output=$(sb output/scripts/make_all.sh)" ;;
  all)            echo "metrics=$(sb e4_eval/eval_metrics.sh) discretization=$(sb e4_eval/eval_discretization.sh)" ;;
  *) echo "TASK must be metrics|discretization|output|all"; exit 1 ;;
esac
