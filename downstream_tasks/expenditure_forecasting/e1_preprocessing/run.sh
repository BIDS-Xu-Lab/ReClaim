#!/bin/bash
# Submit stage e1.   TASK=test|train|pairs|all (default all)   DRY=1 prints the sbatch commands.
#   test  : held-out cohort prediction points + features   (build_test_cohort.sh)
#   train : supervision pool, split, features               (build_train_cohort.sh)
#   pairs : post-training pairs (needs train)               (build_posttrain_pairs.sh)
set -euo pipefail; cd "$(dirname "$0")/.."
sb() { if [ "${DRY:-0}" = 1 ]; then echo "  sbatch $*" >&2; echo DRY; else sbatch --parsable "$@"; fi; }
case "${TASK:-all}" in
  test)  echo "test=$(sb e1_preprocessing/build_test_cohort.sh)" ;;
  train) echo "train=$(sb e1_preprocessing/build_train_cohort.sh)" ;;
  pairs) echo "pairs=$(sb e1_preprocessing/build_posttrain_pairs.sh)" ;;
  all)   T=$(sb e1_preprocessing/build_test_cohort.sh); TR=$(sb e1_preprocessing/build_train_cohort.sh)
         P=$(sb --dependency=afterok:$TR e1_preprocessing/build_posttrain_pairs.sh)
         echo "test=$T train=$TR pairs=$P" ;;
  *) echo "TASK must be test|train|pairs|all"; exit 1 ;;
esac
