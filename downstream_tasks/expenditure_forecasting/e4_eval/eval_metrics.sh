#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_job_logs/%x_%j.out
#SBATCH --error=slurm_job_logs/%x_%j.err
# Stage e4: score every model with predictions on the evaluation cohort, both sections:
# regression + HNHC classification (eval_cost_main.py), ROC/PR-AUC (eval_binary_auc.py), summary and
# inference-cost tables; then the paper's tables and figures (output/scripts/make_all.sh).
# Scored: LightGBM (HISTORY_LENGTHS), ANCHOR_MODELS, RECLAIM_MODELS (every trajectory count N as
# its own id reclaim-<size>-pretrain-<N>), the roster's post-trained models, and EXTRA_MODELS
# ("name1 name2", or auto = every model_gen/ folder). Metric CSVs are merge-on-save keyed by
# (source, model, cost_type): a partial run updates only its models' rows.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
. ./env.sh || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
. output/scripts/roster.sh
load_modules
BASE=(); CLS=()
EM="${EXTRA_MODELS:-}"; XTRA=(${EM//,/ })
if [ "${EM}" = "auto" ]; then
    XTRA=($(for SEC in retrospective prospective; do ls "${DATA_DIR}/${SPLIT}/${SEC}/model_gen" 2>/dev/null; done | sort -u))
fi
for H in "${HISTORY_LENGTHS[@]}"; do W="$(hist_wtag "${H}")"
    BASE+=("lightgbm-reg-${W}" "lightgbm-cls-${W}"); CLS+=("lightgbm-cls-${W}")
done
for SECTION in retrospective prospective; do
    SD="${DATA_DIR}/${SPLIT}/${SECTION}"; RD="${RESULT_DIR}/${SECTION}"; GOLD="${SD}/preprocessed_input.parquet"
    ${CONDA_PYTHON} e4_eval/eval_cost_main.py \
        --model_list "${BASE[@]}" "${RECLAIM_MODELS[@]}" "${POSTTRAIN_MODELS[@]}" "${ANCHOR_MODELS[@]}" "${XTRA[@]}" \
        --gold_path "${GOLD}" --model_gen_dir "${SD}/model_gen" --result_dir "${RD}" \
        --cutoff_list "${CUTOFF_HNHC}"
    ${CONDA_PYTHON} e4_eval/eval_binary_auc.py \
        --model_list "${RECLAIM_MODELS[@]}" "${CLS[@]}" "${POSTTRAIN_MODELS[@]}" "${XTRA[@]}" \
        --gold_path "${GOLD}" --model_gen_dir "${SD}/model_gen" --result_dir "${RD}" \
        --cutoff "${CUTOFF_HNHC}" --gen_num all --num_proc "${SLURM_CPUS_PER_TASK:-8}"
done
${CONDA_PYTHON} e4_eval/summarize_results.py --result_root "${RESULT_DIR}"
${CONDA_PYTHON} e4_eval/summarize_inference_cost.py --data_root "${DATA_DIR}/${SPLIT}" --result_root "${RESULT_DIR}" || echo "WARN: inference-cost summary failed"
bash output/scripts/make_all.sh || echo "WARN: output/scripts/make_all.sh failed (metric tables are complete; rerun it by hand)"
echo "eval_metrics completed"
