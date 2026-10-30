# Reported roster

Built by `output/scripts/final_tables.py` from `output/scripts/roster.sh`. Every id is looked up exactly.

* Monte-Carlo anchor trajectories (FINAL_N_GENERATION): **64**
* LightGBM window shown in the figure (FINAL_LIGHTGBM_WINDOW): **1year**
* HNHC cutoff: **$30,000**

## Models

| role | size | model id | table label |
|---|---|---|---|
| anchor | — | `trainmean` | Train-mean anchor |
| anchor | — | `persistence` | Persistence (last-year cost) |
| lightgbm | — | `lightgbm-reg-1year` | LightGBM regressor, 1-yr window |
| lightgbm | — | `lightgbm-cls-1year` | LightGBM classifier, 1-yr window |
| lightgbm_ablation | — | `lightgbm-reg-allhist` | LightGBM regressor, all history |
| lightgbm_ablation | — | `lightgbm-cls-allhist` | LightGBM classifier, all history |
| pretrain | S | `reclaim-s-pretrain-64` | ReClaim-S, pre-training (MC, 64 trajectories) |
| pretrain | M | `reclaim-m-pretrain-64` | ReClaim-M, pre-training (MC, 64 trajectories) |
| pretrain | L | `reclaim-l-pretrain-64` | ReClaim-L, pre-training (MC, 64 trajectories) |
| posttrain | S | `reclaim-s-posttrain` | ReClaim-S, post-training |
| posttrain | M | `reclaim-m-posttrain` | ReClaim-M, post-training |
| posttrain | L | `reclaim-l-posttrain` | ReClaim-L, post-training |

## Files read

* `results/test/retrospective/regression/regression_metrics.csv`
* `results/test/retrospective/classification/30000/classification_metrics.csv`
* `results/test/retrospective/binary_auc/30000/auc_metrics.csv`
* `results/test/prospective/regression/regression_metrics.csv`
* `results/test/prospective/classification/30000/classification_metrics.csv`
* `results/test/prospective/binary_auc/30000/auc_metrics.csv`
* `results/test/inference_cost.csv`

## Missing cells

none: every roster cell is present
