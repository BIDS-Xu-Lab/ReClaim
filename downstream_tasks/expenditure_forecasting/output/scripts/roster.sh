#!/bin/bash
# The reported roster: the one place that decides which model ids the paper's expenditure tables
# and figure are built from. Sourced after env.sh by eval_metrics.sh and make_all.sh. final_tables.py
# looks every id up exactly and echoes them into output/tables/ROSTER.md.
export FINAL_N_GENERATION=64             # trajectories of the reported Monte-Carlo anchor
export FINAL_LIGHTGBM_WINDOW="1year"     # feature window of the reported LightGBM pair
export FINAL_BACKBONES=("s" "m" "l")

# ---- derived (do not edit) ----
export FINAL_PRETRAIN_MODELS=(); export POSTTRAIN_MODELS=()
for _b in "${FINAL_BACKBONES[@]}"; do
    FINAL_PRETRAIN_MODELS+=("reclaim-${_b}-pretrain-${FINAL_N_GENERATION}")
    POSTTRAIN_MODELS+=("reclaim-${_b}-posttrain")
done; unset _b
# Tables: non-learned anchors, LightGBM in both feature windows, Monte-Carlo anchors, post-trained
# models. The figure shows the FINAL_LIGHTGBM_WINDOW pair and the ReClaim rows only.
export FINAL_TABLE_MODELS=("trainmean" "persistence"
                           "lightgbm-reg-1year" "lightgbm-cls-1year" "lightgbm-reg-allhist" "lightgbm-cls-allhist"
                           "${FINAL_PRETRAIN_MODELS[@]}" "${POSTTRAIN_MODELS[@]}")
