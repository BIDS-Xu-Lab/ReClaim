#!/bin/bash
#
# Draw the disease-embedding figures (UMAP and t-SNE) for Delphi and ReClaim,
# then score each model's embeddings against the ICD-10 chapters.
#
# Usage (from visualization/disease_embeddings/):
#   DELPHI_CKPT=/path/to/delphi RECLAIM_CKPT=/path/to/reclaim \
#       bash run_disease_embedding_figures.sh [OUT_DIR]
#
# OUT_DIR defaults to ../../figures/disease_embeddings. Set CORPUS_DIR to the
# tokenized corpus to size markers by prevalence; without it every marker is
# drawn at one size.
#

set -euo pipefail

OUT_DIR="${1:-../../figures/disease_embeddings}"
PYTHON="${PYTHON:-python}"

: "${DELPHI_CKPT:?set DELPHI_CKPT to the Delphi checkpoint directory}"
: "${RECLAIM_CKPT:?set RECLAIM_CKPT to the ReClaim model directory}"

mkdir -p "$OUT_DIR"

COUNT_ARGS=()
if [ -n "${CORPUS_DIR:-}" ]; then
    COUNT_ARGS=(--data_file "$CORPUS_DIR" --count_cache_dir "$OUT_DIR/.count_cache")
fi

MODELS=("Delphi|${DELPHI_CKPT}|delphi|delphi" "ReClaim|${RECLAIM_CKPT}|hf|reclaim")

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path type key <<< "$entry"
    "$PYTHON" plot_umap.py \
        --model_path "$path" --model_type "$type" \
        --out_dir "$OUT_DIR" --stem "umap_${key}" \
        --n_neighbors 30 --min_dist 0.05 --seed 1413 \
        "${COUNT_ARGS[@]}"
done

# n_jobs stays at 1: openTSNE's parallel neighbour search is not deterministic,
# and the run costs seconds at this size.
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path type key <<< "$entry"
    "$PYTHON" plot_tsne.py \
        --model_path "$path" --model_type "$type" \
        --out_dir "$OUT_DIR" --stem "tsne_${key}" \
        --perplexity 16 --exaggeration 2 --learning_rate 10 \
        --fit_scope disease --n_jobs 1 --seed 1413 \
        "${COUNT_ARGS[@]}"
done

# --align_tokenizer_path restricts Delphi to the disease tokens the ReClaim
# tokenizer also holds, so both models are scored over the same diseases.
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path type key <<< "$entry"
    "$PYTHON" cluster_metrics.py \
        --model_path "$path" --model_type "$type" --model_name "$name" \
        --out_file "$OUT_DIR/metrics_${key}.json" \
        --align_tokenizer_path "$RECLAIM_CKPT" \
        --n_seeds 10 --seed 1413
done

echo "Done. Figures and metrics in $OUT_DIR"
