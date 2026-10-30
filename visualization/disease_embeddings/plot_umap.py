#!/usr/bin/env python3
"""UMAP projection of disease token embeddings, coloured by ICD-10 chapter.

Paper panels: Fig. A6a (Delphi) and Fig. A6b (ReClaim).

Position comes from UMAP of the model's embedding row for each disease token,
colour from the disease's ICD-10 chapter, and marker size from how often the
disease occurs in the corpus. The model is never shown the chapter, so any
chapter structure in the layout was learned.

Outputs in --out_dir, each prefixed with --stem:
  coords.npy                          the 2-D coordinates
  labels.csv                          one row per plotted disease, with coordinates
  disease_embeddings.{png,pdf}        the figure
  disease_embeddings_legend_size.*    the marker-size legend strip

Everything except the projection itself lives in common.py, which plot_tsne.py
also imports, so the UMAP and t-SNE figures stay directly comparable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import umap

from common import (
    DEFAULT_EXPORT_DPI,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_LABELS_FILE,
    NON_DISEASE_CHAPTERS,
    count_tokens_in_data,
    extract_code_from_label,
    extract_disease_mapping,
    load_model_embeddings,
    plot_disease_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="auto",
                        choices=["auto", "delphi", "hf"])
    parser.add_argument("--labels_file", default=str(DEFAULT_LABELS_FILE),
                        help="Disease label table; defaults to the copy in "
                             "downstream_tasks/disease_prediction/delphi_baseline/")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"),
                        help="Figures and coordinates land here")
    parser.add_argument("--stem", default="",
                        help="Prefix on every output name, so two settings can "
                             "share one directory without overwriting")
    parser.add_argument("--data_file",
                        help="Parquet file or directory the token counts that "
                             "size the markers are read from. Omit to draw "
                             "every marker at one size.")
    parser.add_argument("--count_cache_dir",
                        help="Where to cache token counts. Defaults to a "
                             "subdirectory of --out_dir, which recounts "
                             "per run; point it at a shared path to reuse.")
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.05)
    parser.add_argument("--circle_size_scale", type=float, default=1.0)
    parser.add_argument("--figure_width", type=float,
                        default=DEFAULT_FIGURE_WIDTH)
    parser.add_argument("--figure_height", type=float,
                        default=DEFAULT_FIGURE_HEIGHT)
    parser.add_argument("--export_dpi", type=int, default=DEFAULT_EXPORT_DPI)
    parser.add_argument("--center_embeddings", action="store_true",
                        help="Subtract the mean disease-token vector before "
                             "reducing. Corrects embedding anisotropy, which "
                             "otherwise makes a model whose vectors share a "
                             "dominant direction look artificially well "
                             "clustered. Apply to every model in a comparison "
                             "or to none of them.")
    parser.add_argument("--fit_scope", default="model_default",
                        choices=["model_default", "disease_only"],
                        help="model_default reproduces the historical "
                             "behaviour, where Delphi fits the full vocabulary "
                             "and HF models fit disease tokens only. "
                             "disease_only fits disease tokens for every "
                             "model, the like-for-like setting for a "
                             "head-to-head comparison.")
    parser.add_argument("--seed", type=int, default=1413)
    args = parser.parse_args()

    # Imported here so --help does not pay for loading CUDA.
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.count_cache_dir:
        args.count_cache_dir = os.path.join(args.out_dir, "count_cache")

    def with_tag(name: str) -> str:
        return f"{args.stem}_{name}" if args.stem else name

    # Printed so a figure can be traced back to the run that made it.
    print("Settings: %s" % (vars(args),))

    print("Loading model from %s" % (args.model_path,))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type, tokenizer, wte = load_model_embeddings(
        args.model_path, args.model_type, device)
    vocab_size = wte.shape[0]
    print("  Embedding shape: %s" % (wte.shape,))

    print("Loading labels from %s" % (args.labels_file,))
    labels = pd.read_csv(args.labels_file)
    labels["index"] = labels["index"].astype(int)
    labels["code"] = labels["name"].map(extract_code_from_label)
    disease_labels = labels[
        ~labels["ICD-10 Chapter"].isin(NON_DISEASE_CHAPTERS)].copy()

    # Delphi's row index is the label index; an HF model needs its vocabulary
    # searched for the matching <DX-MAJOR_*> token.
    if model_type == "delphi":
        delphi_to_token = {int(i): int(i)
                           for i in disease_labels["index"] if i < vocab_size}
    elif tokenizer:
        delphi_to_token = extract_disease_mapping(tokenizer, labels)
    else:
        raise ValueError("No tokenizer available for HF model")

    delphi_ids = [d for d in sorted(delphi_to_token.keys())
                  if delphi_to_token[d] < vocab_size]
    token_ids = [delphi_to_token[d] for d in delphi_ids]
    mapped_wte = wte[token_ids]
    print("  Mapped %d disease tokens" % (len(delphi_ids),))

    # The disease-token mean, not the whole-vocabulary mean, so the correction
    # is defined on the same set of points in every model.
    if args.center_embeddings:
        mean_vec = mapped_wte.mean(axis=0, keepdims=True)
        wte = wte - mean_vec
        mapped_wte = mapped_wte - mean_vec
        print("  Centered on the disease-token mean (norm %.4f)" % (float(np.linalg.norm(mean_vec)),))

    token_counts = {}
    if args.data_file:
        counts = count_tokens_in_data(
            args.data_file, labels, args.count_cache_dir)
        code_lookup = labels.set_index("index")["code"].to_dict()
        token_counts = {d: counts.get(code_lookup.get(d), 0)
                        for d in delphi_ids}

    # UMAP requires n_neighbors below the point count.
    effective_nn = min(args.n_neighbors, len(delphi_ids) - 1)
    fit_full_vocab = (model_type == "delphi"
                      and args.fit_scope == "model_default")
    print("Running UMAP (n_neighbors=%d, min_dist=%s, fit_scope=%s, "
             "centered=%s)..." % (effective_nn, args.min_dist, "full_vocab" if fit_full_vocab else "disease_only", args.center_embeddings,))

    reducer = umap.UMAP(random_state=args.seed, n_neighbors=effective_nn,
                        min_dist=args.min_dist, metric="cosine")

    if fit_full_vocab:
        # The whole vocabulary shapes the layout, then the disease rows are
        # sliced out, so they keep their position relative to the full map.
        reducer.fit(wte)
        full_coords = reducer.transform(wte)
        full_coords = -(full_coords - np.median(full_coords, axis=0))
        umap_coords = full_coords[token_ids]
    else:
        umap_coords = reducer.fit_transform(mapped_wte)
        umap_coords = -(umap_coords - np.median(umap_coords, axis=0))

    np.save(os.path.join(args.out_dir, with_tag("coords.npy")),
            umap_coords)

    delphi_to_umap = {d: (umap_coords[i, 0], umap_coords[i, 1])
                      for i, d in enumerate(delphi_ids)}
    labels_plot = disease_labels[
        disease_labels["index"].isin(delphi_ids)].copy()
    labels_plot["UMAP1"] = labels_plot["index"].map(
        lambda x: delphi_to_umap[x][0])
    labels_plot["UMAP2"] = labels_plot["index"].map(
        lambda x: delphi_to_umap[x][1])

    if token_counts:
        labels_plot["count"] = labels_plot["index"].map(
            lambda x: token_counts.get(x, 0))
        # A disease absent from the corpus has no size; NaN draws the default
        # rather than the smallest marker, which would imply "seen once".
        labels_plot.loc[labels_plot["count"] == 0, "count"] = np.nan

    # A stray point far outside the cloud would stretch the axes and flatten
    # everything else.
    labels_plot = labels_plot[(labels_plot["UMAP1"].abs() < 20)
                              & (labels_plot["UMAP2"].abs() < 20)]
    labels_plot.to_csv(
        os.path.join(args.out_dir, with_tag("labels.csv")), index=False)
    print("  %d tokens for visualization" % (len(labels_plot),))

    plot_disease_embedding(
        labels_plot,
        os.path.join(args.out_dir,
                     with_tag("disease_embeddings.png")),
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        export_dpi=args.export_dpi,
        size_scale=args.circle_size_scale,
    )

    print("=" * 60)
    print("Output: %s" % (args.out_dir,))
    print("=" * 60)


if __name__ == "__main__":
    main()
