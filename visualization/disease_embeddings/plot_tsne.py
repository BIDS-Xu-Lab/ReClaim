#!/usr/bin/env python3
"""t-SNE projection of disease token embeddings, coloured by ICD-10 chapter.

Paper panels: Fig. A7a (Delphi) and Fig. A7b (ReClaim).

Position comes from a t-SNE of the model's embedding row for each disease
token, colour from the disease's ICD-10 chapter, and marker size from how often
the disease occurs in the corpus.

Everything except the projection lives in common.py, which plot_umap.py also
imports, so the two figures are directly comparable.

Outputs in --out_dir, each prefixed with --stem:
  coords.npy                          the 2-D coordinates
  labels.csv                          one row per plotted disease, with coordinates
  disease_embeddings.{png,pdf}        the figure
  disease_embeddings_legend_size.{png,pdf}  the marker-size legend strip

The optimiser is openTSNE rather than scikit-learn, because the published
settings use a main-phase ``exaggeration`` that scikit-learn does not expose,
separate from the early phase.

On the two fit modes. ``--fit_scope disease`` fits the disease tokens alone and
is the only like-for-like head-to-head: Delphi's vocabulary is about 99% disease
tokens, so fitting its full vocabulary is nearly the same computation, while for
ReClaim it is a 20862-token fit in which diseases are a 6% sample. Use
``--fit_scope vocab`` only for a statement about one model, never for a
comparison between two.

    python plot_tsne.py --model_path <checkpoint> --model_type hf \\
        --perplexity 16 --exaggeration 2 --learning_rate 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_tsne(perplexity, seed, metric, learning_rate, n_iter,
               early_exaggeration, early_exaggeration_iter, exaggeration,
               negative_gradient_method, n_jobs):
    """Construct an openTSNE estimator with every knob set explicitly."""
    import openTSNE

    print("  openTSNE(perplexity=%s, metric=%s, learning_rate=%s, "
             "n_iter=%d, early_exaggeration=%s, early_exaggeration_iter=%d, "
             "exaggeration=%s, negative_gradient_method=%s)" % (perplexity, metric, learning_rate, n_iter, early_exaggeration, early_exaggeration_iter, exaggeration, negative_gradient_method,))
    return openTSNE.TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        initialization="pca",
        learning_rate=learning_rate,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        early_exaggeration_iter=early_exaggeration_iter,
        exaggeration=exaggeration,
        negative_gradient_method=negative_gradient_method,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )


def center_and_flip(coords: np.ndarray) -> np.ndarray:
    """Median-centre then negate, matching the UMAP post-processing so the
    orientation of every disease figure is comparable."""
    return -(coords - np.median(coords, axis=0))


def run_reduction(tsne, matrix: np.ndarray,
                  row_ids: list[int] | None = None) -> np.ndarray:
    """Fit and return 2D coordinates, optionally keeping only ``row_ids``."""
    print("Fitting openTSNE on %d points of dimension %d" % (matrix.shape[0], matrix.shape[1],))
    coords = np.asarray(tsne.fit(matrix))
    coords = center_and_flip(coords)
    return coords if row_ids is None else coords[row_ids]


def parse_args() -> argparse.Namespace:
    """Every path and knob is a flag, so a run is reproducible from its output."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="auto",
                   choices=["auto", "delphi", "hf"])
    p.add_argument("--labels_file", default=str(DEFAULT_LABELS_FILE),
                   help="Disease label table; defaults to the copy in "
                        "downstream_tasks/disease_prediction/delphi_baseline/")
    p.add_argument("--out_dir", type=Path, default=Path("figures"),
                   help="Figures and coordinates land here")
    p.add_argument("--stem", default="")
    p.add_argument("--data_file", help="Parquet file/dir for token counting")
    p.add_argument("--count_cache_dir")
    p.add_argument("--fit_scope", default="disease",
                   choices=["disease", "vocab"],
                   help="disease fits the disease tokens alone and is the only "
                        "like-for-like head-to-head; vocab fits the whole "
                        "vocabulary and is a statement about one model only")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="Effective neighbourhood size")
    p.add_argument("--metric", default="cosine",
                   help="Distance metric, matching the UMAP figure")
    p.add_argument("--learning_rate", default="auto",
                   help="'auto' resolves to n_samples/early_exaggeration")
    p.add_argument("--n_iter", type=int, default=750,
                   help="Main-phase iterations, after the early phase")
    p.add_argument("--early_exaggeration", default="auto",
                   help="Exaggeration during the early phase")
    p.add_argument("--early_exaggeration_iter", type=int, default=250)
    p.add_argument("--exaggeration", default="none",
                   help="Main-phase exaggeration; 'none' leaves it off")
    p.add_argument("--negative_gradient_method", default="auto",
                   choices=["auto", "bh", "fft"],
                   help="openTSNE picks bh below roughly 10k points")
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--center_embeddings", action="store_true",
                   help="Subtract the mean disease-token vector before "
                        "reducing. Corrects embedding anisotropy. Apply to "
                        "every model in a comparison or to none of them.")
    p.add_argument("--circle_size_scale", type=float, default=1.0)
    p.add_argument("--figure_width", type=float, default=DEFAULT_FIGURE_WIDTH)
    p.add_argument("--figure_height", type=float, default=DEFAULT_FIGURE_HEIGHT)
    p.add_argument("--export_dpi", type=int, default=DEFAULT_EXPORT_DPI)
    p.add_argument("--seed", type=int, default=1413)
    return p.parse_args()


def coerce_numeric(value, name: str):
    """openTSNE accepts 'auto'/None or a number; coerce numeric strings."""
    if isinstance(value, str):
        if value.lower() in ("auto", "none"):
            return None if value.lower() == "none" else "auto"
        try:
            return float(value)
        except ValueError:
            raise SystemExit(f"--{name} must be auto, none or a number, "
                             f"got {value!r}")
    return value


def main() -> None:
    args = parse_args()
    args.learning_rate = coerce_numeric(args.learning_rate, "learning_rate")
    args.early_exaggeration = coerce_numeric(
        args.early_exaggeration, "early_exaggeration")
    args.exaggeration = coerce_numeric(args.exaggeration, "exaggeration")

    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.count_cache_dir:
        args.count_cache_dir = os.path.join(args.out_dir, "count_cache")

    def with_tag(name: str) -> str:
        return f"{args.stem}_{name}" if args.stem else name

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

    if model_type == "delphi":
        delphi_to_token = {int(i): int(i)
                           for i in disease_labels["index"] if i < vocab_size}
    elif tokenizer:
        delphi_to_token = extract_disease_mapping(tokenizer, labels)
    else:
        raise ValueError("No tokenizer available for HF model")

    delphi_ids = [d for d in sorted(delphi_to_token)
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

    n_fit = vocab_size if args.fit_scope == "vocab" else len(delphi_ids)
    eff_perp = float(min(args.perplexity, max(5.0, (n_fit - 1) / 3.0)))
    if eff_perp != args.perplexity:
        print("  Perplexity capped to %g for %d points" % (eff_perp, n_fit,))

    tsne = build_tsne(
        eff_perp, args.seed, args.metric, args.learning_rate, args.n_iter,
        args.early_exaggeration, args.early_exaggeration_iter,
        args.exaggeration, args.negative_gradient_method, args.n_jobs)

    if args.fit_scope == "vocab":
        coords = run_reduction(tsne, wte, row_ids=token_ids)
    else:
        coords = run_reduction(tsne, mapped_wte)

    np.save(os.path.join(args.out_dir, with_tag("coords.npy")), coords)

    xy_of = {d: (coords[i, 0], coords[i, 1])
             for i, d in enumerate(delphi_ids)}
    plot = disease_labels[disease_labels["index"].isin(delphi_ids)].copy()
    # common.py draws both figures and names its coordinate columns UMAP1
    # and UMAP2, so the t-SNE coordinates go in under those names.
    plot["UMAP1"] = plot["index"].map(lambda x: xy_of[x][0])
    plot["UMAP2"] = plot["index"].map(lambda x: xy_of[x][1])
    if token_counts:
        plot["count"] = plot["index"].map(lambda x: token_counts.get(x, 0))
        plot.loc[plot["count"] == 0, "count"] = np.nan

    csv_df = plot.rename(columns={"UMAP1": "TSNE1", "UMAP2": "TSNE2"})
    csv_df.to_csv(os.path.join(args.out_dir, with_tag("labels.csv")),
                  index=False)
    print("  %d tokens for visualization" % (len(plot),))

    plot_disease_embedding(
        plot,
        os.path.join(args.out_dir, with_tag("disease_embeddings.png")),
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
