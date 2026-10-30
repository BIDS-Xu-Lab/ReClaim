#!/usr/bin/env python3
"""
Score a model's disease embeddings against the ICD-10 chapters.

Clusters the embeddings with k-means into as many clusters as there are
chapters, then reports normalized mutual information and the adjusted Rand
index against the chapter labels, averaged over several k-means restarts.

Model loading and disease-token mapping come from common.py, so the embeddings
scored here are the ones the figures are drawn from.

Outputs --out_file, a JSON holding both scores per restart and their means,
alongside the embedding matrix and token ids it scored.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from common import (
    DEFAULT_LABELS_FILE,
    NON_DISEASE_CHAPTERS,
    extract_code_from_label,
    extract_disease_mapping,
    load_model_embeddings,
)


def get_disease_embeddings(model_path: str | Path, model_type: str,
                           labels_df: pd.DataFrame,
                           align_tokenizer_path: str | None = None
                           ) -> tuple[list[int], np.ndarray]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtype, tokenizer, wte = load_model_embeddings(model_path, model_type, device)
    vocab_size = wte.shape[0]

    align_keys = None
    if align_tokenizer_path:
        from transformers import AutoTokenizer
        align_tok = AutoTokenizer.from_pretrained(align_tokenizer_path)
        align_keys = set(extract_disease_mapping(align_tok, labels_df).keys())
        print(f"  Aligning to {len(align_keys)} tokens from {align_tokenizer_path}")

    disease_labels = labels_df[~labels_df["ICD-10 Chapter"].isin(NON_DISEASE_CHAPTERS)]
    if mtype == "delphi":
        delphi_to_token = {int(i): int(i) for i in disease_labels["index"] if i < vocab_size}
    elif tokenizer is not None:
        delphi_to_token = extract_disease_mapping(tokenizer, labels_df)
    else:
        raise ValueError("No tokenizer available for HF model")

    if align_keys is not None:
        delphi_to_token = {d: t for d, t in delphi_to_token.items() if d in align_keys}

    delphi_ids = sorted(d for d in delphi_to_token if delphi_to_token[d] < vocab_size)
    token_ids = [delphi_to_token[d] for d in delphi_ids]
    return delphi_ids, wte[token_ids]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="auto", choices=["auto", "delphi", "hf"])
    p.add_argument("--labels_file", default=str(DEFAULT_LABELS_FILE),
                   help="Disease label table; defaults to the copy in "
                        "downstream_tasks/disease_prediction/delphi_baseline/")
    p.add_argument("--out_file", type=Path, default=Path("figures/metrics.json"),
                   help="JSON with NMI and ARI per seed, plus their means")
    p.add_argument("--model_name", default="model")
    p.add_argument("--n_seeds", type=int, default=10,
                   help="Independent KMeans runs (different seeds) to average over")
    p.add_argument("--seed", type=int, default=1413)
    p.add_argument("--align_tokenizer_path", default=None,
                   help="HF model/tokenizer path. If set, restrict the disease "
                        "token set to codes that have a <DX-MAJOR_*> token in "
                        "this tokenizer (so Delphi and HF runs evaluate the "
                        "same disease tokens).")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading labels from {args.labels_file}")
    labels = pd.read_csv(args.labels_file)
    labels["index"] = labels["index"].astype(int)
    labels["code"] = labels["name"].map(extract_code_from_label)

    print(f"Loading model from {args.model_path}")
    delphi_ids, emb = get_disease_embeddings(
        args.model_path, args.model_type, labels,
        align_tokenizer_path=args.align_tokenizer_path,
    )
    print(f"  {len(delphi_ids)} disease tokens, embedding dim={emb.shape[1]}")

    chapter_by_idx = labels.set_index("index")["ICD-10 Chapter"].to_dict()
    chapters = np.array([chapter_by_idx[d] for d in delphi_ids])
    n_clusters = int(len(np.unique(chapters)))
    print(f"  {n_clusters} ICD-10 chapters as ground truth")

    out_base = os.path.splitext(os.path.abspath(args.out_file))[0]
    emb_file = out_base + "_embeddings.npy"
    ids_file = out_base + "_delphi_ids.npy"
    np.save(emb_file, emb.astype(np.float32))
    np.save(ids_file, np.asarray(delphi_ids, dtype=np.int64))
    print(f"  Saved embeddings to {emb_file}")
    print(f"  Saved delphi ids   to {ids_file}")

    # L2-normalize so Euclidean KMeans approximates cosine clustering,
    # matching the metric='cosine' used by the UMAP step.
    X = normalize(emb.astype(np.float32), norm="l2", axis=1)

    rng = np.random.default_rng(args.seed)
    seeds = [int(s) for s in rng.integers(0, 10**9, size=args.n_seeds)]
    nmis, aris = [], []
    for s in seeds:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=s)
        pred = km.fit_predict(X)
        nmis.append(normalized_mutual_info_score(chapters, pred))
        aris.append(adjusted_rand_score(chapters, pred))

    result = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "model_type": args.model_type,
        "align_tokenizer_path": args.align_tokenizer_path,
        "embedding_file": emb_file,
        "delphi_ids_file": ids_file,
        "n_tokens": int(len(delphi_ids)),
        "embedding_dim": int(emb.shape[1]),
        "n_clusters": n_clusters,
        "n_seeds": int(args.n_seeds),
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "nmi_per_seed": [float(x) for x in nmis],
        "ari_per_seed": [float(x) for x in aris],
        "kmeans_seeds": seeds,
    }

    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"NMI={result['nmi_mean']:.4f}+/-{result['nmi_std']:.4f}  "
        f"ARI={result['ari_mean']:.4f}+/-{result['ari_std']:.4f}")
    print(f"Saved {args.out_file}")


if __name__ == "__main__":
    main()
