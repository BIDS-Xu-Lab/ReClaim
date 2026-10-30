#!/usr/bin/env python3
"""
reduce_embeddings.py -- PCA plug-in: dimension-matched low-dim variants of existing embeddings.

Reads cohort_with_<model>_embeddings.parquet and writes, per requested k,
cohort_with_<model>-pca<k>_embeddings.parquet beside it (same schema: person_id [+ carried
columns] + rep_0..rep_{k-1}, float32). The spec name <model>-pca<k>-only then flows through
ladder_driver/estimate_ladder with NO pipeline changes (spec -> parquet name resolution is
mechanical), so this whole analysis is a droppable side branch.

Design notes:
  * ks are dimension-matched to the study's baselines (120 = Delphi's native width, 500 = hdPS k;
    60 = a typical expert covariate count).
  * One PCA fit per (parquet) at max(ks) components; smaller ks take the leading components
    of the same fit (nested by construction), so all ks share one deterministic decomposition.
  * Deterministic: fit on an evenly-strided subsample of rows sorted by person_id,
    randomized-SVD PCA with a fixed seed. Unsupervised: neither treatment nor outcome is used.

Usage:
  python b2_embeddings/reduce_embeddings.py \
    --embed_dir intermediate/mdcr/embeddings/positive_control_celecoxib_gi_MDCR \
    --model reclaim-m --ks 60,120,500
"""
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.decomposition import PCA


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embed_dir", required=True, help="dir holding cohort_with_<model>_embeddings.parquet")
    p.add_argument("--model", required=True, help="source model key, e.g. reclaim-m")
    p.add_argument("--ks", default="60,120,500")
    p.add_argument("--fit_rows", type=int, default=500_000,
                   help="max rows for the PCA fit (evenly strided over person_id order)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ks = sorted({int(k) for k in args.ks.split(",")})
    src = os.path.join(args.embed_dir, f"cohort_with_{args.model}_embeddings.parquet")
    if not os.path.exists(src):
        sys.exit(f"ERROR: source parquet missing: {src}")
    outs = {k: os.path.join(args.embed_dir, f"cohort_with_{args.model}-pca{k}_embeddings.parquet")
            for k in ks}
    # cache is per person: an output is reused only if it has as many rows as the source
    n_src = pq.read_metadata(src).num_rows
    todo = {k: o for k, o in outs.items()
            if not os.path.exists(o) or pq.read_metadata(o).num_rows != n_src}
    if not todo:
        log(f"all outputs cached for {args.model} in {args.embed_dir}; nothing to do")
        return

    log(f"reading {src}")
    tab = pq.read_table(src)
    rep_cols = [c for c in tab.column_names if c.startswith("rep_")]
    carry_cols = [c for c in tab.column_names if not c.startswith("rep_")]
    X = np.column_stack([tab.column(c).to_numpy(zero_copy_only=False) for c in rep_cols]).astype(np.float32)
    n, d = X.shape
    log(f"n={n:,} d={d} carry={carry_cols}")
    max_k = max(ks)
    if max_k >= d:
        sys.exit(f"ERROR: max k ({max_k}) must be < source dim ({d})")

    # deterministic fit subsample: even stride over person_id-sorted rows
    order = np.argsort(tab.column("person_id").to_numpy(zero_copy_only=False), kind="stable")
    m = min(args.fit_rows, n)
    idx = order[np.linspace(0, n - 1, m).astype(np.int64)]
    log(f"fitting PCA(k={max_k}) on {m:,} strided rows (seed={args.seed})")
    pca = PCA(n_components=max_k, svd_solver="randomized", random_state=args.seed)
    pca.fit(X[idx])
    cum = np.cumsum(pca.explained_variance_ratio_)
    for k in ks:
        log(f"  cumulative explained variance at k={k}: {cum[k-1]:.4f}")

    Z = (X - pca.mean_) @ pca.components_[:max_k].T  # n x max_k, float32 math
    carry = {c: tab.column(c) for c in carry_cols}
    for k, out in todo.items():
        cols = dict(carry)
        for j in range(k):
            cols[f"rep_{j}"] = pa.array(Z[:, j].astype(np.float32))
        pq.write_table(pa.table(cols), out)
        log(f"wrote {out}  ({n:,} x {k})")
    log("done")


if __name__ == "__main__":
    main()
