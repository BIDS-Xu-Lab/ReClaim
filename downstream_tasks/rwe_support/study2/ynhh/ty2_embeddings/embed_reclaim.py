#!/usr/bin/env python3
"""
embed_reclaim.py -- ReClaim patient embeddings (rep_*) for the union cohort (ty2_1).

Takes a prepare_sequences `cohort_with_pre_entry_seq.parquet` and writes
`person_id + rep_0..rep_{d-1}` as parquet (float32 + snappy; estimate_ladder reads it
column-pruned via nanoparquet). Two interchangeable inference backends behind one CLI:

  --backend hf     HuggingFace transformers (reference implementation).
  --backend vllm   vLLM pooling runner; faster under the very skewed sequence-length
                   distribution (continuous batching). Backend equivalence verified
                   (median cosine 0.99998).

The embedding:
  * pooling      : last NON-PAD token of the final hidden state (for a causal LM only that
                   position has attended the whole sequence)
  * normalise    : L2 to unit vectors
  * truncation   : LEFT (keep the most RECENT tokens) at --max_length, the only mode. Pooling
                   reads the final token, so the embedding is "the patient as of the last
                   token kept". Right truncation would anchor it at a mid-history point
                   whose date varies with history length, i.e. with healthcare utilisation,
                   itself correlated with treatment: a confound injected rather than
                   controlled. Left truncation drops the <sos><SEX><DOBYR><AGE> header on the
                   rare over-length sequences; age and sex are explicit PS covariates in
                   every spec, so nothing the model needs is lost.
  * dtype        : bfloat16 where supported, else float16; eval mode, no_grad

Usage: python ty2_embeddings/embed_reclaim.py --model reclaim-s --model_path $RECLAIM_S_CKPT \
    --input <prevseq.parquet> --output <cohort_with_reclaim-s_embeddings.parquet> --backend vllm
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, time_execution  # noqa: E402

INDENT2 = "    "


def parse_args():
    p = argparse.ArgumentParser(description="Generate ReClaim rep_* embeddings for an prepare_sequences prevseq file",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="reclaim-s",
                   help="Model key (reclaim-s | reclaim-m | reclaim-l). Drives logging and the output "
                        "filename tag; --model_path supplies the weights. The spec '<key>-only' in "
                        "estimate_ladder maps back to cohort_with_<key>_embeddings.parquet.")
    p.add_argument("--model_path", required=True, help="Checkpoint directory")
    p.add_argument("--input", required=True, help="prepare_sequences cohort_with_pre_entry_seq*.parquet")
    p.add_argument("--output", required=True, help="Output .parquet path")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--max_length", type=int, default=4096,
                   help="Max prompt tokens, keeping the LAST max_length (left truncation)")
    p.add_argument("--batch_size", type=int, default=8, help="hf backend only")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vllm only")
    p.add_argument("--max_num_seqs", type=int, default=256, help="vllm only")
    p.add_argument("--vllm_chunk_size", type=int, default=50_000,
                   help="vllm only: patients per embed() call. Bounds host RAM (tokenised prompts + "
                        "returned Python float lists), not GPU throughput. Lower it if the node is small")
    p.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=True, help="vllm only")
    p.add_argument("--chunked_prefill", action=argparse.BooleanOptionalAction, default=False, help="vllm only")
    p.add_argument("--limit", type=int, default=None, help="Embed only the first N rows (smoke tests)")
    p.add_argument("--gpus", default="auto",
                   help="Data parallelism: number of GPUs to shard across (the torchrun-style "
                        "self-launch below). 'auto' = count of visible GPUs (1 if none/unknown). "
                        "With N>1 the parent re-executes itself N times, one process per GPU "
                        "(CUDA_VISIBLE_DEVICES pins each) on contiguous row shards, then "
                        "concatenates the shard outputs in order. N=1 runs inline, as ever.")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="INTERNAL (set by the --gpus self-launch; usable manually too): embed only "
                        "the K-th of N contiguous row shards. Concatenating shard outputs in shard "
                        "order restores the original row order because shards are contiguous.")
    p.add_argument("--keep_seq_col", action="store_true",
                   help="Keep pre_entry_seq in the output (default drops it: it is most of the file "
                        "size and nothing downstream reads it)")
    return p.parse_args()


def load_sequences(path, limit=None):
    """Read the prepare_sequences prevseq file. Every cohort row must carry a sequence.

    Null pre_entry_seq rows are NOT dropped: prepare_sequences guarantees one row per cohort
    patient (header-only for patients with no pre-index events), so a null means the file is
    truncated or corrupt, and silently dropping rows is how a partial embedding file would
    reach estimate_ladder. Fail loudly instead.
    """
    if not path.endswith(".parquet"):
        raise ValueError(f"expected an prepare_sequences cohort_with_pre_entry_seq*.parquet, got: {path}")
    df = pd.read_parquet(path)
    if "pre_entry_seq" not in df.columns:
        raise ValueError(f"{path} has no pre_entry_seq column (got {list(df.columns)[:8]}). "
                         "Expected an prepare_sequences cohort_with_pre_entry_seq*.parquet.")
    if limit:
        df = df.head(limit).copy()
        logging.info("%s--limit %d: smoke-test subset", INDENT2, limit)
    n_null = int(df["pre_entry_seq"].isna().sum())
    if n_null:
        raise ValueError(f"{n_null:,} of {len(df):,} rows have a null pre_entry_seq. prepare_sequences emits a "
                         "header-only sequence for patients with no in-window events, so nulls mean "
                         "the file is truncated or corrupt. Refusing to proceed -- dropping them "
                         "would silently shrink coverage and estimate_ladder would zero-fill the rest.")
    return df, df["pre_entry_seq"].astype(str).tolist()


def embed_hf(seqs, model_path, max_length, batch_size):
    """Reference implementation: last-non-pad-token pooling, L2-normalised. Deliberately not
    optimised (fixed batches padded to the longest member): length-sorted batching would add
    reorder bookkeeping and change batch composition; continuous batching is the vllm
    backend's job. For speed with this backend use --gpus data parallelism."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("%sdevice=%s%s", INDENT2, device,
                 f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"          # keep the most RECENT events

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True,
                                      device_map="auto" if device.type == "cuda" else None)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    logging.info("%shidden_size=%s dtype=%s", INDENT2, model.config.hidden_size, dtype)

    out = []
    nb = (len(seqs) + batch_size - 1) // batch_size
    for b in tqdm(range(nb), desc="hf batches"):
        batch = seqs[b * batch_size:(b + 1) * batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc.pop("token_type_ids", None)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        last = mask.sum(dim=1) - 1                      # index of the last NON-PAD token
        pooled = hs[torch.arange(hs.shape[0], device=device), last.long()]
        out.append(pooled.float().cpu().numpy())
        if device.type == "cuda" and b % 50 == 0:
            torch.cuda.empty_cache()
    return np.vstack(out)


def embed_vllm(seqs, model_path, max_length, gpu_util, max_num_seqs, enforce_eager, chunked_prefill,
               chunk_size=50_000):
    """vLLM pooling runner. PoolerConfig(LAST, normalize) matches the hf pooling/normalisation.

    Sequences are tokenised HERE with the same tokenizer the hf path uses, and explicit token
    ids are passed: letting vLLM truncate raw text slices a slightly different window for
    over-length sequences (the tokenizer truncates around its special tokens; vLLM slices
    token_ids[-k:] afterwards), so the two backends would disagree exactly on those patients.
    Chunked so a large cohort fits in host RAM (tokenised prompts and the returned Python
    float lists peak together); the engine is built once outside the loop and vLLM's own
    scheduler still batches within each chunk.
    """
    import gc
    from vllm import LLM, TokensPrompt
    from vllm.config import PoolerConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    llm = LLM(model=model_path, runner="pooling", convert="embed",
              # pooler_config (override_pooler_config is deprecated)
              pooler_config=PoolerConfig(pooling_type="LAST", normalize=True),
              dtype="bfloat16", enforce_eager=enforce_eager,
              enable_chunked_prefill=chunked_prefill,
              gpu_memory_utilization=gpu_util, max_num_seqs=max_num_seqs,
              max_model_len=max_length, trust_remote_code=True)

    n, n_trunc, parts = len(seqs), 0, []
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, n, chunk_size), 1):
        block = seqs[start:start + chunk_size]
        enc = tok(block, truncation=True, max_length=max_length)["input_ids"]
        n_trunc += sum(1 for e in enc if len(e) >= max_length)
        prompts = [TokensPrompt(prompt_token_ids=e) for e in enc]
        del enc
        outs = llm.embed(prompts, use_tqdm=True)   # already truncated; no truncate_prompt_tokens
        if len(outs) != len(block):
            raise RuntimeError(f"vLLM returned {len(outs)} embeddings for {len(block)} prompts")
        parts.append(np.asarray([o.outputs.embedding for o in outs], dtype=np.float32))
        del outs, prompts
        gc.collect()
        logging.info("%schunk %d/%d done (%s/%s seq)", INDENT2, ci, n_chunks,
                     f"{min(start + chunk_size, n):,}", f"{n:,}")

    logging.info("%stokenised %s seq(s) with the hf tokenizer; %d at the %d-token cap",
                 INDENT2, f"{n:,}", n_trunc, max_length)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


@time_execution
def _resolve_gpus(spec):
    if str(spec) != "auto":
        return max(1, int(spec))
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return max(1, len([d for d in vis.split(",") if d.strip() != ""]))
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return max(1, len([l for l in out.stdout.splitlines() if l.strip()]))
    except Exception:
        return 1


def launch_data_parallel(args, n):
    """Self-launch N child copies of this script, one per GPU, each on the K-th contiguous row
    shard; shard outputs are concatenated in shard order (contiguity restores the original row
    order). Children are plain single-GPU runs (no torch.distributed: inference has nothing to
    synchronise), so independent processes fail independently."""
    import subprocess, time
    # each child gets the k-th entry of the parent's visible-device list (SLURM may have
    # restricted/renumbered it), not a literal physical id
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [d.strip() for d in vis.split(",")] if vis else [str(k) for k in range(n)]
    shard_out = lambda k: f"{args.output[:-len('.parquet')]}.shard{k}of{n}.parquet"
    # child argv = parent argv minus --gpus, plus the shard; new flags are inherited automatically
    base = [a for a in sys.argv[1:]]
    for flag in ("--gpus",):
        while flag in base:
            i = base.index(flag); del base[i:i+2]
    procs = []
    for k in range(n):
        argv = list(base)
        i = argv.index("--output"); argv[i + 1] = shard_out(k)
        argv += ["--shard", f"{k}/{n}", "--gpus", "1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[k])
        logging.info("%slaunching shard %d/%d on GPU %s", INDENT2, k, n, devices[k])
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv, env=env))
        time.sleep(2)   # stagger: child log filenames are second-resolution
    rc = 0
    for k, p in enumerate(procs):
        if p.wait() != 0:
            logging.error("shard %d/%d FAILED (exit %d)", k, n, p.returncode); rc = 1
    if rc:
        raise SystemExit(1)
    import pyarrow as pa, pyarrow.parquet as pq
    tabs = [pq.read_table(shard_out(k)) for k in range(n)]
    pq.write_table(pa.concat_tables(tabs), args.output, compression="snappy")
    logging.info("%sconcatenated %d shards -> %s (%s rows)",
                 INDENT2, n, args.output, f"{sum(x.num_rows for x in tabs):,}")
    for k in range(n):
        os.remove(shard_out(k))


def main():
    args = parse_args()
    # Validate the output path before any compute: the write is the last step.
    if not args.output.endswith(".parquet"):
        raise ValueError(f"--output must be a .parquet path, got: {args.output}")
    if not args.model.startswith("reclaim-"):
        raise ValueError(f"--model must be a 'reclaim-*' id, got: {args.model}")
    n_gpus = _resolve_gpus(args.gpus)
    if n_gpus > 1 and args.shard is None:
        setup_logging(level=logging.INFO)
        logging.info("DATA-PARALLEL LAUNCHER (backend=%s, model=%s, gpus=%d)",
                     args.backend, args.model, n_gpus)
        launch_data_parallel(args, n_gpus)
        return
    log_file = setup_logging(level=logging.INFO)
    logging.info("RECLAIM EMBEDDINGS STARTING (backend=%s, model=%s)", args.backend, args.model)
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    df, seqs = load_sequences(args.input, args.limit)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        if not (0 <= k < n):
            raise ValueError(f"--shard {args.shard}: need 0 <= K < N")
        lo, hi = len(df) * k // n, len(df) * (k + 1) // n
        df, seqs = df.iloc[lo:hi].copy(), seqs[lo:hi]
        logging.info("%sshard %d/%d: rows [%s, %s) -> %s patients",
                     INDENT2, k, n, f"{lo:,}", f"{hi:,}", f"{len(df):,}")
        if not len(df):
            raise ValueError(f"--shard {args.shard} selected 0 rows (cohort smaller than N shards)")
    logging.info("%s%s patients from %s", INDENT2, f"{len(seqs):,}", args.input)

    if args.backend == "hf":
        emb = embed_hf(seqs, args.model_path, args.max_length, args.batch_size)
    else:
        emb = embed_vllm(seqs, args.model_path, args.max_length, args.gpu_memory_utilization,
                         args.max_num_seqs, args.enforce_eager, args.chunked_prefill,
                         args.vllm_chunk_size)
    logging.info("%sembeddings %s", INDENT2, emb.shape)

    # L2-normalise unconditionally (vLLM already does via PoolerConfig) so the two backends stay
    # comparable and a future upstream default change cannot matter.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{int((norms == 0).sum())} zero-norm embedding(s) -- a zero vector is off the "
                         "unit sphere every real embedding lives on and would poison the PS model.")
    emb = emb / norms
    n = np.linalg.norm(emb, axis=1)
    logging.info("%snorms min=%.6f max=%.6f mean=%.6f", INDENT2, n.min(), n.max(), n.mean())

    out = df if args.keep_seq_col else df.drop(columns=["pre_entry_seq"])
    rep = pd.DataFrame(emb.astype(np.float32),
                       columns=[f"rep_{i}" for i in range(emb.shape[1])], index=out.index)
    out = pd.concat([out, rep], axis=1)
    if len(out) != len(df):
        raise RuntimeError("row count changed while attaching embeddings")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    logging.info("%swriting %s patients x %s cols -> %s", INDENT2, f"{len(out):,}",
                 f"{out.shape[1]:,}", args.output)
    out.to_parquet(args.output, compression="snappy", index=False)
    logging.info("RECLAIM EMBEDDINGS COMPLETED. Log: %s", log_file)


if __name__ == "__main__":
    main()
