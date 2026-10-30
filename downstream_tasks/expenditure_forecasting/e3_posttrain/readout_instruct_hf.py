#!/usr/bin/env python3
"""
REFERENCE readout of a cost-instruct checkpoint on HF transformers — the original
implementation, kept for cross-engine verification and as the fallback if vLLM is
unavailable.  STANDALONE (its twin, readout_instruct_vllm.py, is the production engine with
the same interface and the same math).

Math: prompt = context_seq + " <INSTRUCT-COST>" as token ids (left-truncated to seq_len-6),
left-padded batches of --batch_size in cohort order, one prefill + 3 KV-cached greedy steps;
the softmax restricted to the 59 <COST-me> bin tokens at each step is the predictive
distribution.  Columns: predicted_<target> = sum p(bin) * E[$ | bin] (EV capped at $1M) and
prob_ge_<cutoff> = bin mass at/above the cutoff on the total-cost position.

Data-parallel: one model copy per GPU (--dp_size), cohort sharded round-robin, CSVs merged.
Writes <section>/model_gen/<model_name>/{cost_predictions.csv,timing.json}.  Roughly 5-7x
the GPU time of the vLLM readout (padding waste of fixed-size batches over a long-tailed
length profile).
"""
import os, json, argparse, logging, tempfile, time, traceback
import numpy as np, pandas as pd

from utils import setup_logging
from sequence_parsing import GOLD_COST_COLS
from posttrain_bins import BinSpec
from inference_cost import SectionTimer, timing_record, write_timing, local_bin_artifact, log_evaluation_total
from vllm_engine import join_fail_fast   # engine-neutral fail-fast join (no vllm import)

ENGINE = "hf"


def parse_args():
    p = argparse.ArgumentParser(description="HF distributional cost readout (reference)")
    p.add_argument("--model_dir", required=True, help="Trained model dir (the .../final save)")
    p.add_argument("--bin_artifact", default=None, help="Override the checkpoint's bin_artifact_ref.json")
    p.add_argument("--section_dirs", nargs="+", required=True)
    p.add_argument("--model_name", required=True, help="Output folder under <section>/model_gen/")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--max_rows", type=int, default=None, help="Smoke tests")
    p.add_argument("--dp_size", type=int, default=1, help="Data-parallel replicas, one GPU each")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def decode_batch(model, input_ids, attention_mask, bin_ids_t):
    """Returns probs (B, 4, K): the bin distribution at each response slot."""
    import torch, torch.nn.functional as F
    with torch.no_grad():
        B = input_ids.shape[0]; all_probs = []
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        for _ in range(N_STEPS):
            probs = F.softmax(logits[:, bin_ids_t].float(), dim=-1)          # restricted softmax
            all_probs.append(probs)
            next_tok = bin_ids_t[probs.argmax(dim=-1)]                        # greedy within the bins
            attention_mask = torch.cat([attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype,
                                                                   device=attention_mask.device)], dim=1)
            out = model(input_ids=next_tok.unsqueeze(1), attention_mask=attention_mask,
                        past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
        return torch.stack(all_probs, dim=1)


def run_shard(args_dict, shard_parquet, out_csv, gpu_id, bin_ids):
    """One replica: load the model on one GPU, read out its shard in batches, write a CSV + stats json."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args = argparse.Namespace(**args_dict)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device).eval()
    load_s = time.time() - t0
    df = pd.read_parquet(shard_parquet)
    ids_list = encode_prompts(tokenizer, df["context_seq"].tolist(), args.seq_len - (N_STEPS + 2))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    bin_ids_t = torch.tensor(bin_ids, dtype=torch.long, device=device)
    B = args.batch_size; n_batches = (len(ids_list) + B - 1) // B; probs_all = []
    t1 = time.time()
    for bi in range(n_batches):
        chunk = ids_list[bi * B:(bi + 1) * B]; maxlen = max(len(x) for x in chunk)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)   # left padding
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, x in enumerate(chunk):
            input_ids[i, maxlen - len(x):] = torch.tensor(x); attn[i, maxlen - len(x):] = 1
        probs_all.append(decode_batch(model, input_ids.to(device), attn.to(device), bin_ids_t).cpu().numpy())
        if (bi + 1) % 50 == 0:
            logging.info("  batch %d/%d", bi + 1, n_batches)
    gen_s = time.time() - t1
    probs = np.concatenate(probs_all).astype(np.float64)
    predictions_frame(df["enrollee_id"].values, probs, BinSpec(args.bin_artifact_resolved)).to_csv(out_csv, index=False)
    json.dump(dict(load_s=load_s, gen_s=gen_s, patients=len(df), prompt_tokens=int(sum(len(x) for x in ids_list)),
                   extra=dict(batch_size=B)), open(out_csv + ".stats.json", "w"))
    logging.info("  shard done: %d patients, load %.0fs, generate %.0fs -> %s", len(df), load_s, gen_s, out_csv)

CUTOFFS = [30000.0]                    # HNHC threshold only
N_STEPS = len(GOLD_COST_COLS)          # 1 response position: total
EV_CAP = 1_000_000.0                   # censored target E[min(cost, $1M)]


def encode_prompts(tokenizer, contexts, max_prompt):
    """Token ids for context + ' <INSTRUCT-COST>', left-truncated (most recent history kept)."""
    out = []
    for c in contexts:
        ids = tokenizer(c + " <INSTRUCT-COST>", add_special_tokens=False)["input_ids"]
        out.append(ids[-max_prompt:] if len(ids) > max_prompt else ids)
    return out


def predictions_frame(eids, probs, binspec):
    """probs (N, 4, K) -> the cost_predictions.csv columns."""
    ev = {tc: np.minimum(np.asarray(binspec.ev_dollars[tc], dtype=float), EV_CAP) for tc in GOLD_COST_COLS}
    cw = {c: np.array([1.0 if q >= c else 0.0 for q in binspec.bin_ids]) for c in CUTOFFS}
    rec = {"enrollee_id": np.asarray(eids)}
    for p_idx, tc in enumerate(GOLD_COST_COLS):
        rec[f"predicted_{tc}"] = (probs[:, p_idx, :] * ev[tc]).sum(-1)
    for c in CUTOFFS:
        rec[f"prob_ge_{int(c)}"] = (probs[:, 0, :] * cw[c]).sum(-1)
    return pd.DataFrame(rec)


def preview(section_dir, result):
    """Quick-look line in the job log (formal metrics: e4_eval)."""
    gold = pd.read_parquet(os.path.join(section_dir, "preprocessed_input.parquet"), columns=["enrollee_id", "next_annual_cost"])
    m = gold.merge(result[["enrollee_id", "predicted_next_annual_cost", "prob_ge_30000"]], on="enrollee_id")
    y = np.minimum(m["next_annual_cost"].to_numpy(float), EV_CAP); pdl = m["predicted_next_annual_cost"].to_numpy(float)
    r2 = 1 - ((y - pdl) ** 2).sum() / ((y - y.mean()) ** 2).sum(); auc_txt = ""
    yb = (y >= 30_000).astype(int)
    if 0 < yb.sum() < len(yb):
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_txt = " ROC=%.3f PR=%.3f" % (roc_auc_score(yb, m["prob_ge_30000"]), average_precision_score(yb, m["prob_ge_30000"]))
    logging.info("  PREVIEW [%s] R2=%.3f MAE=%.0f%s (censored gold, HNHC $30k; formal metrics via e4_eval)",
                 os.path.basename(section_dir), r2, np.abs(y - pdl).mean(), auc_txt)


def resolve_bins(args):
    """Pin the bin artifact (checkpoint ref, with the local fallback) and the bin token ids."""
    ref = args.bin_artifact or json.load(open(os.path.join(args.model_dir, "bin_artifact_ref.json")))["bin_artifact"]
    args.bin_artifact_resolved = local_bin_artifact(ref, os.environ.get("DATA_DIR", ""))
    binspec = BinSpec(args.bin_artifact_resolved)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    bin_ids = [tok.convert_tokens_to_ids(t) for t in binspec.tokens]
    assert all(i is not None and i >= 0 for i in bin_ids)
    logging.info("  Bin artifact: %s (%d bins)", args.bin_artifact_resolved, len(bin_ids))
    return bin_ids


def _worker(args_dict, shard_parquet, out_csv, gpu_id, bin_ids):
    setup_logging(script_path=f"readout_{ENGINE}_rank{gpu_id}")
    try:
        run_shard(args_dict, shard_parquet, out_csv, gpu_id, bin_ids)
    except Exception:
        logging.error("readout shard on GPU %s failed:\n%s", gpu_id, traceback.format_exc()); raise


def main():
    """Data-parallel driver: cohort sharded round-robin, one worker process per GPU (fail-fast
    join), shard CSVs merged into model_gen/<name>/cost_predictions.csv + timing.json."""
    args = parse_args()
    log_file = setup_logging(level=logging.INFO)
    logging.info("READOUT (%s) STARTING...  model=%s name=%s dp=%d", ENGINE, args.model_dir, args.model_name, args.dp_size)
    bin_ids = resolve_bins(args)
    args_dict = vars(args).copy()
    tmp = tempfile.mkdtemp(prefix=f"readout_{ENGINE}_", dir=os.environ.get("TMPDIR"))
    n = max(1, args.dp_size); totals = []
    for section_dir in args.section_dirs:
        ip = os.path.join(section_dir, "preprocessed_input.parquet")
        if not os.path.exists(ip):
            logging.warning("  Missing input, skipping: %s", ip); continue
        df = pd.read_parquet(ip, columns=["enrollee_id", "context_seq"])
        if args.max_rows: df = df.head(args.max_rows)
        sec = os.path.basename(section_dir)
        logging.info("  [%s] %d patients", sec, len(df))
        out_dir = os.path.join(section_dir, "model_gen", args.model_name); os.makedirs(out_dir, exist_ok=True)
        with SectionTimer() as T:
            shard_csvs = []
            if n == 1:
                sp_ = os.path.join(tmp, f"{sec}.parquet"); df.to_parquet(sp_); csv = os.path.join(tmp, f"{sec}.csv")
                run_shard(args_dict, sp_, csv, None, bin_ids); shard_csvs.append(csv)
            else:
                import multiprocessing as mp
                ctx = mp.get_context("spawn"); procs = []
                for r in range(n):
                    part = df.iloc[r::n]; sp_ = os.path.join(tmp, f"{sec}.shard{r}.parquet"); part.to_parquet(sp_)
                    csv = os.path.join(tmp, f"{sec}.shard{r}.csv"); shard_csvs.append(csv)
                    p = ctx.Process(target=_worker, args=(args_dict, sp_, csv, r, bin_ids)); p.start(); procs.append(p)
                code = join_fail_fast(procs)
                if code != 0:
                    raise RuntimeError(f"{ENGINE} readout workers failed (exit code {code})")
        result = pd.concat([pd.read_csv(c) for c in shard_csvs], ignore_index=True)
        stats = [json.load(open(c + ".stats.json")) for c in shard_csvs]
        result.to_csv(os.path.join(out_dir, "cost_predictions.csv"), index=False)
        preview(section_dir, result)
        rec = timing_record(model=args.model_name, section=sec, engine=f"{ENGINE} dp={n}", patients=len(result), wall_s=T.wall_s,
                            n_gpus=n, tokens_generated_per_patient=float(N_STEPS), prompt_tokens=sum(s["prompt_tokens"] for s in stats),
                            trajectories=1, tokens_generated_per_trajectory=float(N_STEPS),
                            engine_load_s=round(max(s["load_s"] for s in stats), 1), generate_s=round(max(s["gen_s"] for s in stats), 1),
                            sampling=dict(temperature=0, max_tokens=N_STEPS, min_tokens=N_STEPS, allowed_token_ids=len(bin_ids)),
                            **stats[0].get("extra", {}))
        write_timing(out_dir, rec); totals.append(rec)
        logging.info("  [%s] Saved %d rows to %s/cost_predictions.csv", sec, len(result), out_dir)
    log_evaluation_total(args.model_name, totals)
    logging.info("READOUT (%s) COMPLETED", ENGINE); logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
