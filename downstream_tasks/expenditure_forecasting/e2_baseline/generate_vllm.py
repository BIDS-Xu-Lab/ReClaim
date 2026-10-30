#!/usr/bin/env python3
"""
Monte Carlo trajectory generation for cost prediction using vLLM.

Prompts a pretrained ReClaim model with each patient's context_seq and samples
K future trajectories (temperature=1, top-p=1 by default).  Cost tokens in the
first generated year are later aggregated into the expenditure estimate.

Design: uses n=1 with repeated interleaved prompts instead of n=K, which keeps
one decode sequence per request in the KV cache and lets vLLM run far more
concurrent requests; prefix caching makes the repeated prefills nearly free.

Note: context_seq from preprocessing already starts with <sos> and ends at
a bare <NY> token; no <eos> is present.
"""

import os
import argparse
import json
import logging
import tempfile
import multiprocessing as mp
import numpy as np
from time import sleep

from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer

from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='MC generation for cost prediction (vLLM)')
    parser.add_argument("--processed_data_base_dir", type=str, required=True,
                        help='Directory containing <section>/preprocessed_input.parquet')
    parser.add_argument('--model', type=str, default='reclaim-s-pretrain',
                        help='Model name (used for the output folder)')
    parser.add_argument("--model_path", type=str, required=True,
                        help='Path to pretrained model (HF format)')
    parser.add_argument("--num_generation", type=int, default=20,
                        help='Number of sampled trajectories per patient')
    parser.add_argument("--seq_len", type=int, default=4096,
                        help='Model context length')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--min_tokens", type=int, default=2,
                        help='Minimum generated tokens (prevents immediate EOS)')
    parser.add_argument("--dp_size", type=int, default=1,
                        help='Data-parallel replicas (1 GPU each)')
    parser.add_argument("--section", type=str, default="both",
                        choices=["both", "retrospective", "prospective"])
    return parser.parse_args()


def load_vllm_model(args):
    """Engine from the SHARED config (e0_common/vllm_engine.py) so the MC anchor and the
    vLLM readout are built identically; the values below are this script's historical ones."""
    from vllm_engine import build_llm
    logging.info("Loading model from: %s", args.model_path)
    llm, cfg = build_llm(args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
                         max_model_len=args.seq_len, dtype=args.dtype, seed=args.seed)
    logging.info("Model loaded successfully")
    return llm, cfg


def prepare_dataset(section_dir, args, tokenizer):
    """Load dataset and left-truncate long contexts (keep most recent history)."""
    input_path = os.path.join(section_dir, 'preprocessed_input.parquet')
    logging.info("Loading data from: %s", input_path)
    dataset = Dataset.from_parquet(input_path)
    logging.info("Loaded %d sequences", len(dataset))

    if 'context_seq' not in dataset.column_names:
        raise ValueError("Data must contain 'context_seq' column as the input")

    generation_buffer = 512  # MC simulation of one year of events
    min_tokens = args.min_tokens

    max_prompt_len = args.seq_len - generation_buffer
    logging.info("Left-truncating to max %d prompt tokens (buffer %d, model=%s)",
                 max_prompt_len, generation_buffer, args.model)

    def truncate_sequence(example):
        tokens = tokenizer(example['context_seq'], add_special_tokens=False,
                           truncation=False)
        input_ids = tokens['input_ids']
        was_truncated = len(input_ids) > max_prompt_len
        if was_truncated:
            input_ids = input_ids[-max_prompt_len:]
        truncated_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        return {'context_seq': truncated_text, '_was_truncated': was_truncated, '_n_prompt_tokens': len(input_ids)}

    dataset = dataset.map(truncate_sequence, desc="Truncating sequences")
    truncated_count = sum(dataset['_was_truncated'])
    prompt_tokens = int(sum(dataset['_n_prompt_tokens']))          # for the inference-cost record
    dataset = dataset.remove_columns(['_was_truncated', '_n_prompt_tokens'])
    if truncated_count > 0:
        logging.warning("Left-truncated %d sequences", truncated_count)

    return dataset, generation_buffer, min_tokens, prompt_tokens


def create_sampling_params(args, generation_buffer, min_tokens):
    if args.num_generation > 1:
        sampling_params = SamplingParams(
            temperature=args.temperature, top_p=args.top_p,
            min_tokens=min_tokens, max_tokens=generation_buffer)
        logging.info("Sampling mode (n=1 x %d copies): T=%s, top_p=%s, max_tokens=%d",
                     args.num_generation, args.temperature, args.top_p, generation_buffer)
    else:
        sampling_params = SamplingParams(
            temperature=0, min_tokens=min_tokens, max_tokens=generation_buffer)
        logging.info("Greedy decoding, max_tokens=%d", generation_buffer)
    return sampling_params


def build_interleaved_prompts(unique_prompts, K):
    """[p0]*K + [p1]*K + ... — adjacency maximises vLLM prefix-cache hits."""
    flat_prompts = []
    for p in unique_prompts:
        flat_prompts.extend([p] * K)
    return flat_prompts


def regroup_flat_outputs(flat_texts, n_unique, K):
    assert len(flat_texts) == n_unique * K, \
        f"Expected {n_unique * K} outputs, got {len(flat_texts)}"
    return [flat_texts[i * K: (i + 1) * K] for i in range(n_unique)]


def generate_predictions(llm, prompts, sampling_params):
    """Returns (texts, total generated tokens as counted by the engine)."""
    logging.info("Starting inference on %d prompts (continuous batching, n=1)", len(prompts))
    # use_tqdm=False: batch logs need no bar, and vLLM's bar code divides by the
    # bar's elapsed time — with TQDM_DISABLE=1 in the job env (env.sh) a disabled
    # bar reports elapsed=0 -> ZeroDivisionError in every data-parallel rank.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generated_texts = [output.outputs[0].text for output in outputs]

    finish_reasons, empty_count, token_counts = {}, 0, []
    for i, output in enumerate(outputs):
        comp = output.outputs[0]
        finish_reasons[comp.finish_reason] = finish_reasons.get(comp.finish_reason, 0) + 1
        token_counts.append(len(comp.token_ids))
        if not comp.text.strip():
            empty_count += 1
            if empty_count <= 5:
                logging.warning("Empty output #%d (idx=%d): finish_reason=%s, prompt_end='%s'",
                                empty_count, i, comp.finish_reason, prompts[i][-200:])

    logging.info("Finish reasons: %s | empty: %d/%d", finish_reasons, empty_count, len(outputs))
    if token_counts:
        logging.info("Output tokens: avg=%.1f, min=%d, max=%d",
                     sum(token_counts) / len(token_counts), min(token_counts), max(token_counts))
    return generated_texts, int(sum(token_counts))


def save_results(dataset, generated_per_prompt, section_dir, model, num_generation):
    result_dataset = dataset.add_column('generated_seq', generated_per_prompt)
    output_dir = os.path.join(section_dir, 'model_gen', model)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'generated_data_{num_generation}.parquet')
    result_dataset.to_parquet(output_path)
    logging.info("Saved %d rows (%d generations each) to: %s",
                 len(result_dataset), num_generation, output_path)


def dp_worker(rank, gpu_id, args_dict, unique_prompts_shard, K,
              generation_buffer, min_tokens, output_file):
    import traceback as tb
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_logging(script_path=f"dp_rank{rank}")
    args = argparse.Namespace(**args_dict)
    try:
        import time
        n_unique = len(unique_prompts_shard)
        logging.info("DP rank %d (GPU %d): %d unique x %d reps", rank, gpu_id, n_unique, K)
        t0 = time.time(); llm, cfg = load_vllm_model(args); load_s = time.time() - t0
        sampling_params = create_sampling_params(args, generation_buffer, min_tokens)
        flat_prompts = build_interleaved_prompts(unique_prompts_shard, K)
        t1 = time.time(); flat_outputs, gen_tokens = generate_predictions(llm, flat_prompts, sampling_params); gen_s = time.time() - t1
        generated_per_prompt = regroup_flat_outputs(flat_outputs, n_unique, K)
        with open(output_file, 'w') as f:
            json.dump(generated_per_prompt, f)
        with open(output_file + ".stats.json", 'w') as f:      # inference-cost record, same fields as the readout
            json.dump(dict(load_s=load_s, gen_s=gen_s, gen_tokens=gen_tokens, engine_cfg=cfg), f)
        logging.info("DP rank %d: saved %d prompt results (load %.0fs, generate %.0fs)", rank, len(generated_per_prompt), load_s, gen_s)
    except Exception:
        error_msg = tb.format_exc()
        logging.error("DP rank %d failed:\n%s", rank, error_msg)
        with open(output_file + ".error", 'w') as f:
            f.write(error_msg)
        raise
    sleep(1)


def run_dp_generation(prompts, args, K, generation_buffer, min_tokens, tmp_base):
    import subprocess
    dp_size = args.dp_size
    try:
        nv_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        n_gpus = len(nv_out.strip().splitlines())
    except Exception:
        n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
    if n_gpus < dp_size:
        raise RuntimeError(f"dp_size={dp_size} but only {n_gpus} GPU(s) visible.")
    logging.info("Data parallelism: %d replicas across %d GPUs", dp_size, n_gpus)

    n_unique = len(prompts)
    shards = []
    for rank in range(dp_size):
        u_start = rank * n_unique // dp_size
        u_end = (rank + 1) * n_unique // dp_size
        shards.append(prompts[u_start:u_end])
        logging.info("DP rank %d: %d unique x %d reps", rank, u_end - u_start, K)

    os.makedirs(tmp_base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="vllm_dp_", dir=tmp_base)
    args_dict = vars(args).copy()

    try:
        ctx = mp.get_context("spawn")
        procs, tmp_files = [], []
        for rank in range(dp_size):
            tmp_file = os.path.join(tmp_dir, f"rank_{rank}.json")
            tmp_files.append(tmp_file)
            p = ctx.Process(target=dp_worker,
                            args=(rank, rank, args_dict, shards[rank], K,
                                  generation_buffer, min_tokens, tmp_file))
            p.start()
            procs.append(p)

        from vllm_engine import join_fail_fast   # shared with readout_instruct_vllm.py
        exit_code = join_fail_fast(procs)

        if exit_code != 0:
            for rank in range(dp_size):
                err_file = tmp_files[rank] + ".error"
                if os.path.exists(err_file):
                    with open(err_file) as f:
                        logging.error("DP rank %d traceback:\n%s", rank, f.read())
            raise RuntimeError(f"Data parallel workers failed (exit code {exit_code})")

        generated_per_prompt, stats = [], []
        for rank, tmp_file in enumerate(tmp_files):
            with open(tmp_file) as f:
                generated_per_prompt.extend(json.load(f))
            with open(tmp_file + ".stats.json") as f:
                stats.append(json.load(f))
        assert len(generated_per_prompt) == n_unique
        logging.info("Total gathered: %d prompts", len(generated_per_prompt))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ranks ran concurrently: load / generate = slowest rank; tokens = sum
    return generated_per_prompt, dict(load_s=max(s["load_s"] for s in stats), gen_s=max(s["gen_s"] for s in stats),
                                      gen_tokens=sum(s["gen_tokens"] for s in stats), engine_cfg=stats[0]["engine_cfg"])


def run_single_gpu_generation(prompts, args, generation_buffer, min_tokens, K):
    """One engine per section (loaded INSIDE the section timer, exactly like the DP ranks and
    the readout), so every timing.json wall means the same thing: load + generate."""
    import time
    t0 = time.time(); llm, cfg = load_vllm_model(args); load_s = time.time() - t0
    sampling_params = create_sampling_params(args, generation_buffer, min_tokens)
    flat_prompts = build_interleaved_prompts(prompts, K) if K > 1 else prompts
    t1 = time.time(); flat_outputs, gen_tokens = generate_predictions(llm, flat_prompts, sampling_params); gen_s = time.time() - t1
    out = regroup_flat_outputs(flat_outputs, len(prompts), K) if K > 1 else [[t] for t in flat_outputs]
    del llm
    return out, dict(load_s=load_s, gen_s=gen_s, gen_tokens=gen_tokens, engine_cfg=cfg)


def main():
    args = parse_args()
    log_file = setup_logging()
    logging.info("Arguments: %s", vars(args))

    base_dir = args.processed_data_base_dir
    sections = ["retrospective", "prospective"] if args.section == "both" else [args.section]
    logging.info("Sections: %s | base dir: %s", sections, base_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    K = max(args.num_generation, 1)
    timing_totals = []

    for section in sections:
        section_dir = os.path.join(base_dir, section)
        input_path = os.path.join(section_dir, 'preprocessed_input.parquet')
        if not os.path.exists(input_path):
            logging.warning("[%s] Input not found, skipping: %s", section, input_path)
            continue

        logging.info("=" * 60)
        logging.info("[%s] Starting generation", section.upper())
        logging.info("=" * 60)

        dataset, generation_buffer, min_tokens, prompt_tokens = prepare_dataset(section_dir, args, tokenizer)
        prompts = dataset['context_seq']
        if not prompts:
            logging.warning("[%s] Dataset empty - skipping", section)
            continue
        logging.info("[%s] Sample prompt (first 200 chars): %s...", section, prompts[0][:200])

        # ---- inference-cost record: SAME fields and the same wall definition (engine load +
        # generation, per section) as the post-train readout (e3_posttrain/readout_instruct_vllm.py),
        # so results/test/inference_cost*.csv compares pretrain-MC and post-train like for like.
        from inference_cost import SectionTimer, timing_record, write_timing, log_evaluation_total
        n = max(args.dp_size, 1)
        with SectionTimer() as T:
            if n > 1:
                tmp_base = os.path.join(base_dir, "_vllm_tmp")
                generated_per_prompt, st = run_dp_generation(
                    prompts, args, K, generation_buffer, min_tokens, tmp_base)
            else:
                generated_per_prompt, st = run_single_gpu_generation(
                    prompts, args, generation_buffer, min_tokens, K)
        gen_tok = st["gen_tokens"] / max(len(prompts), 1)          # engine-counted tokens, all K trajectories
        timing_totals.append(
                     timing_record(model=args.model, section=section, engine=f"vllm dp={n}",
                                   patients=len(prompts), wall_s=T.wall_s, n_gpus=n,
                                   tokens_generated_per_patient=round(gen_tok, 1), prompt_tokens=prompt_tokens,
                                   trajectories=K, tokens_generated_per_trajectory=round(gen_tok / K, 1),
                                   engine_load_s=round(st["load_s"], 1), generate_s=round(st["gen_s"], 1),
                                   sampling=dict(temperature=args.temperature if K > 1 else 0.0,   # K == 1 decodes greedily (create_sampling_params); record what ran
                                                 top_p=args.top_p if K > 1 else None, greedy=(K == 1),
                                                 max_tokens=generation_buffer, min_tokens=min_tokens),
                                   engine_cfg=st["engine_cfg"]))
        # per-N file (timing_<N>.json), never the shared timing.json: each N is its own model id downstream
        write_timing(os.path.join(section_dir, "model_gen", args.model), timing_totals[-1], filename=f"timing_{K}.json")

        if generated_per_prompt and generated_per_prompt[0]:
            logging.info("[%s] Sample output (first 200 chars): %s...",
                         section, generated_per_prompt[0][0][:200])

        save_results(dataset, generated_per_prompt, section_dir, args.model,
                     args.num_generation)
        logging.info("[%s] Section completed", section.upper())

    from inference_cost import log_evaluation_total
    log_evaluation_total(args.model, timing_totals)
    logging.info("Generation completed successfully!")
    logging.info("Log file: %s", log_file)


if __name__ == "__main__":
    main()
