#!/usr/bin/env python3
"""
Zero-shot inference script for cost prediction task using vLLM.
Uses context_seq from preprocessed data to generate predictions.
Leverages vLLM for high-throughput inference with continuous batching.

Key design choice: uses n=1 with repeated prompts (interleaved) instead of
n=K.  This allows far more concurrent requests in the KV cache, dramatically
improving GPU utilization on large-memory GPUs (H200).  Prefix caching
ensures repeated identical prompts reuse the already-computed KV, so the
"extra" prefill cost is negligible.
"""

import os
import argparse
import json
import logging
import tempfile
import multiprocessing as mp
from time import sleep

from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer

from utils import setup_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Zero-shot inference for cost prediction using vLLM')
    parser.add_argument("--processed_data_base_dir", type=str, 
                       default=os.path.join(
                           os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                           "evaluation",
                           "versions",
                           "v6",
                           "data",
                       ),
                       help='Base directory containing processed data')
    parser.add_argument('--model', type=str,
                       default='v6-s-pretrain',
                       help='Model')
    parser.add_argument("--model_path", type=str, 
                       default=os.environ.get("RECLAIM_MODEL_PATH", "/path/to/reclaim_model_checkpoint"),
                       help='Path to pretrained model')
    parser.add_argument("--num_generation", type=int, default=1,
                       help='Number of repetitions for each sample (for sampling diversity)')
    parser.add_argument("--seq_len", type=int, default=4096,
                       help='Maximum sequence length (model context length)')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help='GPU memory utilization (0.0-1.0)')
    parser.add_argument("--temperature", type=float, default= 1.0,
                       help='Sampling temperature (only used if num_generation > 1)')
    parser.add_argument("--top_p", type=float, default= 1.0,
                       help='Top-p sampling (only used if num_generation > 1)')
    parser.add_argument("--dtype", type=str, default='auto',
                       choices=['auto', 'float16', 'bfloat16', 'float32'],
                       help='Data type for model weights')
    parser.add_argument("--seed", type=int, default=66,
                       help='Random seed for reproducibility')
    parser.add_argument("--min_tokens", type=int, default=2,
                       help='Minimum tokens (ATT and 1+ event token) to generate (prevents immediate EOS)')
    parser.add_argument("--dp_size", type=int, default=1,
                       help='Number of data parallel replicas (each uses 1 GPU). Set >1 to shard prompts across GPUs.')
    parser.add_argument("--section", type=str, default="both",
                       choices=["both", "retrospective", "prospective"],
                       help='Which section(s) to generate for (default: both)')
    
    return parser.parse_args()


def load_vllm_model(args):
    """Load model using vLLM"""
    logging.info("Loading model from: %s", args.model_path)
    logging.info("GPU memory utilization: %s", args.gpu_memory_utilization)
    logging.info("Max model length: %s", args.seq_len)
    
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.seq_len,
        dtype=args.dtype,
        seed=args.seed,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    
    logging.info("Model loaded successfully")
    return llm


def prepare_dataset(section_dir, args, tokenizer):
    """Load and prepare dataset, truncating sequences that exceed max length"""
    input_path = os.path.join(section_dir, 'preprocessed_input.parquet')
    logging.info("Loading data from: %s", input_path)
    
    # Load directly into HuggingFace Dataset
    dataset = Dataset.from_parquet(input_path)
    logging.info("Loaded %d sequences", len(dataset))
    logging.info("Columns: %s", dataset.column_names)
    
    # Check for required column
    if 'context_seq' not in dataset.column_names:
        raise ValueError("Data must contain 'context_seq' column as the input")
 
    if "posttrain" in args.model.lower():
        generation_buffer = 48  # theoretically no more than 36 tokens
        min_tokens = args.min_tokens
    else:
        generation_buffer = 512  # for pretrain models
        min_tokens = args.min_tokens
    # Left-truncate sequences to leave room for generation (keep most recent context)
    max_prompt_len = args.seq_len - generation_buffer
    logging.info("Left-truncating sequences to max %d tokens (reserving %d for generation based on model=%s)", max_prompt_len, generation_buffer, args.model)
    
    def truncate_sequence(example):
        # Tokenize without truncation first
        tokens = tokenizer(
            example['context_seq'], 
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = tokens['input_ids']
        was_truncated = len(input_ids) > max_prompt_len
        
        # Left truncation: keep the last max_prompt_len tokens
        if was_truncated:
            input_ids = input_ids[-max_prompt_len:]
        
        truncated_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        return {'context_seq': truncated_text, '_was_truncated': was_truncated}
    
    dataset = dataset.map(truncate_sequence, desc="Truncating sequences")
    
    truncated_count = sum(dataset['_was_truncated'])
    dataset = dataset.remove_columns(['_was_truncated'])
    
    if truncated_count > 0:
        logging.warning("Left-truncated %d sequences to %d tokens", truncated_count, max_prompt_len)
    
    logging.info("Total unique samples: %d", len(dataset))
    
    return dataset, generation_buffer, min_tokens


def create_sampling_params(args, generation_buffer, min_tokens):
    """Create vLLM SamplingParams.

    Always uses n=1.  When num_generation > 1, prompts are repeated externally
    so that each copy is a separate request (with n=1).  This keeps only one
    decode sequence per request in the KV cache, allowing far more concurrent
    requests → higher GPU utilization.
    """
    max_tokens = generation_buffer

    if args.num_generation > 1:
        # Sampling for diversity (n=1 per request, prompts repeated externally)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        logging.info("Sampling mode (n=1, %d repeated copies): temperature=%s, top_p=%s, min_tokens=%d, max_tokens=%d",
                     args.num_generation, args.temperature, args.top_p, min_tokens, max_tokens)
    else:
        # Greedy decoding for single generation
        sampling_params = SamplingParams(
            temperature=0,  # Greedy
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        logging.info("Greedy decoding (num_generation=1), min_tokens=%d, max_tokens=%d", min_tokens, max_tokens)

    logging.info("Generation: stops at EOS or when hitting max_tokens (%d)", max_tokens)
    return sampling_params


def build_interleaved_prompts(unique_prompts, K):
    """Repeat each prompt K times, interleaved so copies are adjacent.

    Returns (flat_prompts, prompt_indices) where:
    - flat_prompts: [p0, p0, ..., p0 (K), p1, p1, ..., p1 (K), ...]
    - prompt_indices: [0]*K + [1]*K + ... (maps each flat position to its unique prompt index)

    Adjacent identical prompts maximise vLLM prefix-cache hits.
    """
    flat_prompts = []
    prompt_indices = []
    for i, p in enumerate(unique_prompts):
        flat_prompts.extend([p] * K)
        prompt_indices.extend([i] * K)
    return flat_prompts, prompt_indices


def regroup_flat_outputs(flat_texts, n_unique, K):
    """Regroup a flat list of N*K texts into a list-of-lists (one inner list of K per unique prompt)."""
    assert len(flat_texts) == n_unique * K, \
        f"Expected {n_unique * K} outputs, got {len(flat_texts)}"
    grouped = []
    for i in range(n_unique):
        grouped.append(flat_texts[i * K : (i + 1) * K])
    return grouped


def generate_predictions(llm, prompts, sampling_params):
    """Generate predictions using vLLM with n=1.

    Returns a flat list of generated strings (one per prompt in the input list).
    """
    logging.info("Starting inference on %d prompts (vLLM continuous batching, n=1)",
                 len(prompts))
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    generated_texts = [output.outputs[0].text for output in outputs]
    logging.info("Generated %d outputs", len(generated_texts))

    # Diagnostics
    finish_reasons = {}
    empty_count = 0
    token_counts = []

    for i, output in enumerate(outputs):
        comp = output.outputs[0]
        reason = comp.finish_reason
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        num_tokens = len(comp.token_ids)
        token_counts.append(num_tokens)
        if comp.text == '' or comp.text.strip() == '':
            empty_count += 1
            if empty_count <= 5:
                prompt_preview = prompts[i][-200:] if len(prompts[i]) > 200 else prompts[i]
                logging.warning("Empty output #%d (idx=%d): finish_reason=%s, tokens=%d, prompt_end='%s'",
                                empty_count, i, reason, num_tokens, prompt_preview)

    logging.info("Finish reasons distribution: %s", finish_reasons)
    logging.info("Empty outputs: %d/%d (%.1f%%)", empty_count, len(outputs),
                 100 * empty_count / len(outputs) if outputs else 0)
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        non_empty_tokens = [t for t in token_counts if t > 0]
        avg_non_empty = sum(non_empty_tokens) / len(non_empty_tokens) if non_empty_tokens else 0
        logging.info("Output token stats: avg=%.1f, avg_non_empty=%.1f, min=%d, max=%d",
                     avg_tokens, avg_non_empty, min(token_counts), max(token_counts))

    return generated_texts


def save_results(dataset, generated_per_prompt, section_dir, model, num_generation):
    """Save results to parquet.

    Each row keeps N rows (one per unique prompt).  The ``generated_seq``
    column is a list of K strings (one per generation) for that prompt.
    """
    result_dataset = dataset.add_column('generated_seq', generated_per_prompt)

    output_dir = os.path.join(section_dir, 'model_gen', model)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'generated_data_{num_generation}.parquet')
    result_dataset.to_parquet(output_path)
    logging.info("Saved %d rows (%d generations each) to: %s",
                 len(result_dataset), num_generation, output_path)


def dp_worker(rank, gpu_id, args_dict, unique_prompts_shard, num_reps,
              generation_buffer, min_tokens, output_file):
    """Worker for a single data-parallel rank.

    Repeats each unique prompt K times (interleaved), generates with n=1,
    then regroups into list-of-lists.  Writes JSON to *output_file*.
    """
    import traceback as tb

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_logging(script_path=f"dp_rank{rank}")
    args = argparse.Namespace(**args_dict)

    try:
        n_unique = len(unique_prompts_shard)
        K = num_reps
        logging.info("DP rank %d (GPU %d): %d unique prompts × %d reps = %d total (n=1, interleaved)",
                     rank, gpu_id, n_unique, K, n_unique * K)

        llm = load_vllm_model(args)
        sampling_params = create_sampling_params(args, generation_buffer, min_tokens)

        # Build interleaved prompt list for optimal prefix-cache reuse
        flat_prompts, _ = build_interleaved_prompts(unique_prompts_shard, K)
        flat_outputs = generate_predictions(llm, flat_prompts, sampling_params)

        # Regroup flat results into list-of-K per unique prompt
        generated_per_prompt = regroup_flat_outputs(flat_outputs, n_unique, K)

        with open(output_file, 'w') as f:
            json.dump(generated_per_prompt, f)

        logging.info("DP rank %d: saved %d prompt results to %s", rank, len(generated_per_prompt), output_file)
    except Exception:
        error_msg = tb.format_exc()
        logging.error("DP rank %d failed:\n%s", rank, error_msg)
        with open(output_file + ".error", 'w') as f:
            f.write(error_msg)
        raise

    sleep(1)


def resolve_sections(section_arg):
    """Return the list of sections to process."""
    if section_arg == "both":
        return ["retrospective", "prospective"]
    return [section_arg]


def run_dp_generation(prompts, args, K, generation_buffer, min_tokens):
    """Run generation with data parallelism (multiple GPUs)."""
    import subprocess

    dp_size = args.dp_size
    try:
        nv_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        n_gpus = len(nv_out.strip().splitlines())
    except Exception:
        n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
    if n_gpus < dp_size:
        raise RuntimeError(
            f"dp_size={dp_size} but only {n_gpus} GPU(s) visible. "
            f"Request at least {dp_size} GPUs (e.g. --gres=gpu:{dp_size} in SLURM)."
        )
    logging.info("Using data parallelism with %d replicas across %d GPUs", dp_size, n_gpus)

    n_unique = len(prompts)

    shards = []
    for rank in range(dp_size):
        u_start = rank * n_unique // dp_size
        u_end = (rank + 1) * n_unique // dp_size
        shards.append(prompts[u_start:u_end])
        logging.info("DP rank %d (GPU %d): %d unique prompts × %d reps = %d total",
                     rank, rank, u_end - u_start, K, (u_end - u_start) * K)

    tmp_base = os.environ.get("VLLM_TMP_ROOT", "/tmp/reclaim_cost_vllm")
    os.makedirs(tmp_base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="vllm_dp_", dir=tmp_base)
    args_dict = vars(args).copy()

    try:
        ctx = mp.get_context("spawn")
        procs = []
        tmp_files = []
        for rank in range(dp_size):
            tmp_file = os.path.join(tmp_dir, f"rank_{rank}.json")
            tmp_files.append(tmp_file)
            p = ctx.Process(
                target=dp_worker,
                args=(rank, rank, args_dict, shards[rank], K,
                      generation_buffer, min_tokens, tmp_file),
            )
            p.start()
            procs.append(p)

        exit_code = 0
        for p in procs:
            p.join(timeout=172800)
            if p.exitcode is None:
                logging.error("Killing DP rank process %d (timed out after 48h)", p.pid)
                p.kill()
                exit_code = 1
            elif p.exitcode != 0:
                exit_code = p.exitcode

        if exit_code != 0:
            for rank in range(dp_size):
                err_file = tmp_files[rank] + ".error"
                if os.path.exists(err_file):
                    with open(err_file) as f:
                        logging.error("DP rank %d traceback:\n%s", rank, f.read())
            raise RuntimeError(f"Data parallel workers failed (exit code {exit_code})")

        generated_per_prompt = []
        for rank, tmp_file in enumerate(tmp_files):
            with open(tmp_file) as f:
                shard_results = json.load(f)
            generated_per_prompt.extend(shard_results)
            logging.info("Gathered %d prompt results from DP rank %d", len(shard_results), rank)

        assert len(generated_per_prompt) == n_unique, \
            f"Expected {n_unique} prompt results, got {len(generated_per_prompt)}"
        logging.info("Total gathered: %d prompts", len(generated_per_prompt))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return generated_per_prompt


def run_single_gpu_generation(prompts, llm, sampling_params, K):
    """Run generation on a single GPU (model already loaded)."""
    if K > 1:
        n_unique = len(prompts)
        flat_prompts, _ = build_interleaved_prompts(prompts, K)
        logging.info("Expanded %d unique prompts × %d reps = %d total (interleaved, n=1)",
                     n_unique, K, len(flat_prompts))
        flat_outputs = generate_predictions(llm, flat_prompts, sampling_params)
        return regroup_flat_outputs(flat_outputs, n_unique, K)
    else:
        flat_outputs = generate_predictions(llm, prompts, sampling_params)
        return [[t] for t in flat_outputs]


def main():
    """Main function for generating predictions"""
    args = parse_args()

    log_file = setup_logging()
    logging.info("Logging to: %s", log_file)
    logging.info("Arguments: %s", vars(args))

    if args.num_generation > 1:
        logging.info("Generating %d sequences with temperature %s and top_p %s",
                     args.num_generation, args.temperature, args.top_p)
    else:
        logging.info("Generating 1 sequence using greedy decoding")

    base_dir = args.processed_data_base_dir
    sections = resolve_sections(args.section)
    logging.info("Sections to process: %s", sections)
    logging.info("Base data directory: %s", base_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    dp_size = args.dp_size
    K = max(args.num_generation, 1)

    # For single-GPU, load the model once and reuse across sections
    llm = None
    if dp_size <= 1:
        llm = load_vllm_model(args)

    for section in sections:
        section_dir = os.path.join(base_dir, section)
        input_path = os.path.join(section_dir, 'preprocessed_input.parquet')
        if not os.path.exists(input_path):
            logging.warning("[%s] Input not found, skipping: %s", section, input_path)
            continue

        logging.info("=" * 60)
        logging.info("[%s] Starting generation", section.upper())
        logging.info("=" * 60)

        dataset, generation_buffer, min_tokens = prepare_dataset(section_dir, args, tokenizer)

        prompts = dataset['context_seq']
        if "posttrain" in args.model.lower():
            prompts = [p + " <INSTRUCT-COST>" for p in prompts]
            logging.info("[%s] Appended ' <INSTRUCT-COST>' to %d prompts (posttrain model).",
                         section, len(prompts))
        if len(prompts) == 0:
            logging.warning("[%s] Dataset is empty - skipping", section)
            continue
        logging.info("[%s] Sample prompt (first 200 chars): %s...", section, prompts[0][:200])

        if dp_size > 1:
            generated_per_prompt = run_dp_generation(
                prompts, args, K, generation_buffer, min_tokens)
        else:
            sampling_params = create_sampling_params(args, generation_buffer, min_tokens)
            generated_per_prompt = run_single_gpu_generation(
                prompts, llm, sampling_params, K)

        if generated_per_prompt and generated_per_prompt[0]:
            logging.info("[%s] Sample output (first 200 chars): %s...",
                         section, generated_per_prompt[0][0][:200])

        save_results(dataset, generated_per_prompt, section_dir, args.model, args.num_generation)
        logging.info("[%s] Section completed", section.upper())

    logging.info("Zero-shot inference completed successfully!")
    logging.info("Log file: %s", log_file)


if __name__ == "__main__":
    main()
