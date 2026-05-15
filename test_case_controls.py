#!/usr/bin/env python3
"""
Script to evaluate AUC for disease prediction from patient sequences.
Transformed from get_logit.ipynb, now supports multi-GPU processing.

OPTIMIZED VERSION:
- Pre-computes ALL unique (patient_id, seq_point) logits in batched mode
- Uses global cache to avoid duplicate computation across diseases
- Proper GPU batching for maximum throughput
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.evaluate_auc import get_auc_delong_var 
from collections import defaultdict
import torch.multiprocessing as mp
from functools import partial

def aggregate_normals(group):
    """Aggregate normal distributions for AUC variance calculation."""
    n = len(group)
    mean = group["auc_delong"].mean()
    return pd.Series(
        {
            "auc": mean,
            "n_samples": n,
            "n_diseased": group["n_diseased"].sum(),
            "n_healthy": group["n_healthy"].sum(),
        }
    )

def get_major_disease_id_map(tokenizer):
    """Get mapping of major disease tokens to their token IDs."""
    major_disease_ids_map = {}
    for token, idx in tokenizer.get_vocab().items():
        if token.startswith("<DX-MAJOR"):
            major_disease_ids_map[token] = idx
    major_disease_ids_map = dict(
        sorted(major_disease_ids_map.items(), key=lambda item: item[1])
    )
    return major_disease_ids_map

def get_predict_seq(data, patient_id, seq_point):
    """Get the prediction sequence for a patient at a given sequence point."""
    seq = data.loc[patient_id, "eval_seq"]
    return " ".join(seq.split(" ")[:seq_point])

def get_predict_seq_list(data, indices, points):
    """Get list of prediction sequences."""
    seqs = data.iloc[indices]["eval_seq"]
    seq_list = []
    for seq, seq_point in zip(seqs, points):
        seq_list.append(" ".join(seq.split(" ")[:seq_point]))
    return seq_list


class UniqueSequenceDataset(Dataset):
    """Dataset for unique (row_index, seq_point) pairs."""
    
    def __init__(self, unique_pairs, data, last_str):
        """
        Args:
            unique_pairs: List of (row_index, seq_point) tuples where row_index is an integer
            data: DataFrame with patient data
            last_str: String to append to each sequence
        """
        self.unique_pairs = unique_pairs
        self.data = data
        self.last_str = last_str
    
    def __len__(self):
        return len(self.unique_pairs)
    
    def __getitem__(self, idx):
        row_index, seq_point = self.unique_pairs[idx]
        
        # Get the sequence using iloc (row_index is an integer index into data)
        seq = self.data.iloc[row_index]["eval_seq"]
        
        # Truncate to seq_point and append last_str (same as original get_predict_seq_list)
        truncated_seq = " ".join(seq.split(" ")[:seq_point]) + self.last_str
        
        return {
            "seq": truncated_seq,
            "row_index": row_index,
            "seq_point": seq_point,
            "idx": idx
        }


def collate_fn_unique(batch, tokenizer, max_length=4096):
    """Collate function for unique sequence batching."""
    seqs = [item["seq"] for item in batch]
    row_indices = [item["row_index"] for item in batch]
    seq_points = [item["seq_point"] for item in batch]
    indices = [item["idx"] for item in batch]

    # Batch tokenization with padding
    tokens = tokenizer(
        seqs,
        return_tensors="pt",
        max_length=max_length,
        padding="longest",  # Pad to the longest sequence in the batch
    )

    return {
        "tokens": tokens,
        "row_indices": row_indices,
        "seq_points": seq_points,
        "indices": indices,
    }


def collect_unique_sequences_and_diseases(case_control_ids_list, data):
    """
    Collect all unique (row_index, seq_point) pairs and disease token indices.
    case_ids and control_ids are row indices into the data DataFrame.
    """
    unique_pairs = set()
    disease_tokens = set()
    
    for disease_info in case_control_ids_list:
        case_ids = disease_info["case_ids"]
        case_seq_points = disease_info["case_seq_points"]
        control_ids = disease_info["control_ids"]
        control_seq_points = disease_info["control_seq_points"]
        disease_token = disease_info["disease_token"]
        
        # Collect disease token
        disease_tokens.add(disease_token)
        
        # Add case pairs (row_index, seq_point)
        for row_idx, sp in zip(case_ids, case_seq_points):
            unique_pairs.add((row_idx, sp))
        
        # Add control pairs (row_index, seq_point)
        for row_idx, sp in zip(control_ids, control_seq_points):
            unique_pairs.add((row_idx, sp))
    
    disease_tokens = sorted(list(disease_tokens))  # Sort for consistent ordering
    print(f"Collected {len(unique_pairs)} unique (row_index, seq_point) pairs")
    print(f"Collected {len(disease_tokens)} unique disease tokens")
    return list(unique_pairs), disease_tokens


def gpu_worker(gpu_id, model_path, batches_data, tokenizer_path, disease_tokens, result_queue=None, progress_queue=None, max_length=4096):
    """
    Worker function that runs in a separate process for each GPU.
    Only extracts logits for the specified disease tokens to save memory.
    """
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # Report loading status
    if progress_queue:
        progress_queue.put(("status", gpu_id, f"Loading model on GPU {gpu_id}..."))
    
    # Load model in this process
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer in this process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Convert disease_tokens to tensor for efficient indexing
    disease_tokens_tensor = torch.tensor(disease_tokens, device=device)
    
    if progress_queue:
        progress_queue.put(("status", gpu_id, f"GPU {gpu_id} ready, processing {len(batches_data)} batches"))
    
    results = []
    total_batches = len(batches_data)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(batches_data):
            seqs, row_indices, seq_points = batch_data
            
            # Tokenize
            tokens = tokenizer(
                seqs,
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Forward pass
            outputs = model(**tokens)
            
            # Get logits for last real token
            attention_mask = tokens["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1)
            last_token_indices = seq_lengths - 1
            
            batch_size_actual = outputs.logits.shape[0]
            last_token_logits = outputs.logits[
                torch.arange(batch_size_actual, device=device),
                last_token_indices,
                :
            ]
            
            # Only extract logits for disease tokens (saves memory!)
            disease_logits = last_token_logits[:, disease_tokens_tensor]
            
            # Move to CPU and store
            disease_logits_cpu = disease_logits.cpu().numpy()
            
            for i, (row_idx, sp) in enumerate(zip(row_indices, seq_points)):
                results.append((row_idx, sp, disease_logits_cpu[i]))
            
            # Report progress
            if progress_queue:
                progress_queue.put(("progress", gpu_id, batch_idx + 1, total_batches))
    
    # Put results in queue
    result_queue.put((gpu_id, results))
    
    if progress_queue:
        progress_queue.put(("done", gpu_id, len(results)))
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


def precompute_all_logits(
    unique_pairs, 
    disease_tokens,
    data, 
    tokenizer, 
    model_path, 
    last_str,
    num_gpus, 
    batch_size=8,
    max_length=4096
):
    """
    Pre-compute logits for all unique sequences using parallel multi-GPU inference.
    Only saves logits for the specified disease tokens to save memory.
    
    Returns:
        logits_cache: dict mapping (row_index, seq_point) -> logits array (only disease tokens)
        disease_tokens: list of disease token indices (for mapping back)
    """
    print(f"Pre-computing logits for {len(unique_pairs)} unique sequences...")
    print(f"Only saving logits for {len(disease_tokens)} disease tokens (memory efficient)")
    
    # Prepare all batch data (sequences as strings, not tensors - for multiprocessing)
    # Group unique_pairs into batches
    all_batch_data = []
    
    # Create data index for fast lookup
    for i in range(0, len(unique_pairs), batch_size):
        batch_pairs = unique_pairs[i:i+batch_size]
        seqs = []
        row_indices = []
        seq_points = []
        
        for row_idx, sp in batch_pairs:
            seq = data.iloc[row_idx]["eval_seq"]
            truncated_seq = " ".join(seq.split(" ")[:sp]) + last_str
            seqs.append(truncated_seq)
            row_indices.append(row_idx)
            seq_points.append(sp)
        
        all_batch_data.append((seqs, row_indices, seq_points))
    
    print(f"Created {len(all_batch_data)} batches of size {batch_size}")
    
    # Create disease token to index mapping
    disease_token_to_idx = {token: idx for idx, token in enumerate(disease_tokens)}
    
    if num_gpus <= 1:
        # Single GPU or CPU: sequential processing
        device = "cuda:0" if num_gpus == 1 else "cpu"
        print(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if num_gpus == 1:
            model = model.cuda()
        model.eval()
        
        # Convert disease_tokens to tensor for efficient indexing
        disease_tokens_tensor = torch.tensor(disease_tokens, device=device)
        
        logits_cache = {}
        
        with torch.no_grad():
            for batch_data in tqdm(all_batch_data, desc="Computing logits"):
                seqs, row_indices, seq_points = batch_data
                
                tokens = tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True,
                )
                if num_gpus == 1:
                    tokens = {k: v.cuda() for k, v in tokens.items()}
                
                outputs = model(**tokens)
                
                attention_mask = tokens["attention_mask"]
                seq_lengths = attention_mask.sum(dim=1)
                last_token_indices = seq_lengths - 1
                
                batch_size_actual = outputs.logits.shape[0]
                last_token_logits = outputs.logits[
                    torch.arange(batch_size_actual, device=outputs.logits.device),
                    last_token_indices,
                    :
                ]
                
                # Only extract logits for disease tokens
                disease_logits = last_token_logits[:, disease_tokens_tensor]
                disease_logits_cpu = disease_logits.cpu().numpy()
                
                for i, (row_idx, sp) in enumerate(zip(row_indices, seq_points)):
                    logits_cache[(row_idx, sp)] = disease_logits_cpu[i]
        
        del model
        if num_gpus > 0:
            torch.cuda.empty_cache()
        
        return logits_cache, disease_token_to_idx
    
    # Multi-GPU: parallel processing using multiprocessing
    print(f"Running parallel inference on {num_gpus} GPUs...")
    
    # Distribute batches to GPUs
    gpu_batches = [[] for _ in range(num_gpus)]
    for batch_idx, batch_data in enumerate(all_batch_data):
        gpu_id = batch_idx % num_gpus
        gpu_batches[gpu_id].append(batch_data)
    
    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_batches[gpu_id])} batches")
    
    # Use multiprocessing for parallel execution
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, model_path, gpu_batches[gpu_id], model_path, disease_tokens, result_queue, progress_queue, max_length),
        )
        p.start()
        processes.append(p)
    
    # Track progress from all GPUs
    logits_cache = {}
    results_collected = 0
    gpu_progress = {gpu_id: 0 for gpu_id in range(num_gpus)}
    gpu_totals = {gpu_id: len(gpu_batches[gpu_id]) for gpu_id in range(num_gpus)}
    total_batches = len(all_batch_data)
    gpus_done = set()
    
    # Create progress bar
    pbar = tqdm(total=total_batches, desc="Computing logits")
    
    # Monitor progress and results
    while results_collected < num_gpus:
        # Check for progress updates (non-blocking)
        try:
            while True:
                msg = progress_queue.get_nowait()
                if msg[0] == "status":
                    _, gpu_id, status_msg = msg
                    tqdm.write(f"  {status_msg}")
                elif msg[0] == "progress":
                    _, gpu_id, batch_done, batch_total = msg
                    old_progress = gpu_progress[gpu_id]
                    gpu_progress[gpu_id] = batch_done
                    pbar.update(batch_done - old_progress)
                elif msg[0] == "done":
                    _, gpu_id, num_results = msg
                    gpus_done.add(gpu_id)
        except:
            pass
        
        # Check for results (non-blocking with small timeout)
        try:
            gpu_id, results = result_queue.get(timeout=0.1)
            for row_idx, sp, logits in results:
                logits_cache[(row_idx, sp)] = logits
            results_collected += 1
            tqdm.write(f"  GPU {gpu_id} completed: {len(results)} sequences")
        except:
            pass
    
    pbar.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print(f"Pre-computed {len(logits_cache)} logit vectors")
    return logits_cache, disease_token_to_idx


def compute_auc_from_cache(disease_info, logits_cache, disease_token_to_idx):
    """
    Compute AUC for a single disease using pre-computed logits.
    case_ids and control_ids are row indices into the data DataFrame.
    
    Args:
        disease_info: dict with disease_token, case_ids, control_ids, etc.
        logits_cache: dict mapping (row_idx, seq_point) -> logits array (only disease tokens)
        disease_token_to_idx: dict mapping disease_token -> index in logits array
    """
    disease_token = disease_info["disease_token"]
    case_ids = disease_info["case_ids"]  # These are row indices
    case_seq_points = disease_info["case_seq_points"]
    control_ids = disease_info["control_ids"]  # These are row indices
    control_seq_points = disease_info["control_seq_points"]
    sex = disease_info["sex"]
    age = disease_info["age"]
    
    # Get the index for this disease token in the reduced logits array
    disease_idx = disease_token_to_idx.get(disease_token)
    if disease_idx is None:
        print(f"Warning: Disease token {disease_token} not found in disease_token_to_idx")
        return None
    
    # Look up case logits using (row_index, seq_point) as cache key
    case_logits = []
    for row_idx, sp in zip(case_ids, case_seq_points):
        logit_vector = logits_cache.get((row_idx, sp))
        if logit_vector is not None:
            case_logits.append(logit_vector[disease_idx])
        else:
            print(f"Warning: Missing logits for case (row={row_idx}, sp={sp})")
    
    # Look up control logits using (row_index, seq_point) as cache key
    control_logits = []
    for row_idx, sp in zip(control_ids, control_seq_points):
        logit_vector = logits_cache.get((row_idx, sp))
        if logit_vector is not None:
            control_logits.append(logit_vector[disease_idx])
        else:
            print(f"Warning: Missing logits for control (row={row_idx}, sp={sp})")
    
    if len(case_logits) == 0 or len(control_logits) == 0:
        return None
    
    # Compute AUC
    auc_value_delong, auc_variance_delong = get_auc_delong_var(
        control_logits, case_logits
    )
    
    return {
        "auc_delong": auc_value_delong,
        "sex": sex,
        "disease_token": disease_token,
        "age": age,
        "n_diseased": len(case_logits),
        "n_healthy": len(control_logits),
        "case_logits": case_logits,
        "control_logits": control_logits,
    }


def evaluate_pipeline(args, tokenizer, data, case_control_ids_list_of_dict, device, num_gpus, model_path):
    """Main evaluation pipeline with optimized batched inference."""
    
    last_str = args.last_str
    # last_str = " <NY> <ATT-0> <VT-outpatient> <DX-PRINCIPAL>"
    aucs = []
    logits_data = []

    logits_file = os.path.join(args.output_dir, "logits.pkl")
    logits_cache_file = os.path.join(args.output_dir, "logits_cache.pkl")
    logits_exists = os.path.isfile(logits_file)
    logits_cache_exists = os.path.isfile(logits_cache_file)

    if args.skip_logits and logits_exists:
        print(f"Skipping logit computation, loading logits from {logits_file}.")
        with open(logits_file, "rb") as f:
            logits_data = pickle.load(f)
        # Calculate AUCs using loaded logits
        for out, logits in zip(case_control_ids_list_of_dict, logits_data):
            disease_token = out["disease_token"]
            sex = out["sex"]
            age = out["age"]
            case_ids = out["case_ids"]
            control_ids = out["control_ids"]
            case_logits = logits["case_logits"]
            control_logits = logits["control_logits"]
            auc_value_delong, auc_variance_delong = get_auc_delong_var(
                control_logits, case_logits
            )
            auc_result = {
                "auc_delong": auc_value_delong,
                "sex": sex,
                "disease_token": disease_token,
                "age": age,
                "n_diseased": len(case_ids),
                "n_healthy": len(control_ids),
            }
            aucs.append(auc_result)
    else:
        print("Computing logits using optimized batched inference...")
        
        # Check if we have a cached logits_cache
        if args.skip_logits and logits_cache_exists:
            print(f"Loading pre-computed logits cache from {logits_cache_file}...")
            with open(logits_cache_file, "rb") as f:
                cache_data = pickle.load(f)
                logits_cache = cache_data["logits_cache"]
                disease_token_to_idx = cache_data["disease_token_to_idx"]
        else:
            # Step 1: Collect all unique (patient_id, seq_point) pairs and disease tokens
            unique_pairs, disease_tokens = collect_unique_sequences_and_diseases(case_control_ids_list_of_dict, data)
            
            # Step 2: Pre-compute logits only for disease tokens (memory efficient)
            logits_cache, disease_token_to_idx = precompute_all_logits(
                unique_pairs=unique_pairs,
                disease_tokens=disease_tokens,
                data=data,
                tokenizer=tokenizer,
                model_path=model_path,
                last_str=last_str,
                num_gpus=num_gpus,
                batch_size=args.batch_size if hasattr(args, 'batch_size') else 8,
                max_length=args.max_length if hasattr(args, 'max_length') else 4096,
            )
            
            # Save logits cache for future use
            print(f"Saving logits cache to {logits_cache_file}...")
            with open(logits_cache_file, "wb") as f:
                pickle.dump({
                    "logits_cache": logits_cache,
                    "disease_token_to_idx": disease_token_to_idx,
                }, f)
        
        # Step 3: Compute AUCs using cached logits
        print("Computing AUCs from cached logits...")
        for disease_info in tqdm(case_control_ids_list_of_dict, desc="Computing AUCs"):
            result = compute_auc_from_cache(disease_info, logits_cache, disease_token_to_idx)
            if result is not None:
                aucs.append({k: v for k, v in result.items() if k not in ["case_logits", "control_logits"]})
                logits_data.append({
                    "case_logits": result["case_logits"],
                    "control_logits": result["control_logits"],
                })
        
        # Save logits to file
        with open(logits_file, "wb") as f:
            pickle.dump(logits_data, f)
        print(f"Saved logits to {logits_file}")

    aucs_df = pd.DataFrame(aucs)
    auc_df = (
        aucs_df.groupby(["disease_token"])
        .apply(aggregate_normals, include_groups=False)
        .reset_index()
    )
    output_file = os.path.join(args.output_dir, "auc_results.csv")
    auc_df.to_csv(output_file, index=False)
    print(f"Saved AUC results to {output_file}")

    output_file_unpooled = os.path.join(args.output_dir, "auc_results_unpooled.csv")
    aucs_df.to_csv(output_file_unpooled, index=False)
    print(f"Saved unpooled AUC results to {output_file_unpooled}")

    print(f"\nSummary:")
    print(f"  Total diseases evaluated: {len(auc_df)}")
    print(f"  Mean AUC: {auc_df['auc'].mean():.4f}")
    print(f"  Diseases with AUC < 0.5: {(auc_df['auc'] < 0.5).sum()}")
    return auc_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate AUC for disease prediction")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to CSV file with patient sequences",
    )
    parser.add_argument(
        "--sample_frac", type=float, default=1.0, help="Fraction of data to sample"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--case_control_ids_file", type=str, required=True, help="Path to case control IDs file"
    )
    parser.add_argument(
        "--skip_logits",
        action="store_true",
        help="Skip logit computation if already done",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")',
    )
    parser.add_argument(
        "--auc_workers",
        type=int,
        default=8,
        help="Number of parallel workers for AUC calculation (default: number of CPU cores)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU for inference (default: 8)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization (default: 4096)",
    )
    parser.add_argument(
        "--last_str",
        type=str,
        default=" <NY> <INSTRUCT-DX>",
        help='Suffix appended to each truncated sequence (default: " <NY> <INSTRUCT-DX>")',
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process data
    data = pd.read_parquet(args.data_file)
    print(f"Loaded {len(data)} samples from {args.data_file}")

    if args.sample_frac < 1.0:
        # Select the first N fraction of data for fast debugging
        n = int(len(data) * args.sample_frac)
        data = data.iloc[:n]

    # Load tokenizer (not models here as we will load per worker for each device)
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding (same as original)
    tokenizer.truncation_side = "right"  # Right truncation (same as original)
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # GPU setup
    if torch.cuda.is_available():
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
            num_gpus = len(gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            device = "cuda:0"
            print(f"Using GPUs: {gpu_ids} (visible as cuda:0-{num_gpus-1})")
        elif args.num_gpus:
            num_gpus = min(args.num_gpus, torch.cuda.device_count())
            device = "cuda:0"
            print(f"Using {num_gpus} GPUs")
        else:
            num_gpus = torch.cuda.device_count()
            device = "cuda:0" if num_gpus > 0 else "cpu"
            if num_gpus > 0:
                print(f"Using all available {num_gpus} GPUs")
    else:
        num_gpus = 0
        device = "cpu"
        print("No CUDA available, using CPU")

    case_control_ids_list_of_dict = pd.read_parquet(args.case_control_ids_file)
    # If loaded as DataFrame, convert each row to dict
    if isinstance(case_control_ids_list_of_dict, pd.DataFrame):
        case_control_ids_list_of_dict = case_control_ids_list_of_dict.to_dict(orient='records')

    auc_df = evaluate_pipeline(args, tokenizer, data, case_control_ids_list_of_dict, device, num_gpus, args.model_path)

if __name__ == "__main__":
    main()
