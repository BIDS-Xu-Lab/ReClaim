#!/usr/bin/env python3
"""
Script to prepare disease case-control IDs from evaluation data.
Processes patient sequences and generates case-control pairs for different diseases.
"""

import pandas as pd
import numpy as np
import sys
import argparse
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.data_ops import (
    get_major_disease_id_map,
    get_disease_case_control_ids_from_precomputed,
    _precompute_disease_case_control_shared,
    prepare_eval_inputs,
)


class DiseasePredictionDataset(Dataset):
    """Dataset for disease prediction."""

    def __init__(self, data, tokenizer, seq_len, padding_length=200):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.padding_length = padding_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        seq = item["seq"]
        disease_points = item["disease_points"]
        disease_points_ages = item["disease_points_ages"]
        if "disease_points_visit_location" in item:
            disease_points_visit_location = item["disease_points_visit_location"]
        else:
            disease_points_visit_location = None
        assert len(disease_points) == len(disease_points_ages)
        padded_points = np.zeros(self.padding_length, dtype=int)
        padded_points_ages = (
            np.ones(self.padding_length, dtype=float) * 1000 * 365.25
        )  ## max age 1000 years
        if self.padding_length < len(disease_points):
            disease_points = disease_points[:self.padding_length]
            disease_points_ages = disease_points_ages[:self.padding_length]
        else:
            padded_points[:len(disease_points)] = disease_points
            padded_points_ages[:len(disease_points_ages)] = disease_points_ages
        if disease_points_visit_location is not None:
            padded_points_visit_location = np.zeros(self.padding_length, dtype=int)
            if self.padding_length < len(disease_points_visit_location):
                disease_points_visit_location = disease_points_visit_location[:self.padding_length]
            else:
                padded_points_visit_location[:len(disease_points_visit_location)] = disease_points_visit_location
        else:
            padded_points_visit_location = None

        try:
            patient_id = item["enrollee_id"]
        except Exception:
            patient_id = item["person_id"]

        # Tokenization will happen in batch (for padding efficiency)
        return {
            "seq": seq,
            "disease_points": padded_points,
            "disease_points_ages": padded_points_ages,
            "disease_points_visit_location": padded_points_visit_location,
            "patient_id": patient_id,
            "idx": idx,
        }


def collate_fn(batch, tokenizer, seq_len):
    """Collate function for batching."""
    seqs = [item["seq"] for item in batch]
    disease_points = np.stack([item["disease_points"] for item in batch])
    disease_points_ages = np.stack([item["disease_points_ages"] for item in batch])
    disease_points_visit_location = (
        np.stack([item["disease_points_visit_location"] for item in batch])
        if any("disease_points_visit_location" in item for item in batch)
        else None
    )
    patient_ids = [item["patient_id"] for item in batch]
    indices = [item["idx"] for item in batch]

    # Batch tokenization
    batch_tokens = tokenizer(
        seqs,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )

    return {
        "tokens": batch_tokens,
        "disease_points": disease_points,
        "disease_points_ages": disease_points_ages,
        "disease_points_visit_location": disease_points_visit_location,
        "patient_ids": patient_ids,
        "indices": indices,
    }


def process_disease(disease_ind, precomputed):
    """Process a single disease and return case-control IDs using precomputed shared data."""
    try:
        outs = get_disease_case_control_ids_from_precomputed(disease_ind, precomputed)
        return outs
    except Exception as e:
        print(f"Error processing disease {disease_ind}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Prepare disease case-control IDs from evaluation data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the parquet file containing trajectory data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="disease_case_control_ids.parquet",
        help="Output parquet filename (default: disease_case_control_ids.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for the parquet file (default: current directory)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for data loading (default: 128)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=365.25,
        help="Offset in days for prediction (default: 365.25)",
    )
    parser.add_argument(
        "--age_index",
        type=int,
        default=3,
        help="Index where age starts (default: 3)",
    )
    parser.add_argument(
        "--age_group_range",
        type=str,
        default="45,80,5",
        help="Range of ages to process (default: 45,80,5)",
    )
    parser.add_argument(
        "--sex_index",
        type=int,
        default=1,
        help="Index where sex information is located (default: 1)",
    )
    parser.add_argument(
        "--icd_code_map_file",
        type=str,
        default=None,
        help="Path to CSV file with ICD codes to process. If None, uses all major diseases (default: None)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for processing different diseases in parallel (default: 4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader (default: 0)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    age_start, age_end, age_step = map(int, args.age_group_range.split(","))
    age_groups = np.arange(age_start, age_end, age_step)

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Loading data from: {args.data_dir}")
    data = pd.read_parquet(args.data_dir)
    print(f"Loaded {len(data)} samples")

    dataset = DiseasePredictionDataset(data, tokenizer, args.seq_len, padding_length=600)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.seq_len),
    )

    print("Preparing evaluation inputs...")
    sex_token_id = tokenizer.encode("<SEX-1>")[0]
    all_disease_tokens, all_ages, all_seq_points, all_sex = prepare_eval_inputs(
        dataloader, tokenizer, args.sex_index, sex_token_id
    )
    print(f"Prepared inputs for {len(all_disease_tokens)} samples")

    # Precompute shared data once (expensive O(n*m^2) work) - reused for all diseases
    print("Precomputing shared case-control data...")
    precomputed = _precompute_disease_case_control_shared(
        all_disease_tokens,
        all_ages,
        all_seq_points,
        all_sex,
        age_groups=age_groups,
        offset=args.offset,
    )
    print("Precomputation done.")

    # Get list of diseases to process
    if args.icd_code_map_file is None:
        print("Using all major diseases from tokenizer")
        diseases_id_map = get_major_disease_id_map(tokenizer)
        diseases_inds = list(diseases_id_map.values())
    else:
        print(f"Loading ICD codes from: {args.icd_code_map_file}")
        icd_code_map = pd.read_csv(args.icd_code_map_file)
        icd_codes = icd_code_map["ICD code"].tolist()
        diseases_inds = [
            tokenizer.encode(f"<DX-MAJOR_{icd_code}>")[0]
            for icd_code in icd_codes
            if tokenizer.encode(f"<DX-MAJOR_{icd_code}>")[0] != 0
        ]
    
    print(f"Processing {len(diseases_inds)} diseases using {args.num_threads} threads")

    # Process diseases in parallel using threads (each uses shared precomputed data)
    all_outs = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        future_to_disease = {
            executor.submit(process_disease, disease_ind, precomputed): disease_ind
            for disease_ind in diseases_inds
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_disease), total=len(diseases_inds), desc="Processing diseases"):
            disease_ind = future_to_disease[future]
            try:
                outs = future.result()
                all_outs.extend(outs)
            except Exception as e:
                print(f"Error processing disease {disease_ind}: {e}")

    print(f"Generated {len(all_outs)} case-control pairs")

    # Save results
    outs_df = pd.DataFrame(all_outs)
    output_path = os.path.join(args.output_dir, args.output_file)
    print(f"Saving results to: {output_path}")
    outs_df.to_parquet(output_path, index=False)
    print(f"Successfully saved {len(outs_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
