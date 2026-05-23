import os
import re
import argparse
from pathlib import Path
from glob import glob
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import PreTrainedTokenizerFast
import pyarrow.parquet as pq
from datasets import load_dataset

def parse_args():	
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=" ")
    p.add_argument("--file_dir", type=str, default=" ")
    p.add_argument("--data_version", type=str, default="v6")
    p.add_argument("--data_source", type=str, default="full_combined")
    p.add_argument("--data_size", type=int, default=100000)
    p.add_argument("--tokenizer_dir", type=str, default=" ")
    p.add_argument("--split_by", type=str, default="visit", choices=["visit", "first_year", "demo", "year", "random"])
    p.add_argument("--instruct_type", type=str, default="split", choices=["split", "mask"])
    p.add_argument("--instruct_position", type=str, default="after", choices=["before", "after"])
    return p.parse_args()

args = parse_args()

data_version = args.data_version
data_source = args.data_source
data_size = args.data_size
split_by = args.split_by
instruct_type = args.instruct_type
instruct_position = args.instruct_position

# 0. Set up HF cache directory
base_dir = args.base_dir
cache_dir = os.path.join(base_dir, "hfcache")
Path(cache_dir).mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
print(f"[INFO] HF Cache dir set to {cache_dir}")

# 1. Load the data
file_dir = args.file_dir
file_path = os.path.join(base_dir, file_dir, "traj_train_pd.parquet")
raw_train = load_dataset("parquet", data_files={"train": file_path}, split="train")

# 2. Check if any split_point > 4096
if instruct_type == "split":
    def keep_example(sp):
        return sp < 4096

    raw_train = raw_train.filter(
        keep_example,
        input_columns=["split_point"], 
        num_proc=60,
        desc=f"filter train split_point < {4096}",
    )

# 3. Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
print(f"Tokenizer loaded from {args.tokenizer_dir}")

# 4. Tokenize the data and build labels
def tokenize_post_train_data(ex):
    text = ex["instruct_seq"]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=4096,
        padding=False,
    )

    input_ids = enc["input_ids"]
    labels = [-100] * len(input_ids)

    if instruct_type == "split":
        sp = int(ex["split_point"])
        sp = min(sp, len(input_ids))
        labels[sp:] = input_ids[sp:]

    elif instruct_type == "mask":
        dx_positions = ex["dx_positions"]
        for pos in dx_positions:
            if pos < 4096:
                labels[pos] = input_ids[pos]

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }

tokenized_train = raw_train.map(
    tokenize_post_train_data,
    remove_columns=raw_train.column_names,
    num_proc=60,
    desc="tokenizing",
)

# 5. Save the data
save_folder = os.path.join(base_dir, f"tokens/tokens_{instruct_type}_size_{data_size}_pos_{instruct_position}_split_{split_by}")
Path(save_folder).mkdir(parents=True, exist_ok=True)
tokenized_train.save_to_disk(save_folder, num_proc=60)
print(f"Tokenized training dataset saved to {save_folder}")

