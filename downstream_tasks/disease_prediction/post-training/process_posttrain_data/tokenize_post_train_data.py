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
    p.add_argument("--tok_dir", type=str, default=" ")
    p.add_argument("--base_dir", type=str, default=" ")
    p.add_argument("--file_dir", type=str, default=" ")
    p.add_argument("--token_folder", type=str, default=" ")
    p.add_argument("--data_version", type=str, default="vf")
    p.add_argument("--data_source", type=str, default="1000k")
    p.add_argument("--data_size", type=int, default=100000)
    p.add_argument("--split_by", type=str, default="visit", choices=["visit", "first_year", "demo", "random"])
    p.add_argument("--instruct_position", type=str, default="after", choices=["before", "after"])
    return p.parse_args()

args = parse_args()

data_version = args.data_version
data_source = args.data_source
data_size = args.data_size
split_by = args.split_by
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
def keep_example(sp):
    return sp < 4096

raw_train = raw_train.filter(
    keep_example,
    input_columns=["split_point"], 
    num_proc=60,
    desc=f"filter train split_point < {4096}",
)

# 3. Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tok_dir)

# 4. Tokenize the data and build labels
MAX_SEQ_LEN = 4096

def tokenize_post_train_data(ex):
    text = ex["instruct_seq"]
    sp = int(ex["split_point"])

    enc = tokenizer(
        text,
        truncation=False,
        padding=False,
    )

    input_ids = enc["input_ids"]
    sp = min(sp, len(input_ids))

    sup_start = min(sp + 1, len(input_ids))

    labels = [-100] * len(input_ids)
    labels[sup_start:] = input_ids[sup_start:]

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "seq_len": len(input_ids),
        "n_supervised": len(input_ids) - sup_start,
    }

tokenized_train = raw_train.map(
    tokenize_post_train_data,
    remove_columns=raw_train.column_names,
    num_proc=60,
    desc="tokenizing",
)

n_before = len(tokenized_train)
tokenized_train = tokenized_train.filter(
    lambda seq_len, n_supervised: (seq_len <= MAX_SEQ_LEN) and (n_supervised > 0),
    input_columns=["seq_len", "n_supervised"],
    num_proc=60,
    desc=f"drop seq_len > {MAX_SEQ_LEN} or no supervised tokens",
)
print(f"Kept {len(tokenized_train)}/{n_before} examples after length/label filter")
tokenized_train = tokenized_train.remove_columns(["seq_len", "n_supervised"])

# 5. Save the data
save_folder = os.path.join(base_dir, f"{args.token_folder}/tokens_size_{data_size}_pos_{instruct_position}_split_{split_by}")
Path(save_folder).mkdir(parents=True, exist_ok=True)
tokenized_train.save_to_disk(save_folder, num_proc=60)
print(f"Tokenized training dataset saved to {save_folder}")
