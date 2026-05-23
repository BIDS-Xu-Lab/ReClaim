import os
import re
import argparse
from pathlib import Path
from glob import glob
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def main():
    parser = argparse.ArgumentParser(description="Preprocess text data for pretraining")
    parser.add_argument("--input_folder_path", type=str, required=True, help="Path to the input folder containing corpus and tokenizer")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for BERT input")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for tokenization")
    args = parser.parse_args()

    input_folder_path = args.input_folder_path
    max_seq_len = args.max_seq_len
    num_proc = args.num_proc

    print("max_seq_len set to ", max_seq_len)

    data_version = input_folder_path.split("/")[-1]
    data_source = input_folder_path.split("/")[-2]
    base_dir = f"***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/{data_source}_{data_version}_seqlen_{max_seq_len}"

    cache_dir = os.path.join(base_dir, "hfcache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    print(f"[INFO] HF Cache dir set to {cache_dir}")

    from datasets import load_dataset

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    # Save tokenizer to data directory
    tokenizer_copy_path = f"{base_dir}/tokenizer"
    Path(tokenizer_copy_path).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_copy_path)

    # Inspect vocab
    df = pd.DataFrame(list(tokenizer.vocab.items()), columns=["token", "id"])
    print(f"Tokenizer vocab size: {len(df)}")

    # Load train/val files
    train_root = Path(f"{input_folder_path}/final_data/train/")
    train_files = [str(p) for p in list(train_root.rglob("*.parquet"))]

    val_root = Path(f"{input_folder_path}/final_data/val/")
    val_files = [str(p) for p in list(val_root.rglob("*.parquet"))]

    print(f"Found {len(train_files)} train files and {len(val_files)} val files")

    # Load dataset
    raw_train = load_dataset("parquet", data_files={"train": train_files}, split="train")
    raw_val = load_dataset("parquet", data_files={"val": val_files}, split="val")

    # Prepare save folders
    train_token_folder = f"{base_dir}/data/pre_training_data/train_tokens"
    val_token_folder = f"{base_dir}/data/pre_training_data/val_tokens"
    Path(train_token_folder).mkdir(parents=True, exist_ok=True)
    Path(val_token_folder).mkdir(parents=True, exist_ok=True)

    # Tokenization
    def tokenize(batch):
        return {"input_ids": tokenizer(batch["seq"], add_special_tokens=False)["input_ids"]}

    # Group into blocks of max_seq_len
    def group(batch):
        all_ids = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
        blocks = [all_ids[i:i+max_seq_len] for i in range(0, len(all_ids), max_seq_len)]
        return {"input_ids": blocks}

    print(f"CPU count (using): {num_proc}")

    # Process training data
    toked_train = raw_train.map(tokenize, batched=True, batch_size=20000, num_proc=num_proc, remove_columns=raw_train.column_names)
    packed_train = toked_train.map(group, batched=True, batch_size=20000, num_proc=num_proc)

    # Process validation data
    toked_val = raw_val.map(tokenize, batched=True, batch_size=20000, num_proc=num_proc, remove_columns=raw_val.column_names)
    packed_val = toked_val.map(group, batched=True, batch_size=20000, num_proc=num_proc)

    # Save processed datasets
    packed_train.save_to_disk(train_token_folder, num_proc=num_proc)
    packed_val.save_to_disk(val_token_folder, num_proc=num_proc)
    print(f"Tokenized training dataset saved to {train_token_folder}")
    print(f"Tokenized validation dataset saved to {val_token_folder}")


if __name__ == "__main__":

    main()
