#!/usr/bin/env python3
"""
Feature engineering for cost prediction task.

Reads preprocessed data (context_seq, future_seq) and a vocabulary file to
construct:
  - Outcome: next_annual_cost (sum of COST tokens in future_seq up to next <NY>)
  - Features:
      * Demographics: sex, age
      * Clinical: binary indicators (present/absent) of DX/RX tokens from
        vocab, extracted over a configurable history window ("1year" or "all")

Uses HuggingFace datasets for parallel processing and sklearn CountVectorizer
for efficient sparse token counting, then binarises counts to 0/1.

Column names are sanitised (angle brackets removed) for downstream model
compatibility (e.g. XGBoost).

Outputs a single parquet file with the feature matrix and outcome column.
"""

import os
import argparse
import logging
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

from utils import setup_logging


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature engineering for cost prediction")
    parser.add_argument("--processed_data_dir", type=str, required=True,
        help="Path to preprocessed_input.parquet from preprocess step")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"],
        help="Split to process (default: train)")
    parser.add_argument( "--vocab_path", type=str, required=True,
        help="Path to vocab CSV/parquet with columns: category, token")
    parser.add_argument("--history_length", type=str, default="1year",
        choices=["1year", "all"],
        help="How much context history to use for clinical features (default: 1year)")
    parser.add_argument("--token_feature_type", type=str, default="binary",
        choices=["binary", "count"],
        help="Token feature type: binary (0/1) or count (default: binary)")
    parser.add_argument("--num_proc", type=int, default=None,
        help="Number of worker processes (default: cpu_count - 1)")
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def sanitise_column_name(name: str) -> str:
    """Remove angle brackets from a column name for downstream compatibility."""
    return name.replace("<", "").replace(">", "")


# =============================================================================
# Cost parsing (used for prev_cost feature)
# =============================================================================
def parse_cost_value(token: str) -> int:
    """Parse a single COST token to an integer dollar amount."""
    if token == "<COST-MISSING>":
        return 0
    if not (token.startswith("<COST-") and token.endswith(">")):
        return 0
    cost_code = token[6:-1]
    if len(cost_code) >= 2:
        mantissa = int(cost_code[0])
        exponent = int(cost_code[1:])
        return mantissa * (10 ** exponent)
    return 0


# =============================================================================
# Demographics
# =============================================================================
def extract_demographics(example: dict) -> dict:
    """Extract sex and age from context_seq."""
    tokens = example["context_seq"].split()

    sex = -1
    age_at_anchor = -1
    n_ny_after_age = 0
    seen_age = False

    for t in tokens:
        if t.startswith("<SEX-") and t.endswith(">"):
            val = t[5:-1]
            sex = 1 if val == "1" else (0 if val == "2" else -1)
        elif t.startswith("<AGE-") and t.endswith(">"):
            try:
                age_at_anchor = int(t[5:-1])
                seen_age = True
            except ValueError:
                pass
        elif seen_age and t.startswith("<NY") and t.endswith(">"):
            n_ny_after_age += 1

    age = age_at_anchor + n_ny_after_age if age_at_anchor >= 0 else -1
    return {"sex": sex, "age": age}


# =============================================================================
# History windowing
# =============================================================================

def extract_relevant_tokens(example: dict, history_length: str) -> dict:
    """
    Extract the relevant tokens from context_seq.

    "1year" : tokens between the second-to-last <NY> and end of context_seq.
    "all"   : tokens after <DOBYR-...> through end of context_seq.

    The returned string can be fed directly to CountVectorizer.
    """
    tokens = example["context_seq"].strip().split()

    if history_length == "1year":
        ny_positions = [i for i, t in enumerate(tokens) if t.startswith("<NY")]
        if len(ny_positions) >= 2:
            start = ny_positions[-2] + 1
        else:
            return {"relevant_tokens": "", "prev_cost": 0}
        relevant_tokens = tokens[start:]

    elif history_length == "all":
        start_idx = 2 # dobyr suppose to be 2
        for i, t in enumerate(tokens):
            if t.startswith("<DOBYR-"):
                start_idx = i + 1
                break
        relevant_tokens = tokens[start_idx:]

    else:
        raise ValueError(f"Unsupported history_length: {history_length}")

    # Extract and aggregate COST tokens from relevant_tokens
    prev_cost = 0
    for token in relevant_tokens:
        if token.startswith("<COST-"):
            prev_cost += parse_cost_value(token)

    return {"relevant_tokens": " ".join(relevant_tokens), "prev_cost": prev_cost}


# =============================================================================
# Vocab loading
# =============================================================================
def load_vocab_features(vocab_path: str) -> list[str]:
    """Load vocab  and return sorted list of relevant tokens."""
    vocab_ds = load_dataset("parquet", data_files=os.path.join(vocab_path, "*.parquet"), split="train")
    logging.info("  Vocab: loaded %d rows", len(vocab_ds))

    # filter to relevant categories and exclude special tokens
    special_tokens = {
        "<DX-MISSING>", "<DX-NOMAP>", "<DX-PRINCIPAL>", "<DX-SECONDARY>",
        "<RX-MISSING>", "<RX-NOMAP>", "<RX-COMBSTART>", "<RX-COMBEND>"
    }
    
    # only keep ICD10 major codes
    vocab_relevant = vocab_ds.filter(
        lambda x: (
            x["token"] is not None
            and (
                (x["token"].startswith("<DX-MAJOR_") and re.match(r"<DX-MAJOR_[A-Z]", x["token"])) or
                (x["category"] == "RX" and x["token"] not in special_tokens)
            )
        )
    )

    relevant_tokens = sorted(set(vocab_relevant["token"]))
    logging.info("  Vocab: %d DX/RX feature tokens", len(relevant_tokens))
    return relevant_tokens


def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("FEATURE ENGINEERING STARTING...")

    args = parse_args()
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("  Config: history_length=%s, token_feature_type=%s, num_proc=%d",
                 args.history_length, args.token_feature_type, num_proc)
    logging.info("  Input:  %s", args.processed_data_dir)
    logging.info("  Vocab:  %s", args.vocab_path)
    logging.info("  Output: %s", args.processed_data_dir)

    # ------------------------------------------------------------------
    # 1. Load preprocessed data as a HF Dataset
    # ------------------------------------------------------------------
    ds = load_dataset("parquet", data_files=os.path.join(args.processed_data_dir, args.split, "preprocessed_input.parquet"), split="train")
    logging.info("  Loaded %d rows", len(ds))

    VT_TYPES = ("inpatient", "outpatient", "pharmacy")
    TARGET_COLS = ["next_annual_cost"] + [f"next_annual_cost_{vt}" for vt in VT_TYPES]

    for col in ("context_seq", "future_seq", "enrollee_id", "next_annual_cost"):
        if col not in ds.column_names:
            raise ValueError(f"Missing required column: {col}")

    available_targets = [c for c in TARGET_COLS if c in ds.column_names]
    logging.info("  Available target columns: %s", available_targets)

    # ------------------------------------------------------------------
    # 2. Extract demographics (parallel)
    # ------------------------------------------------------------------
    ds = ds.map(extract_demographics, num_proc=num_proc, desc="Extracting demographics")

    # ------------------------------------------------------------------
    # 4. Extract history window as text (parallel)
    # ------------------------------------------------------------------
    ds = ds.map(
        lambda ex: extract_relevant_tokens(ex, args.history_length),
        num_proc=num_proc,
        desc=f"Extracting relevant tokens ({args.history_length})"
    )

    # ------------------------------------------------------------------
    # 5. Load vocab → build fixed-vocabulary CountVectorizer
    # ------------------------------------------------------------------
    feature_tokens = load_vocab_features(args.vocab_path)
    token2idx = {t: i for i, t in enumerate(feature_tokens)}

    # CountVectorizer with a fixed vocabulary.
    # We use a simple whitespace tokenizer since tokens are already
    # space-delimited and may contain special chars like < > -.
    vectorizer = CountVectorizer(
        vocabulary=token2idx,
        tokenizer=str.split,       # simple whitespace split
        lowercase=False,            # tokens are case-sensitive
        token_pattern=None,         # disable default regex (using tokenizer)
    )

    # ------------------------------------------------------------------
    # 6. Vectorize clinical tokens → binarise or keep counts
    # ------------------------------------------------------------------
    logging.info("  Vectorizing clinical tokens with CountVectorizer...")
    history_texts = ds["relevant_tokens"]  # list of strings
    clinical_sparse = vectorizer.transform(history_texts)  # scipy sparse matrix
    logging.info("  Clinical matrix shape: %s, nnz: %d",
                 clinical_sparse.shape, clinical_sparse.nnz)

    # Apply feature type transformation
    if args.token_feature_type == "binary":
        # Binarise: any count > 0 becomes 1
        clinical_sparse = (clinical_sparse > 0).astype(np.int32)
        logging.info("  Binarised clinical features (count -> 0/1)")
    else:  # count
        # Keep counts as-is
        clinical_sparse = clinical_sparse.astype(np.int32)
        logging.info("  Kept clinical feature counts")

    # ------------------------------------------------------------------
    # 7. Assemble final DataFrame
    # ------------------------------------------------------------------
    logging.info("  Assembling final DataFrame...")

    # Sanitise clinical column names: <DX-MAJOR_A01> → DX-MAJOR_A01
    feature_names = [sanitise_column_name(t) for t in feature_tokens]

    result_df = pd.DataFrame({
        "enrollee_id": ds["enrollee_id"],
        "sex": ds["sex"],
        "age": ds["age"],
        "prev_cost": ds["prev_cost"],
    })

    # Convert sparse matrix to DataFrame
    clinical_df = pd.DataFrame.sparse.from_spmatrix(
        clinical_sparse, columns=feature_names
    )
    # Use int32 for count features, int8 for binary features
    dtype = np.int32 if args.token_feature_type == "count" else np.int8
    clinical_df = clinical_df.sparse.to_dense().astype(dtype)

    result_df = pd.concat([result_df, clinical_df], axis=1)
    for tc in available_targets:
        result_df[tc] = ds[tc]

    # ------------------------------------------------------------------
    # 8. Summary stats
    # ------------------------------------------------------------------
    logging.info("  Result shape: %s", result_df.shape)
    for tc in available_targets:
        logging.info("  %-40s mean=%.2f  median=%.2f  max=%d  zero_frac=%.3f",
                     tc, result_df[tc].mean(), result_df[tc].median(),
                     result_df[tc].max(), (result_df[tc] == 0).mean())

    clinical_nonzero = (clinical_sparse > 0).sum(axis=1).A1
    logging.info("  Clinical features per patient: mean=%.1f  median=%.1f  max=%d",
                 clinical_nonzero.mean(), np.median(clinical_nonzero),
                 clinical_nonzero.max())

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.join(args.processed_data_dir, args.split), exist_ok=True)
    out_path = os.path.join(args.processed_data_dir, args.split, f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet")
    result_df.to_parquet(out_path, index=False)
    logging.info("  Saved to: %s", out_path)

    logging.info("FEATURE ENGINEERING COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
