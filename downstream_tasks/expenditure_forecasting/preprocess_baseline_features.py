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
    parser.add_argument("--split", type=str, default="test_100k", choices=["test_100k", "test_1m"],
        help="Split to process (default: test_100k)")
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


def run_feature_pipeline(section: str, args, feature_tokens: list[str],
                         token2idx: dict, num_proc: int):
    """Run the full feature engineering pipeline for one section."""
    logging.info("=" * 60)
    logging.info("  [%s] Feature engineering", section.upper())
    logging.info("=" * 60)

    VT_TYPES = ("inpatient", "outpatient", "pharmacy")
    TARGET_COLS = ["next_annual_cost"] + [f"next_annual_cost_{vt}" for vt in VT_TYPES]

    # ------------------------------------------------------------------
    # 1. Load preprocessed data
    # ------------------------------------------------------------------
    input_path = os.path.join(args.processed_data_dir, args.split, section, "preprocessed_input.parquet")
    if not os.path.exists(input_path):
        logging.warning("  [%s] Input not found, skipping: %s", section, input_path)
        return
    ds = load_dataset("parquet", data_files=input_path, split="train")
    logging.info("  [%s] Loaded %d rows", section, len(ds))

    for col in ("context_seq", "future_seq", "enrollee_id", "next_annual_cost"):
        if col not in ds.column_names:
            raise ValueError(f"[{section}] Missing required column: {col}")

    available_targets = [c for c in TARGET_COLS if c in ds.column_names]
    logging.info("  [%s] Available target columns: %s", section, available_targets)

    # ------------------------------------------------------------------
    # 2. Extract demographics (parallel)
    # ------------------------------------------------------------------
    ds = ds.map(extract_demographics, num_proc=num_proc,
                desc=f"[{section}] Extracting demographics")

    # ------------------------------------------------------------------
    # 3. Extract history window as text (parallel)
    # ------------------------------------------------------------------
    ds = ds.map(
        lambda ex: extract_relevant_tokens(ex, args.history_length),
        num_proc=num_proc,
        desc=f"[{section}] Extracting relevant tokens ({args.history_length})"
    )

    # ------------------------------------------------------------------
    # 4. Vectorize clinical tokens
    # ------------------------------------------------------------------
    vectorizer = CountVectorizer(
        vocabulary=token2idx,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
    )

    logging.info("  [%s] Vectorizing clinical tokens...", section)
    history_texts = ds["relevant_tokens"]
    clinical_sparse = vectorizer.transform(history_texts)
    logging.info("  [%s] Clinical matrix shape: %s, nnz: %d",
                 section, clinical_sparse.shape, clinical_sparse.nnz)

    if args.token_feature_type == "binary":
        clinical_sparse = (clinical_sparse > 0).astype(np.int32)
        logging.info("  [%s] Binarised clinical features (count -> 0/1)", section)
    else:
        clinical_sparse = clinical_sparse.astype(np.int32)
        logging.info("  [%s] Kept clinical feature counts", section)

    # ------------------------------------------------------------------
    # 5. Assemble final DataFrame
    # ------------------------------------------------------------------
    logging.info("  [%s] Assembling final DataFrame...", section)

    feature_names = [sanitise_column_name(t) for t in feature_tokens]

    result_df = pd.DataFrame({
        "enrollee_id": ds["enrollee_id"],
        "sex": ds["sex"],
        "age": ds["age"],
        "prev_cost": ds["prev_cost"],
    })

    clinical_df = pd.DataFrame.sparse.from_spmatrix(
        clinical_sparse, columns=feature_names
    )
    dtype = np.int32 if args.token_feature_type == "count" else np.int8
    clinical_df = clinical_df.sparse.to_dense().astype(dtype)

    result_df = pd.concat([result_df, clinical_df], axis=1)
    for tc in available_targets:
        result_df[tc] = ds[tc]

    # ------------------------------------------------------------------
    # 6. Summary stats
    # ------------------------------------------------------------------
    logging.info("  [%s] Result shape: %s", section, result_df.shape)
    for tc in available_targets:
        logging.info("  [%s] %-40s mean=%.2f  median=%.2f  max=%d  zero_frac=%.3f",
                     section, tc, result_df[tc].mean(), result_df[tc].median(),
                     result_df[tc].max(), (result_df[tc] == 0).mean())

    clinical_nonzero = (clinical_sparse > 0).sum(axis=1).A1
    logging.info("  [%s] Clinical features per patient: mean=%.1f  median=%.1f  max=%d",
                 section, clinical_nonzero.mean(), np.median(clinical_nonzero),
                 clinical_nonzero.max())

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_dir = os.path.join(args.processed_data_dir, args.split, section)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet")
    result_df.to_parquet(out_path, index=False)
    logging.info("  [%s] Saved to: %s", section, out_path)


def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("FEATURE ENGINEERING STARTING...")

    args = parse_args()
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("  Config: history_length=%s, token_feature_type=%s, num_proc=%d",
                 args.history_length, args.token_feature_type, num_proc)
    logging.info("  Input:  %s", args.processed_data_dir)
    logging.info("  Vocab:  %s", args.vocab_path)

    # ------------------------------------------------------------------
    # Load vocab once (shared across sections)
    # ------------------------------------------------------------------
    feature_tokens = load_vocab_features(args.vocab_path)
    token2idx = {t: i for i, t in enumerate(feature_tokens)}

    # ------------------------------------------------------------------
    # Run pipeline for each section
    # ------------------------------------------------------------------
    for section in ("retrospective", "prospective"):
        run_feature_pipeline(section, args, feature_tokens, token2idx, num_proc)

    logging.info("FEATURE ENGINEERING COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
