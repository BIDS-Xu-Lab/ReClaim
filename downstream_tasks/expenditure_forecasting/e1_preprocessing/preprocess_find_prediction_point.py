#!/usr/bin/env python3
"""
Preprocess sequence data for the expenditure-forecasting task.

Modes
-----
test  : build both evaluation sections from the held-out test cohort
          - retrospective : random middle-<NY> split within seq_thru_cutoff
                            (pre-cutoff years)
          - prospective   : fixed split at the training cutoff boundary
                            (first post-cutoff year)
train : build baseline-development data (random middle-<NY> split on
        seq_thru_cutoff) from a sample of the training corpus, then split
        90/10 into train/val for supervised baselines.

Input format
------------
- Columns: seq_thru_cutoff (history up to the training cutoff) and
  event_seq_from_cutoff (events after it).
- Sequences arrive wrapped as `<sos> ... <eos>`: no <sos> is prepended and the
  trailing <eos> is stripped from context sequences.
- <NY> tokens are bare (no year value); the split logic uses token positions only.
- Enrollees whose first event is post-cutoff have demographics-only
  seq_thru_cutoff (no <AGE> token); they fail the age filter and are excluded.
"""

import os
import hashlib
import argparse
import logging
from glob import glob

import numpy as np
from datasets import load_dataset, Dataset, disable_caching

disable_caching()

from utils import (setup_logging, ID_COL, hash_salt, hash64_sql, smallest_n_sql, duck_connect,
                   select_smallest_ids, corpus_fingerprint, fingerprint_str)
from sequence_parsing import GOLD_COST_COLS, parse_age_value, strip_wrappers, extract_outcome


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess patient sequences for cost prediction task')
    parser.add_argument("--mode", type=str, default='test', choices=['test', 'train'],
                        help='test: evaluation sections; train: baseline train/val data')
    parser.add_argument("--reclaim_data_dir", type=str, required=True,
                        help='data root: final_data/{train,test} + processed_data/vocab')
    parser.add_argument("--sample_n", type=int, default=5_000_000,
                        help='[mode=train] number of patients to sample from final_data/train')
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--min_age", type=int, default=16,
                        help="Minimum ANCHOR age (first observed year); 16 guarantees "
                             "adult (>=18) prediction points -- see the filter comment")
    return parser.parse_args()


def tokenize(sequence: str) -> np.ndarray:
    return np.array(sequence.strip().split())


def _ineligible_result(n_tokens: int) -> dict:
    return {
        'context_seq': None,
        'future_seq': None,
        'n_valid_points': 0,
        'n_tokens': n_tokens,
        'is_eligible': False,
    }


# =============================================================================
# Retrospective processing (random middle-NY split on seq_thru_cutoff)
# =============================================================================

def process_example(example, seed):
    """Split at a random middle <NY> token of seq_thru_cutoff.

    The per-patient RNG is seeded from (seed, hash(enrollee_id)) so the chosen
    prediction point is a property of the patient, independent of file layout,
    row order or worker count (reproducible across clusters).

    All <NY> positions are found; the first and last are excluded.  Excluding
    the first <NY> guarantees at least one full observed calendar year of
    history (the <AGE> anchor only marks January of the first event year, so
    splitting at the first <NY> could leave under a year of claims).  Excluding
    the last <NY> guarantees a complete follow-up year inside the pre-cutoff
    data.  The chosen <NY> token is kept at the end of the context.
    """
    eid_hash = int(hashlib.sha256(example['enrollee_id'].encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng([seed, eid_hash])

    tokens = tokenize(strip_wrappers(example['seq_thru_cutoff']))
    n = len(tokens)

    is_ny = np.char.startswith(tokens, '<NY')
    ny_indices = np.where(is_ny)[0]

    if len(ny_indices) < 3:
        return _ineligible_result(n)

    middle_indices = ny_indices[1:-1]
    n_valid = len(middle_indices)
    split_idx = int(rng.choice(middle_indices))

    return {
        'context_seq': ' '.join(tokens[:split_idx + 1]),
        'future_seq': ' '.join(tokens[split_idx + 1:]),
        'n_valid_points': n_valid,
        'n_tokens': n,
        'is_eligible': True,
    }


# =============================================================================
# Prospective processing (fixed prediction point at the cutoff boundary)
# =============================================================================

def process_example_prospective(example):
    """Fixed split at the training-cutoff year boundary.

    Eligibility:
      - event_seq_from_cutoff is not null and starts with (or contains) <NY>
      - event_seq_from_cutoff contains >= 2 <NY> tokens (a complete first
        post-cutoff outcome year)
      - seq_thru_cutoff contains >= 1 <NY> token (>= 1 year of history)

    context_seq = seq_thru_cutoff (minus <eos>) + first <NY> from
                  event_seq_from_cutoff
    future_seq  = event_seq_from_cutoff after that leading <NY>
    """
    tokens_hist = tokenize(strip_wrappers(example['seq_thru_cutoff']))
    n_tokens = len(tokens_hist)

    seq_fut = example.get('event_seq_from_cutoff')
    if seq_fut is None or (isinstance(seq_fut, float) and np.isnan(seq_fut)):
        return _ineligible_result(n_tokens)

    tokens_fut = tokenize(str(seq_fut))
    n_tokens += len(tokens_fut)

    ny_mask_fut = np.char.startswith(tokens_fut, '<NY')
    ny_mask_hist = np.char.startswith(tokens_hist, '<NY')

    if int(ny_mask_fut.sum()) < 2 or int(ny_mask_hist.sum()) < 1:
        return _ineligible_result(n_tokens)

    first_ny_idx = int(np.where(ny_mask_fut)[0][0])
    ny_token = tokens_fut[first_ny_idx]

    context_seq = ' '.join(tokens_hist) + ' ' + ny_token
    future_seq = ' '.join(tokens_fut[first_ny_idx + 1:])

    return {
        'context_seq': context_seq,
        'future_seq': future_seq,
        'n_valid_points': 0,
        'n_tokens': n_tokens,
        'is_eligible': True,
    }


# =============================================================================
# Shared helpers
# =============================================================================

# --- Content-keyed corpus sampling (train mode) -------------------------------
# The 5M supervision pool is the sample_n patients with the SMALLEST salted
# id-hashes — a property of the ids, not of which part file or row they sit
# in.  DuckDB does the work in two SQL steps over the raw parquet files:
#   1. rank the NARROW id column only -> the n selected ids (a few GB of state);
#   2. stream every row once through a semi-join on those ids -> pool parquet.
# The corpus is never sorted as full rows nor materialised as an arrow cache;
# memory is bounded by the 5M-id join build side, and DuckDB spills to the
# job-local temp dir if it must.  load_dataset then consumes the pool file.


def _duck_for_job(num_proc):
    """DuckDB tuned to the SLURM allocation: all cores, ~75% of the job memory,
    node-local, job-scoped spill dir (never $HOME)."""
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0)
    tmp = os.path.join(os.environ.get("TMPDIR") or "/tmp",   # env.sh sets TMPDIR to the job-local scratch
                       f"duck_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    os.makedirs(tmp, exist_ok=True)
    return duck_connect(threads=num_proc, memory_limit_gb=(mem_mb * 0.75 / 1024) if mem_mb else None, temp_dir=tmp)


def build_hash_pool(parquet_files, sample_n, seed, num_proc, pool_path):
    """Write the sample_n content-keyed pool rows to pool_path; returns (total, n_pool, cutoff)."""
    con = _duck_for_job(num_proc)
    src = f"read_parquet({parquet_files!r})"
    fp = corpus_fingerprint(con, src)
    logging.info("  Corpus fingerprint (layout-independent): %s over %d files", fingerprint_str(fp), len(parquet_files))
    total = fp[0]
    pool_salt = hash_salt(seed, "pool")
    n_pool = min(sample_n, total)
    if sample_n >= total:
        logging.warning("  Corpus (%d) <= sample_n (%d): keeping every row", total, sample_n)
    con.execute(f"CREATE TABLE sel AS {smallest_n_sql(src, n_pool, pool_salt, select=ID_COL)}")
    cutoff = con.execute(f"SELECT max({hash64_sql(pool_salt)}) FROM sel").fetchone()[0]
    logging.info("  Hash-pool rule: salt=%r cutoff=0x%016x -> %d/%d rows (sample_n=%d)",
                 pool_salt, cutoff, n_pool, total, sample_n)
    con.execute(f"COPY (SELECT c.* FROM {src} c SEMI JOIN sel USING ({ID_COL})) TO '{pool_path}' (FORMAT PARQUET)")
    n_written = con.execute(f"SELECT count(*) FROM read_parquet('{pool_path}')").fetchone()[0]
    con.close()
    if n_written != n_pool:
        raise RuntimeError(f"pool file has {n_written} rows != expected {n_pool}")
    return total, n_pool, cutoff


def load_input_dataset(args, num_proc: int) -> Dataset:
    if args.mode == 'train':
        trajectory_dir = os.path.join(args.reclaim_data_dir, "final_data", "train")
    else:  # test: the 1M held-out cohort
        trajectory_dir = os.path.join(args.reclaim_data_dir, "final_data", "test")
    parquet_files = sorted(glob(os.path.join(trajectory_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {trajectory_dir}")

    if args.mode == 'train' and args.sample_n is not None:
        os.makedirs(args.processed_data_dir, exist_ok=True)
        pool_path = os.path.join(args.processed_data_dir, "_pool_duck.parquet")   # transient
        total, n_pool, _ = build_hash_pool(parquet_files, args.sample_n, args.seed, num_proc, pool_path)
        dataset = load_dataset('parquet', data_files=[pool_path], split='train')
        os.remove(pool_path)   # load_dataset holds its own arrow copy
        if len(dataset) != n_pool:
            raise RuntimeError(f"pool size {len(dataset)} != expected {n_pool}")
        logging.info("  Loaded %d sequences (hash pool) from %s", len(dataset), trajectory_dir)
        return dataset

    dataset = load_dataset('parquet', data_files=parquet_files, split='train')
    logging.info("  Loaded %d sequences from %s", len(dataset), trajectory_dir)
    return dataset


def canonical_order(ds: Dataset) -> Dataset:
    """Sort rows by enrollee id so saved artifacts are byte-comparable across
    clusters (the hash pool fixes WHICH rows; this fixes their ORDER)."""
    eids = np.asarray(ds[ID_COL], dtype=str)
    return ds.select(np.argsort(eids, kind="stable"))


def hash_split(ds: Dataset, val_frac: float, seed: int):
    """90/10 split keyed on a second salt: val = the round(val_frac*n) patients
    with the smallest split-hashes.  Membership is a property of the id."""
    eids = ds[ID_COL]
    val_ids = select_smallest_ids(duck_connect(), eids, int(round(val_frac * len(eids))), hash_salt(seed, "split"))
    val_mask = np.fromiter((e in val_ids for e in eids), dtype=bool, count=len(eids))
    return ds.select(np.flatnonzero(~val_mask)), ds.select(np.flatnonzero(val_mask))


def log_outcome_stats(ds: Dataset):
    for col in GOLD_COST_COLS:
        vals = np.asarray(ds[col])
        logging.info("  %-40s mean=%.2f  median=%.2f  max=%d  zero_frac=%.3f",
                     col, vals.mean(), np.median(vals), vals.max(), (vals == 0).mean())


def log_examples(ds: Dataset):
    if len(ds) > 0:
        ex = ds[0]
        logging.info("  Example context (last 10): %s", ' '.join(ex['context_seq'].split()[-10:]))
        logging.info("  Example future (first 10): %s", ' '.join(ex['future_seq'].split()[:10]))


def run_section(dataset: Dataset, section: str, args, num_proc: int) -> Dataset:
    """Apply a split function, filter eligible rows, and extract the outcome."""
    logging.info("%s processing", section.upper())

    if section == 'prospective':
        processed = dataset.map(process_example_prospective, num_proc=num_proc,
                                desc=f"[{section}] Processing")
    else:
        processed = dataset.map(lambda x: process_example(x, args.seed),
                                num_proc=num_proc,
                                desc=f"[{section}] Processing")

    eligible = processed.filter(lambda x: x['is_eligible'], num_proc=num_proc,
                                desc=f"[{section}] Filtering")
    logging.info("  Eligible: %d/%d (%.1f%%)",
                 len(eligible), len(dataset), 100 * len(eligible) / max(len(dataset), 1))

    eligible = eligible.map(lambda x: extract_outcome(x['future_seq']),
                            num_proc=num_proc,
                            desc=f"[{section}] Extracting cost outcome")
    log_outcome_stats(eligible)
    log_examples(eligible)
    return eligible


def save_section(ds: Dataset, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preprocessed_input.parquet")
    ds.to_parquet(out_path)
    logging.info("  Saved %d rows to: %s", len(ds), out_path)


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    args = parse_args()
    logging.info("PREPROCESS STARTING (mode=%s)...", args.mode)
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("Config: mode=%s, min_age=%d", args.mode, args.min_age)
    logging.info("  Input: %s", args.reclaim_data_dir)
    logging.info("  Output: %s", args.processed_data_dir)

    dataset = load_input_dataset(args, num_proc)

    # Age filter -- gates the ANCHOR age (parse_age_value returns the raw
    # <AGE-XX> value = age in the FIRST observed year), and also drops
    # demographics-only rows, which have no <AGE> token.
    #
    # Rationale for 16: every eligible prediction point sits >= 2 observed
    # years after the anchor (retrospective: the middle-<NY> split excludes
    # the first and last year markers; prospective: empirically the same
    # bound holds), so anchor >= 16 STRICTLY GUARANTEES prediction-point
    # age >= 18 -- i.e., adults only, consistent with the followed prior
    # work (Osawa et al. 2020: adults >= 18).
    # Do NOT raise this to 18: that would gate the anchor, dropping ~4.6%
    # of the cohort (anchor-16/17 patients) whose prediction points are all
    # fully compliant adults.  Verified empirically: the realized
    # minimum prediction-point age is exactly 18 in both sections.
    initial_count = len(dataset)
    dataset = dataset.filter(
        lambda x: (age := parse_age_value(x['seq_thru_cutoff'])) is not None
        and age >= args.min_age,
        num_proc=num_proc, desc="Age filter")
    logging.info("  Age filter: %d -> %d (kept %.1f%%)",
                 initial_count, len(dataset), 100 * len(dataset) / initial_count)

    if args.mode == 'test':
        retrospective = run_section(dataset, 'retrospective', args, num_proc)
        save_section(retrospective, os.path.join(args.processed_data_dir, "test", "retrospective"))

        prospective = run_section(dataset, 'prospective', args, num_proc)
        save_section(prospective,
                     os.path.join(args.processed_data_dir, "test", "prospective"))
    else:  # train
        eligible = canonical_order(run_section(dataset, 'retrospective', args, num_proc))
        train_ds, val_ds = hash_split(eligible, 0.1, args.seed)
        logging.info("  Train/val split (id-hash, salt=%r): %d train, %d val",
                     hash_salt(args.seed, "split"), len(train_ds), len(val_ds))
        save_section(train_ds, os.path.join(args.processed_data_dir, "train"))
        save_section(val_ds, os.path.join(args.processed_data_dir, "val"))

    logging.info("PREPROCESS COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
