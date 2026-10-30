"""
Cost Prediction Binary AUC Evaluation Script

Evaluates binary cost category predictions (below vs. above a single cutoff)
using ROC-AUC and PR-AUC metrics. Reads raw generated_data_N.parquet files
(not the pre-averaged cost_predictions CSVs) so that each individual generation
contributes an independent binary vote per enrollee.

For each enrollee across all individual generations:
    positive_score = count(generations with parsed_cost >= cutoff) / total_generations
"""

import os
import re
import glob
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import Dataset

from utils import setup_logging, time_execution

# =============================================================================
# Constants
# =============================================================================

INDENT1 = "  "
INDENT2 = "    "

TRUE_COLUMN = 'next_annual_cost'
CAP = 1_000_000


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate binary cost prediction AUC metrics')
    parser.add_argument("--model_list", type=str, nargs='+', required=True,
                        help='List of models to evaluate')
    parser.add_argument("--gold_path", type=str, required=True,
                        help='Path to preprocessed_input.parquet with gold outcomes')
    parser.add_argument("--model_gen_dir", type=str, required=True,
                        help='Base directory containing <model>/generated_data_N.parquet')
    parser.add_argument("--result_dir", type=str, required=True,
                        help='Directory for combined results')
    parser.add_argument("--cutoff", type=float, default=15000,
                        help='Single cost cutoff for binary classification (below vs. >= cutoff)')
    parser.add_argument("--gen_num", type=str, required=True,
                        help="Which N to score for Monte-Carlo models: an integer (one N) or 'all' -- every N "
                             "that has a cost_predictions_<N>.csv with a prob_ge column or a generated_data_<N>.parquet. "
                             "Each N is saved as its own row <model>-<N>.")
    parser.add_argument("--num_proc", type=int, default=None,
                        help='Number of processes for dataset.map (default: CPU count - 1)')
    return parser.parse_args()


# =============================================================================
# Cost Parsing (shared token helpers)
# =============================================================================

from sequence_parsing import extract_outcome as _extract_outcome_dict


def extract_outcome(sequence: str) -> int:
    """Total next-year cost parsed from a generated sequence."""
    return _extract_outcome_dict(sequence)["next_annual_cost"]


# =============================================================================
# Dataset.map helpers (module-level for picklability with num_proc)
# =============================================================================

def _expand_and_threshold(batch, cutoff):
    """Batched map fn for list-format generated_seq: explode and threshold each generation."""
    eids, positives = [], []
    for eid, seqs in zip(batch['enrollee_id'], batch['generated_seq']):
        for seq in seqs:
            cost = min(extract_outcome(seq), CAP)
            eids.append(eid)
            positives.append(int(cost >= cutoff))
    return {'enrollee_id': eids, 'is_positive': positives}


def _threshold_single(example, cutoff):
    """Row-level map fn for single-string generated_seq."""
    cost = min(extract_outcome(example['generated_seq']), CAP)
    return {'is_positive': int(cost >= cutoff)}


# =============================================================================
# File Discovery
# =============================================================================

def find_generation_file(model_dir: str, model: str,
                         gen_num: int) -> Optional[str]:
    """
    Find generated_data_N.parquet for the requested N.

    Args:
        model_dir: Directory containing generated data files
        model: Model name (for logging)
        gen_num: The N in generated_data_N.parquet

    Returns:
        Path to the parquet file, or None if not found
    """
    fpath = os.path.join(model_dir, f"generated_data_{gen_num}.parquet")
    if os.path.exists(fpath):
        logging.info("%sFound %s for model '%s'",
                     INDENT1, os.path.basename(fpath), model)
        return fpath

    logging.warning("%sModel '%s': generated_data_%d.parquet not found. Skipping.",
                    INDENT1, model, gen_num)
    return None


# =============================================================================
# Data Loading & Probability Construction
# =============================================================================

@time_execution
def load_and_build_probabilities(parquet_path: str,
                                  gold_path: str,
                                  cutoff: float,
                                  num_proc: int) -> pd.DataFrame:
    """
    Load a single generated_data_N.parquet, parse costs from generated sequences,
    and compute per-person probability of cost >= cutoff across all individual
    generations within the file.

    For each enrollee:
        positive_score = count(individual generations with cost >= cutoff) / total_generations

    Returns:
        DataFrame with columns: enrollee_id, source, true_label (0/1),
        positive_score, n_generations
    """
    positive_label = f'>=${cutoff:,.0f}'
    negative_label = f'<${cutoff:,.0f}'

    ds = Dataset.from_parquet(parquet_path)
    n_rows = len(ds)
    first_seq = ds[0]['generated_seq']
    is_list = isinstance(first_seq, list)

    if is_list:
        ds = ds.map(
            _expand_and_threshold, batched=True, batch_size=1000,
            remove_columns=ds.column_names, num_proc=num_proc,
            fn_kwargs={'cutoff': cutoff},
            desc="Parsing generations",
        )
    else:
        ds = ds.map(
            _threshold_single, num_proc=num_proc,
            fn_kwargs={'cutoff': cutoff},
            desc="Parsing generations",
        )

    stacked = ds.to_pandas()[['enrollee_id', 'is_positive']]
    n_gens = len(stacked)
    n_pos = stacked['is_positive'].sum()

    logging.info("%s%d rows -> %d generations (%d positive) from %s",
                 INDENT1, n_rows, n_gens, n_pos, os.path.basename(parquet_path))

    # Per-enrollee: positive_score = fraction of generations >= cutoff
    agg = stacked.groupby('enrollee_id')['is_positive'].agg(['mean', 'count'])
    agg.columns = ['positive_score', 'n_generations']
    agg = agg.reset_index()
    logging.info("%sTotal individual generations: %d", INDENT1, n_gens)
    logging.info("%sUnique enrollees with predictions: %d", INDENT1, len(agg))

    # Load gold
    gold_df = pd.read_parquet(gold_path, columns=['enrollee_id', TRUE_COLUMN])
    logging.info("%sLoaded gold data: %d rows", INDENT1, len(gold_df))

    # Merge
    merged = agg.merge(gold_df, on='enrollee_id', how='inner')
    logging.info("%sMerged: %d enrollees with both predictions and gold", INDENT1, len(merged))

    n_pred_only = len(agg) - len(merged)
    n_gold_only = len(gold_df) - len(merged)
    if n_pred_only > 0:
        logging.warning("%s%d prediction enrollees had no matching gold enrollee_id",
                        INDENT2, n_pred_only)
    if n_gold_only > 0:
        logging.info("%s%d gold enrollees had no matching prediction",
                     INDENT2, n_gold_only)

    # True binary label
    merged['true_label'] = (merged[TRUE_COLUMN] >= cutoff).astype(int)
    merged['source'] = merged['enrollee_id'].astype(str).str[:2]

    # Log distributions
    logging.info("%sCutoff: $%s", INDENT1, f"{cutoff:,.0f}")
    logging.info("%sPositive class: %s | Negative class: %s",
                 INDENT1, positive_label, negative_label)
    n_pos = merged['true_label'].sum()
    n_neg = len(merged) - n_pos
    logging.info("%sTrue label distribution: %s=%d (%.2f%%), %s=%d (%.2f%%)",
                 INDENT1, positive_label, n_pos, 100 * n_pos / len(merged),
                 negative_label, n_neg, 100 * n_neg / len(merged))
    logging.info("%sSource distribution: %s", INDENT1,
                 merged['source'].value_counts().to_dict())
    logging.info("%sGenerations per enrollee: min=%d, max=%d, mean=%.1f",
                 INDENT1, int(merged['n_generations'].min()),
                 int(merged['n_generations'].max()), merged['n_generations'].mean())

    return merged


@time_execution
def load_probability_csv(model_dir: str, model: str, cutoff: float,
                         gold_path: str, csv_name: str = 'cost_predictions.csv') -> Optional[pd.DataFrame]:
    """
    Alternative loader for models that emit probabilities directly
    (per-cutoff classifier baselines, post-trained distributional readouts,
    and the Monte-Carlo anchor's aggregated cost_predictions_<N>.csv,
    whose prob_ge_<cutoff> column is the fraction of the N trajectories at/above
    the cutoff, exactly what load_and_build_probabilities derives from the parquet):
    a CSV containing a prob_ge_<int(cutoff)> column.

    Returns the same merged frame schema as load_and_build_probabilities,
    or None if no suitable file/column exists.
    """
    csv_path = os.path.join(model_dir, csv_name)
    prob_col = f"prob_ge_{int(cutoff)}"
    if not os.path.exists(csv_path):
        return None
    pred_df = pd.read_csv(csv_path)
    if prob_col not in pred_df.columns:
        return None
    logging.info("%sModel '%s': using probability column %s from %s",
                 INDENT1, model, prob_col, os.path.basename(csv_path))

    agg = pred_df[['enrollee_id', prob_col]].rename(columns={prob_col: 'positive_score'})
    agg['n_generations'] = 0

    gold_df = pd.read_parquet(gold_path, columns=['enrollee_id', TRUE_COLUMN])
    merged = agg.merge(gold_df, on='enrollee_id', how='inner')
    logging.info("%sMerged: %d enrollees with both predictions and gold",
                 INDENT1, len(merged))
    if merged.empty:
        return None

    merged['true_label'] = (merged[TRUE_COLUMN] >= cutoff).astype(int)
    merged['source'] = merged['enrollee_id'].astype(str).str[:2]
    n_pos = merged['true_label'].sum()
    logging.info("%sTrue label distribution: pos=%d (%.2f%%), neg=%d",
                 INDENT1, n_pos, 100 * n_pos / len(merged), len(merged) - n_pos)
    return merged


# =============================================================================
# AUC Metrics Computation
# =============================================================================

def compute_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Compute binary ROC-AUC and PR-AUC.

    Args:
        y_true: Binary true labels (0/1)
        y_score: Positive class probability scores

    Returns:
        Dict with n_samples, n_positive, n_negative, roc_auc, pr_auc
    """
    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_samples - n_positive

    if n_samples == 0 or n_positive == 0 or n_negative == 0:
        return {
            'n_samples': n_samples, 'n_positive': n_positive,
            'n_negative': n_negative, 'roc_auc': np.nan, 'pr_auc': np.nan,
        }

    return {
        'n_samples': n_samples,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'roc_auc': roc_auc_score(y_true, y_score),
        'pr_auc': average_precision_score(y_true, y_score),
    }


def _fmt(val) -> str:
    """Format a metric value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NaN"
    return f"{val:.4f}"


def log_auc_metrics(metrics: Dict):
    """Log AUC metrics."""
    logging.info("%sSamples: %d (positive=%d, negative=%d)",
                 INDENT2, metrics['n_samples'], metrics['n_positive'], metrics['n_negative'])
    logging.info("%sROC-AUC: %s", INDENT2, _fmt(metrics['roc_auc']))
    logging.info("%sPR-AUC:  %s", INDENT2, _fmt(metrics['pr_auc']))


# =============================================================================
# Per-Subset AUC Collection
# =============================================================================

def collect_auc_for_subset(df: pd.DataFrame, source_label: str,
                           model: str, cutoff: float) -> Tuple[Dict, Dict, List[str]]:
    """
    Compute AUC metrics for a data subset (total or a single source).

    Returns:
        Tuple of (record_for_csv, metrics_dict, report_lines)
    """
    y_true = df['true_label'].values
    y_score = df['positive_score'].values

    metrics = compute_auc_metrics(y_true, y_score)

    record = {
        'model': model, 'cutoff': cutoff, 'source': source_label,
        'n_samples': metrics['n_samples'],
        'n_positive': metrics['n_positive'],
        'n_negative': metrics['n_negative'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
    }

    cutoff_str = f"${cutoff:,.0f}"
    reports = [
        f"AUC Report - Model: {model}, Cutoff: {cutoff_str}, Source: {source_label}",
        "=" * 80,
        f"Binary: < {cutoff_str} vs. >= {cutoff_str}",
        f"Samples: {metrics['n_samples']} (positive={metrics['n_positive']}, "
        f"negative={metrics['n_negative']})",
        f"  ROC-AUC: {_fmt(metrics['roc_auc'])}",
        f"  PR-AUC:  {_fmt(metrics['pr_auc'])}",
        "=" * 80 + "\n",
    ]

    return record, metrics, reports


# =============================================================================
# Main Evaluation Logic
# =============================================================================

def evaluate_auc_metrics(merged_df: pd.DataFrame, model: str,
                         cutoff: float) -> Tuple[List[Dict], List[str]]:
    """
    Evaluate AUC metrics for total and by source.

    Returns:
        Tuple of (all_records, all_report_lines)
    """
    sources = sorted(merged_df['source'].unique())
    all_records, all_reports = [], []

    # Overall
    logging.info("AUC METRICS - OVERALL (source=total)")
    record, metrics, reports = collect_auc_for_subset(
        merged_df, 'total', model, cutoff
    )
    log_auc_metrics(metrics)
    all_records.append(record)
    all_reports.extend(reports)

    # By source
    logging.info("AUC METRICS - BY SOURCE")
    for source in sources:
        source_df = merged_df[merged_df['source'] == source]
        logging.info("SOURCE: %s (%d enrollees)", source, len(source_df))

        record, metrics, reports = collect_auc_for_subset(
            source_df, source, model, cutoff
        )
        log_auc_metrics(metrics)
        all_records.append(record)
        all_reports.extend(reports)

    return all_records, all_reports


# =============================================================================
# Saving Results
# =============================================================================

def save_results(result_dir: str, cutoff: float,
                 auc_records: List[Dict], report_lines: List[str]):
    """
    Save AUC metrics to:
        result_dir/binary_auc/<cutoff>/auc_metrics.csv
        result_dir/binary_auc/<cutoff>/auc_reports.txt
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("SAVING AUC RESULTS")
    logging.info("=" * 80)

    if not auc_records:
        logging.warning("%sNo AUC records to save", INDENT1)
        return

    cutoff_dir = os.path.join(result_dir, "binary_auc", str(int(cutoff)))
    os.makedirs(cutoff_dir, exist_ok=True)

    priority_cols = ['source', 'model']
    df = pd.DataFrame(auc_records)
    if 'cutoff' in df.columns:
        df = df.drop(columns=['cutoff'])
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    path = os.path.join(cutoff_dir, "auc_metrics.csv")
    # Partial runs must not clobber other models' rows: merge on the keys,
    # this run's rows winning.
    if os.path.exists(path):
        old = pd.read_csv(path)
        keep = old[~old.set_index(priority_cols).index.isin(df.set_index(priority_cols).index)]
        df = pd.concat([keep, df], ignore_index=True)
    df = (df.drop_duplicates(subset=priority_cols, keep="last")      # older files carried duplicate rows
            .sort_values(priority_cols).reset_index(drop=True))
    df.to_csv(path, index=False)
    logging.info("%sAUC metrics saved to: %s", INDENT1, path)

    if report_lines:
        path = os.path.join(cutoff_dir, "auc_reports.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        logging.info("%sAUC reports saved to: %s", INDENT1, path)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function."""
    setup_logging()
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    cutoff = args.cutoff
    num_proc = args.num_proc or max(1, os.cpu_count() - 1)

    logging.info("=" * 80)
    logging.info("BINARY AUC EVALUATION - Cost >= $%s vs. < $%s",
                 f"{cutoff:,.0f}", f"{cutoff:,.0f}")
    logging.info("=" * 80)
    logging.info("%sModels: %s", INDENT1, args.model_list)
    logging.info("%sCutoff: $%s", INDENT1, f"{cutoff:,.0f}")
    logging.info("%sGeneration num: %s", INDENT1, args.gen_num)
    logging.info("%sGold data: %s", INDENT1, args.gold_path)
    logging.info("%sResult directory: %s", INDENT1, args.result_dir)
    logging.info("%sNum proc: %d", INDENT1, num_proc)
    logging.info("=" * 80)

    all_records: List[Dict] = []
    all_reports: List[str] = []

    import re as _re
    prob_col = f"prob_ge_{int(cutoff)}"

    def discover_ns(model_dir):
        """Every N with a usable per-N source: cost_predictions_<N>.csv carrying the prob column, or generated_data_<N>.parquet."""
        ns = set()
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                m = _re.match(r"^cost_predictions_(\d+)\.csv$", f)
                if m and prob_col in pd.read_csv(os.path.join(model_dir, f), nrows=0).columns:
                    ns.add(int(m.group(1)))
                m = _re.match(r"^generated_data_(\d+)\.parquet$", f)
                if m: ns.add(int(m.group(1)))
        return sorted(ns)

    for model in args.model_list:
        model_dir = os.path.join(args.model_gen_dir, model)
        # Which N to score for this model: 'all' -> every N with a per-N source; an
        # integer -> that N; models with no per-N source (classifier baselines,
        # post-trained readouts) are scored once from their plain cost_predictions.csv.
        ns = discover_ns(model_dir) if args.gen_num == "all" else [int(args.gen_num)]
        if not ns or not any(os.path.exists(os.path.join(model_dir, f))
                             for n in ns for f in (f"cost_predictions_{n}.csv", f"generated_data_{n}.parquet")):
            ns = [None]
        for n in ns:
            logging.info("***** Model=%s, Cutoff=$%s, N=%s *****", model, f"{cutoff:,.0f}", n if n is not None else "-")
            # Resolution order: (1) the Monte-Carlo anchor's aggregated
            # cost_predictions_<N>.csv with a prob_ge column, (2) the trajectory
            # parquet generated_data_<N>.parquet, (3) a plain cost_predictions.csv.
            # Rows from (1)/(2) are saved as <model>-<N>, matching the regression /
            # classification rows built from cost_predictions_<N>.csv, so every N of
            # an anchor is an independent model id in the metric tables.
            row_model = model
            try:
                merged_df = None
                if n is not None:
                    merged_df = load_probability_csv(model_dir, model, cutoff, args.gold_path,
                                                     csv_name=f"cost_predictions_{n}.csv")
                    if merged_df is None:
                        parquet_path = find_generation_file(model_dir, model, n)
                        if parquet_path is not None:
                            merged_df = load_and_build_probabilities(parquet_path, args.gold_path, cutoff, num_proc)
                    if merged_df is not None:
                        row_model = f"{model}-{n}"
                if merged_df is None:
                    merged_df = load_probability_csv(model_dir, model, cutoff, args.gold_path)
                    if merged_df is None:
                        logging.warning("%sModel '%s' (N=%s): no generations and no usable "
                                        "probability CSV - skipping", INDENT1, model, n)
                        continue
            except Exception as e:
                logging.error("%sError loading data for %s (N=%s): %s", INDENT1, model, n, str(e))
                continue

            records, reports = evaluate_auc_metrics(merged_df, row_model, cutoff)
            all_records.extend(records)
            all_reports.extend(reports)

    save_results(args.result_dir, cutoff, all_records, all_reports)

    logging.info("")
    logging.info("AUC evaluation completed successfully")
    logging.info("%sModels processed: %d", INDENT1, len(args.model_list))
    logging.info("%sResults saved to: %s", INDENT1, args.result_dir)


if __name__ == "__main__":
    main()
