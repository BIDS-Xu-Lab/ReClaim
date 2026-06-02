"""
Cost Prediction Evaluation Script

Evaluates predicted_next_annual_cost from cost_predictions.csv against
the ground truth next_annual_cost from preprocessed_input.parquet,
computing both regression metrics (MAE, MSE, RMSE, R2) and classification
metrics for cost categories.
"""

import os
import re
import glob
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from utils import setup_logging, time_execution

# =============================================================================
# Constants
# =============================================================================

INDENT1 = "  "
INDENT2 = "    "
INDENT3 = "      "

VT_TYPES = ("inpatient", "outpatient", "pharmacy")
COST_TYPES = [
    ("total", "next_annual_cost", "predicted_next_annual_cost"),
    *((vt, f"next_annual_cost_{vt}", f"predicted_next_annual_cost_{vt}") for vt in VT_TYPES),
]


# =============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate cost prediction task')
    parser.add_argument("--model_list", type=str, nargs='+', default=['xgb_count'],
                        help='List of models to evaluate (e.g., xgb_count xgb_binary)')
    parser.add_argument("--gold_path", type=str, default=os.path.join(
                            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                            "evaluation",
                            "versions",
                            "v6",
                            "data",
                            "test",
                            "preprocessed_input.parquet",
                        ),
                        help='Path to gold data')
    parser.add_argument("--model_gen_dir", type=str,
                        default=os.path.join(
                            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                            "evaluation",
                            "versions",
                            "v6",
                            "data",
                            "model_gen",
                        ),
                        help='Base directory containing versioned data')
    parser.add_argument("--result_dir", type=str,
                        default=os.path.join(
                            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
                            "evaluation",
                            "versions",
                            "v6",
                            "results",
                        ),
                        help='Directory for combined results')
    parser.add_argument("--cutoff_list", type=str, nargs='+',
                        default=['1500,15000,30000'],
                        help='List of cutoff configurations, each as comma-separated values')
    return parser.parse_args()


def parse_cutoff_string(cutoff_str: str) -> List[float]:
    """Parse comma-separated cutoff string to list of floats."""
    return [float(x.strip()) for x in cutoff_str.split(',')]


def find_prediction_files(model_dir: str, model: str) -> List[Tuple[str, str]]:
    """
    find cost prediction CSV files in a model directory.

    Looks for numbered variants like cost_predictions_1.csv, cost_predictions_20.csv.
    If numbered files exist, returns each as a separate (model_id, csv_path) entry
    with the number appended to the model name (e.g., model-1, model-20).
    If no numbered files exist, falls back to the base cost_predictions.csv.

    Args:
        model_dir: Directory containing prediction CSV files
        model: Original model name

    Returns:
        List of (model_id, csv_path) tuples
    """
    base_path = os.path.join(model_dir, "cost_predictions.csv")

    # Find all numbered variants: cost_predictions_N.csv
    pattern = os.path.join(model_dir, "cost_predictions_*.csv")
    numbered_files = sorted(glob.glob(pattern))

    results = []
    for fpath in numbered_files:
        fname = os.path.basename(fpath)
        match = re.match(r'^cost_predictions_(\d+)\.csv$', fname)
        if match:
            num = match.group(1)
            results.append((f"{model}-{num}", fpath))

    if results:
        # Sort by the numeric suffix
        results.sort(key=lambda x: int(x[0].rsplit('-', 1)[-1]))
        logging.info("%sFound %d numbered prediction files for model '%s': %s",
                     INDENT1, len(results), model,
                     [os.path.basename(r[1]) for r in results])
    else:
        # Fall back to base file
        results.append((model, base_path))

    return results


# =============================================================================
# Cost Categorization
# =============================================================================

def categorize_cost(values: pd.Series, cutoffs: List[float]) -> Tuple[pd.Series, List[str]]:
    """
    Categorize cost values into labeled classes using pd.cut().

    Example with cutoffs=[1500, 15000, 30000]:
        cost < 1500 -> '<$1,500'
        1500 <= cost < 15000 -> '$1,500-$14,999'
        15000 <= cost < 30000 -> '$15,000-$29,999'
        cost >= 30000 -> '>=$30,000'
    """
    cutoffs = sorted(cutoffs)
    bins = [-np.inf] + cutoffs + [np.inf]

    labels = [f'<${cutoffs[0]:,.0f}']
    labels += [f'${cutoffs[i]:,.0f}-${cutoffs[i+1]-1:,.0f}' for i in range(len(cutoffs) - 1)]
    labels.append(f'>=${cutoffs[-1]:,.0f}')

    categories = pd.cut(values, bins=bins, labels=labels, right=False)
    return categories, labels


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute MAE, MSE, RMSE, R2 for valid (non-NaN) pairs."""
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid, y_pred_valid = y_true[valid_mask], y_pred[valid_mask]

    if len(y_true_valid) == 0:
        return {'n_valid': 0, 'n_total': len(y_true),
                'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    mse = mean_squared_error(y_true_valid, y_pred_valid)
    return {
        'n_valid': len(y_true_valid),
        'n_total': len(y_true),
        'MAE': mean_absolute_error(y_true_valid, y_pred_valid),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_true_valid, y_pred_valid)
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    labels: List[str] = None,
                                    is_binary: bool = False,
                                    positive_label: str = None) -> Dict:
    """
    Compute classification metrics: accuracy, precision, recall, F1.
    Supports weighted, macro averages, and binary metrics.
    """
    n_valid, n_total = len(y_true), len(y_true)

    if n_valid == 0:
        return {
            'n_valid': 0, 'n_total': n_total, 'accuracy': 0.0, 'balanced_accuracy': 0.0,
            'precision-weighted': 0.0, 'recall-weighted': 0.0, 'f1-weighted': 0.0,
            'precision-macro': 0.0, 'recall-macro': 0.0, 'f1-macro': 0.0,
            'precision-binary': None, 'recall-binary': None, 'f1-binary': None,
            'classification_report': "No valid samples",
            'confusion_matrix': None, 'confusion_matrix_labels': []
        }

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    p_b = r_b = f1_b = None
    if is_binary and positive_label:
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=positive_label, zero_division=0)

    present_labels = [l for l in (labels or []) if l in y_true or l in y_pred]
    report_labels = present_labels if present_labels else None

    class_report = classification_report(y_true, y_pred, labels=report_labels, zero_division=0)
    cm_labels = present_labels if present_labels else sorted(set(y_true) | set(y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=cm_labels)

    return {
        'n_valid': n_valid, 'n_total': n_total,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision-weighted': p_w, 'recall-weighted': r_w, 'f1-weighted': f1_w,
        'precision-macro': p_m, 'recall-macro': r_m, 'f1-macro': f1_m,
        'precision-binary': p_b, 'recall-binary': r_b, 'f1-binary': f1_b,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_labels
    }


# =============================================================================
# Formatting and Logging Helpers
# =============================================================================

def format_confusion_matrix(conf_matrix: np.ndarray, labels: List[str]) -> str:
    """Format confusion matrix as a readable string."""
    if conf_matrix is None:
        return "No confusion matrix available"

    label_width = max(len(str(l)) for l in labels) if labels else 10
    col_width = max(len(str(conf_matrix.max())), label_width) + 2

    lines = ["Confusion Matrix (rows=true, cols=predicted):"]
    lines.append(" " * (label_width + 2) + "".join(f"{str(l):>{col_width}}" for l in labels))

    for i, label in enumerate(labels):
        row = "".join(f"{conf_matrix[i, j]:>{col_width}}" for j in range(len(labels)))
        lines.append(f"{str(label):<{label_width + 2}}{row}")

    return "\n".join(lines)


def log_classification_metrics(metrics: Dict):
    """Log classification metrics."""
    logging.info("%sValid samples: %d/%d", INDENT2, metrics['n_valid'], metrics['n_total'])
    logging.info("%sAccuracy:           %.4f", INDENT2, metrics['accuracy'])
    logging.info("%sBalanced Accuracy:  %.4f", INDENT2, metrics['balanced_accuracy'])
    logging.info("%sPrecision-weighted: %.4f", INDENT2, metrics['precision-weighted'])
    logging.info("%sRecall-weighted:    %.4f", INDENT2, metrics['recall-weighted'])
    logging.info("%sF1-weighted:        %.4f", INDENT2, metrics['f1-weighted'])
    logging.info("%sPrecision-macro:    %.4f", INDENT2, metrics['precision-macro'])
    logging.info("%sRecall-macro:       %.4f", INDENT2, metrics['recall-macro'])
    logging.info("%sF1-macro:           %.4f", INDENT2, metrics['f1-macro'])
    if metrics.get('precision-binary') is not None:
        logging.info("%sPrecision-binary:   %.4f", INDENT2, metrics['precision-binary'])
        logging.info("%sRecall-binary:      %.4f", INDENT2, metrics['recall-binary'])
        logging.info("%sF1-binary:          %.4f", INDENT2, metrics['f1-binary'])


def log_dataframe(df: pd.DataFrame, indent: str = INDENT1):
    """Log a DataFrame with proper formatting."""
    table_str = df.to_string(index=False, float_format=lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:.2e}")
    for line in table_str.split("\n"):
        logging.info("%s%s", indent, line)


# =============================================================================
# Data Loading
# =============================================================================

@time_execution
def load_and_prepare_data(model_pred_path: str,
                          gold_path: str,
                          cost_cutoffs: List[float]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load predictions and ground truth, join by enrollee_id, and categorize costs.

    Args:
        model_pred_path: Path to cost_predictions.csv (enrollee_id, predicted_next_annual_cost)
        gold_path: Path to preprocessed_input.parquet (enrollee_id, next_annual_cost, ...)
        cost_cutoffs: List of cutoff values for cost categorization

    Returns:
        Tuple of (merged_df, category_labels)
    """
    # Load predictions
    pred_df = pd.read_csv(model_pred_path)
    logging.info("%sLoaded predictions: %d rows from %s", INDENT1, len(pred_df), model_pred_path)
    logging.info("%sPrediction columns: %s", INDENT2, pred_df.columns.tolist())

    # Load gold/ground truth (all cost columns present in the file)
    gold_all = pd.read_parquet(gold_path)
    gold_cost_cols = [tc for _, tc, _ in COST_TYPES if tc in gold_all.columns]
    gold_df = gold_all[['enrollee_id'] + gold_cost_cols]
    del gold_all
    logging.info("%sLoaded gold data: %d rows from %s", INDENT1, len(gold_df), gold_path)
    logging.info("%sGold cost columns: %s", INDENT2, gold_cost_cols)

    # Join by enrollee_id
    merged_df = pred_df.merge(gold_df, on='enrollee_id', how='inner')
    logging.info("%sMerged data: %d rows (inner join on enrollee_id)", INDENT1, len(merged_df))

    n_pred_only = len(pred_df) - len(merged_df)
    n_gold_only = len(gold_df) - len(merged_df)
    if n_pred_only > 0:
        logging.warning("%s%d prediction rows had no matching gold enrollee_id", INDENT2, n_pred_only)
    if n_gold_only > 0:
        logging.info("%s%d gold rows had no matching prediction (expected if not all were predicted)", INDENT2, n_gold_only)

    # Cap predicted costs at $1,000,000
    CAP = 1_000_000
    for cost_label, _, pred_col in COST_TYPES:
        if pred_col not in merged_df.columns:
            continue
        n_capped = (merged_df[pred_col] > CAP).sum()
        if n_capped > 0:
            logging.info("%sCapping %d predictions in '%s' that exceed $%s",
                         INDENT1, n_capped, pred_col, f"{CAP:,}")
            merged_df[pred_col] = merged_df[pred_col].clip(upper=CAP)

    # Derive source from enrollee_id prefix
    merged_df['source'] = merged_df['enrollee_id'].astype(str).str[:2]
    logging.info("%sSource distribution: %s", INDENT1, merged_df['source'].value_counts().to_dict())

    # Categorize costs for each cost type
    logging.info("%sCost cutoffs: %s", INDENT1, sorted(cost_cutoffs))
    category_labels = None
    for cost_label, true_col, pred_col in COST_TYPES:
        if true_col not in merged_df.columns or pred_col not in merged_df.columns:
            logging.info("%sSkipping cost type '%s': columns not found", INDENT2, cost_label)
            continue
        cat_true = f"true_cost_category_{cost_label}"
        cat_pred = f"pred_cost_category_{cost_label}"
        merged_df[cat_true], labels = categorize_cost(merged_df[true_col], cost_cutoffs)
        merged_df[cat_pred], _ = categorize_cost(merged_df[pred_col], cost_cutoffs)
        if category_labels is None:
            category_labels = labels
        logging.info("%s[%s] True distribution: %s", INDENT2, cost_label,
                     merged_df[cat_true].value_counts().to_dict())

    logging.info("%sCategory labels: %s", INDENT1, category_labels)
    return merged_df, category_labels


# =============================================================================
# Metrics Collection Helper
# =============================================================================

def collect_metrics_for_subset(df: pd.DataFrame, source_label: str,
                                model: str, cutoffs_label: str,
                                cost_label: str, true_col: str, pred_col: str,
                                cat_true_col: str, cat_pred_col: str,
                                category_labels: List[str], is_binary: bool,
                                positive_label: str) -> Tuple[pd.DataFrame, List[Dict], List[str], Dict]:
    """
    Collect regression and classification metrics for a data subset.

    Returns:
        Tuple of (continuous_metrics_df, classification_metrics_list, report_lines, class_metrics)
    """
    # Regression metrics
    cont_metrics = compute_continuous_metrics(
        df[true_col].values, df[pred_col].values
    )
    cont_metrics['cost_type'] = cost_label
    cont_metrics['source'] = source_label
    cont_metrics['model'] = model
    cont_metrics['cutoffs'] = cutoffs_label
    cont_df = pd.DataFrame([cont_metrics])

    # Classification metrics
    y_true = df[cat_true_col].astype(str)
    y_pred = df[cat_pred_col].astype(str)
    valid_mask = (y_true != 'nan') & (y_pred != 'nan')

    class_metrics = compute_classification_metrics(
        y_true[valid_mask].values, y_pred[valid_mask].values,
        labels=category_labels, is_binary=is_binary, positive_label=positive_label
    )
    class_metrics['n_total'] = len(y_true)

    # Build classification record for CSV output
    class_record = {
        'model': model, 'cutoffs': cutoffs_label,
        'cost_type': cost_label, 'source': source_label,
        'n_valid': class_metrics['n_valid'], 'n_total': class_metrics['n_total'],
        'accuracy': class_metrics['accuracy'],
        'balanced_accuracy': class_metrics['balanced_accuracy'],
        'precision-weighted': class_metrics['precision-weighted'],
        'recall-weighted': class_metrics['recall-weighted'],
        'f1-weighted': class_metrics['f1-weighted'],
        'precision-macro': class_metrics['precision-macro'],
        'recall-macro': class_metrics['recall-macro'],
        'f1-macro': class_metrics['f1-macro'],
        'precision-binary': class_metrics['precision-binary'],
        'recall-binary': class_metrics['recall-binary'],
        'f1-binary': class_metrics['f1-binary']
    }

    # Build report text
    reports = []
    reports.append(f"Classification Report - Model: {model}, Cost: {cost_label}, Cutoffs: {cutoffs_label}, Source: {source_label}")
    reports.append("=" * 80)
    reports.append(f"Categories: {category_labels}")
    if is_binary:
        reports.append(f"Binary classification: positive class = '{positive_label}' (above cutoff)")
    reports.append(class_metrics['classification_report'])
    reports.append("")
    reports.append(format_confusion_matrix(class_metrics['confusion_matrix'], class_metrics['confusion_matrix_labels']))
    reports.append("\n" + "=" * 80 + "\n")

    return cont_df, [class_record], reports, class_metrics


# =============================================================================
# Main Evaluation Logic
# =============================================================================

def evaluate_all_metrics(merged_df: pd.DataFrame, category_labels: List[str],
                          model: str, cutoffs_label: str,
                          is_binary: bool = False) -> Tuple[List, List, List]:
    """
    Evaluate all metrics (total and by source, for each cost type) for a
    single model-cutoff configuration.

    Returns:
        Tuple of (continuous_metrics_list, classification_metrics_list, report_lines)
    """
    positive_label = category_labels[-1] if is_binary else None
    sources = sorted(merged_df['source'].unique())

    all_cont, all_class, all_reports = [], [], []

    # Determine which cost types are available in this dataframe
    available_cost_types = [
        (cl, tc, pc) for cl, tc, pc in COST_TYPES
        if tc in merged_df.columns and pc in merged_df.columns
    ]

    for cost_label, true_col, pred_col in available_cost_types:
        cat_true = f"true_cost_category_{cost_label}"
        cat_pred = f"pred_cost_category_{cost_label}"
        if cat_true not in merged_df.columns:
            continue

        logging.info("--- Cost type: %s ---", cost_label)

        # Overall metrics
        logging.info("EVALUATION METRICS - OVERALL (source=total, cost_type=%s)", cost_label)

        cont_df, class_recs, reports, class_metrics = collect_metrics_for_subset(
            merged_df, 'total', model, cutoffs_label,
            cost_label, true_col, pred_col, cat_true, cat_pred,
            category_labels, is_binary, positive_label
        )

        logging.info("### Regression Metrics ###")
        log_dataframe(cont_df)
        all_cont.append(cont_df)

        logging.info("### Classification Metrics ###")
        log_classification_metrics(class_metrics)
        all_class.extend(class_recs)
        all_reports.extend(reports)

        # Metrics by source (payer type)
        logging.info("EVALUATION METRICS - BY SOURCE (cost_type=%s)", cost_label)
        for source in sources:
            logging.info("SOURCE: %s", source)

            source_df = merged_df[merged_df['source'] == source]
            logging.info("%sNumber of samples: %d", INDENT1, len(source_df))

            cont_df, class_recs, reports, class_metrics = collect_metrics_for_subset(
                source_df, source, model, cutoffs_label,
                cost_label, true_col, pred_col, cat_true, cat_pred,
                category_labels, is_binary, positive_label
            )

            logging.info("%sRegression Metrics for Source %s:", INDENT1, source)
            log_dataframe(cont_df, INDENT2)
            all_cont.append(cont_df)

            logging.info("%sClassification Metrics for Source %s:", INDENT1, source)
            log_classification_metrics(class_metrics)
            all_class.extend(class_recs)
            all_reports.extend(reports)

    return all_cont, all_class, all_reports


def save_results(result_dir: str, continuous_metrics: List[pd.DataFrame],
                 classification_metrics: List[Dict],
                 reports_by_cutoff: Dict[str, List[str]]):
    """
    Save all collected metrics to files in structured subfolders.

    Directory structure:
        result_dir/
            regression/
                continuous_metrics.csv
            classification/
                <cutoff_str>/
                    classification_metrics.csv
                    classification_reports.txt
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("SAVING COMBINED RESULTS")
    logging.info("=" * 80)

    priority_cols = ['source', 'model', 'cost_type']

    # --- Regression metrics -> result_dir/regression/ ---
    if continuous_metrics:
        reg_dir = os.path.join(result_dir, "regression")
        os.makedirs(reg_dir, exist_ok=True)

        df = pd.concat(continuous_metrics, ignore_index=True)
        if 'cutoffs' in df.columns:
            df = df.drop(columns=['cutoffs']).drop_duplicates().reset_index(drop=True)
        other_cols = [c for c in df.columns if c not in priority_cols]
        df = df[priority_cols + other_cols].sort_values(priority_cols).reset_index(drop=True)
        path = os.path.join(reg_dir, "regression_metrics.csv")
        df.to_csv(path, index=False)
        logging.info("%sRegression metrics saved to: %s", INDENT1, path)

    # --- Classification metrics -> result_dir/classification/<cutoff_str>/ ---
    if classification_metrics:
        cls_priority_cols = ['source', 'model', 'cost_type']
        all_cls_df = pd.DataFrame(classification_metrics)

        for cutoff_str, grp_df in all_cls_df.groupby('cutoffs'):
            cutoff_dir = os.path.join(result_dir, "classification", str(cutoff_str))
            os.makedirs(cutoff_dir, exist_ok=True)

            grp_df = grp_df.drop(columns=['cutoffs'])
            other_cols = [c for c in grp_df.columns if c not in cls_priority_cols]
            grp_df = grp_df[cls_priority_cols + other_cols].sort_values(cls_priority_cols).reset_index(drop=True)
            path = os.path.join(cutoff_dir, "classification_metrics.csv")
            grp_df.to_csv(path, index=False)
            logging.info("%sClassification metrics saved to: %s", INDENT1, path)

            # Save corresponding reports
            if cutoff_str in reports_by_cutoff and reports_by_cutoff[cutoff_str]:
                path = os.path.join(cutoff_dir, "classification_reports.txt")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(reports_by_cutoff[cutoff_str]))
                logging.info("%sClassification reports saved to: %s", INDENT1, path)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function."""
    setup_logging()
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    cutoff_configs = [parse_cutoff_string(c) for c in args.cutoff_list]

    logging.info("=" * 80)
    logging.info("OUTCOME EVALUATION - Predicted vs True next_annual_cost")
    logging.info("=" * 80)
    logging.info("%sModels: %s", INDENT1, args.model_list)
    logging.info("%sCutoff configurations: %s", INDENT1, cutoff_configs)
    logging.info("%sGold data: %s", INDENT1, args.gold_path)
    logging.info("%sResult directory: %s", INDENT1, args.result_dir)
    logging.info("=" * 80)

    all_cont, all_class = [], []
    all_reports_by_cutoff: Dict[str, List[str]] = {}
    config_num = 0

    # Expand model_list: find numbered prediction files for each model
    expanded_model_entries = []  # list of (model_id, model_pred_path)
    for model in args.model_list:
        model_dir = os.path.join(args.model_gen_dir, model)
        pred_files = find_prediction_files(model_dir, model)
        expanded_model_entries.extend(pred_files)

    total_configs = len(expanded_model_entries) * len(cutoff_configs)
    logging.info("%sExpanded model entries: %s", INDENT1,
                 [entry[0] for entry in expanded_model_entries])

    for model_id, model_pred_path in expanded_model_entries:
        for cutoff_str, cost_cutoffs in zip(args.cutoff_list, cutoff_configs):
            config_num += 1


            logging.info("***** Configuration %d/%d: Model=%s, Cutoffs=%s *****", config_num, total_configs, model_id, cutoff_str)
            logging.info("%sPredictions: %s", INDENT1, model_pred_path)

            # Load and prepare data
            try:
                merged_df, category_labels = load_and_prepare_data(
                    model_pred_path, args.gold_path, cost_cutoffs
                )
            except Exception as e:
                logging.error("%sError loading data for %s: %s", INDENT1, model_id, str(e))
                continue

            # Evaluate metrics
            is_binary = len(cost_cutoffs) == 1
            cont, cls, reports = evaluate_all_metrics(
                merged_df, category_labels, model_id, cutoff_str, is_binary
            )

            all_cont.extend(cont)
            all_class.extend(cls)
            # Group reports by cutoff string
            if cutoff_str not in all_reports_by_cutoff:
                all_reports_by_cutoff[cutoff_str] = []
            all_reports_by_cutoff[cutoff_str].extend(reports)

    # Save results
    save_results(args.result_dir, all_cont, all_class, all_reports_by_cutoff)

    logging.info("")
    logging.info("Evaluation completed successfully")
    logging.info("%sTotal configurations processed: %d", INDENT1, config_num)
    logging.info("%sResults saved to: %s", INDENT1, args.result_dir)


if __name__ == "__main__":
    main()
