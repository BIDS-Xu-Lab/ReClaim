#!/usr/bin/env python3
"""
XGBoost prediction for cost prediction task.

Loads trained XGBoost models (one per target, saved by regressor_xgboost.py)
and a feature-engineered parquet file, generates predictions for all available
targets, and saves a single CSV with enrollee_id + all prediction columns.

Model directory structure expected:
    model_dir/
        next_annual_cost/xgb_best_model.json
        next_annual_cost_inpatient/xgb_best_model.json
        next_annual_cost_outpatient/xgb_best_model.json
        next_annual_cost_pharmacy/xgb_best_model.json
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd

import xgboost as xgb

from utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XGBoost prediction for cost prediction")
    parser.add_argument("--model", type=str, required=True,
        help="Model name (xgb_binary)")
    parser.add_argument("--model_dir", type=str, required=True,
        help="Directory containing per-target model subdirs")
    parser.add_argument("--data_path", type=str, required=True,
        help="Path to baseline_features parquet file (test split)")
    parser.add_argument("--output_dir", type=str, default="output",
        help="Directory to save predictions (default: output)")
    return parser.parse_args()


ID_COL = "enrollee_id"
MODEL_FILENAME = "xgb_best_model.json"
VT_TYPES = ("inpatient", "outpatient", "pharmacy")
ALL_TARGET_COLS = [
    "next_annual_cost",
    *(f"next_annual_cost_{vt}" for vt in VT_TYPES),
]


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("XGBOOST PREDICTION STARTING...")

    args = parse_args()
    logging.info("  Model dir: %s", args.model_dir)
    logging.info("  Data:      %s", args.data_path)
    logging.info("  Output:    %s", args.output_dir)

    # ------------------------------------------------------------------
    # 1. Discover available target models
    # ------------------------------------------------------------------
    available_targets = []
    for target in ALL_TARGET_COLS:
        model_path = os.path.join(args.model_dir, target, MODEL_FILENAME)
        if os.path.isfile(model_path):
            available_targets.append((target, model_path))
    logging.info("  Found models for %d targets: %s",
                 len(available_targets), [t for t, _ in available_targets])

    if not available_targets:
        logging.error("  No trained models found in %s", args.model_dir)
        return

    # ------------------------------------------------------------------
    # 2. Load data (once)
    # ------------------------------------------------------------------
    df = pd.read_parquet(args.data_path)
    logging.info("  Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    exclude_cols = set([ID_COL] + ALL_TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    logging.info("  Features: %d", len(feature_cols))

    # ------------------------------------------------------------------
    # 3. Predict for each target
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({"enrollee_id": df[ID_COL]})

    for target, model_path in available_targets:
        logging.info("  --- Target: %s ---", target)
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        logging.info("    Model loaded (%d features expected)", model.n_features_in_)

        if len(feature_cols) != model.n_features_in_:
            logging.warning("    Feature count mismatch: data has %d, model expects %d",
                            len(feature_cols), model.n_features_in_)

        y_pred = model.predict(X)
        pred_col = "predicted_" + target
        pred_df[pred_col] = y_pred
        logging.info("    %s: mean=%.2f  median=%.2f  std=%.2f",
                     pred_col, y_pred.mean(), np.median(y_pred), y_pred.std())

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "cost_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info("  Saved predictions (%d columns): %s",
                 len(pred_df.columns), pred_path)

    logging.info("XGBOOST PREDICTION COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
