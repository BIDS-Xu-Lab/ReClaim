#!/usr/bin/env python3
"""
LightGBM baseline prediction for the cost-prediction task.

Loads per-target trained models (from train_lightgbm.py) and a
feature-engineered test parquet, generates predictions for all available
targets, and writes a single cost_predictions.csv with enrollee_id + all
prediction columns (same schema as the ReClaim aggregation output).
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import setup_logging
from sequence_parsing import GOLD_COST_COLS

ID_COL = "enrollee_id"
MODEL_FILENAME = "lgbm_best_model.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM baseline prediction for cost task")
    parser.add_argument("--task", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="classification: per-cutoff binary models (ge_<c> subdirs) "
                             "-> prob_ge_<c> probability columns")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing per-target (or per-cutoff) model subdirs")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to baseline_features parquet (test split)")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    log_file = setup_logging(level=logging.INFO)
    args = parse_args()
    logging.info("BASELINE PREDICTION STARTING...")
    logging.info("  Model dir: %s", args.model_dir)
    logging.info("  Data:      %s", args.data_path)
    logging.info("  Output:    %s", args.output_dir)

    if args.task == "classification":
        import re
        subdirs = sorted(
            (int(m.group(1)), os.path.join(args.model_dir, d, MODEL_FILENAME))
            for d in os.listdir(args.model_dir)
            if (m := re.match(r'^ge_(\d+)$', d))
            and os.path.isfile(os.path.join(args.model_dir, d, MODEL_FILENAME)))
        available_targets = [(f"prob_ge_{c}", p) for c, p in subdirs]
    else:
        available_targets = []
        for target in GOLD_COST_COLS:
            model_path = os.path.join(args.model_dir, target, MODEL_FILENAME)
            if os.path.isfile(model_path):
                available_targets.append((target, model_path))
    logging.info("  Found models for %d targets: %s",
                 len(available_targets), [t for t, _ in available_targets])
    if not available_targets:
        logging.error("  No trained models found in %s", args.model_dir)
        return

    df = pd.read_parquet(args.data_path)
    logging.info("  Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    exclude_cols = set([ID_COL] + GOLD_COST_COLS)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    logging.info("  Features: %d", len(feature_cols))

    pred_df = pd.DataFrame({ID_COL: df[ID_COL]})
    for target, model_path in available_targets:
        logging.info("  --- Target: %s ---", target)
        model = lgb.Booster(model_file=model_path)
        if len(feature_cols) != model.num_feature():
            logging.warning("    Feature count mismatch: data has %d, model expects %d",
                            len(feature_cols), model.num_feature())
        # Booster.predict on a binary objective returns P(positive) directly
        y_pred = model.predict(X)
        out_col = target if args.task == "classification" else "predicted_" + target
        pred_df[out_col] = y_pred
        logging.info("    %s: mean=%.4f  median=%.4f  std=%.4f",
                     out_col, y_pred.mean(), np.median(y_pred), y_pred.std())

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "cost_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info("  Saved predictions (%d cols): %s", len(pred_df.columns), pred_path)

    logging.info("BASELINE PREDICTION COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
