#!/usr/bin/env python3
"""
LightGBM regression for cost prediction task.

Reads the feature-engineered parquet (from feature_engineering.py) and trains
a LightGBM regressor to predict next_annual_cost on the full training set.

The upstream feature engineering script already handles all preprocessing:
  - Continuous features (age, prev_cost) are numeric
  - Categorical feature (sex) is integer-encoded
  - Clinical features are binarised (0/1)

This script loads the data as-is, trains with fixed hyperparameters and early
stopping using a separately provided validation set, and saves the best model
and diagnostics.  Evaluation on a held-out test set is performed separately.

Hyperparameters:
  n_estimators = 10,000 (with early stopping patience of 100)
  max_depth = 6
  learning_rate = 0.1
  subsample = 0.8
  All other parameters follow LightGBM library defaults.
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import setup_logging


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LightGBM regression for cost prediction")
    parser.add_argument("--processed_data_dir", type=str, required=True,
        help="Path to processed data directory")
    parser.add_argument("--history_length", type=str, default="1year",
        choices=["1year", "all"],
        help="History length (default: 1year)") 
    parser.add_argument("--token_feature_type", type=str, default="binary",
        choices=["binary", "count"],
        help="Token feature type (default: binary)")
    parser.add_argument("--output_dir", type=str,
        default=os.path.join(
            os.environ.get("COST_TASK_ROOT", "/path/to/expenditure_forecasting_root"),
            "baseline", "versions", "v6", "model"),
        help="Directory to save output")
    parser.add_argument("--random_seed", type=int, default=66,
        help="Random seed for reproducibility (default: 66)")
    return parser.parse_args() 

# =============================================================================
# Constants
# =============================================================================

ID_COL = "enrollee_id"
VT_TYPES = ("inpatient", "outpatient", "pharmacy")
TARGET_COLS = [
    "next_annual_cost",
    *(f"next_annual_cost_{vt}" for vt in VT_TYPES),
]

# Fixed hyperparameters
LGBM_PARAMS = {
    "n_estimators": 10_000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
}

EARLY_STOPPING_ROUNDS = 100


# =============================================================================
# Training
# =============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_seed: int,
) -> tuple[lgb.LGBMRegressor, dict]:
    """Train LightGBM with early stopping and return (model, info_dict)."""

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=random_seed,
        verbosity=-1,
        **LGBM_PARAMS,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ],
    )

    best_iteration = model.best_iteration_
    best_val_rmse = model.best_score_["valid_0"]["rmse"]
    logging.info("  Best iteration: %d / %d", best_iteration, LGBM_PARAMS["n_estimators"])
    logging.info("  Best val RMSE:  %.2f", best_val_rmse)

    info = {
        "best_iteration": best_iteration,
        "best_val_rmse": best_val_rmse,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        **LGBM_PARAMS,
    }

    return model, info


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("LIGHTGBM REGRESSION STARTING...")

    args = parse_args()

    logging.info("  Config: seed=%d", args.random_seed)
    logging.info("  Hyperparams: %s", LGBM_PARAMS)
    logging.info("  Early stopping patience: %d rounds", EARLY_STOPPING_ROUNDS)
    logging.info("  Train input: %s", os.path.join(args.processed_data_dir, "train", f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet"))
    logging.info("  Val input:   %s", os.path.join(args.processed_data_dir, "val", f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet"))
    logging.info("  Output:      %s", os.path.join(args.output_dir, args.history_length, args.token_feature_type))

    # ------------------------------------------------------------------
    # 1. Load feature data
    # ------------------------------------------------------------------
    df_train = pd.read_parquet(os.path.join(args.processed_data_dir, "train", f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet"))
    df_val = pd.read_parquet(os.path.join(args.processed_data_dir, "val", f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet"))
    logging.info("  Loaded train: %d rows, %d columns", df_train.shape[0], df_train.shape[1])
    logging.info("  Loaded val:   %d rows, %d columns", df_val.shape[0], df_val.shape[1])

    # ------------------------------------------------------------------
    # 2. Separate features (exclude all target columns)
    # ------------------------------------------------------------------
    exclude_cols = set([ID_COL] + TARGET_COLS)
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    logging.info("  Features: %d total", len(feature_cols))

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]

    # ------------------------------------------------------------------
    # 3. Train one model per target
    # ------------------------------------------------------------------
    available_targets = [t for t in TARGET_COLS if t in df_train.columns]
    logging.info("  Targets to train: %s", available_targets)

    for target_col in available_targets:
        logging.info("")
        logging.info("=" * 60)
        logging.info("  TARGET: %s", target_col)
        logging.info("=" * 60)

        y_train = df_train[target_col]
        y_val = df_val[target_col]

        logging.info("  Train target stats:")
        logging.info("    mean=%.2f  median=%.2f  std=%.2f  zero_frac=%.3f",
                     y_train.mean(), y_train.median(), y_train.std(), (y_train == 0).mean())
        logging.info("  Val target stats:")
        logging.info("    mean=%.2f  median=%.2f  std=%.2f  zero_frac=%.3f",
                     y_val.mean(), y_val.median(), y_val.std(), (y_val == 0).mean())

        logging.info("  Training LightGBM with early stopping...")
        model, train_info = train_model(X_train, y_train, X_val, y_val, args.random_seed)
        train_info["target"] = target_col

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        logging.info("  Top 10 features by importance:")
        for _, row in importance_df.head(10).iterrows():
            logging.info("    %-30s %.4f", row["feature"], row["importance"])

        # Save artefacts
        save_dir = os.path.join(args.output_dir, args.history_length, args.token_feature_type, "lightgbm", target_col)
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "lgbm_best_model.txt")
        importance_path = os.path.join(save_dir, "feature_importance.csv")
        metrics_path = os.path.join(save_dir, "training_metrics.csv")

        model.booster_.save_model(model_path)
        importance_df.to_csv(importance_path, index=False)
        pd.DataFrame([train_info]).to_csv(metrics_path, index=False)

        logging.info("  Saved model:      %s", model_path)
        logging.info("  Saved importance:  %s", importance_path)
        logging.info("  Saved metrics:     %s", metrics_path)

    logging.info("LIGHTGBM REGRESSION COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
