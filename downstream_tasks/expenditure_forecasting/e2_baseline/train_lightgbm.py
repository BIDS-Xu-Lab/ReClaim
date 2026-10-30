#!/usr/bin/env python3
"""
LightGBM baseline training for the cost-prediction task.

Trains one LightGBM regressor per target (total + per-visit-type next-year
cost) on the feature-engineered train split, with early stopping on the val
split.

Hyperparameters (fixed):
  n_estimators = 10,000 (early stopping patience 100)
  max_depth = 6, learning_rate = 0.1, subsample = 0.8
  All other parameters are library defaults.

"""

import os
import argparse
import logging

import pandas as pd
import lightgbm as lgb

from utils import setup_logging
from sequence_parsing import GOLD_COST_COLS

ID_COL = "enrollee_id"
MODEL_FILENAME = "lgbm_best_model.txt"

SHARED_PARAMS = {
    "n_estimators": 10_000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    # Bit-reproducible training regardless of thread count / run
    # (LightGBM docs: deterministic requires a fixed histogram build mode).
    "deterministic": True,
    "force_row_wise": True,
}
EARLY_STOPPING_ROUNDS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM baseline training for cost prediction")
    parser.add_argument("--processed_data_dir", type=str, required=True,
                        help="Directory containing train/ and val/ feature parquets")
    parser.add_argument("--task", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="regression: one regressor on total cost; "
                             "classification: one binary HNHC classifier per cutoff on total cost")
    parser.add_argument("--cutoffs", type=float, nargs='+',
                        default=[30000.0],
                        help="[task=classification] cost cutoffs, one binary model each")
    parser.add_argument("--history_length", type=str, default="1year",
                        choices=["1year", "all"])
    parser.add_argument("--token_feature_type", type=str, default="binary",
                        choices=["binary", "count"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=66)
    parser.add_argument("--n_estimators", type=int, default=None,
                        help="Override n_estimators (smoke tests only; default 10,000)")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Training threads (default: all cores; set from SLURM_CPUS_PER_TASK)")
    return parser.parse_args()


def train_lightgbm(X_train, y_train, X_val, y_val, seed, n_jobs):
    model = lgb.LGBMRegressor(
        objective="regression", random_state=seed, verbosity=-1, n_jobs=n_jobs, **SHARED_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                   lgb.log_evaluation(period=100)],
    )
    info = {
        "best_iteration": model.best_iteration_,
        "best_val_rmse": model.best_score_["valid_0"]["rmse"],
    }
    return model, info


def train_lightgbm_classifier(X_train, y_train, X_val, y_val, seed, n_jobs):
    """Binary classifier with natural class weighting (keeps probabilities
    calibrated for downstream AUC/calibration comparisons)."""
    model = lgb.LGBMClassifier(
        objective="binary", random_state=seed, verbosity=-1, n_jobs=n_jobs,
        **SHARED_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                   lgb.log_evaluation(period=100)],
    )
    info = {
        "best_iteration": model.best_iteration_,
        "best_val_auc": model.best_score_["valid_0"]["auc"],
        "best_val_logloss": model.best_score_["valid_0"]["binary_logloss"],
    }
    return model, info


def run_classification(args, df_train, df_val, X_train, X_val, feature_cols, n_jobs):
    """Train one binary classifier per cutoff on total next-year cost."""
    target_col = "next_annual_cost"
    for cutoff in sorted(args.cutoffs):
        logging.info("=" * 60)
        logging.info("  CLASSIFIER: %s >= $%s", target_col, f"{cutoff:,.0f}")
        logging.info("=" * 60)
        y_train = (df_train[target_col] >= cutoff).astype(int)
        y_val = (df_val[target_col] >= cutoff).astype(int)
        logging.info("  Positive rate: train=%.4f  val=%.4f",
                     y_train.mean(), y_val.mean())

        model, info = train_lightgbm_classifier(
            X_train, y_train, X_val, y_val, args.random_seed, n_jobs)
        logging.info("  Best iteration: %s | val AUC: %.4f | val logloss: %.4f",
                     info["best_iteration"], info["best_val_auc"],
                     info["best_val_logloss"])

        info.update({"target": target_col, "cutoff": cutoff,
                     "pos_rate_train": y_train.mean(), "pos_rate_val": y_val.mean(),
                     "n_train": len(X_train), "n_val": len(X_val),
                     **SHARED_PARAMS})

        importance_df = pd.DataFrame({
            "feature": feature_cols, "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logging.info("  Top 10 features: %s",
                     importance_df.head(10)["feature"].tolist())

        # output_dir IS the standardized model dir (models/lightgbm-cls-<window>)
        save_dir = os.path.join(args.output_dir, f"ge_{int(cutoff)}")
        os.makedirs(save_dir, exist_ok=True)
        model.booster_.save_model(os.path.join(save_dir, MODEL_FILENAME))
        importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
        pd.DataFrame([info]).to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)
        logging.info("  Saved model artefacts to: %s", save_dir)


def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("BASELINE TRAINING STARTING...")

    args = parse_args()
    if args.n_estimators is not None:
        SHARED_PARAMS["n_estimators"] = args.n_estimators
        logging.warning("  n_estimators overridden to %d", args.n_estimators)
    feat_name = f"baseline_features_{args.history_length}_{args.token_feature_type}.parquet"
    train_path = os.path.join(args.processed_data_dir, "train", feat_name)
    val_path = os.path.join(args.processed_data_dir, "val", feat_name)

    logging.info("  Objective: l2 | seed: %d", args.random_seed)
    logging.info("  Hyperparams: %s (early stopping %d)", SHARED_PARAMS, EARLY_STOPPING_ROUNDS)
    logging.info("  Train: %s", train_path)
    logging.info("  Val:   %s", val_path)

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    logging.info("  Loaded train: %s, val: %s", df_train.shape, df_val.shape)

    exclude_cols = set([ID_COL] + GOLD_COST_COLS)
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    X_train, X_val = df_train[feature_cols], df_val[feature_cols]
    logging.info("  Features: %d", len(feature_cols))

    n_jobs = args.n_jobs or os.cpu_count()

    if args.task == "classification":
        run_classification(args, df_train, df_val, X_train, X_val,
                           feature_cols, n_jobs)
        logging.info("BASELINE TRAINING COMPLETED")
        logging.info("Log: %s", log_file)
        return

    available_targets = [t for t in GOLD_COST_COLS if t in df_train.columns]

    for target_col in available_targets:
        logging.info("=" * 60)
        logging.info("  TARGET: %s", target_col)
        logging.info("=" * 60)
        y_train, y_val = df_train[target_col], df_val[target_col]
        logging.info("  Train target: mean=%.2f median=%.2f zero_frac=%.3f",
                     y_train.mean(), y_train.median(), (y_train == 0).mean())

        model, info = train_lightgbm(
            X_train, y_train, X_val, y_val, args.random_seed, n_jobs)
        logging.info("  Best iteration: %s | best val RMSE: %.2f",
                     info["best_iteration"], info["best_val_rmse"])

        info.update({"target": target_col, "objective": "l2",
                     "n_train": len(X_train), "n_val": len(X_val),
                     **SHARED_PARAMS})

        importance_df = pd.DataFrame({
            "feature": feature_cols, "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logging.info("  Top 10 features: %s",
                     importance_df.head(10)["feature"].tolist())

        # output_dir IS the standardized model dir (models/lightgbm-reg-<window>)
        save_dir = os.path.join(args.output_dir, target_col)
        os.makedirs(save_dir, exist_ok=True)
        model.booster_.save_model(os.path.join(save_dir, MODEL_FILENAME))
        importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
        pd.DataFrame([info]).to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)
        logging.info("  Saved model artefacts to: %s", save_dir)

    logging.info("BASELINE TRAINING COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
