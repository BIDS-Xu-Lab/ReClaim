"""LightGBM model training and inference."""

import os
import warnings
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier


def train_model(
    x_train: csr_matrix,
    y_train: np.ndarray,
    x_val: Optional[csr_matrix] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: int = 10000,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = 1,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
) -> LGBMClassifier:
    """Train LGBMClassifier with optional early stopping."""
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=-1,
    )
    callbacks = []
    if early_stopping_rounds > 0 and x_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    fit_kwargs: dict = {"callbacks": callbacks} if callbacks else {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )
        if x_val is not None and y_val is not None:
            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="auc",
                **fit_kwargs,
            )
        else:
            model.fit(x_train, y_train, **fit_kwargs)
    return model


def predict_proba(model: LGBMClassifier, x: csr_matrix) -> np.ndarray:
    """Predict probabilities, suppressing sklearn feature-name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )
        return model.predict_proba(x)[:, 1]


def save_model(model: LGBMClassifier, path: str) -> None:
    """Save model to file (LightGBM native format)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.booster_.save_model(path)


def load_model(path: str) -> LGBMClassifier:
    """Load model from LightGBM .txt file."""
    booster = lgb.Booster(model_file=path)
    n_features = int(booster.num_feature())
    model = LGBMClassifier()
    model._Booster = booster
    # Keep compatibility across LightGBM / sklearn versions:
    # sklearn validation may read different internal attributes for feature count.
    model._n_features = n_features
    model._n_features_in = n_features
    model.n_features_in_ = n_features
    model._n_classes = 2
    model._classes = np.array([0, 1])
    model.fitted_ = True
    return model
