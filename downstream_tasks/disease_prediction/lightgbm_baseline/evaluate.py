"""AUC evaluation by strata."""

from typing import Dict, List

import pandas as pd

from utils.evaluate_auc import get_auc_delong_var


def evaluate_by_strata(
    scored_df: pd.DataFrame,
    target_token_value: object,
    target_token: str,
) -> pd.DataFrame:
    """Compute DeLong AUC per (sex, age) stratum."""
    results: List[Dict] = []
    for (sex, age), group in scored_df.groupby(["sex", "age"]):
        labels = group["label"].to_numpy()
        scores = group["score"].to_numpy()
        case_scores = scores[labels == 1]
        control_scores = scores[labels == 0]
        if len(case_scores) == 0 or len(control_scores) == 0:
            continue
        auc_delong, _ = get_auc_delong_var(control_scores, case_scores)
        results.append({
            "auc_delong": float(auc_delong),
            "sex": sex,
            "disease_token": target_token_value,
            "disease_token_str": target_token,
            "age": int(age),
            "n_diseased": int(len(case_scores)),
            "n_healthy": int(len(control_scores)),
        })
    return pd.DataFrame(results)


def aggregate_normals(group: pd.DataFrame) -> pd.Series:
    """Aggregate normal distributions for pooled AUC variance."""
    n = len(group)
    mean = group["auc_delong"].mean()
    return pd.Series({
        "auc": mean,
        "n_samples": n,
        "n_diseased": group["n_diseased"].sum(),
        "n_healthy": group["n_healthy"].sum(),
    })
