"""
AUC and Case/Control Helpers

The area under the ROC curve by DeLong's method, with its variance, and the three helpers that turn
a set of strata into the scores an area is taken over.

This is the scoring every model in this repository is measured with, so an AUC from this baseline
means the same thing as an AUC from ReClaim itself.
"""

import pandas as pd

import numpy as np


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Reference:
     Xu Sun and Weichao Xu, "Fast Implementation of DeLong's Algorithm for
     Comparing the Areas Under Correlated Receiver Operating Characteristic
     Curves", IEEE Signal Processing Letters 21(11), 1389-1393 (2014).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    """
    Computes ROC AUC value and variance using DeLong's method.
    """
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert (
        len(aucs) == 1
    ), "There is a bug in the code, please forward this to the developers"

    return aucs[0], delongcov


def aggregate_normals(group):
    """Aggregate normal distributions for AUC variance calculation."""
    n = len(group)
    mean = group["auc_delong"].mean()
    return pd.Series(
        {
            "auc": mean,
            "n_samples": n,
            "n_diseased": group["n_diseased"].sum(),
            "n_healthy": group["n_healthy"].sum(),
        }
    )


def collect_unique_sequences_and_diseases(case_control_ids_list, data):
    """
    Collect all unique (row_index, seq_point) pairs and disease token indices.
    case_ids and control_ids are row indices into the data DataFrame.
    """
    unique_pairs = set()
    disease_tokens = set()
    
    for disease_info in case_control_ids_list:
        case_ids = disease_info["case_ids"]
        case_seq_points = disease_info["case_seq_points"]
        control_ids = disease_info["control_ids"]
        control_seq_points = disease_info["control_seq_points"]
        disease_token = disease_info["disease_token"]
        
        # Collect disease token
        disease_tokens.add(disease_token)
        
        # Add case pairs (row_index, seq_point)
        for row_idx, sp in zip(case_ids, case_seq_points):
            unique_pairs.add((row_idx, sp))
        
        # Add control pairs (row_index, seq_point)
        for row_idx, sp in zip(control_ids, control_seq_points):
            unique_pairs.add((row_idx, sp))
    
    disease_tokens = sorted(list(disease_tokens))  # Sort for consistent ordering
    print(f"Collected {len(unique_pairs)} unique (row_index, seq_point) pairs")
    print(f"Collected {len(disease_tokens)} unique disease tokens")
    return list(unique_pairs), disease_tokens


def compute_auc_from_cache(disease_info, logits_cache, disease_token_to_idx):
    """
    Compute AUC for a single disease using pre-computed logits.
    case_ids and control_ids are row indices into the data DataFrame.
    
    Args:
        disease_info: dict with disease_token, case_ids, control_ids, etc.
        logits_cache: dict mapping (row_idx, seq_point) -> logits array (only disease tokens)
        disease_token_to_idx: dict mapping disease_token -> index in logits array
    """
    disease_token = disease_info["disease_token"]
    case_ids = disease_info["case_ids"]  # These are row indices
    case_seq_points = disease_info["case_seq_points"]
    control_ids = disease_info["control_ids"]  # These are row indices
    control_seq_points = disease_info["control_seq_points"]
    sex = disease_info["sex"]
    age = disease_info["age"]
    
    # Get the index for this disease token in the reduced logits array
    disease_idx = disease_token_to_idx.get(disease_token)
    if disease_idx is None:
        print(f"Warning: Disease token {disease_token} not found in disease_token_to_idx")
        return None
    
    # Look up case logits using (row_index, seq_point) as cache key
    case_logits = []
    for row_idx, sp in zip(case_ids, case_seq_points):
        logit_vector = logits_cache.get((row_idx, sp))
        if logit_vector is not None:
            case_logits.append(logit_vector[disease_idx])
        else:
            print(f"Warning: Missing logits for case (row={row_idx}, sp={sp})")
    
    # Look up control logits using (row_index, seq_point) as cache key
    control_logits = []
    for row_idx, sp in zip(control_ids, control_seq_points):
        logit_vector = logits_cache.get((row_idx, sp))
        if logit_vector is not None:
            control_logits.append(logit_vector[disease_idx])
        else:
            print(f"Warning: Missing logits for control (row={row_idx}, sp={sp})")
    
    if len(case_logits) == 0 or len(control_logits) == 0:
        return None
    
    # Compute AUC
    auc_value_delong, auc_variance_delong = get_auc_delong_var(
        control_logits, case_logits
    )
    
    return {
        "auc_delong": auc_value_delong,
        "sex": sex,
        "disease_token": disease_token,
        "age": age,
        "n_diseased": len(case_logits),
        "n_healthy": len(control_logits),
        "case_logits": case_logits,
        "control_logits": control_logits,
    }
