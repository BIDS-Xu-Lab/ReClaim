"""AUC computation for Delphi on a predefined case/control cohort (DeLong AUC per stratum)."""

import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
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
    Computes ROC AUC value and variance using DeLong's method

    Args:
        healthy_scores: Values for class 0 (healthy/controls)
        diseased_scores: Values for class 1 (diseased/cases)
    Returns:
        AUC value and variance
    """
    # Create ground truth labels (1 for diseased, 0 for healthy)
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    # Compute statistics needed for DeLong method
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Calculate AUC and covariance
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

    return aucs[0], delongcov


# =============================================================================
# Predefined Cohort Evaluation Pipeline
# =============================================================================

def _collect_predefined_unique_pairs(predefined_cohort):
    pairs = set()
    disease_tokens = set()
    for record in predefined_cohort:
        disease_tokens.add(int(record["disease_token"]))
        pairs.update((int(idx), int(seq_pt)) for idx, seq_pt in zip(record["case_patient_indices"], record["case_seq_points"]))
        pairs.update((int(idx), int(seq_pt)) for idx, seq_pt in zip(record["control_patient_indices"], record["control_seq_points"]))
    pairs = sorted(pairs, key=lambda item: (item[1], item[0]))
    disease_tokens = sorted(disease_tokens)
    return pairs, disease_tokens


def _build_prefix_batch(prefix_samples, device):
    mask_time = -10000.0
    batch_size = len(prefix_samples)
    max_len = max(len(tokens) for tokens, _ in prefix_samples)

    x = torch.zeros((batch_size, max_len), dtype=torch.long)
    a = torch.full((batch_size, max_len), mask_time, dtype=torch.float32)
    last_positions = torch.full((batch_size,), max_len - 1, dtype=torch.long)

    for i, (tokens, ages) in enumerate(prefix_samples):
        length = len(tokens)
        x[i, -length:] = torch.as_tensor(tokens, dtype=torch.long) + 1
        a[i, -length:] = torch.as_tensor(ages, dtype=torch.float32)

    if device != "cpu":
        x = x.to(device)
        a = a.to(device)
        last_positions = last_positions.to(device)

    return x, a, last_positions


def _precompute_predefined_logits(model, val, val_p2i, unique_pairs, disease_tokens, batch_size, device):
    disease_tokens_tensor = torch.tensor(disease_tokens, dtype=torch.long, device=device)
    disease_token_to_idx = {token: idx for idx, token in enumerate(disease_tokens)}
    logits_cache = {}
    invalid_pairs = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(unique_pairs), batch_size), desc="Predefined prefix inference"):
            pairs_batch = unique_pairs[batch_start : batch_start + batch_size]
            prefix_samples = []
            valid_pairs = []

            for patient_idx, seq_pt in pairs_batch:
                if patient_idx < 0 or patient_idx >= len(val_p2i):
                    invalid_pairs.append((patient_idx, seq_pt, "patient_idx_out_of_range"))
                    continue

                start, length = val_p2i[patient_idx]
                if seq_pt < 0 or seq_pt >= length:
                    invalid_pairs.append((patient_idx, seq_pt, f"seq_point_out_of_range_0_{length}"))
                    continue

                # seq_pt is the last Delphi record included in model input.
                patient_slice = val[start : start + seq_pt + 1]
                if len(patient_slice) == 0:
                    invalid_pairs.append((patient_idx, seq_pt, "empty_prefix"))
                    continue

                prefix_samples.append((patient_slice[:, 2].astype(np.int64), patient_slice[:, 1].astype(np.float32)))
                valid_pairs.append((patient_idx, seq_pt))

            if not valid_pairs:
                continue

            x, a, last_positions = _build_prefix_batch(prefix_samples, device)
            outputs = model(x, a)[0]
            batch_logits = outputs[
                torch.arange(outputs.shape[0], device=outputs.device),
                last_positions,
                :,
            ][:, disease_tokens_tensor].cpu().numpy()

            for pair, logits in zip(valid_pairs, batch_logits):
                logits_cache[pair] = logits.astype(np.float32, copy=False)

    return logits_cache, disease_token_to_idx, invalid_pairs


def _summarize_predefined_record(record, limit=3):
    return {
        "disease_token": int(record["disease_token"]),
        "disease_code": record.get("disease_code"),
        "sex": record["sex"],
        "age": int(record["age"]),
        "n_cases": len(record.get("case_patient_indices", [])),
        "n_controls": len(record.get("control_patient_indices", [])),
        "case_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(record.get("case_patient_indices", [])[:limit], record.get("case_seq_points", [])[:limit])
        ],
        "control_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(record.get("control_patient_indices", [])[:limit], record.get("control_seq_points", [])[:limit])
        ],
    }


def _summarize_prefix_pair(patient_idx, seq_pt, val, val_p2i, limit=6):
    start, length = val_p2i[patient_idx]
    prefix = val[start : start + seq_pt + 1]
    return {
        "patient_idx": int(patient_idx),
        "seq_point": int(seq_pt),
        "patient_length": int(length),
        "prefix_length": int(len(prefix)),
        "token_tail": prefix[:, 2].astype(int).tolist()[-limit:],
        "age_tail": prefix[:, 1].astype(int).tolist()[-limit:],
    }


def evaluate_auc_pipeline_predefined(
    model,
    val,
    val_p2i,
    predefined_cohort,
    delphi_labels,
    offset=365.25,
    batch_size=64,
    device="cpu",
    seed=1337,
    debug_examples=0,
):
    """
    Evaluate AUC using a transformed predefined case-control cohort.

    Each record is expected to contain Delphi patient indices and prefix endpoints in
    Delphi patient-record space. transform_predefined_cohort.py converts raw eval_seq
    positions into these endpoints so the model sees exactly the prefix that ReClaim
    used, then predicts the next event.
    """
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)

    if not predefined_cohort:
        cols = ["token", "auc", "n_diseased", "n_healthy"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    unique_pairs, disease_tokens = _collect_predefined_unique_pairs(predefined_cohort)
    print(f"Predefined cohort contains {len(disease_tokens)} unique diseases")
    print(f"Total strata: {len(predefined_cohort)}")
    print(f"Unique (patient_idx, seq_point) pairs: {len(unique_pairs)}")
    if debug_examples:
        print("Sample predefined strata:")
        print(json.dumps([_summarize_predefined_record(record) for record in predefined_cohort[:debug_examples]], indent=2))
        sample_pairs = unique_pairs[:debug_examples]
        print("Sample prefix pairs:")
        print(json.dumps([_summarize_prefix_pair(patient_idx, seq_pt, val, val_p2i) for patient_idx, seq_pt in sample_pairs], indent=2))

    logits_cache, disease_token_to_idx, invalid_pairs = _precompute_predefined_logits(
        model=model,
        val=val,
        val_p2i=val_p2i,
        unique_pairs=unique_pairs,
        disease_tokens=disease_tokens,
        batch_size=batch_size,
        device=device,
    )
    if invalid_pairs:
        print(f"Warning: skipped {len(invalid_pairs)} invalid predefined pairs")
        print(f"Sample invalid pairs: {invalid_pairs[:5]}")

    all_aucs = []
    for record in tqdm(predefined_cohort, desc="Computing AUC for predefined strata"):
        disease_token = int(record["disease_token"])
        disease_idx = disease_token_to_idx.get(disease_token)
        if disease_idx is None:
            continue

        case_logits = []
        for patient_idx, seq_pt in zip(record["case_patient_indices"], record["case_seq_points"]):
            logits = logits_cache.get((int(patient_idx), int(seq_pt)))
            if logits is not None:
                case_logits.append(float(logits[disease_idx]))

        control_logits = []
        for patient_idx, seq_pt in zip(record["control_patient_indices"], record["control_seq_points"]):
            logits = logits_cache.get((int(patient_idx), int(seq_pt)))
            if logits is not None:
                control_logits.append(float(logits[disease_idx]))

        if len(case_logits) == 0 or len(control_logits) == 0:
            continue

        try:
            auc_value, auc_variance = get_auc_delong_var(np.array(control_logits), np.array(case_logits))
        except Exception as e:
            print(f"Warning: AUC computation failed for disease {disease_token}: {e}")
            continue

        out = {
            "token": disease_token,
            "sex": record["sex"],
            "age": int(record["age"]),
            "auc": auc_value,
            "auc_delong": auc_value,
            "auc_variance_delong": auc_variance,
            "n_diseased": len(case_logits),
            "n_healthy": len(control_logits),
            "predefined_cohort": True,
        }
        all_aucs.append(out)

    df_auc_unpooled = pd.DataFrame(all_aucs)
    if df_auc_unpooled.empty:
        print("Warning: No AUCs were calculated from predefined cohort.")
        cols = ["token", "auc", "n_diseased", "n_healthy"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    print(f"Computed AUC for {len(df_auc_unpooled)} strata")

    delphi_labels_subset = delphi_labels[["index", "ICD-10 Chapter (short)", "name", "color", "count"]]
    df_auc_unpooled_merged = df_auc_unpooled.merge(
        delphi_labels_subset, left_on="token", right_on="index", how="left"
    )

    def aggregate_strata(group):
        n = len(group)
        out = {
            "auc": group["auc_delong"].mean(),
            "auc_variance_delong": group["auc_variance_delong"].sum() / (n ** 2) if n > 0 else np.nan,
            "n_samples": n,
            "n_diseased": group["n_diseased"].sum(),
            "n_healthy": group["n_healthy"].sum(),
        }
        return pd.Series(out)

    df_auc = df_auc_unpooled.groupby(["token"]).apply(aggregate_strata).reset_index()
    df_auc_merged = df_auc.merge(delphi_labels, left_on="token", right_on="index", how="left")

    print(f"Final aggregated results: {len(df_auc_merged)} diseases")
    print(f"Mean AUC: {df_auc_merged['auc'].mean():.4f}")

    return df_auc_unpooled_merged, df_auc_merged
