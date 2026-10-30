"""Feature extraction with shared CountVectorizer."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from lightgbm_baseline.data import PatientHistory

FEATURE_TOKEN_PREFIXES = ("<DX-MAJOR_", "<PROC-", "<RX-", "<ATT-")

# Feature scaling must match how the models being loaded were TRAINED.
#   "raw"     - raw token counts (default; used for the reported results)
#   "relfreq" - per-patient L1 relative frequency
# Scoring a model with the other scaling silently costs ~0.06 AUC, so it is an
# explicit switch (BASELINE_FEATURE_MODE) rather than something inferred.
FEATURE_MODE = os.environ.get("BASELINE_FEATURE_MODE", "raw").lower()
if FEATURE_MODE not in ("raw", "relfreq"):
    raise ValueError(
        f"BASELINE_FEATURE_MODE must be 'raw' or 'relfreq', got {FEATURE_MODE!r}"
    )


def _scale_sparse(x_sparse):
    """Apply the configured token-count scaling."""
    if FEATURE_MODE == "relfreq":
        return normalize(x_sparse, norm="l1", axis=1, copy=False)
    return x_sparse


def extract_feature_text(prefix_seq: str) -> str:
    """Extract only DX-MAJOR, PROC, RX, ATT tokens for sparse features."""
    tokens = prefix_seq.split(" ")
    filtered = [t for t in tokens if any(t.startswith(p) for p in FEATURE_TOKEN_PREFIXES)]
    return " ".join(filtered)


def extract_dense_features(prefix_seq: str, pred_age_days: float, sex: str) -> List[float]:
    tokens = prefix_seq.split(" ")
    dx_count = float(sum(token.startswith("<DX-MAJOR_") for token in tokens))
    proc_count = float(sum(token.startswith("<PROC-") for token in tokens))
    rx_count = float(sum(token.startswith("<RX-") for token in tokens))
    visit_count = float(sum(token.startswith("<ATT-") for token in tokens))
    L = len(tokens)
    age_years = float(pred_age_days / 365.25)
    male_flag = 1.0 if sex == "male" else 0.0
    if FEATURE_MODE == "relfreq":
        denom = float(L) if L > 0 else 1.0
        return [dx_count / denom, proc_count / denom, rx_count / denom,
                visit_count / denom, float(np.log1p(L)), age_years, male_flag]
    return [dx_count, proc_count, rx_count, visit_count, float(L), age_years, male_flag]


def cohort_to_prefixes_and_dense(
    cohort_df: pd.DataFrame,
    histories: List[PatientHistory],
    seq_tokens_cache: Optional[Dict[int, List[str]]] = None,
    history_map: Optional[Dict[int, "PatientHistory"]] = None,
) -> Tuple[List[str], List[List[float]]]:
    """Extract prefix sequences and dense features from cohort. Uses itertuples + optional cache."""
    if seq_tokens_cache is None:
        seq_tokens_cache = {}
    if history_map is None:
        history_map = {h.patient_idx: h for h in histories}
    prefixes: List[str] = []
    dense_features: List[List[float]] = []
    df = cohort_df.reset_index(drop=True)
    for row in df.itertuples(index=False):
        pid = int(row.patient_idx)
        seq_point = int(row.seq_point)
        pred_age_days = float(row.pred_age_days)
        sex = str(row.sex)
        if pid not in seq_tokens_cache:
            seq_tokens_cache[pid] = history_map[pid].eval_seq.split(" ")
        seq_tokens = seq_tokens_cache[pid]
        prefix_tokens = seq_tokens[:seq_point]
        prefix_seq = " ".join(prefix_tokens)
        filtered = [t for t in prefix_tokens if any(t.startswith(p) for p in FEATURE_TOKEN_PREFIXES)]
        prefixes.append(" ".join(filtered))
        dx_c = sum(1 for t in prefix_tokens if t.startswith("<DX-MAJOR_"))
        proc_c = sum(1 for t in prefix_tokens if t.startswith("<PROC-"))
        rx_c = sum(1 for t in prefix_tokens if t.startswith("<RX-"))
        visit_c = sum(1 for t in prefix_tokens if t.startswith("<ATT-"))
        L = len(prefix_tokens)
        male_flag = 1.0 if sex == "male" else 0.0
        if FEATURE_MODE == "relfreq":
            denom = float(L) if L > 0 else 1.0
            dense_features.append([
                dx_c / denom, proc_c / denom, rx_c / denom, visit_c / denom,
                float(np.log1p(L)), pred_age_days / 365.25, male_flag,
            ])
        else:
            dense_features.append([
                float(dx_c), float(proc_c), float(rx_c), float(visit_c),
                float(L), pred_age_days / 365.25, male_flag,
            ])
    return prefixes, dense_features


def materialize_features(
    cohort_df: pd.DataFrame,
    histories: List[PatientHistory],
    vectorizer: Optional[CountVectorizer] = None,
    max_features: int = 20000,
    fit_vectorizer: bool = True,
) -> Tuple[csr_matrix, pd.DataFrame, CountVectorizer]:
    """Transform cohort to feature matrix."""
    prefixes, dense_features = cohort_to_prefixes_and_dense(cohort_df, histories)
    work_df = cohort_df.copy().reset_index(drop=True)
    work_df["feature_text"] = prefixes

    if fit_vectorizer:
        vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=str.split,
            token_pattern=None,
            max_features=max_features,
        )
        x_sparse = vectorizer.fit_transform(prefixes)
    else:
        if vectorizer is None:
            raise ValueError("vectorizer must not be None when fit_vectorizer=False")
        x_sparse = vectorizer.transform(prefixes)

    x_sparse = _scale_sparse(x_sparse)

    x_dense = csr_matrix(np.asarray(dense_features, dtype=np.float32))
    x = hstack([x_sparse, x_dense], format="csr")
    return x, work_df, vectorizer



def build_vectorizer_from_tokenizer(
    tokens_file: str,
) -> CountVectorizer:
    """Build CountVectorizer from the feature vocabulary (JSON map or tokenizer directory)."""
    from lightgbm_baseline.utils import load_feature_vocab

    full_vocab: Dict[str, int] = load_feature_vocab(tokens_file)
    filtered = {
        token: idx for token, idx in full_vocab.items()
        if any(token.startswith(p) for p in FEATURE_TOKEN_PREFIXES)
    }
    # CountVectorizer vocabulary: token -> column index (0, 1, 2, ...)
    vocabulary = {t: i for i, t in enumerate(sorted(filtered.keys()))}
    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
        vocabulary=vocabulary,
    )
    return vectorizer


def fit_global_vectorizer(
    all_prefixes: List[str],
    max_features: int = 30000,
) -> CountVectorizer:
    """Fit CountVectorizer on prefix sequences (slow, prefer build_vectorizer_from_tokenizer)."""
    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
        max_features=max_features,
    )
    vectorizer.fit(all_prefixes)
    return vectorizer


def transform_cohort_with_vectorizer(
    cohort_df: pd.DataFrame,
    histories: List[PatientHistory],
    vectorizer: CountVectorizer,
    seq_tokens_cache: Optional[Dict[int, List[str]]] = None,
    history_map: Optional[Dict[int, PatientHistory]] = None,
) -> Tuple[csr_matrix, pd.DataFrame]:
    """Transform cohort to features using pre-fitted vectorizer."""
    prefixes, dense_features = cohort_to_prefixes_and_dense(
        cohort_df, histories, seq_tokens_cache=seq_tokens_cache, history_map=history_map
    )
    work_df = cohort_df.copy().reset_index(drop=True)
    work_df["feature_text"] = prefixes
    x_sparse = _scale_sparse(vectorizer.transform(prefixes))
    x_dense = csr_matrix(np.asarray(dense_features, dtype=np.float32))
    x = hstack([x_sparse, x_dense], format="csr")
    return x, work_df


def save_vectorizer(vectorizer: CountVectorizer, path: str) -> None:
    """Save vectorizer vocabulary to JSON. Converts to plain Python types for reliable serialization."""
    vocab = getattr(vectorizer, "vocabulary_", None) or getattr(vectorizer, "vocabulary", None)
    if vocab is None:
        raise ValueError("Vectorizer has no vocabulary (neither vocabulary_ nor vocabulary)")
    payload = {
        "vocabulary": {k: int(v) for k, v in vocab.items()},
        "max_features": int(vectorizer.max_features) if vectorizer.max_features is not None else len(vocab),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_vectorizer(path: str) -> CountVectorizer:
    """Load vectorizer from saved vocabulary."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
        max_features=payload.get("max_features", 30000),
        vocabulary=payload["vocabulary"],
    )
    return vectorizer
