"""Data loading and cohort building for disease baseline."""

import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.data_ops import add_disease_points_ages_and_locations

SEX_TOKEN_MALE = "<SEX-1>"
SEX_TOKEN_FEMALE = "<SEX-2>"


def canonicalize_dx_major_token(token: str) -> str:
    """Normalize DX-MAJOR token to angle-bracket form."""
    token = str(token).strip()
    if token.startswith("<DX-MAJOR_") and token.endswith(">"):
        return token
    if token.startswith("DX-MAJOR_"):
        return f"<{token}>"
    return token


@dataclass
class PatientHistory:
    patient_idx: int
    eval_seq: str
    disease_tokens: List[str]
    disease_ages: List[float]
    seq_points: List[int]
    sex: str


def load_parquet_recursive(data_dir: str, max_files: Optional[int] = None) -> pd.DataFrame:
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    if max_files is not None:
        parquet_files = parquet_files[:max_files]
    return pd.concat((pd.read_parquet(path, engine="pyarrow") for path in parquet_files))


def prepare_eval_dataframe(
    raw_data: pd.DataFrame,
    demo_end_index: int = 3,
    age_index: int = 3,
    seq_len: int = 4096,
    att_type: str = "month",
    min_seq_len: int = 10,
) -> pd.DataFrame:
    return add_disease_points_ages_and_locations(
        raw_data,
        demo_end_index=demo_end_index,
        age_index=age_index,
        seq_len=seq_len,
        att_type=att_type,
        min_seq_len=min_seq_len,
        first_occurrence=True,
    )


def _detect_sex(tokens: List[str]) -> Optional[str]:
    if SEX_TOKEN_MALE in tokens:
        return "male"
    if SEX_TOKEN_FEMALE in tokens:
        return "female"
    return None


def build_histories(eval_df: pd.DataFrame) -> List[PatientHistory]:
    histories: List[PatientHistory] = []
    for patient_idx, row in eval_df.reset_index(drop=True).iterrows():
        tokens = row["eval_seq"].split(" ")
        sex = _detect_sex(tokens)
        if sex is None:
            continue
        disease_points = list(row["disease_points"])
        disease_ages = list(row["disease_points_ages"])
        seq_points = list(row["disease_points_visit_location"])
        disease_tokens = []
        for point in disease_points:
            if 0 <= point < len(tokens):
                disease_tokens.append(canonicalize_dx_major_token(tokens[point]))
            else:
                disease_tokens.append("")
        if len(disease_tokens) != len(disease_ages) or len(disease_tokens) != len(seq_points):
            continue
        histories.append(
            PatientHistory(
                patient_idx=patient_idx,
                eval_seq=row["eval_seq"],
                disease_tokens=disease_tokens,
                disease_ages=disease_ages,
                seq_points=seq_points,
                sex=sex,
            )
        )
    return histories


def histories_to_df(histories: List[PatientHistory]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "patient_idx": h.patient_idx,
            "eval_seq": h.eval_seq,
            "disease_tokens": h.disease_tokens,
            "disease_ages": h.disease_ages,
            "seq_points": h.seq_points,
            "sex": h.sex,
        }
        for h in histories
    ])


def df_to_histories(df: pd.DataFrame) -> List[PatientHistory]:
    return [
        PatientHistory(
            patient_idx=int(row["patient_idx"]),
            eval_seq=str(row["eval_seq"]),
            disease_tokens=list(row["disease_tokens"]),
            disease_ages=list(row["disease_ages"]),
            seq_points=list(row["seq_points"]),
            sex=str(row["sex"]),
        )
        for _, row in df.iterrows()
    ]


def collect_disease_tokens(histories: List[PatientHistory]) -> List[str]:
    diseases: Set[str] = set()
    for h in histories:
        for token in h.disease_tokens:
            canonical_token = canonicalize_dx_major_token(token)
            if canonical_token.startswith("<DX-MAJOR_"):
                diseases.add(canonical_token)
    return sorted(diseases)


def _pred_idx_for_event(disease_ages: List[float], event_idx: int, offset: float) -> int:
    if event_idx <= 0 or event_idx >= len(disease_ages):
        return -1
    threshold = disease_ages[event_idx] - offset
    pred_idx = -1
    for idx in range(len(disease_ages) - 1):
        if disease_ages[idx] < threshold:
            pred_idx = idx
    return pred_idx


def _pred_indices_all_events(disease_ages: np.ndarray, offset: float) -> np.ndarray:
    """Vectorized: compute pred_idx for all event_idx in one pass. O(L) total."""
    n = len(disease_ages)
    if n < 2:
        return np.array([], dtype=np.int64)
    thresholds = disease_ages[1:n] - offset
    pred_indices = np.searchsorted(disease_ages[: n - 1], thresholds, side="right") - 1
    pred_indices[pred_indices < 0] = -1
    return pred_indices


def _process_one_history(h: PatientHistory, offset: float) -> Optional[Dict]:
    """Process single history for precompute. Used by parallel workers."""
    disease_tokens = h.disease_tokens
    disease_ages = np.asarray(h.disease_ages, dtype=np.float64)
    seq_points = h.seq_points
    if len(disease_tokens) < 2:
        return None
    pred_indices = _pred_indices_all_events(disease_ages, offset)
    case_info: Dict[str, Tuple[int, float, int]] = {}
    for i, event_idx in enumerate(range(1, len(disease_tokens))):
        token = canonicalize_dx_major_token(disease_tokens[event_idx])
        if not token.startswith("<DX-MAJOR_"):
            continue
        pred_idx = int(pred_indices[i]) if i < len(pred_indices) else -1
        if pred_idx < 0:
            continue
        if token not in case_info:
            case_info[token] = (pred_idx, float(disease_ages[pred_idx]), int(seq_points[pred_idx]))
    control_info = None
    for i in range(len(pred_indices)):
        if pred_indices[i] >= 0:
            pred_idx = int(pred_indices[i])
            control_info = (pred_idx, float(disease_ages[pred_idx]), int(seq_points[pred_idx]))
            break
    return {
        "patient_idx": h.patient_idx,
        "sex": h.sex,
        "case_info": case_info,
        "control_info": control_info,
    }


def _process_chunk_precompute(args: Tuple[List[PatientHistory], float]) -> List[Dict]:
    """Module-level worker for multiprocessing; must be picklable."""
    chunk, offset = args
    out = []
    for h in chunk:
        r = _process_one_history(h, offset)
        if r is not None:
            out.append(r)
    return out


def save_precomputed_index(precomputed: List[Dict], path: str) -> None:
    """Save precomputed case/control index to disk (pickle)."""
    with open(path, "wb") as f:
        pickle.dump(precomputed, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_precomputed_index(path: str) -> List[Dict]:
    """Load precomputed case/control index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def precompute_case_control_index(
    histories: List[PatientHistory],
    offset: float,
    num_workers: int = 0,
) -> List[Dict]:
    """Precompute case_info and control_info per patient. Supports parallelization for 1M+ patients."""

    if num_workers is not None and num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunk_size = max(1, len(histories) // (num_workers * 4))
        chunks = [histories[i : i + chunk_size] for i in range(0, len(histories), chunk_size)]
        chunks_with_offset = [(c, offset) for c in chunks]
        result: List[Dict] = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            it = ex.map(_process_chunk_precompute, chunks_with_offset)
            for chunk_result in tqdm(
                it,
                total=len(chunks),
                desc="Precompute index",
                unit="chunk",
                unit_scale=True,
                smoothing=0.1,
            ):
                result.extend(chunk_result)
        return result

    result: List[Dict] = []
    for h in tqdm(
        histories,
        desc="Precompute index",
        unit="patient",
        unit_scale=True,
        smoothing=0.1,
    ):
        r = _process_one_history(h, offset)
        if r is not None:
            result.append(r)
    return result


def build_case_control_rows_from_precomputed(
    precomputed: List[Dict],
    target_token: str,
    age_groups: np.ndarray,
    age_step: int,
    stratify_by_sex_age: bool = True,
) -> pd.DataFrame:
    """Build cohort for one disease from precomputed index. Single pass O(M) over precomputed."""
    age_groups_set: Set[int] = set(int(a) for a in age_groups)
    rows: List[Dict] = []
    for entry in precomputed:
        sex = entry["sex"]
        case_info = entry["case_info"]
        control_info = entry["control_info"]
        pred_age_days: float
        seq_point: int
        label: int
        if target_token in case_info:
            pred_idx, pred_age_days, seq_point = case_info[target_token]
            label = 1
        elif control_info is not None:
            pred_idx, pred_age_days, seq_point = control_info
            label = 0
        else:
            continue
        pred_age = pred_age_days / 365.25
        age_low = int(pred_age // age_step) * age_step if age_step > 0 else int(pred_age)
        if stratify_by_sex_age and age_low not in age_groups_set:
            continue
        rows.append({
            "patient_idx": entry["patient_idx"],
            "sex": sex,
            "age": age_low,
            "age_step": age_step,
            "label": label,
            "seq_point": seq_point,
            "pred_age_days": pred_age_days,
        })
    return pd.DataFrame(rows)


def build_case_control_lookup_from_precomputed(
    precomputed: List[Dict],
    age_groups: np.ndarray,
    age_step: int,
    stratify_by_sex_age: bool = True,
) -> Dict[str, Any]:
    """Build reusable lookup for fast per-disease val cohorts.

    The lookup stores:
    - case_rows_by_token: token -> rows where patient is a case for token
    - case_patient_ids_by_token: token -> set(patient_idx)
    - control_rows: rows from patients with valid control_info
    """
    age_groups_set: Set[int] = set(int(a) for a in age_groups)
    case_rows_by_token: Dict[str, List[Dict]] = {}
    case_patient_ids_by_token: Dict[str, Set[int]] = {}
    control_rows: List[Dict] = []

    for entry in precomputed:
        patient_idx = int(entry["patient_idx"])
        sex = entry["sex"]
        case_info = entry["case_info"]
        control_info = entry["control_info"]

        for token, (_, pred_age_days, seq_point) in case_info.items():
            pred_age = pred_age_days / 365.25
            age_low = int(pred_age // age_step) * age_step if age_step > 0 else int(pred_age)
            if stratify_by_sex_age and age_low not in age_groups_set:
                continue
            row = {
                "patient_idx": patient_idx,
                "sex": sex,
                "age": age_low,
                "age_step": age_step,
                "label": 1,
                "seq_point": int(seq_point),
                "pred_age_days": float(pred_age_days),
            }
            case_rows_by_token.setdefault(token, []).append(row)
            case_patient_ids_by_token.setdefault(token, set()).add(patient_idx)

        if control_info is not None:
            _, pred_age_days, seq_point = control_info
            pred_age = pred_age_days / 365.25
            age_low = int(pred_age // age_step) * age_step if age_step > 0 else int(pred_age)
            if stratify_by_sex_age and age_low not in age_groups_set:
                continue
            control_rows.append(
                {
                    "patient_idx": patient_idx,
                    "sex": sex,
                    "age": age_low,
                    "age_step": age_step,
                    "label": 0,
                    "seq_point": int(seq_point),
                    "pred_age_days": float(pred_age_days),
                }
            )

    return {
        "case_rows_by_token": case_rows_by_token,
        "case_patient_ids_by_token": case_patient_ids_by_token,
        "control_rows": control_rows,
    }


def build_case_control_rows_from_lookup(
    lookup: Dict[str, Any],
    target_token: str,
) -> pd.DataFrame:
    """Build cohort for one disease from pre-built lookup."""
    case_rows = lookup["case_rows_by_token"].get(target_token, [])
    case_patients = lookup["case_patient_ids_by_token"].get(target_token, set())
    control_rows = lookup["control_rows"]
    if not case_rows and not control_rows:
        return pd.DataFrame()
    if case_patients:
        filtered_controls = [r for r in control_rows if r["patient_idx"] not in case_patients]
    else:
        filtered_controls = control_rows
    return pd.DataFrame(case_rows + filtered_controls)


def build_case_control_rows(
    histories: List[PatientHistory],
    target_token: str,
    age_groups: np.ndarray,
    offset: float,
    stratify_by_sex_age: bool = True,
) -> pd.DataFrame:
    age_step = int(age_groups[1] - age_groups[0])
    rows: List[Dict] = []

    if stratify_by_sex_age:
        for sex in ["male", "female"]:
            sex_histories = [h for h in histories if h.sex == sex]
            for age_low in age_groups:
                age_high = age_low + age_step
                for h in sex_histories:
                    disease_tokens = h.disease_tokens
                    disease_ages = h.disease_ages
                    seq_points = h.seq_points
                    if len(disease_tokens) < 2:
                        continue
                    has_target = target_token in disease_tokens
                    if has_target:
                        first_target_event_idx = -1
                        for event_idx in range(1, len(disease_tokens)):
                            if disease_tokens[event_idx] == target_token:
                                first_target_event_idx = event_idx
                                break
                        if first_target_event_idx == -1:
                            continue
                        pred_idx = _pred_idx_for_event(disease_ages, first_target_event_idx, offset)
                        if pred_idx == -1:
                            continue
                        pred_age = disease_ages[pred_idx] / 365.25
                        if not (age_low <= pred_age < age_high):
                            continue
                        rows.append({
                            "patient_idx": h.patient_idx, "sex": sex, "age": int(age_low),
                            "age_step": age_step, "label": 1, "pred_idx": pred_idx,
                            "seq_point": int(seq_points[pred_idx]),
                            "pred_age_days": float(disease_ages[pred_idx]),
                        })
                    else:
                        selected_control = None
                        for event_idx in range(1, len(disease_tokens)):
                            pred_idx = _pred_idx_for_event(disease_ages, event_idx, offset)
                            if pred_idx == -1:
                                continue
                            pred_age = disease_ages[pred_idx] / 365.25
                            if age_low <= pred_age < age_high:
                                selected_control = (pred_idx, disease_ages[pred_idx])
                                break
                        if selected_control is None:
                            continue
                        pred_idx, pred_age_days = selected_control
                        rows.append({
                            "patient_idx": h.patient_idx, "sex": sex, "age": int(age_low),
                            "age_step": age_step, "label": 0, "pred_idx": pred_idx,
                            "seq_point": int(seq_points[pred_idx]),
                            "pred_age_days": float(pred_age_days),
                        })
    else:
        for h in histories:
            disease_tokens = h.disease_tokens
            disease_ages = h.disease_ages
            seq_points = h.seq_points
            sex = h.sex
            if len(disease_tokens) < 2:
                continue
            has_target = target_token in disease_tokens
            if has_target:
                first_target_event_idx = -1
                for event_idx in range(1, len(disease_tokens)):
                    if disease_tokens[event_idx] == target_token:
                        first_target_event_idx = event_idx
                        break
                if first_target_event_idx == -1:
                    continue
                pred_idx = _pred_idx_for_event(disease_ages, first_target_event_idx, offset)
                if pred_idx == -1:
                    continue
                pred_age = disease_ages[pred_idx] / 365.25
                age_low = int(pred_age // age_step) * age_step if age_step > 0 else int(pred_age)
                rows.append({
                    "patient_idx": h.patient_idx, "sex": sex, "age": int(age_low),
                    "age_step": age_step, "label": 1, "pred_idx": pred_idx,
                    "seq_point": int(seq_points[pred_idx]),
                    "pred_age_days": float(disease_ages[pred_idx]),
                })
            else:
                selected_control = None
                for event_idx in range(1, len(disease_tokens)):
                    pred_idx = _pred_idx_for_event(disease_ages, event_idx, offset)
                    if pred_idx == -1:
                        continue
                    selected_control = (pred_idx, disease_ages[pred_idx])
                    break
                if selected_control is None:
                    continue
                pred_idx, pred_age_days = selected_control
                pred_age = pred_age_days / 365.25
                age_low = int(pred_age // age_step) * age_step if age_step > 0 else int(pred_age)
                rows.append({
                    "patient_idx": h.patient_idx, "sex": sex, "age": int(age_low),
                    "age_step": age_step, "label": 0, "pred_idx": pred_idx,
                    "seq_point": int(seq_points[pred_idx]),
                    "pred_age_days": float(pred_age_days),
                })
    return pd.DataFrame(rows)


def build_val_cohort_from_case_control_ids(
    case_control_ids_df: pd.DataFrame,
    target_token: str,
    target_token_value: object,
    age_groups: np.ndarray,
) -> pd.DataFrame:
    """Rows of the shared case/control file for one disease.

    Matches on `disease_token_str` when the file has it: the integer
    `disease_token` is an id of whichever tokenizer built the file and need not
    agree with the feature vocabulary. Older files without the string column
    fall back to the integer id under the feature vocabulary.
    """
    if "disease_token_str" in case_control_ids_df.columns:
        subset = case_control_ids_df[case_control_ids_df["disease_token_str"] == target_token]
    else:
        target_int = int(target_token_value) if isinstance(target_token_value, (int, float)) else None
        if target_int is None:
            return pd.DataFrame()
        subset = case_control_ids_df[case_control_ids_df["disease_token"].astype(int) == target_int]
    if subset.empty:
        return pd.DataFrame()
    age_step = int(age_groups[1] - age_groups[0]) if len(age_groups) >= 2 else 5

    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            return x.tolist()
        return list(x)

    rows: List[Dict] = []
    for _, r in subset.iterrows():
        case_ids = _ensure_list(r["case_ids"])
        control_ids = _ensure_list(r["control_ids"])
        case_seq_points = _ensure_list(r["case_seq_points"])
        control_seq_points = _ensure_list(r["control_seq_points"])
        age = int(r["age"])
        sex = str(r["sex"])
        pred_age_days = (age + age_step / 2.0) * 365.25
        for pid, sp in zip(case_ids, case_seq_points):
            rows.append({
                "patient_idx": int(pid), "sex": sex, "age": age, "age_step": age_step,
                "label": 1, "seq_point": int(sp), "pred_age_days": pred_age_days,
            })
        for pid, sp in zip(control_ids, control_seq_points):
            rows.append({
                "patient_idx": int(pid), "sex": sex, "age": age, "age_step": age_step,
                "label": 0, "seq_point": int(sp), "pred_age_days": pred_age_days,
            })
    return pd.DataFrame(rows)


def sanitize_token_for_filename(token: str) -> str:
    return (
        token.replace("<", "").replace(">", "").replace("/", "_")
        .replace(":", "_").replace(" ", "_")
    )
