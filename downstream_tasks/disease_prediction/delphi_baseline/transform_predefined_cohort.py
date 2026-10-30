#!/usr/bin/env python3
"""
Prepare a predefined case/control cohort for Delphi evaluation.

The cohort is the one written by prepare_disease_case_controls_fast.py, so
Delphi is scored on exactly the same patients and prediction points as
ReClaim. This script does three things:
1. Maps cohort endpoints to Delphi disease tokens via the ICD-10 code in
   `disease_token_str` (independent of the tokenizer that built the cohort).
2. Maps cohort row ids from traj_test_pd.parquet to Delphi patient indices.
3. Converts raw eval_seq seq points into Delphi prefix indices using the same
   disease-only transformation logic as preprocess.py.

The output JSON is designed for evaluate.py --predefined_cohort.
An intermediate parquet with row-level conversions is also written for debugging.
"""

import argparse
import json
import logging
import os
import re
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from preprocess import get_prediction_points_data, load_disease_mapping_csv
from utils import get_p2i


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_dir, f"transform_predefined_cohort_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath),
    ],
)
logger = logging.getLogger("DelphiCohortTransform")
logger.info(f"Log file saved to: {log_filepath}")
logger.info(f"Current node: {socket.gethostname()}")


SEX_MAPPING = {"<SEX-2>": 1, "<SEX-1>": 2}
DISEASE_STR_COLUMN = "disease_token_str"
DX_MAJOR_RE = re.compile(r"<DX-MAJOR_([^>]+)>")

COHORT_COLUMNS = [
    "sex",
    "disease_token",
    "age",
    "age_step",
    "case_ids",
    "control_ids",
    "case_seq_points",
    "control_seq_points",
]


def _append_limited(bucket, payload, limit):
    if limit > 0 and len(bucket) < limit:
        bucket.append(payload)


def _log_json(title, payload):
    if payload:
        logger.info("%s:%s", title, json.dumps(payload, indent=2))


def _serialize_seq_window(raw_tokens, center, window):
    left = max(0, int(center) - window)
    right = min(len(raw_tokens), int(center) + window + 1)
    return [
        {"position": int(idx), "token": raw_tokens[idx]}
        for idx in range(left, right)
    ]


def _summarize_events(events, limit=5):
    return [
        {
            "raw_position": int(event["raw_position"]),
            "age": int(event["age"]),
            "delphi_raw_token": int(event["delphi_raw_token"]),
            "delphi_record_index": int(event["delphi_record_index"]),
        }
        for event in events[:limit]
    ]


def _summarize_cohort_row(row, limit=3):
    return {
        "sex": row["sex"],
        "disease_token": int(row["disease_token"]),
        "age": int(row["age"]),
        "age_step": int(row.get("age_step", 5)),
        "n_cases": len(row.get("case_ids") or []),
        "n_controls": len(row.get("control_ids") or []),
        "case_examples": [
            {"raw_row_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip((row.get("case_ids") or [])[:limit], (row.get("case_seq_points") or [])[:limit])
        ],
        "control_examples": [
            {"raw_row_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip((row.get("control_ids") or [])[:limit], (row.get("control_seq_points") or [])[:limit])
        ],
    }


def _summarize_prepared_row(row, limit=3):
    return {
        "disease_token": int(row["disease_token"]),
        "disease_code": row.get("disease_code"),
        "delphi_raw_token": int(row.get("delphi_raw_token", -1)),
        "sex": row["sex"],
        "age": int(row["age"]),
        "n_cases": int(row.get("n_cases", len(row.get("case_patient_indices", [])))),
        "n_controls": int(row.get("n_controls", len(row.get("control_patient_indices", [])))),
        "case_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(row.get("case_patient_indices", [])[:limit], row.get("case_seq_points", [])[:limit])
        ],
        "control_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(row.get("control_patient_indices", [])[:limit], row.get("control_seq_points", [])[:limit])
        ],
    }


def load_pid_mapping(pid_mapping_path: str):
    """Load person_id -> Delphi patient index mapping."""
    pid_map = {}
    mapping_file = Path(pid_mapping_path)
    if not mapping_file.exists():
        raise FileNotFoundError(f"PID mapping file not found: {pid_mapping_path}")

    if mapping_file.suffix.lower() == ".tsv":
        with mapping_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                person_id, idx = line.split("	")
                pid_map[str(person_id)] = int(idx)
    else:
        with mapping_file.open() as f:
            raw = json.load(f)
        pid_map = {str(k): int(v) for k, v in raw.items()}

    return pid_map


def load_case_control_rows(case_control_file: str):
    """Load the cohort parquet into Python dict rows.

    `disease_token_str` is the tokenizer-independent endpoint identity (e.g.
    `<DX-MAJOR_A36>`); the integer `disease_token` depends on whichever tokenizer
    built the cohort and is not used.
    """
    pf = pq.ParquetFile(case_control_file)
    if DISEASE_STR_COLUMN not in pf.schema_arrow.names:
        raise SystemExit(
            f"ERROR: cohort file has no '{DISEASE_STR_COLUMN}' column. Rebuild it with "
            "prepare_disease_case_controls_fast.py, which writes it."
        )
    columns = list(COHORT_COLUMNS) + [DISEASE_STR_COLUMN]
    rows = []
    for batch in tqdm(pf.iter_batches(batch_size=64, columns=columns), desc="Loading cohort parquet"):
        batch_dict = batch.to_pydict()
        n_rows = len(batch_dict["disease_token"])
        for i in range(n_rows):
            rows.append({col: batch_dict[col][i] for col in columns})
    return rows


def make_disease_resolver(disease_mapping):
    """Return a function mapping a cohort row to (delphi_shifted_token, disease_code).

    The three-character ICD-10 code is parsed out of `disease_token_str` and looked
    up in the Delphi disease map. Returns (None, code) if the code is not a Delphi
    endpoint.
    """
    def resolve(row):
        raw = str(row.get(DISEASE_STR_COLUMN) or "")
        match = DX_MAJOR_RE.match(raw)
        if match is None:
            return None, raw
        code = match.group(1)
        raw_token = disease_mapping.get(code)
        if raw_token is None:
            return None, code
        return int(raw_token) + 1, code

    return resolve


def collect_needed_rows_and_seq_points(case_control_rows):
    """Collect all referenced trajectory row ids and raw seq points."""
    needed_rows = set()
    seq_points_by_row = defaultdict(set)
    for cohort_row_idx, row in enumerate(case_control_rows):
        for label in ("case", "control"):
            row_ids = row.get(f"{label}_ids") or []
            seq_points = row.get(f"{label}_seq_points") or []
            if len(row_ids) != len(seq_points):
                raise ValueError(
                    f"Cohort row {cohort_row_idx} has mismatched {label} ids/seq_points lengths: "
                    f"{len(row_ids)} ids vs {len(seq_points)} seq points"
                )
            for row_idx, seq_pt in zip(row_ids, seq_points):
                row_idx = int(row_idx)
                seq_pt = int(seq_pt)
                needed_rows.add(row_idx)
                seq_points_by_row[row_idx].add(seq_pt)
    return needed_rows, seq_points_by_row


def build_expected_delphi_events(seq: str, disease_mapping: dict, att_time_unit: str, att_time_mode: str):
    """
    Replay preprocess.py disease extraction for one raw sequence.

    Returns the sex token id and the disease events exactly as preprocess.py
    would write them into the Delphi binary.
    """
    raw_positions, ages, disease_ids, sex_id = get_prediction_points_data(
        seq,
        disease_mapping=disease_mapping,
        sex_mapping=SEX_MAPPING,
        demo_end_index=3,
        age_index=2,
        sex_index=0,
        first_occurrence=True,
        att_time_unit=att_time_unit,
        att_time_mode=att_time_mode,
    )

    events = []
    for order, (raw_pos, age, did) in enumerate(zip(raw_positions, ages, disease_ids)):
        if did == -1:
            continue
        events.append(
            {
                "raw_position": int(raw_pos),
                "age": int(age),
                "delphi_raw_token": int(did),
                "order": int(order),
            }
        )

    events.sort(key=lambda item: (item["age"], item["order"]))
    for record_idx, event in enumerate(events, start=1):
        event["delphi_record_index"] = record_idx

    return int(sex_id), events


def build_patient_cache(
    traj_parquet_path: str,
    needed_rows,
    seq_points_by_row,
    pid_map,
    disease_mapping,
    pid_col: str,
    seq_col: str,
    att_time_unit: str,
    att_time_mode: str,
    debug_examples=0,
    debug_token_window=5,
):
    """Stream only the referenced trajectory rows and prepare Delphi-aligned metadata."""
    pf = pq.ParquetFile(traj_parquet_path)
    patient_cache = {}
    patient_debug_examples = []
    row_offset = 0
    remaining_rows = set(int(x) for x in needed_rows)

    for batch in tqdm(pf.iter_batches(batch_size=2048, columns=[pid_col, seq_col]), desc="Scanning trajectory parquet"):
        batch_dict = batch.to_pydict()
        pids = batch_dict[pid_col]
        seqs = batch_dict[seq_col]
        batch_size = len(pids)

        for i in range(batch_size):
            raw_row_idx = row_offset + i
            if raw_row_idx not in remaining_rows:
                continue

            person_id = str(pids[i])
            seq = seqs[i]
            raw_tokens = seq.split(" ")
            sex_id, events = build_expected_delphi_events(
                seq=seq,
                disease_mapping=disease_mapping,
                att_time_unit=att_time_unit,
                att_time_mode=att_time_mode,
            )

            position_to_event = {event["raw_position"]: event for event in events}
            seq_points = sorted(int(seq_pt) for seq_pt in seq_points_by_row[raw_row_idx])
            token_hits = {
                int(seq_pt): position_to_event.get(int(seq_pt))
                for seq_pt in seq_points
            }
            events_by_token = defaultdict(list)
            for event in events:
                events_by_token[int(event["delphi_raw_token"])].append(event)
            seq_windows = {
                int(seq_pt): _serialize_seq_window(raw_tokens, int(seq_pt), debug_token_window)
                for seq_pt in seq_points[: max(debug_examples, 0)]
            }

            patient_cache[raw_row_idx] = {
                "person_id": person_id,
                "delphi_patient_index": pid_map.get(person_id),
                "raw_seq_length": len(raw_tokens),
                "sex_id": sex_id,
                "events": events,
                "events_by_token": dict(events_by_token),
                "token_hits": token_hits,
                "seq_windows": seq_windows,
            }
            _append_limited(
                patient_debug_examples,
                {
                    "raw_row_idx": int(raw_row_idx),
                    "person_id": person_id,
                    "delphi_patient_index": (
                        None if pid_map.get(person_id) is None else int(pid_map.get(person_id))
                    ),
                    "raw_seq_length": int(len(raw_tokens)),
                    "seq_windows": seq_windows,
                    "event_examples": _summarize_events(events, limit=max(debug_examples, 1)),
                },
                debug_examples,
            )
            remaining_rows.remove(raw_row_idx)

        row_offset += batch_size
        if not remaining_rows:
            break

    return patient_cache, sorted(remaining_rows), patient_debug_examples


def validate_patient_alignment(val, val_p2i, delphi_patient_index: int, sex_id: int, events):
    """Validate that replayed preprocess output matches the Delphi binary exactly."""
    if delphi_patient_index is None or delphi_patient_index >= len(val_p2i):
        return False, "patient_index_missing_or_out_of_range"

    start, length = val_p2i[delphi_patient_index]
    patient_tokens = val[start : start + length, 2].astype(int).tolist()
    patient_ages = val[start : start + length, 1].astype(int).tolist()

    expected_tokens = [sex_id] + [event["delphi_raw_token"] for event in events]
    expected_ages = [0] + [event["age"] for event in events]

    if patient_tokens != expected_tokens:
        return False, "token_mismatch"
    if patient_ages != expected_ages:
        return False, "age_mismatch"
    return True, "ok"


def raw_seq_point_to_delphi_prefix_index(raw_seq_point: int, events, hit_event):
    """
    Convert a raw eval_seq boundary into the Delphi prefix endpoint.

    ReClaim case/control evaluation uses tokens strictly before raw seq_point,
    i.e. seq.split(" ")[:seq_point]. When seq_point lands on a disease token,
    that token is the next target and must be excluded from the Delphi prefix.

    The returned index is the last Delphi record included in model input:
    - 0 means sex-only input
    - k > 0 means include records 0..k and predict record k+1
    """
    if hit_event is not None:
        return int(hit_event["delphi_record_index"]) - 1

    prior_record_indices = [
        int(event["delphi_record_index"])
        for event in events
        if event["raw_position"] < raw_seq_point
    ]
    return max(prior_record_indices, default=0)


def build_predefined_cohort_records(
    case_control_rows,
    patient_cache,
    val,
    val_p2i,
    resolve_disease,
    collect_intermediate=False,
    debug_examples=0,
):
    """Convert cohort rows into Delphi-evaluable JSON records and optional row-level debug rows."""
    prepared_rows = []
    intermediate_rows = [] if collect_intermediate else None
    stats = defaultdict(int)
    debug_summary = {
        "prepared_examples": [],
        "dropped_case_examples": [],
        "dropped_control_examples": [],
        "alignment_failure_examples": [],
        "empty_strata_examples": [],
    }

    for cohort_row_idx, row in enumerate(tqdm(case_control_rows, desc="Preparing cohort metadata")):
        delphi_shifted_token, disease_code = resolve_disease(row)
        if delphi_shifted_token is None:
            stats["missing_disease_mapping"] += 1
            continue

        delphi_raw_token = int(delphi_shifted_token) - 1
        prepared = {
            "disease_token": int(delphi_shifted_token),
            "disease_code": disease_code,
            "delphi_raw_token": delphi_raw_token,
            "sex": row["sex"],
            "age": int(row["age"]),
            "age_step": int(row.get("age_step", 5)),
            "case_patient_indices": [],
            "control_patient_indices": [],
            "case_seq_points": [],
            "control_seq_points": [],
        }

        for label in ("case", "control"):
            row_ids = row[f"{label}_ids"]
            seq_points = row[f"{label}_seq_points"]

            for raw_row_idx, raw_seq_point in zip(row_ids, seq_points):
                raw_row_idx = int(raw_row_idx)
                raw_seq_point = int(raw_seq_point)
                patient_info = patient_cache.get(raw_row_idx)
                if patient_info is None:
                    stats[f"missing_patient_row_{label}"] += 1
                    continue

                delphi_patient_index = patient_info["delphi_patient_index"]
                if delphi_patient_index is None:
                    stats[f"missing_pid_mapping_{label}"] += 1
                    continue

                if raw_seq_point < 0 or raw_seq_point >= patient_info["raw_seq_length"]:
                    stats[f"raw_seq_point_out_of_range_{label}"] += 1
                    continue

                alignment_ok, alignment_msg = validate_patient_alignment(
                    val,
                    val_p2i,
                    delphi_patient_index,
                    patient_info["sex_id"],
                    patient_info["events"],
                )
                if not alignment_ok:
                    stats[f"alignment_failed_{label}"] += 1
                    _append_limited(
                        debug_summary["alignment_failure_examples"],
                        {
                            "cohort_row_idx": int(cohort_row_idx),
                            "label": label,
                            "raw_row_idx": int(raw_row_idx),
                            "delphi_patient_index": int(delphi_patient_index),
                            "alignment_msg": alignment_msg,
                        },
                        debug_examples,
                    )
                    continue

                hit_event = patient_info["token_hits"].get(raw_seq_point)
                hit_any_disease = hit_event is not None
                hit_target_at_boundary = hit_any_disease and hit_event["delphi_raw_token"] == delphi_raw_token
                target_events = patient_info["events_by_token"].get(delphi_raw_token, [])
                target_positions = [int(event["raw_position"]) for event in target_events]
                has_target_after_prefix = any(pos >= raw_seq_point for pos in target_positions)
                has_target_anywhere = bool(target_positions)

                if label == "case" and not has_target_after_prefix:
                    stats["case_target_not_after_prefix"] += 1
                    _append_limited(
                        debug_summary["dropped_case_examples"],
                        {
                            "cohort_row_idx": int(cohort_row_idx),
                            "raw_row_idx": int(raw_row_idx),
                            "raw_seq_point": int(raw_seq_point),
                            "delphi_raw_token": int(delphi_raw_token),
                            "target_positions": target_positions[:10],
                            "hit_event": None if hit_event is None else {
                                "raw_position": int(hit_event["raw_position"]),
                                "delphi_raw_token": int(hit_event["delphi_raw_token"]),
                            },
                            "seq_window": patient_info["seq_windows"].get(raw_seq_point, []),
                        },
                        debug_examples,
                    )

                if label == "control" and has_target_anywhere:
                    stats["control_contains_target_disease"] += 1
                    _append_limited(
                        debug_summary["dropped_control_examples"],
                        {
                            "cohort_row_idx": int(cohort_row_idx),
                            "raw_row_idx": int(raw_row_idx),
                            "raw_seq_point": int(raw_seq_point),
                            "delphi_raw_token": int(delphi_raw_token),
                            "target_positions": target_positions[:10],
                            "seq_window": patient_info["seq_windows"].get(raw_seq_point, []),
                        },
                        debug_examples,
                    )

                delphi_seq_point = raw_seq_point_to_delphi_prefix_index(
                    raw_seq_point=raw_seq_point,
                    events=patient_info["events"],
                    hit_event=hit_event,
                )
                patient_length = int(val_p2i[delphi_patient_index][1])
                if delphi_seq_point < 0 or delphi_seq_point >= patient_length:
                    stats[f"delphi_seq_point_out_of_range_{label}"] += 1
                    continue

                prepared[f"{label}_patient_indices"].append(int(delphi_patient_index))
                prepared[f"{label}_seq_points"].append(int(delphi_seq_point))
                stats[f"valid_{label}"] += 1

                if collect_intermediate:
                    intermediate_rows.append(
                        {
                            "cohort_row_idx": cohort_row_idx,
                            "label": label,
                            "sex": row["sex"],
                            "age": int(row["age"]),
                            "age_step": int(row.get("age_step", 5)),
                            "disease_code": disease_code,
                            "delphi_shifted_token": int(delphi_shifted_token),
                            "delphi_raw_token": delphi_raw_token,
                            "raw_row_idx": raw_row_idx,
                            "delphi_patient_index": int(delphi_patient_index),
                            "raw_seq_point": raw_seq_point,
                            "delphi_seq_point": int(delphi_seq_point),
                            "raw_seq_length": int(patient_info["raw_seq_length"]),
                            "raw_seq_point_hits_disease_event": hit_any_disease,
                            "raw_seq_point_hits_target_disease": hit_target_at_boundary,
                            "target_present_after_prefix": has_target_after_prefix,
                            "target_present_anywhere": has_target_anywhere,
                            "target_positions_preview": target_positions[:10],
                            "raw_seq_point_window_json": json.dumps(patient_info["seq_windows"].get(raw_seq_point, [])),
                            "alignment_ok": alignment_ok,
                            "alignment_msg": alignment_msg,
                        }
                    )

        if prepared["case_patient_indices"] and prepared["control_patient_indices"]:
            prepared["n_cases"] = len(prepared["case_patient_indices"])
            prepared["n_controls"] = len(prepared["control_patient_indices"])
            prepared_rows.append(prepared)
            stats["valid_strata"] += 1
            _append_limited(debug_summary["prepared_examples"], _summarize_prepared_row(prepared), debug_examples)
        else:
            stats["empty_strata"] += 1
            _append_limited(
                debug_summary["empty_strata_examples"],
                {
                    "cohort_row_idx": int(cohort_row_idx),
                    "disease_code": disease_code,
                    "delphi_shifted_token": int(delphi_shifted_token),
                    "cases_kept": len(prepared["case_patient_indices"]),
                    "controls_kept": len(prepared["control_patient_indices"]),
                },
                debug_examples,
            )

    intermediate_df = pd.DataFrame(intermediate_rows) if collect_intermediate else None
    return prepared_rows, intermediate_df, dict(stats), debug_summary


def main():
    parser = argparse.ArgumentParser(description="Prepare a predefined case/control cohort for Delphi evaluation")
    parser.add_argument("--case_control_file", type=str, required=True, help="Path to raw cohort parquet")
    parser.add_argument("--traj_parquet", type=str, required=True, help="Path to raw trajectory parquet")
    parser.add_argument("--delphi_bin", type=str, required=True, help="Path to Delphi-format test.bin")
    parser.add_argument("--pid_mapping", type=str, required=True, help="Path to Delphi pid mapping TSV/JSON")
    parser.add_argument("--delphi_disease_map", type=str, default="./nature_disease_map.csv", help="Path to Delphi disease CSV")
    parser.add_argument("--output", type=str, required=True, help="Output JSON used by evaluate.py --predefined_cohort")
    parser.add_argument("--intermediate_output", type=str, default=None, help="Optional row-level intermediate parquet")
    parser.add_argument("--debug_output", type=str, default=None, help="Optional sampled debug JSON output")
    parser.add_argument("--pid_col", type=str, default="person_id", help="PID column in trajectory parquet")
    parser.add_argument("--seq_col", type=str, default="eval_seq", help="Raw evaluation sequence column in trajectory parquet")
    parser.add_argument("--att_time_unit", type=str, default="months", help="Same ATT unit used during preprocess.py")
    parser.add_argument("--att_time_mode", type=str, default="relative", help="Same ATT mode used during preprocess.py")
    parser.add_argument("--debug_examples", type=int, default=5, help="Number of representative examples to log")
    parser.add_argument("--debug_token_window", type=int, default=5, help="Token window size for raw sequence debug examples")
    args = parser.parse_args()

    output_path = Path(args.output)
    intermediate_output = None
    if args.intermediate_output:
        intermediate_output = Path(args.intermediate_output)
        if intermediate_output.suffix == "":
            intermediate_output = intermediate_output.with_name(intermediate_output.name + "_samples.parquet")
    debug_output = Path(args.debug_output) if args.debug_output else output_path.with_name(output_path.stem + "_debug.json")

    logger.info("--- Configuration Summary ---")
    logger.info("Case/control file : %s", args.case_control_file)
    logger.info("Trajectory parquet: %s", args.traj_parquet)
    logger.info("Delphi binary     : %s", args.delphi_bin)
    logger.info("PID mapping       : %s", args.pid_mapping)
    logger.info("Disease map CSV   : %s", args.delphi_disease_map)
    logger.info("Output JSON       : %s", args.output)
    logger.info("Debug JSON        : %s", debug_output)
    logger.info("Intermediate out  : %s", intermediate_output or "disabled")
    logger.info("Seq column        : %s", args.seq_col)
    logger.info("ATT unit/mode     : %s / %s", args.att_time_unit, args.att_time_mode)
    logger.info("Debug examples    : %s", args.debug_examples)
    logger.info("Token window      : %s", args.debug_token_window)

    case_control_rows = load_case_control_rows(args.case_control_file)
    logger.info("Loaded %s cohort strata", len(case_control_rows))
    sample_cohort_rows = [_summarize_cohort_row(row) for row in case_control_rows[: args.debug_examples]]
    _log_json("Sample raw cohort rows", sample_cohort_rows)

    needed_rows, seq_points_by_row = collect_needed_rows_and_seq_points(case_control_rows)
    logger.info("Needed raw trajectory rows: %s", len(needed_rows))
    logger.info("Unique raw seq_point groups: %s", len(seq_points_by_row))
    sample_needed_rows = [
        {
            "raw_row_idx": int(raw_row_idx),
            "seq_points": sorted(int(seq_pt) for seq_pt in list(seq_points_by_row[raw_row_idx]))[:10],
        }
        for raw_row_idx in sorted(seq_points_by_row)[: args.debug_examples]
    ]
    _log_json("Sample needed rows and seq_points", sample_needed_rows)

    pid_map = load_pid_mapping(args.pid_mapping)
    logger.info("Loaded %s PID mappings", len(pid_map))
    disease_mapping = load_disease_mapping_csv(args.delphi_disease_map)
    resolve_disease = make_disease_resolver(disease_mapping)

    val = np.fromfile(args.delphi_bin, dtype=np.uint32)
    if val.size == 0:
        raise ValueError(
            f"Delphi binary is empty: {args.delphi_bin}. This usually means preprocess.py filtered out every patient before cohort preparation. For eval_seq inputs, make sure the parser handles the leading <sos> token."
        )
    if val.size % 3 != 0:
        raise ValueError(f"Unexpected Delphi binary size in {args.delphi_bin}: {val.size} uint32 values")
    val = val.reshape(-1, 3)
    val_p2i = get_p2i(val)
    logger.info("Loaded Delphi binary with %s records and %s patients", len(val), len(val_p2i))

    patient_cache, missing_rows, patient_debug_examples = build_patient_cache(
        traj_parquet_path=args.traj_parquet,
        needed_rows=needed_rows,
        seq_points_by_row=seq_points_by_row,
        pid_map=pid_map,
        disease_mapping=disease_mapping,
        pid_col=args.pid_col,
        seq_col=args.seq_col,
        att_time_unit=args.att_time_unit,
        att_time_mode=args.att_time_mode,
        debug_examples=args.debug_examples,
        debug_token_window=args.debug_token_window,
    )
    logger.info("Built patient cache for %s referenced trajectory rows", len(patient_cache))
    _log_json("Sample patient cache entries", patient_debug_examples)
    if missing_rows:
        logger.warning("Missing %s referenced trajectory rows", len(missing_rows))
        _log_json("Sample missing trajectory rows", [int(x) for x in missing_rows[: args.debug_examples]])

    prepared_rows, intermediate_df, stats, debug_summary = build_predefined_cohort_records(
        case_control_rows=case_control_rows,
        patient_cache=patient_cache,
        val=val,
        val_p2i=val_p2i,
        resolve_disease=resolve_disease,
        collect_intermediate=intermediate_output is not None,
        debug_examples=args.debug_examples,
    )
    if missing_rows:
        stats["missing_traj_rows"] = len(missing_rows)

    _log_json("Prepared cohort debug examples", debug_summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(prepared_rows, f, indent=2)

    if intermediate_df is not None and not intermediate_df.empty:
        intermediate_output.parent.mkdir(parents=True, exist_ok=True)
        intermediate_df.to_parquet(intermediate_output, index=False)

    debug_payload = {
        "config": {
            "case_control_file": args.case_control_file,
            "traj_parquet": args.traj_parquet,
            "delphi_bin": args.delphi_bin,
            "pid_mapping": args.pid_mapping,
            "delphi_disease_map": args.delphi_disease_map,
            "output": args.output,
            "intermediate_output": None if intermediate_output is None else str(intermediate_output),
            "seq_col": args.seq_col,
            "att_time_unit": args.att_time_unit,
            "att_time_mode": args.att_time_mode,
            "debug_examples": args.debug_examples,
            "debug_token_window": args.debug_token_window,
        },
        "sample_raw_cohort_rows": sample_cohort_rows,
        "sample_needed_rows": sample_needed_rows,
        "sample_patient_cache_entries": patient_debug_examples,
        "transform_debug": debug_summary,
        "stats": stats,
        "sample_missing_rows": [int(x) for x in missing_rows[: args.debug_examples]],
        "prepared_strata": len(prepared_rows),
        "intermediate_rows": 0 if intermediate_df is None else len(intermediate_df),
    }
    debug_output.parent.mkdir(parents=True, exist_ok=True)
    with debug_output.open("w") as f:
        json.dump(debug_payload, f, indent=2)

    logger.info("Prepared strata: %s", len(prepared_rows))
    logger.info("Intermediate rows: %s", 0 if intermediate_df is None else len(intermediate_df))
    logger.info("Stats: %s", json.dumps(stats, indent=2))
    logger.info("Prepared cohort JSON: %s", output_path)
    logger.info("Debug JSON: %s", debug_output)
    logger.info("Intermediate parquet: %s", intermediate_output or "disabled")


if __name__ == "__main__":
    main()
