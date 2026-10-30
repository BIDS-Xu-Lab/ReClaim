#!/usr/bin/env python3
"""
Phased pipeline for gradient-boosted disease baseline.

Phases:
  1. prepare  - Build all disease train/val cohorts, save to output_dir/cohorts/
  2. features - Build the CountVectorizer from the feature vocabulary, extract features per disease
  3. fit      - Train model per disease, save models, evaluate on val
  4. fit_eval - Alias for fit (train + evaluate)
  5. eval     - Evaluate on val (or inference_only: load saved models, predict on val only)
  6. all      - Run prepare -> features -> fit -> eval
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def _fmt_elapsed(sec: float) -> str:
    """Format seconds as H:MM:SS."""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz, vstack as sparse_vstack

from lightgbm_baseline.data import (
    PatientHistory,
    build_case_control_rows,
    build_case_control_lookup_from_precomputed,
    build_case_control_rows_from_precomputed,
    build_case_control_rows_from_lookup,
    build_histories,
    build_val_cohort_from_case_control_ids,
    load_precomputed_index,
    load_parquet_recursive,
    precompute_case_control_index,
    prepare_eval_dataframe,
    save_precomputed_index,
    sanitize_token_for_filename,
)
from lightgbm_baseline.evaluate import aggregate_normals, evaluate_by_strata
from lightgbm_baseline.features import (
    load_vectorizer,
    save_vectorizer,
    transform_cohort_with_vectorizer,
)
from lightgbm_baseline.model import load_model, predict_proba, save_model, train_model
from lightgbm_baseline.utils import get_dx_major_tokens, get_target_tokens, load_disease_token_id_map


def _load_histories(args: argparse.Namespace) -> Tuple[List[PatientHistory], List[PatientHistory]]:
    """Load or build train and val histories."""
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    sample_size = getattr(args, "train_sample_size", None)
    train_cache_suffix = f"_n{sample_size}" if sample_size is not None else ""
    train_hist_cache = os.path.join(cache_dir, f"train_histories{train_cache_suffix}.parquet")
    val_hist_cache = os.path.join(cache_dir, "val_histories.parquet")

    prep_kw = dict(
        demo_end_index=args.demo_end_index,
        age_index=args.age_index,
        seq_len=args.seq_len,
        att_type=args.att_type,
        min_seq_len=args.min_seq_len,
    )

    use_case_control = bool(getattr(args, "case_control_ids_file", None) and getattr(args, "val_data_file", None))
    if use_case_control:
        args.case_control_ids_df = pd.read_parquet(args.case_control_ids_file)

    if args.resume and os.path.isfile(train_hist_cache) and os.path.isfile(val_hist_cache):
        from lightgbm_baseline.data import df_to_histories
        return df_to_histories(pd.read_parquet(train_hist_cache)), df_to_histories(pd.read_parquet(val_hist_cache))

    train_max_files = getattr(args, "train_max_files", None)
    train_raw = load_parquet_recursive(args.train_data_dir, max_files=train_max_files)
    if sample_size is not None:
        n = min(sample_size, len(train_raw))
        train_raw = train_raw.sample(n=n, random_state=args.random_state)
        print(f"[load] Sampled {n} train records (requested {sample_size})")
    train_eval = prepare_eval_dataframe(train_raw, **prep_kw)

    if use_case_control:
        val_raw = pd.read_parquet(args.val_data_file)
        required = ["eval_seq", "disease_points", "disease_points_ages", "disease_points_visit_location"]
        val_eval = val_raw if all(c in val_raw.columns for c in required) else prepare_eval_dataframe(val_raw, **prep_kw)
    else:
        val_raw = load_parquet_recursive(args.val_data_dir)
        val_eval = prepare_eval_dataframe(val_raw, **prep_kw)

    train_histories = build_histories(train_eval)
    val_histories = build_histories(val_eval)

    if args.resume or args.save_feature_cache:
        from lightgbm_baseline.data import histories_to_df
        histories_to_df(train_histories).to_parquet(train_hist_cache, index=False)
        histories_to_df(val_histories).to_parquet(val_hist_cache, index=False)

    return train_histories, val_histories


def _has_inference_data_override(args: argparse.Namespace) -> bool:
    """Whether inference should use alternate dataset inputs."""
    return bool(
        getattr(args, "inference_val_data_dir", None)
        or getattr(args, "inference_case_control_ids_file", None)
        or getattr(args, "inference_val_data_file", None)
    )


def _validate_inference_override_args(args: argparse.Namespace) -> None:
    """Validate --inference_* argument combinations."""
    has_cc_ids = bool(getattr(args, "inference_case_control_ids_file", None))
    has_val_file = bool(getattr(args, "inference_val_data_file", None))
    if has_cc_ids != has_val_file:
        raise ValueError(
            "--inference_case_control_ids_file and --inference_val_data_file must be provided together"
        )
    if not has_cc_ids and not getattr(args, "inference_val_data_dir", None):
        raise ValueError(
            "Need --inference_val_data_dir or "
            "(--inference_case_control_ids_file and --inference_val_data_file) for inference-only"
        )


def _prepare_inference_features(
    args: argparse.Namespace,
    target_tokens: List[str],
    inference_output_dir: str,
) -> str:
    """Build val cohorts/features for an alternate inference dataset using saved training vectorizer."""
    t0 = time.perf_counter()
    _validate_inference_override_args(args)

    features_dir = os.path.join(inference_output_dir, "features")
    cohorts_dir = os.path.join(inference_output_dir, "cohorts")
    cache_dir = os.path.join(inference_output_dir, "cache")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(cohorts_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    with open(os.path.join(cohorts_dir, "target_tokens.json"), "w") as f:
        json.dump(target_tokens, f, indent=2)

    t_vec = time.perf_counter()
    base_vectorizer_path = getattr(args, "vectorizer_file", None) or os.path.join(args.output_dir, "features", "vectorizer.json")
    if not os.path.isfile(base_vectorizer_path):
        raise FileNotFoundError(f"Training vectorizer not found: {base_vectorizer_path}")
    vectorizer = load_vectorizer(base_vectorizer_path)
    save_vectorizer(vectorizer, os.path.join(features_dir, "vectorizer.json"))
    print(f"[inference_features] Vectorizer loaded in {_fmt_elapsed(time.perf_counter() - t_vec)}")

    use_case_control = bool(args.inference_case_control_ids_file and args.inference_val_data_file)
    resume = bool(getattr(args, "resume", False))

    # ── Fast path: when resuming, if every per-disease feature artifact already exists
    # there is nothing to extract, so skip the (very expensive) val data load entirely.
    if resume and all(
        _inference_outputs_exist(os.path.join(features_dir, sanitize_token_for_filename(t)))
        for t in target_tokens
    ):
        print(
            f"[inference_features] resume=True; all {len(target_tokens)} per-disease features cached, "
            f"skipping val data + histories load."
        )
        return features_dir

    prep_kw = dict(
        demo_end_index=args.demo_end_index,
        age_index=args.age_index,
        seq_len=args.seq_len,
        att_type=args.att_type,
        min_seq_len=args.min_seq_len,
    )

    val_data_path = args.inference_val_data_file or args.inference_val_data_dir
    val_hist_cache_path = os.path.join(cache_dir, "infer_val_histories.parquet")
    val_hist_meta_path = os.path.join(cache_dir, "infer_val_histories.meta.json")

    def _val_hist_cache_valid() -> bool:
        if not (os.path.isfile(val_hist_cache_path) and os.path.isfile(val_hist_meta_path)):
            return False
        try:
            with open(val_hist_meta_path) as f:
                meta = json.load(f)
            return meta.get("val_data_path") == val_data_path
        except Exception:
            return False

    t_data = time.perf_counter()
    case_control_ids_df = None
    if resume and _val_hist_cache_valid():
        from lightgbm_baseline.data import df_to_histories
        val_histories = df_to_histories(pd.read_parquet(val_hist_cache_path))
        if use_case_control:
            case_control_ids_df = pd.read_parquet(args.inference_case_control_ids_file)
        print(
            f"[inference_features] Loaded val histories from cache in "
            f"{_fmt_elapsed(time.perf_counter() - t_data)} | val_histories={len(val_histories)}"
        )
    else:
        if use_case_control:
            case_control_ids_df = pd.read_parquet(args.inference_case_control_ids_file)
            val_raw = pd.read_parquet(args.inference_val_data_file)
        else:
            val_raw = load_parquet_recursive(args.inference_val_data_dir)

        required = ["eval_seq", "disease_points", "disease_points_ages", "disease_points_visit_location"]
        val_eval = val_raw if all(c in val_raw.columns for c in required) else prepare_eval_dataframe(val_raw, **prep_kw)
        val_histories = build_histories(val_eval)
        try:
            from lightgbm_baseline.data import histories_to_df
            histories_to_df(val_histories).to_parquet(val_hist_cache_path, index=False)
            with open(val_hist_meta_path, "w") as f:
                json.dump({"val_data_path": val_data_path, "n_histories": len(val_histories)}, f)
        except Exception as e:
            print(f"[inference_features] WARN: failed to cache val histories: {e}")
        print(
            f"[inference_features] Built val histories in {_fmt_elapsed(time.perf_counter() - t_data)} | "
            f"val_histories={len(val_histories)}"
        )

    age_start, age_end, age_step = map(int, args.age_group_range.split(","))
    age_groups = np.arange(age_start, age_end, age_step)
    age_step = int(age_groups[1] - age_groups[0]) if len(age_groups) >= 2 else 5
    token_value_map = load_disease_token_id_map(target_tokens, args.tokens_file)
    feature_workers = getattr(args, "feature_workers", 0) or 0

    if use_case_control:
        val_lookup = None
    else:
        nw = getattr(args, "num_workers", 0) or 0
        offset_suffix = str(args.offset).replace(".", "_")
        val_precomputed_path = os.path.join(cache_dir, f"infer_val_precomputed_offset{offset_suffix}.pkl")
        val_precomputed_meta_path = os.path.join(cache_dir, f"infer_val_precomputed_offset{offset_suffix}.meta.json")

        def _val_precomputed_cache_valid() -> bool:
            if not os.path.isfile(val_precomputed_path) or not os.path.isfile(val_precomputed_meta_path):
                return False
            try:
                with open(val_precomputed_meta_path) as f:
                    meta = json.load(f)
                return meta.get("n_histories") == len(val_histories) and meta.get("offset") == args.offset
            except Exception:
                return False

        t_precompute = time.perf_counter()
        if _val_precomputed_cache_valid():
            val_precomputed = load_precomputed_index(val_precomputed_path)
            print(
                f"[inference_features] Loaded val precomputed cache in "
                f"{_fmt_elapsed(time.perf_counter() - t_precompute)} ({len(val_precomputed)} entries)"
            )
        else:
            print(f"[inference_features] Precomputing val case/control index (workers={nw})...")
            val_precomputed = precompute_case_control_index(val_histories, args.offset, num_workers=nw)
            save_precomputed_index(val_precomputed, val_precomputed_path)
            with open(val_precomputed_meta_path, "w") as f:
                json.dump({"n_histories": len(val_histories), "offset": args.offset}, f)
            print(
                f"[inference_features] Built val precomputed in "
                f"{_fmt_elapsed(time.perf_counter() - t_precompute)} ({len(val_precomputed)} entries)"
            )

        age_groups_tag = "_".join(map(str, age_groups.tolist()))
        val_lookup_path = os.path.join(cache_dir, f"infer_val_lookup_offset{offset_suffix}_ages{age_groups_tag}.pkl")
        val_lookup_meta_path = os.path.join(
            cache_dir, f"infer_val_lookup_offset{offset_suffix}_ages{age_groups_tag}.meta.json"
        )

        def _val_lookup_cache_valid() -> bool:
            if not os.path.isfile(val_lookup_path) or not os.path.isfile(val_lookup_meta_path):
                return False
            try:
                with open(val_lookup_meta_path) as f:
                    meta = json.load(f)
                return (
                    meta.get("n_histories") == len(val_histories)
                    and meta.get("offset") == args.offset
                    and meta.get("age_group_range") == args.age_group_range
                )
            except Exception:
                return False

        t_lookup = time.perf_counter()
        if _val_lookup_cache_valid():
            val_lookup = load_precomputed_index(val_lookup_path)
            case_tokens_n = len(val_lookup.get("case_rows_by_token", {}))
            print(
                f"[inference_features] Loaded val lookup cache in "
                f"{_fmt_elapsed(time.perf_counter() - t_lookup)} (case_tokens={case_tokens_n})"
            )
        else:
            val_lookup = build_case_control_lookup_from_precomputed(
                val_precomputed, age_groups, age_step, stratify_by_sex_age=True
            )
            save_precomputed_index(val_lookup, val_lookup_path)
            with open(val_lookup_meta_path, "w") as f:
                json.dump(
                    {
                        "n_histories": len(val_histories),
                        "offset": args.offset,
                        "age_group_range": args.age_group_range,
                    },
                    f,
                )
            case_tokens_n = len(val_lookup.get("case_rows_by_token", {}))
            print(
                f"[inference_features] Built val lookup in "
                f"{_fmt_elapsed(time.perf_counter() - t_lookup)} (case_tokens={case_tokens_n})"
            )

    processed = 0
    t_extract = time.perf_counter()

    # Filter already-done diseases in the main process so tqdm shows only pending work.
    if resume:
        pending_tokens = [
            t for t in target_tokens
            if not _inference_outputs_exist(os.path.join(features_dir, sanitize_token_for_filename(t)))
        ]
        n_skip = len(target_tokens) - len(pending_tokens)
        print(
            f"[inference_features] resume=True: {n_skip} already done, "
            f"{len(pending_tokens)} pending (workers={feature_workers})"
        )
    else:
        pending_tokens = target_tokens
        print(f"[inference_features] Extracting val features per disease (workers={feature_workers})...")

    if not pending_tokens:
        print(f"[inference_features] All diseases already done, skipping extraction.")
        return features_dir

    # Pre-build large read-only structures in the MAIN process BEFORE pool creation.
    # Workers inherit them via fork COW — no pickle overhead, no 32× memory duplication.
    t_shared = time.perf_counter()
    _build_infer_shared(val_histories, vectorizer, val_lookup, case_control_ids_df, token_value_map)
    print(f"[inference_features] Shared structures built in {_fmt_elapsed(time.perf_counter() - t_shared)}")

    # Small args only — paths, flags, scalars. These are safe to pickle.
    small_init_args = (cohorts_dir, features_dir, use_case_control, age_groups, age_step, False)  # resume=False: already filtered above

    if feature_workers > 1:
        import multiprocessing as mp

        # maxtasksperchild recycles workers so allocator fragmentation (gc is
        # disabled inside workers to protect COW pages) cannot accumulate over
        # ~1,000 diseases. Re-forks inherit _INFER_SHARED from this process, so
        # recycling is cheap.
        with mp.Pool(
            processes=feature_workers,
            initializer=_init_inference_feature_worker,
            initargs=small_init_args,
            maxtasksperchild=8,
        ) as pool:
            try:
                for n_done in tqdm(
                    pool.imap_unordered(_extract_inference_features_one_worker, pending_tokens, chunksize=1),
                    total=len(pending_tokens),
                    desc="InferenceFeatures",
                    unit="disease",
                    smoothing=0.1,
                ):
                    processed += int(n_done)
            except Exception as e:
                print(f"[inference_features] Pool error: {e}")
                pool.terminate()
                raise
    else:
        _init_inference_feature_worker(*small_init_args)
        for target_token in tqdm(pending_tokens, desc="InferenceFeatures", unit="disease", smoothing=0.1):
            processed += _extract_inference_features_one_worker(target_token)

    elapsed_extract = time.perf_counter() - t_extract
    skipped = max(0, len(target_tokens) - processed)
    avg_sec_per_processed = elapsed_extract / max(processed, 1)
    print(
        f"[inference_features] extraction_done in {_fmt_elapsed(elapsed_extract)} | "
        f"processed={processed} skipped={skipped} avg_sec_per_processed={avg_sec_per_processed:.2f}"
    )
    print(f"[inference_features] Done in {_fmt_elapsed(time.perf_counter() - t0)}")
    return features_dir


def phase_prepare(args: argparse.Namespace) -> None:
    """Build all disease cohorts and save."""
    t0 = time.perf_counter()
    print("[prepare] Loading histories...")
    train_histories, val_histories = _load_histories(args)
    print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t0)}")
    all_tokens = get_dx_major_tokens(args.tokens_file)
    target_tokens = get_target_tokens(
        args.icd_code_map_file, args.tokens_file, all_tokens,
        args.all_diseases,
        args.target_disease_token or "<DX-MAJOR_I10>",
        args.max_diseases,
    )
    token_value_map = load_disease_token_id_map(target_tokens, args.tokens_file)

    age_start, age_end, age_step = map(int, args.age_group_range.split(","))
    age_groups = np.arange(age_start, age_end, age_step)
    use_case_control = bool(getattr(args, "case_control_ids_df", None) is not None)

    cohorts_dir = os.path.join(args.output_dir, "cohorts")
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cohorts_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    offset_suffix = str(args.offset).replace(".", "_")
    train_precomputed_path = os.path.join(cache_dir, f"train_precomputed_offset{offset_suffix}.pkl")
    val_precomputed_path = os.path.join(cache_dir, f"val_precomputed_offset{offset_suffix}.pkl")
    train_precomputed_meta_path = os.path.join(cache_dir, f"train_precomputed_offset{offset_suffix}.meta.json")

    age_step = int(age_groups[1] - age_groups[0]) if len(age_groups) >= 2 else 5
    nw = getattr(args, "num_workers", 0) or 0

    def _train_cache_valid() -> bool:
        if not os.path.isfile(train_precomputed_path) or not os.path.isfile(train_precomputed_meta_path):
            return False
        try:
            with open(train_precomputed_meta_path) as f:
                meta = json.load(f)
            return meta.get("n_histories") == len(train_histories) and meta.get("offset") == args.offset
        except Exception:
            return False

    if _train_cache_valid():
        t1 = time.perf_counter()
        print("[prepare] Loading train case/control index from cache...")
        train_precomputed = load_precomputed_index(train_precomputed_path)
        print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t1)} ({len(train_precomputed)} entries)")
    else:
        t1 = time.perf_counter()
        print(f"[prepare] Precomputing train case/control index (workers={nw})...")
        train_precomputed = precompute_case_control_index(train_histories, args.offset, num_workers=nw)
        print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t1)}")
        save_precomputed_index(train_precomputed, train_precomputed_path)
        with open(train_precomputed_meta_path, "w") as f:
            json.dump({"n_histories": len(train_histories), "offset": args.offset}, f)

    if not use_case_control:
        val_precomputed_meta_path = os.path.join(cache_dir, f"val_precomputed_offset{offset_suffix}.meta.json")

        def _val_cache_valid() -> bool:
            if not os.path.isfile(val_precomputed_path) or not os.path.isfile(val_precomputed_meta_path):
                return False
            try:
                with open(val_precomputed_meta_path) as f:
                    meta = json.load(f)
                return meta.get("n_histories") == len(val_histories) and meta.get("offset") == args.offset
            except Exception:
                return False

        if _val_cache_valid():
            t2 = time.perf_counter()
            print("[prepare] Loading val case/control index from cache...")
            val_precomputed = load_precomputed_index(val_precomputed_path)
            print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t2)} ({len(val_precomputed)} entries)")
        else:
            t2 = time.perf_counter()
            print("[prepare] Precomputing val case/control index...")
            val_precomputed = precompute_case_control_index(val_histories, args.offset, num_workers=nw)
            print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t2)}")
            save_precomputed_index(val_precomputed, val_precomputed_path)
            with open(val_precomputed_meta_path, "w") as f:
                json.dump({"n_histories": len(val_histories), "offset": args.offset}, f)

    train_unstratified = getattr(args, "train_unstratified", False)
    cohort_workers = getattr(args, "cohort_workers", 0) or 0

    def _build_and_save_cohort(target_token: str) -> bool:
        disease_dir = os.path.join(cohorts_dir, sanitize_token_for_filename(target_token))
        os.makedirs(disease_dir, exist_ok=True)
        train_path = os.path.join(disease_dir, "train_cohort.parquet")
        val_path = os.path.join(disease_dir, "val_cohort.parquet")
        if args.resume and os.path.isfile(train_path) and os.path.isfile(val_path):
            return False
        train_cohort = build_case_control_rows_from_precomputed(
            train_precomputed,
            target_token,
            age_groups,
            age_step,
            stratify_by_sex_age=not train_unstratified,
        )
        if use_case_control:
            val_cohort = build_val_cohort_from_case_control_ids(
                args.case_control_ids_df, target_token, token_value_map[target_token], age_groups,
            )
        else:
            val_cohort = build_case_control_rows_from_precomputed(
                val_precomputed, target_token, age_groups, age_step,
            )
        if not train_cohort.empty and not val_cohort.empty:
            train_cohort.to_parquet(train_path, index=False)
            val_cohort.to_parquet(val_path, index=False)
        return True

    t3 = time.perf_counter()
    print(f"[prepare] Building cohorts for {len(target_tokens)} diseases (workers={cohort_workers})...")
    if cohort_workers > 1:
        with ThreadPoolExecutor(max_workers=cohort_workers) as ex:
            futures = {ex.submit(_build_and_save_cohort, t): t for t in target_tokens}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Cohorts",
                unit="disease",
                smoothing=0.1,
            ):
                fut.result()
    else:
        for target_token in tqdm(
            target_tokens,
            desc="Cohorts",
            unit="disease",
            smoothing=0.1,
        ):
            _build_and_save_cohort(target_token)

    print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t3)}")
    with open(os.path.join(cohorts_dir, "target_tokens.json"), "w") as f:
        json.dump(target_tokens, f, indent=2)
    print(f"[prepare] Total: {_fmt_elapsed(time.perf_counter() - t0)} | Cohorts for {len(target_tokens)} diseases.")


# Module-level globals for feature extraction workers (set by _init_feature_worker)
_FEATURE_WORKER_DATA: Optional[Tuple] = None
# Set in the MAIN process BEFORE Pool creation so workers inherit via fork-COW.
# Same pattern as _INFER_SHARED; avoids rebuilding large history maps inside every worker
# (which multiplies memory by the number of workers and can trigger OOM kills).
_FEATURE_SHARED: Optional[Tuple] = None   # (train_history_map, val_history_map, vectorizer)


def _feature_outputs_exist(feat_disease_dir: str) -> bool:
    """Check whether all per-disease feature artifacts exist."""
    required = (
        "x_train.npz",
        "x_val.npz",
        "train_meta.parquet",
        "val_meta.parquet",
        "y_train.npy",
        "y_val.npy",
    )
    return all(os.path.isfile(os.path.join(feat_disease_dir, name)) for name in required)


# Module-level globals for inference-only val feature workers.
# Large read-only data (history_map, seq_tokens_dict) is stored here BEFORE pool creation
# so worker processes inherit it via fork COW — no pickle, no 32× duplication.
_INFER_SHARED: Optional[Tuple] = None   # (history_map, seq_tokens_dict, vectorizer, val_lookup_or_None, case_control_ids_df_or_None, token_value_map)
_INFER_WORKER_DATA: Optional[Tuple] = None

# Rows per transform chunk inside an inference feature worker. Peak memory per
# worker scales linearly with this value (a few GB at 100K rows for million-patient
# cohorts) and is multiplied by feature_workers; raise only with memory to spare.
_INFER_CHUNK_SIZE = int(os.environ.get("BASELINE_INFER_CHUNK_SIZE", "100000"))


def _inference_outputs_exist(feat_disease_dir: str) -> bool:
    """Check whether all inference-only val feature artifacts exist."""
    required = (
        "x_val.npz",
        "val_meta.parquet",
        "y_val.npy",
    )
    return all(os.path.isfile(os.path.join(feat_disease_dir, name)) for name in required)


def _build_infer_shared(val_histories, vectorizer, val_lookup, case_control_ids_df, token_value_map) -> None:
    """Pre-build large read-only structures in the MAIN process before Pool creation.
    Workers inherit them via fork COW — zero pickle overhead, no N-worker duplication.

    NOTE: seq_tokens_dict is NOT pre-built here: tokenized sequences for a large cohort are
    tens of GB of Python string objects, and Python's GC writes to every live object header,
    dirtying those COW pages in every worker. Instead, workers use a per-worker lazy cache.
    """
    global _INFER_SHARED
    history_map = {h.patient_idx: h for h in val_histories}
    _INFER_SHARED = (history_map, vectorizer, val_lookup, case_control_ids_df, token_value_map)


def _init_inference_feature_worker(
    cohorts_dir: str,
    features_dir: str,
    use_case_control: bool,
    age_groups: np.ndarray,
    age_step: int,
    resume: bool,
) -> None:
    """Initialize inference feature worker with small per-worker state.
    Large data (history_map, vectorizer, lookup) is inherited via fork COW from _INFER_SHARED.
    seq_tokens_cache is a per-worker lazy dict — avoids pre-building tokenized sequences in the
    main process (which would be COW-replicated by Python GC across all workers)."""
    global _INFER_WORKER_DATA
    import gc
    gc.disable()   # prevent GC from writing ob_gc_refs and dirtying inherited COW pages
    _INFER_WORKER_DATA = (cohorts_dir, features_dir, use_case_control, age_groups, age_step, resume, {})


def _extract_inference_features_one_worker(target_token: str) -> int:
    """Build val cohort/features for one disease in inference-only mode."""
    global _INFER_WORKER_DATA, _INFER_SHARED
    if _INFER_WORKER_DATA is None or _INFER_SHARED is None:
        return 0
    cohorts_dir, features_dir, use_case_control, age_groups, age_step, resume, seq_tokens_cache = _INFER_WORKER_DATA
    history_map, vectorizer, val_lookup, case_control_ids_df, token_value_map = _INFER_SHARED

    disease_dir = os.path.join(cohorts_dir, sanitize_token_for_filename(target_token))
    feat_disease_dir = os.path.join(features_dir, sanitize_token_for_filename(target_token))
    os.makedirs(disease_dir, exist_ok=True)
    os.makedirs(feat_disease_dir, exist_ok=True)

    val_cohort_path = os.path.join(disease_dir, "val_cohort.parquet")
    if resume and _inference_outputs_exist(feat_disease_dir):
        return 0

    if os.path.isfile(val_cohort_path):
        val_cohort = pd.read_parquet(val_cohort_path)
    else:
        if use_case_control:
            val_cohort = build_val_cohort_from_case_control_ids(
                case_control_ids_df,
                target_token,
                token_value_map[target_token],
                age_groups,
            )
        else:
            val_cohort = build_case_control_rows_from_lookup(val_lookup, target_token)
        if val_cohort.empty:
            return 0
        val_cohort.to_parquet(val_cohort_path, index=False)

    if val_cohort.empty:
        return 0
    # Chunked transform with a fresh per-chunk cache. Reusing one unbounded
    # per-worker cache across all diseases makes every worker cache token lists
    # for (nearly) every patient with gc disabled; a worker killed for memory
    # then makes imap_unordered block forever, so the job looks hung rather than
    # OOM. The train path guards against this the same way.
    x_val, val_scored = _transform_cohort_chunked(
        val_cohort, vectorizer, history_map, chunk_size=_INFER_CHUNK_SIZE
    )
    save_npz(os.path.join(feat_disease_dir, "x_val.npz"), x_val)
    val_scored.to_parquet(os.path.join(feat_disease_dir, "val_meta.parquet"), index=False)
    np.save(os.path.join(feat_disease_dir, "y_val.npy"), val_scored["label"].to_numpy())
    return 1


def _build_feature_shared(train_history_map, val_history_map, vectorizer) -> None:
    """Pre-stash large read-only structures in the MAIN process before Pool creation.
    Workers inherit via fork COW — no pickle, no per-worker rebuild."""
    global _FEATURE_SHARED
    _FEATURE_SHARED = (train_history_map, val_history_map, vectorizer)


def _transform_cohort_chunked(cohort_df, vectorizer, history_map, chunk_size: int = 1_000_000):
    """Transform a cohort in chunks so seq_tokens_cache + prefixes list don't materialize
    for millions of rows at once. Larger chunks amortize per-call sklearn overhead;
    smaller chunks bound peak memory."""
    from lightgbm_baseline.features import transform_cohort_with_vectorizer

    if len(cohort_df) <= chunk_size:
        return transform_cohort_with_vectorizer(
            cohort_df, [], vectorizer,
            seq_tokens_cache={}, history_map=history_map,
        )
    x_chunks = []
    scored_chunks = []
    for start in range(0, len(cohort_df), chunk_size):
        chunk = cohort_df.iloc[start:start + chunk_size]
        # Fresh cache per chunk — bounds peak memory to one chunk's patients.
        x_chunk, scored_chunk = transform_cohort_with_vectorizer(
            chunk, [], vectorizer,
            seq_tokens_cache={}, history_map=history_map,
        )
        x_chunks.append(x_chunk)
        scored_chunks.append(scored_chunk)
        del chunk, x_chunk, scored_chunk
    x = sparse_vstack(x_chunks, format="csr")
    scored = pd.concat(scored_chunks, ignore_index=True)
    return x, scored


def _init_feature_worker(
    cohorts_dir: str,
    features_dir: str,
    resume: bool,
) -> None:
    """Initialize worker with small per-worker state. Large maps inherited via COW
    from _FEATURE_SHARED. gc.disable() prevents Python GC from writing ob_gc_refs
    and dirtying inherited COW pages (which is what blew memory previously)."""
    global _FEATURE_WORKER_DATA
    import gc
    gc.disable()
    # Per-worker seq_tokens_cache (fresh empty dicts — workers' own write-only state).
    _FEATURE_WORKER_DATA = (cohorts_dir, features_dir, resume, {}, {})


def _extract_features_one_worker(target_token: str) -> int:
    """Process one disease. Uses _FEATURE_WORKER_DATA from initializer. Enables per-disease load balancing."""
    global _FEATURE_WORKER_DATA, _FEATURE_SHARED
    if _FEATURE_WORKER_DATA is None or _FEATURE_SHARED is None:
        return 0
    cohorts_dir, features_dir, resume, *_ = _FEATURE_WORKER_DATA
    train_history_map, val_history_map, vectorizer = _FEATURE_SHARED

    disease_dir = os.path.join(cohorts_dir, sanitize_token_for_filename(target_token))
    feat_disease_dir = os.path.join(features_dir, sanitize_token_for_filename(target_token))
    train_path = os.path.join(disease_dir, "train_cohort.parquet")
    val_path = os.path.join(disease_dir, "val_cohort.parquet")
    if not os.path.isfile(train_path) or not os.path.isfile(val_path):
        return 0
    if resume and _feature_outputs_exist(feat_disease_dir):
        return 0
    os.makedirs(feat_disease_dir, exist_ok=True)
    train_cohort = pd.read_parquet(train_path)
    val_cohort = pd.read_parquet(val_path)
    x_train, train_scored = _transform_cohort_chunked(train_cohort, vectorizer, train_history_map)
    del train_cohort
    x_val, val_scored = _transform_cohort_chunked(val_cohort, vectorizer, val_history_map)
    del val_cohort
    save_npz(os.path.join(feat_disease_dir, "x_train.npz"), x_train)
    save_npz(os.path.join(feat_disease_dir, "x_val.npz"), x_val)
    train_scored.to_parquet(os.path.join(feat_disease_dir, "train_meta.parquet"), index=False)
    val_scored.to_parquet(os.path.join(feat_disease_dir, "val_meta.parquet"), index=False)
    np.save(os.path.join(feat_disease_dir, "y_train.npy"), train_scored["label"].to_numpy())
    np.save(os.path.join(feat_disease_dir, "y_val.npy"), val_scored["label"].to_numpy())
    return 1


def phase_features(args: argparse.Namespace) -> None:
    """Fit global vectorizer, extract features per disease."""
    t0 = time.perf_counter()
    print("[features] Loading histories...")
    train_histories, val_histories = _load_histories(args)
    print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t0)}")
    cohorts_dir = os.path.join(args.output_dir, "cohorts")
    features_dir = os.path.join(args.output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    # --target_tokens_file lets the caller restrict feature extraction to a subset
    # (e.g. re-running only the diseases that previously failed). Defaults to the
    # full list written by phase_prepare.
    _tokens_json = getattr(args, "target_tokens_file", None) or os.path.join(cohorts_dir, "target_tokens.json")
    with open(_tokens_json) as f:
        target_tokens = json.load(f)

    from lightgbm_baseline.features import build_vectorizer_from_tokenizer
    t1 = time.perf_counter()
    print(f"[features] Building vectorizer from feature vocab...")
    vectorizer = build_vectorizer_from_tokenizer(args.tokens_file)
    vectorizer_path = os.path.join(features_dir, "vectorizer.json")
    save_vectorizer(vectorizer, vectorizer_path)
    print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t1)}")

    t2 = time.perf_counter()
    feature_workers = getattr(args, "feature_workers", 0) or 0
    resume = bool(getattr(args, "resume", False))
    # Pre-build history maps once; serial path uses fresh per-disease caches to prevent
    # unbounded memory growth (caching split tokens for all patients across all diseases).
    train_history_map = {h.patient_idx: h for h in train_histories}
    val_history_map = {h.patient_idx: h for h in val_histories}

    def _extract_features_one(target_token: str) -> bool:
        disease_dir = os.path.join(cohorts_dir, sanitize_token_for_filename(target_token))
        feat_disease_dir = os.path.join(features_dir, sanitize_token_for_filename(target_token))
        train_path = os.path.join(disease_dir, "train_cohort.parquet")
        val_path = os.path.join(disease_dir, "val_cohort.parquet")
        if not os.path.isfile(train_path) or not os.path.isfile(val_path):
            return False
        if resume and _feature_outputs_exist(feat_disease_dir):
            return False
        os.makedirs(feat_disease_dir, exist_ok=True)
        train_cohort = pd.read_parquet(train_path)
        val_cohort = pd.read_parquet(val_path)
        x_train, train_scored = _transform_cohort_chunked(train_cohort, vectorizer, train_history_map)
        del train_cohort
        x_val, val_scored = _transform_cohort_chunked(val_cohort, vectorizer, val_history_map)
        del val_cohort
        save_npz(os.path.join(feat_disease_dir, "x_train.npz"), x_train)
        save_npz(os.path.join(feat_disease_dir, "x_val.npz"), x_val)
        train_scored.to_parquet(os.path.join(feat_disease_dir, "train_meta.parquet"), index=False)
        val_scored.to_parquet(os.path.join(feat_disease_dir, "val_meta.parquet"), index=False)
        np.save(os.path.join(feat_disease_dir, "y_train.npy"), train_scored["label"].to_numpy())
        np.save(os.path.join(feat_disease_dir, "y_val.npy"), val_scored["label"].to_numpy())
        return True

    print(f"[features] Extracting features per disease (workers={feature_workers})...")
    if feature_workers > 1:
        import multiprocessing as mp

        # Pre-build large read-only structures in the MAIN process BEFORE Pool creation
        # so workers inherit via fork-COW. Without this, _init_feature_worker would
        # rebuild the history maps inside each worker and multiply memory use.
        t_shared = time.perf_counter()
        _build_feature_shared(train_history_map, val_history_map, vectorizer)
        print(f"[features] Shared maps staged in {_fmt_elapsed(time.perf_counter() - t_shared)}")

        init_args = (cohorts_dir, features_dir, resume)
        with mp.Pool(
            processes=feature_workers,
            initializer=_init_feature_worker,
            initargs=init_args,
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_extract_features_one_worker, target_tokens, chunksize=1),
                total=len(target_tokens),
                desc="Features",
                unit="disease",
                smoothing=0.1,
            ):
                pass
    else:
        for target_token in tqdm(target_tokens, desc="Features", unit="disease", smoothing=0.1):
            _extract_features_one(target_token)

    print(f"  -> done in {_fmt_elapsed(time.perf_counter() - t2)}")
    print(f"[features] Total: {_fmt_elapsed(time.perf_counter() - t0)} | Features for {len(target_tokens)} diseases.")


def _fit_and_eval_one(
    target_token: str,
    target_token_value: object,
    features_dir: str,
    model_disease_dir: str,
    args: argparse.Namespace,
    inference_only: bool,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Train (or load) model and evaluate on val."""
    empty = pd.DataFrame(columns=["auc_delong", "sex", "disease_token", "disease_token_str", "age", "n_diseased", "n_healthy"])
    feat_dir = os.path.join(features_dir, sanitize_token_for_filename(target_token))
    model_path = os.path.join(model_disease_dir, "lgb_model.txt")

    if not os.path.isfile(os.path.join(feat_dir, "x_val.npz")):
        return empty, {"status": "not_in_cohort"}
    x_val = load_npz(os.path.join(feat_dir, "x_val.npz"))
    val_scored = pd.read_parquet(os.path.join(feat_dir, "val_meta.parquet"))
    y_val = np.load(os.path.join(feat_dir, "y_val.npy"))

    if val_scored["label"].nunique() < 2:
        return empty, {"status": "single_class_val"}

    if inference_only:
        if not os.path.isfile(model_path):
            return empty, {"status": "model_not_found"}
        model = load_model(model_path)
    else:
        x_train = load_npz(os.path.join(feat_dir, "x_train.npz"))
        y_train = np.load(os.path.join(feat_dir, "y_train.npy"))
        if len(np.unique(y_train)) < 2:
            return empty, {"status": "single_class_train"}
        model = train_model(
            x_train, y_train, x_val, y_val,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            n_jobs=args.xgb_n_jobs,
            random_state=args.random_state,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        save_model(model, model_path)

    val_scored["score"] = predict_proba(model, x_val)
    result_df = evaluate_by_strata(val_scored, target_token_value, target_token)
    return result_df, {"status": "completed", "n_unpooled_rows": len(result_df)}


def phase_fit(args: argparse.Namespace) -> None:
    """Train models and evaluate on val."""
    phase_fit_and_eval(args, inference_only=False)


def phase_eval(args: argparse.Namespace, inference_only: bool = False) -> None:
    """Evaluate on val. If inference_only, load saved models and predict only (no training)."""
    phase_fit_and_eval(args, inference_only=inference_only)


def phase_fit_and_eval(args: argparse.Namespace, inference_only: bool = False) -> None:
    """Fit models and/or evaluate."""
    t0 = time.perf_counter()
    mode = "inference_only" if inference_only else "fit_and_eval"
    print(f"[{mode}] Starting...")

    base_features_dir = os.path.join(args.output_dir, "features")
    model_per_disease_dir = os.path.join(args.output_dir, "per_disease")
    train_cohorts_dir = os.path.join(args.output_dir, "cohorts")
    os.makedirs(model_per_disease_dir, exist_ok=True)

    _tokens_json = getattr(args, "target_tokens_file", None) or os.path.join(train_cohorts_dir, "target_tokens.json")
    with open(_tokens_json) as f:
        target_tokens = json.load(f)

    single = getattr(args, "target_disease_token", None)
    if single:
        from lightgbm_baseline.data import canonicalize_dx_major_token
        single = canonicalize_dx_major_token(single)
        target_tokens = [single]
        print(f"[{mode}] Single disease mode: {single}")

    eval_output_dir = args.output_dir
    features_dir = base_features_dir
    if inference_only and _has_inference_data_override(args):
        eval_output_dir = args.inference_output_dir or os.path.join(args.output_dir, "inference_eval")
        os.makedirs(eval_output_dir, exist_ok=True)
        features_dir = _prepare_inference_features(args, target_tokens, eval_output_dir)
    elif inference_only and args.inference_output_dir:
        eval_output_dir = args.inference_output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    eval_per_disease_dir = os.path.join(eval_output_dir, "per_disease")
    os.makedirs(eval_per_disease_dir, exist_ok=True)

    token_value_map = load_disease_token_id_map(target_tokens, args.tokens_file)

    all_unpooled = []
    skipped = []
    disease_stats = {}
    resume = bool(getattr(args, "resume", False))

    # Separate already-done from pending
    pending_tokens = []
    for target_token in target_tokens:
        eval_disease_dir = os.path.join(eval_per_disease_dir, sanitize_token_for_filename(target_token))
        done_path    = os.path.join(eval_disease_dir, "_DONE")
        metrics_path = os.path.join(eval_disease_dir, "metrics.json")
        unpooled_path = os.path.join(eval_disease_dir, "auc_results_unpooled.csv")
        if resume and os.path.isfile(done_path):
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                if not m.get("error"):
                    disease_stats[target_token] = m
                    if os.path.isfile(unpooled_path):
                        df = pd.read_csv(unpooled_path)
                        if not df.empty:
                            all_unpooled.append(df)
                    continue
            except Exception:
                pass
        pending_tokens.append(target_token)

    eval_workers = getattr(args, "eval_workers", 0) or 0
    # Parallel eval only for inference_only (pure I/O + predict, no training).
    # Training is not thread-safe across LightGBM instances; keep it serial.
    use_parallel = inference_only and eval_workers > 1 and pending_tokens

    def _eval_one_token(target_token: str):
        model_disease_dir = os.path.join(model_per_disease_dir, sanitize_token_for_filename(target_token))
        eval_disease_dir  = os.path.join(eval_per_disease_dir,  sanitize_token_for_filename(target_token))
        os.makedirs(eval_disease_dir, exist_ok=True)
        unpooled_path = os.path.join(eval_disease_dir, "auc_results_unpooled.csv")
        metrics_path  = os.path.join(eval_disease_dir, "metrics.json")
        done_path     = os.path.join(eval_disease_dir, "_DONE")
        try:
            result_df, metrics = _fit_and_eval_one(
                target_token, token_value_map[target_token],
                features_dir, model_disease_dir, args, inference_only=inference_only,
            )
        except Exception as e:
            metrics = {"status": "error", "error": str(e)}
            result_df = None
        metrics["target_token"] = target_token
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        if result_df is not None and not result_df.empty:
            result_df.to_csv(unpooled_path, index=False)
            with open(done_path, "w") as f:
                f.write("completed\n")
            return target_token, metrics, result_df
        return target_token, metrics, None

    if use_parallel:
        print(f"[{mode}] Parallel eval with {eval_workers} threads ({len(pending_tokens)} pending)...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=eval_workers) as ex:
            futures = {ex.submit(_eval_one_token, tok): tok for tok in pending_tokens}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=mode, unit="disease", smoothing=0.1):
                tok, metrics, result_df = fut.result()
                disease_stats[tok] = metrics
                if result_df is not None:
                    all_unpooled.append(result_df)
                else:
                    skipped.append(tok)
    else:
        for target_token in tqdm(pending_tokens, desc=mode, unit="disease", smoothing=0.1):
            tok, metrics, result_df = _eval_one_token(target_token)
            disease_stats[tok] = metrics
            if result_df is not None:
                all_unpooled.append(result_df)
            else:
                skipped.append(tok)

    unpooled_df = pd.concat(all_unpooled, ignore_index=True) if all_unpooled else pd.DataFrame(
        columns=["auc_delong", "sex", "disease_token", "disease_token_str", "age", "n_diseased", "n_healthy"]
    )
    unpooled_path = os.path.join(eval_output_dir, "auc_results_unpooled.csv")
    unpooled_df.to_csv(unpooled_path, index=False)

    if len(unpooled_df) > 0:
        pooled_df = (
            unpooled_df.groupby(["disease_token", "disease_token_str"])
            .apply(aggregate_normals, include_groups=False)
            .reset_index()
            .sort_values("disease_token")
        )
    else:
        pooled_df = pd.DataFrame(
            columns=["disease_token", "disease_token_str", "auc", "n_samples", "n_diseased", "n_healthy"]
        )
    pooled_path = os.path.join(eval_output_dir, "auc_results.csv")
    pooled_df.to_csv(pooled_path, index=False)

    with open(os.path.join(eval_output_dir, "skipped_diseases.json"), "w") as f:
        json.dump(skipped, f, indent=2)
    with open(os.path.join(eval_output_dir, "disease_stats.json"), "w") as f:
        json.dump(disease_stats, f, indent=2)

    print(
        f"[{mode}] Total: {_fmt_elapsed(time.perf_counter() - t0)} | "
        f"Features: {features_dir} | Results: {unpooled_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phased gradient-boosted disease baseline.")
    p.add_argument(
        "--phase",
        choices=["prepare", "features", "fit", "fit_eval", "eval", "all"],
        default="all",
        help="fit_eval: alias for fit (train + evaluate on val)",
    )
    p.add_argument("--inference_only", action="store_true", help="Eval phase: load saved models, predict on val only")
    p.add_argument(
        "--target_tokens_file",
        type=str,
        default=None,
        help="Path to target_tokens.json (inference_only: overrides output_dir/cohorts/target_tokens.json).",
    )
    p.add_argument(
        "--vectorizer_file",
        type=str,
        default=None,
        help="Path to vectorizer.json (inference_only: overrides output_dir/features/vectorizer.json).",
    )
    p.add_argument(
        "--eval_workers",
        type=int,
        default=0,
        help="Parallel threads for the eval loop in inference_only mode (0=serial). "
             "LightGBM inference + disk I/O release the GIL; 8-32 works well.",
    )
    p.add_argument("--train_data_dir", type=str, default=None)
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--case_control_ids_file", type=str, default=None)
    p.add_argument("--val_data_file", type=str, default=None)
    p.add_argument("--inference_val_data_dir", type=str, default=None)
    p.add_argument("--inference_case_control_ids_file", type=str, default=None)
    p.add_argument("--inference_val_data_file", type=str, default=None)
    p.add_argument(
        "--inference_output_dir",
        type=str,
        default=None,
        help=(
            "Eval/inference_only: output dir for alternate-dataset features and results. "
            "Defaults to output_dir/inference_eval when --inference_* inputs are provided."
        ),
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--icd_code_map_file", type=str, default=None)
    p.add_argument(
        "--tokens_file",
        type=str,
        required=True,
        help="Feature vocabulary: a Hugging Face tokenizer/checkpoint directory (its DX-MAJOR, PROC, "
        "RX and ATT tokens are used) or a JSON {token: id} map.",
    )
    p.add_argument("--all_diseases", action="store_true")
    p.add_argument(
        "--target_disease_token",
        type=str,
        default=None,
        help="Single disease mode: prepare uses it when not --all_diseases; fit/eval filter to this disease only.",
    )
    p.add_argument("--max_diseases", type=int, default=None)
    p.add_argument("--offset", type=float, default=365.25)
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel workers for precompute_case_control_index (0=serial). Use 8-32 for 1M+ patients.",
    )
    p.add_argument(
        "--cohort_workers",
        type=int,
        default=0,
        help="Parallel workers for building train/val cohorts per disease (0=serial). Use 4-16 for 1000+ diseases.",
    )
    p.add_argument(
        "--feature_workers",
        type=int,
        default=0,
        help="Parallel workers for feature extraction (0=serial). Uses ProcessPoolExecutor (not threads) to bypass GIL.",
    )
    p.add_argument("--age_group_range", type=str, default="40,80,5")
    p.add_argument("--demo_end_index", type=int, default=3)
    p.add_argument("--age_index", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--att_type", type=str, default="month", choices=["day", "week", "month"])
    p.add_argument("--min_seq_len", type=int, default=10)
    p.add_argument("--max_features", type=int, default=30000)
    p.add_argument("--n_estimators", type=int, default=10000)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--xgb_n_jobs", type=int, default=0)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_feature_cache", action="store_true")
    p.add_argument(
        "--train_unstratified",
        action="store_true",
        help="Train cohort: pool all cases/controls without sex/age stratification.",
    )
    p.add_argument(
        "--train_sample_size",
        type=int,
        default=None,
        help="Sample fixed number of records from train data (for quick testing). Uses random_state for reproducibility.",
    )
    p.add_argument(
        "--train_max_files",
        type=int,
        default=None,
        help="Load only first N parquet files from train_data_dir (for quick testing).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.case_control_ids_file) != bool(args.val_data_file):
        raise ValueError("--case_control_ids_file and --val_data_file must be provided together")
    use_cc = bool(args.case_control_ids_file and args.val_data_file)

    needs_histories = args.phase in ("prepare", "features", "all")
    if needs_histories and not args.train_data_dir:
        raise ValueError("--train_data_dir is required for prepare/features/all")
    if needs_histories and not use_cc and not args.val_data_dir:
        raise ValueError("Need --val_data_dir or (--case_control_ids_file and --val_data_file)")

    if args.phase == "eval" and args.inference_only and _has_inference_data_override(args):
        _validate_inference_override_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.xgb_n_jobs <= 0:
        args.xgb_n_jobs = max(1, (os.cpu_count() or 1) // 2)

    t_total = time.perf_counter()
    if args.phase == "prepare":
        phase_prepare(args)
    elif args.phase == "features":
        phase_features(args)
    elif args.phase in ("fit", "fit_eval"):
        phase_fit(args)
    elif args.phase == "eval":
        phase_eval(args, inference_only=args.inference_only)
    else:
        phase_prepare(args)
        phase_features(args)
        phase_fit_and_eval(args, inference_only=False)
    elapsed = time.perf_counter() - t_total
    print(f"\n>>> Phase '{args.phase}' completed in {_fmt_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
