#!/usr/bin/env python3
"""
ETHOS Model Evaluation on a Predefined Case/Control Cohort

Scores a frozen ETHOS checkpoint on the cohort prepared by transform_predefined_cohort.py and writes
per-disease AUCs.

An onset is the first appearance of the token spelling out an ICD category. A next-token logit belongs
to one token, so a disease is one ICD category: every category is scored and pooled on its own, as
ReClaim scores one, and none are gathered under a clinical name.

Pipeline: load model + timelines -> logit per prediction point -> stratified AUC -> CSV output

    python evaluate.py --timelines <work_dir>/timelines.hdf5 --vocab <vocab.pkl> \
        --model <checkpoint.pt> --predefined_cohort <work_dir>/predefined_cohort.json \
        --output_dir <out_dir>
"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch as th

from evaluate_auc import (aggregate_normals, collect_unique_sequences_and_diseases,
                          compute_auc_from_cache, get_auc_delong_var)
from model import load_model_from_checkpoint
from transform_predefined_cohort import OnsetCohort
from utils import progress, setup_logging
from vocabulary import SpecialToken, Vocabulary

# Logging indentation prefixes
INDENT1 = "  "

# The columns aggregate_normals reads off a group of strata. Pooling hands it these and nothing else.
AGGREGATED_COLUMNS = ["auc_delong", "n_diseased", "n_healthy"]

# Diseases are named one by one in the log up to this many. A sweep over every category is left as a
# count, which says more than a listing of a thousand names.
MAX_DISEASES_TO_LIST = 40


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Score an ETHOS checkpoint on a predefined case/control cohort')
    parser.add_argument("--timelines", type=str, required=True,
                       help='HDF5 file of timelines written by preprocess.py')
    parser.add_argument("--vocab", type=str, default=os.environ.get("ETHOS_VOCAB"),
                       required="ETHOS_VOCAB" not in os.environ,
                       help='Vocabulary the timelines were encoded with. Not distributed here')
    parser.add_argument("--model", type=str, default=os.environ.get("ETHOS_MODEL"),
                       required="ETHOS_MODEL" not in os.environ,
                       help='Checkpoint to evaluate. Not distributed here')
    parser.add_argument("--predefined_cohort", type=str, required=True,
                       help='Cohort written by transform_predefined_cohort.py')
    parser.add_argument("--output_dir", type=str, required=True,
                       help='Directory the AUC tables and the run summary are written to')
    parser.add_argument("--batch_size", type=int, default=32,
                       help='Number of prediction points scored at once')
    parser.add_argument("--device", type=str, default="cuda" if th.cuda.is_available() else "cpu",
                       help='Device the model runs on')
    return parser.parse_args()


def load_predefined_cohort(path):
    """Read the prepared cohort back, restoring the arrays the scoring code indexes with."""
    with open(path) as f:
        prepared = json.load(f)
    for stratum in prepared["strata"]:
        for name in ("case_ids", "control_ids", "case_seq_points", "control_seq_points"):
            stratum[name] = np.asarray(stratum[name], dtype=np.int64)
    return prepared


def pooled_variance(variances: pd.Series) -> float:
    """
    Average the spread of a group of strata, over the strata that have one.

    DeLong needs two cases and two controls to give a variance, so a stratum holding one of either
    returns none. Counting it as a zero would report the pooled area as tighter than the data support.

    Args:
        variances: DeLong variance per stratum, missing where undefined

    Returns:
        float: Variance of the pooled area, or nan when no stratum has one
    """
    defined = variances.dropna()
    if defined.empty:
        return float("nan")
    return float(defined.sum() / len(defined) ** 2)


def construct_seed_timelines(patients, prediction_points, vocab, block_size):
    """
    Build the input the model is given at each prediction point.

    The input is the static context, then the patient's age and the calendar year it starts in, then the
    timeline cut at the prediction point. When the timeline outgrows the window the oldest events are
    dropped, and the age and year are recomputed from the start of what remains.

    Args:
        patients: The per-patient timelines of the cohort
        prediction_points: (patient row, cut point) pairs to build an input for
        vocab: Vocabulary the timelines were encoded with
        block_size: Longest input the model accepts

    Returns:
        dict: (patient row, cut point) to the token IDs of its input
    """
    seeds = {}
    for row, seq_point in prediction_points:
        patient = patients[row]
        context = patient["context"]
        context_len = len(context) + 2                 # the age and the calendar year follow it
        timeline_len = block_size - context_len

        timeline = patient["tokens"][:int(seq_point)]
        timeline_times = patient["times"][:int(seq_point)]
        if len(timeline) > timeline_len:
            window_start = len(timeline) - timeline_len
            age_at_window_start = float(timeline_times[window_start])
            timeline = timeline[window_start:]
        else:
            age_at_window_start = float(timeline_times[0]) if len(timeline_times) else 0.0

        age_token = int(vocab.encode(SpecialToken.age_to_year_token(age_at_window_start)))
        year = patient["age_reference"] + age_at_window_start
        year_token = int(vocab.encode(SpecialToken.age_to_year_token(year - 1970.0)))
        seeds[(row, seq_point)] = np.concatenate([context, [age_token, year_token], timeline]).astype(np.int64)
    return seeds


@th.no_grad()
def logit_scores(patients, prediction_points, disease_tokens, vocab, model, device, block_size,
                 batch_size=32):
    """
    Score every prediction point by the logit the model gives each disease as its next token.

    Only the disease tokens' logits are kept, so one forward pass scores every disease at a point.

    Args:
        patients: The per-patient timelines of the cohort
        prediction_points: The (patient row, cut point) pairs to score
        disease_tokens: Token IDs to score, in the order the cache stores them
        vocab: Vocabulary the timelines were encoded with
        model: Model to score with
        device: Device the model runs on
        block_size: Longest input the model accepts
        batch_size: Prediction points scored at once

    Returns:
        dict: Prediction point to the logit of each disease token
    """
    if not prediction_points:
        return {}
    inputs = construct_seed_timelines(patients, prediction_points, vocab, block_size)
    disease_tokens_tensor = th.tensor(disease_tokens, device=device, dtype=th.long)

    # Inputs of the same length batch together, and only the last position is read, so no padding is
    # needed anywhere and no mask is required to find the last real token
    by_length = {}
    for point, model_input in inputs.items():
        by_length.setdefault(len(model_input), []).append(point)

    logits_cache = {}
    for _, points in progress(by_length.items(), desc="Computing logits", smoothing=0):
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start:batch_start + batch_size]
            x = th.from_numpy(np.stack([inputs[point] for point in batch])).to(device)
            logits, _ = model(x)
            last_token_logits = logits[:, -1, :]
            disease_logits = last_token_logits.index_select(1, disease_tokens_tensor)
            disease_logits_cpu = disease_logits.float().cpu().numpy()
            for position_in_batch, point in enumerate(batch):
                logits_cache[point] = disease_logits_cpu[position_in_batch]
    return logits_cache


def pool(frame, key):
    """Pool the strata of every group under key, both the area and its spread."""
    # Narrowing to the aggregate's own columns keeps the grouping column out of the group it is
    # handed. include_groups would say so directly, but only from pandas 2.2 on: earlier versions
    # pass it down to the aggregate, where it raises.
    pooled = (frame.groupby(key)[AGGREGATED_COLUMNS]
              .apply(aggregate_normals).reset_index())
    spread = (frame.groupby([key])["auc_variance_delong"]
              .agg(auc_variance_delong=pooled_variance, n_strata_with_spread="count")
              .reset_index())
    pooled = pooled.merge(spread, on=key, how="left")
    pooled["auc_std_error"] = np.sqrt(pooled["auc_variance_delong"])
    return pooled


def main():
    """Main execution function."""
    log_file = setup_logging(level=logging.INFO)
    logging.info("EVALUATE STARTING...")
    args = parse_args()

    logging.info("\nConfiguration:")
    logging.info("%sTimelines file: %s", INDENT1, args.timelines)
    logging.info("%sVocabulary: %s", INDENT1, args.vocab)
    logging.info("%sModel: %s", INDENT1, args.model)
    logging.info("%sPrepared cohort: %s", INDENT1, args.predefined_cohort)
    logging.info("%sOutput directory: %s", INDENT1, args.output_dir)

    logging.info("********** 1 Reading the prepared cohort **********")
    prepared = load_predefined_cohort(args.predefined_cohort)
    strata = prepared["strata"]
    labels_of = {entry["token_id"]: entry for entry in prepared["tokens"]}
    target_tokens = set(labels_of)
    logging.info("%sCase-control strata: %s | target tokens to evaluate: %s", INDENT1,
                 f"{len(strata):,}", f"{len(target_tokens):,}")

    vocab = Vocabulary(args.vocab)
    # A stratum names its patients by their row in this cohort, so the cohort has to come out the way
    # it did when the strata were prepared: the same timelines file, and no filter applied to it.
    cohort = OnsetCohort(args.timelines, vocab, min_seq_len=0, min_onsets=0)
    prediction_points, cached_disease_keys = collect_unique_sequences_and_diseases(strata, None)
    # The vocabulary names far more diseases than ReClaim evaluates, and an area averaged over the
    # wider set would not be comparable with the other models. Only the targets may be scored, so a
    # stratum naming anything else is a fault rather than something to drop quietly.
    stray = sorted(set(cached_disease_keys) - target_tokens)
    if stray:
        raise RuntimeError(f"{len(stray)} strata name a token that is not a target, e.g. {stray[:5]}")
    disease_token_to_idx = {key: idx for idx, key in enumerate(cached_disease_keys)}
    logging.info("%sPrediction points to score: %s", INDENT1, f"{len(prediction_points):,}")

    logging.info("********** 2 Scoring with the model **********")
    # The checkpoint stores its configuration as pickled objects, which torch only unpickles when
    # asked to load a file in full
    _original_torch_load = th.load
    th.load = lambda *a, **k: _original_torch_load(*a, **{**k, "weights_only": False})
    model, block_size = load_model_from_checkpoint(args.model, args.device, for_training=False)
    model.eval()
    model.to(args.device)
    logits_cache = logit_scores(cohort.patients, prediction_points, cached_disease_keys, vocab,
                                model, args.device, block_size, args.batch_size)

    logging.info("********** 3 Computing the area under the ROC curve **********")
    aucs = []
    for disease_info in progress(strata, desc="Computing AUCs"):
        result = compute_auc_from_cache(disease_info, logits_cache, disease_token_to_idx)
        if result is None:
            continue
        # A stratum holding one case or one control leaves DeLong no covariance to take, so it warns
        # and returns nan. That is expected, and counted once below rather than warned about per stratum.
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            _, auc_variance = get_auc_delong_var(np.asarray(result["control_logits"]),
                                                 np.asarray(result["case_logits"]))
        labels = labels_of[result["disease_token"]]
        aucs.append({
            **{name: value for name, value in result.items()
               if name not in ("case_logits", "control_logits")},
            "auc_variance_delong": float(np.asarray(auc_variance).reshape(-1)[0]),
            "token": labels["token"], "icd_codes": labels["icd_codes"],
            "disease": labels["disease"],
        })

    aucs_df = pd.DataFrame(aucs)
    if aucs_df.empty:
        raise RuntimeError("No stratum could be evaluated.")

    # A stratum is kept on one case and one control, but DeLong needs two of each for a variance, so
    # some strata give an area and no spread. Worth saying, since the pooled standard error is then
    # averaged over fewer strata than the pooled area.
    n_without_spread = int(aucs_df["auc_variance_delong"].isna().sum())
    if n_without_spread:
        logging.warning("%s of %s strata hold a single case or a single control, so DeLong leaves "
                        "their spread undefined; they count towards the pooled area but are left "
                        "out of the pooled standard error", f"{n_without_spread:,}",
                        f"{len(aucs_df):,}")

    # The only pooling there is: one row per ICD category. ReClaim pools a disease over its strata and
    # stops there, so this script stops there too and never gathers categories under a clinical name.
    auc_df = pool(aucs_df, "disease_token")
    for column in ("token", "icd_codes", "disease"):
        auc_df[column] = auc_df["disease_token"].map(
            aucs_df.drop_duplicates("disease_token").set_index("disease_token")[column])

    logging.info("********** 4 Writing results and summary **********")
    os.makedirs(args.output_dir, exist_ok=True)
    unpooled_path = os.path.join(args.output_dir, "auc_results_unpooled.csv")
    pooled_path = os.path.join(args.output_dir, "auc_results.csv")
    aucs_df.to_csv(unpooled_path, index=False)
    auc_df.to_csv(pooled_path, index=False)
    logging.info("%sWrote %s and %s", INDENT1, pooled_path, unpooled_path)
    if len(auc_df) <= MAX_DISEASES_TO_LIST:
        for row in auc_df.itertuples():
            logging.info("%s%-28s AUC %.4f ± %.4f over %d strata", INDENT1,
                         str(row.disease), row.auc, row.auc_std_error, int(row.n_samples))
    # Every row of auc_df is one target disease, so the mean is over the targets and over nothing
    # else. It is stated with its denominator, because a mean over a different number of diseases is
    # a different number and must not be compared with one.
    logging.info("%sMean AUC %.4f over %s of the %s target diseases", INDENT1,
                 float(auc_df["auc"].mean()), f"{auc_df['disease_token'].nunique():,}",
                 f"{prepared['n_diseases_requested']:,}")
    summary = {
        # The strata are ReClaim's, so the cohort is the patients they name
        "n_cohort_patients": len({int(row) for stratum in strata
                                  for label in ("case_ids", "control_ids")
                                  for row in stratum[label]}),
        "n_target_diseases_requested": prepared["n_diseases_requested"],
        "n_target_tokens_resolved": prepared["n_tokens_evaluated"],
        "n_target_diseases_with_auc": int(auc_df["disease_token"].nunique()),
        "n_strata": int(len(aucs_df)),
        "n_strata_without_spread": n_without_spread,
        "n_prediction_points": len(prediction_points),
        # Averaged over the target diseases that produced an area, which n_target_diseases_with_auc
        # counts. No disease outside target_tokens contributes to it.
        "mean_auroc": float(auc_df["auc"].mean()),
        "median_auroc": float(auc_df["auc"].median()),
        "mean_auroc_unpooled": float(aucs_df["auc_delong"].mean()),
        "n_diseases_below_half": int((auc_df["auc"] < 0.5).sum()),
        "batch_size": args.batch_size,
        "case_control_ids": prepared["case_control_ids"],
        "traj_parquet": prepared["traj_parquet"],
        "target_tokens": prepared["target_tokens"],
        "mapping": prepared["mapping"],
    }
    summary_path = os.path.join(args.output_dir, "onset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("\n%s", json.dumps(summary, indent=2))

    logging.info("Log file: %s", log_file)
    logging.info("EVALUATE COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
