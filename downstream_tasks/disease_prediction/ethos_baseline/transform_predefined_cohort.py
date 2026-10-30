#!/usr/bin/env python3
"""
ETHOS Predefined Cohort Preparation

Prepares a predefined case/control cohort for ETHOS evaluation.

The cohort is the one written by prepare_disease_case_controls_fast.py, so ETHOS is scored on exactly
the same patients and prediction points as ReClaim. This script does three things:
1. Cuts each of ReClaim's target tokens to the ICD-10 category it names and resolves that category to
   the ETHOS token whose first occurrence marks its onset.
2. Maps cohort row ids from traj_test_pd.parquet to the ETHOS timeline of the same patient.
3. Converts each ReClaim seq point into the position the ETHOS timeline is cut at, by locating the
   calendar month and visit kind of the block that seq point closes.

The output JSON is designed for evaluate.py --predefined_cohort. Strata that cannot be carried over
are dropped and counted, and the counts go into the JSON beside the strata.

Two formats describe time differently, and the gap is closed here. A ReClaim sequence recovers an age
by accumulating its time tokens, while an ETHOS timeline carries the age of every token, so the age is
read rather than rebuilt. A ReClaim sequence is cut at the token closing a visit, while an ETHOS
timeline moves between moments at a separator, so that separator is the cut.

    python transform_predefined_cohort.py --timelines <work_dir>/timelines.hdf5 \
        --vocab <vocab.pkl> \
        --case_control_ids <eval_dir>/365.25-20,100,10-disease_case_control_ids.parquet \
        --output <work_dir>/predefined_cohort.json
"""

import argparse
import bisect
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_icd_cm_code_to_name, read_hdf5, setup_logging
from vocabulary import SECONDS_IN_YEAR, SpecialToken, Vocabulary

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "

# The disease selection ships beside this script, so it needs no configuring.
DEFAULT_TARGET_TOKENS = Path(__file__).resolve().parent / "target_tokens.json"

DAYS_PER_YEAR = 365.25

# Pads the age of a patient with fewer onsets than the widest row. Beyond a lifetime, so a padded entry
# is never an earlier onset and never a target.
PAD_AGE_DAYS = 1000 * DAYS_PER_YEAR

# Pads the onset token array. No token encodes to it, so a padded target is distinguishable from a real
# one.
PAD_TOKEN_ID = -1

# Diseases are named one by one in the log up to this many. A sweep over every category is left as a
# count, which says more than a listing of a thousand names.
MAX_DISEASES_TO_LIST = 40

# A ReClaim target token, e.g. <DX-MAJOR_A00>. The name after the prefix is the ICD-10 category.
TARGET_TOKEN_PATTERN = re.compile(r"^<DX-MAJOR_([A-Za-z0-9]+)>$")

# The ranks ReClaim orders the blocks of one month by. A diagnosis only ever opens an outpatient or an
# inpatient block.
OUTPATIENT, PHARMACY, INPATIENT = 2, 3, 4
RECLAIM_VISIT_KINDS = {"<VT-outpatient>": OUTPATIENT, "<VT-pharmacy>": PHARMACY,
                       "<VT-pharmaceutical>": PHARMACY, "<VT-inpatient>": INPATIENT}

# ReClaim's case-control file is built on eval_seq, which ends with 2022. Months are counted from
# January 1970, as numpy's datetime64[M] counts them.
FIRST_MONTH_AFTER_EVAL = (2023 - 1970) * 12


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Map ReClaim's case/control strata onto the ETHOS timelines")
    parser.add_argument("--timelines", type=str, required=True,
                       help='HDF5 file of timelines written by preprocess.py')
    parser.add_argument("--vocab", type=str, default=os.environ.get("ETHOS_VOCAB"),
                       required="ETHOS_VOCAB" not in os.environ,
                       help='Vocabulary the timelines were encoded with. Not distributed here')
    parser.add_argument("--case_control_ids", type=str, required=True,
                       help="ReClaim case-control parquet holding the strata to carry over")
    parser.add_argument("--traj_parquet", type=str, default=None,
                       help='ReClaim trajectory parquet the case-control row ids index into '
                            '(default: traj_test_pd.parquet beside --case_control_ids)')
    parser.add_argument("--target_tokens", type=str, default=str(DEFAULT_TARGET_TOKENS),
                       help='JSON list of the disease tokens ReClaim evaluates, which the strata '
                            'are restricted to. Defaults to the copy shipped beside this script')
    parser.add_argument("--ref_dir", type=str, default=os.environ.get("ETHOS_REF_DIR"),
                       required="ETHOS_REF_DIR" not in os.environ,
                       help="The ETHOS repository's ethos/data directory, holding the ICD-10-CM "
                            "titles a disease is named by. Not distributed here")
    parser.add_argument("--output", type=str, required=True,
                       help='Prepared cohort to write, read by evaluate.py --predefined_cohort')
    return parser.parse_args()


def is_onset_token(token: str) -> bool:
    """
    Decide whether a token marks the onset of a disease.

    Only the 3-character category counts, the piece spelled out as a title. The pieces after it merely
    narrow the diagnosis, and a procedure describes treatment rather than diagnosis.

    Args:
        token: Token string to classify

    Returns:
        bool: True when the token marks a disease onset
    """
    if not isinstance(token, str) or not token.startswith("ICD_"):
        return False
    return not (token.startswith("ICD_4-5_") or token.startswith("ICD_6-_") or token.startswith("ICD_PCS_"))


class OnsetCohort:
    """
    The patients that can be evaluated, reduced to the four arrays the case-control code takes.

    Each patient contributes the first occurrence of every onset token, the age it happened at, and the
    point the timeline is cut at to ask about it. Padding those lists into one row per patient is what
    lets the case-control code work on every patient at once.

    A timeline is cut at a boundary between two moments, not inside one, so everything recorded
    alongside the onset is part of the question. The separator tokens are those boundaries; the end of
    the timeline closes the last moment.
    """

    def __init__(self, timelines_path, vocab, min_seq_len=10, min_onsets=2):
        """
        Args:
            timelines_path: HDF5 file holding the timelines
            vocab: Vocabulary the timelines were encoded with
            min_seq_len: Shortest timeline that can be evaluated
            min_onsets: Fewest onsets a patient needs to give a prediction point
        """
        self.vocab = vocab
        self.onset_ids = {token_id for token, token_id in vocab.stoi.items() if is_onset_token(token)}
        self.separator_ids = {int(token_id) for token_id in vocab.encode(SpecialToken.SEPARATOR_NAMES)}
        self.sex_male_id = int(vocab.encode("SEX_M"))
        self._build(read_hdf5(timelines_path), min_seq_len, min_onsets)

    def _build(self, data, min_seq_len, min_onsets):
        """
        Collect the onsets of every patient whose timeline is long enough to be evaluated.

        Args:
            data: Raw arrays of the timelines file
            min_seq_len: Shortest timeline that can be evaluated
            min_onsets: Fewest onsets a patient needs to give a prediction point
        """
        tokens, times = data["tokens"], data["times"]
        offsets = list(data["patient_data_offsets"])
        bounds = offsets + [len(tokens)]

        rows_tokens, rows_ages, rows_points, rows_male = [], [], [], []
        self.patients = []

        for patient_index in range(len(offsets)):
            start, end = int(bounds[patient_index]), int(bounds[patient_index + 1])
            timeline_tokens, timeline_times = tokens[start:end], times[start:end]
            if len(timeline_tokens) < min_seq_len:
                continue

            onset_tokens, onset_ages, cut_points = self._find_onsets(timeline_tokens, timeline_times)
            if len(onset_tokens) < min_onsets:
                continue

            context = data["patient_context"][patient_index]
            rows_tokens.append(onset_tokens)
            rows_ages.append(onset_ages)
            rows_points.append(cut_points)
            rows_male.append(int(context[0]) == self.sex_male_id)
            self.patients.append({
                "tokens": np.asarray(timeline_tokens, dtype=np.int64),
                "times": np.asarray(timeline_times, dtype=np.float64),
                "context": np.asarray(context, dtype=np.int64),
                "age_reference": float(data["age_reference"][patient_index]),
                "patient_id": data["patient_ids"][patient_index],
            })

        if not self.patients:
            raise RuntimeError("No patient has a timeline long enough with enough onsets to evaluate.")

        # Pad the per-patient lists into arrays of one row per patient
        n_patients = len(rows_tokens)
        n_onsets = max(len(row) for row in rows_tokens)
        self.all_disease_tokens = np.full((n_patients, n_onsets), PAD_TOKEN_ID, dtype=np.int64)
        self.all_seq_points = np.zeros((n_patients, n_onsets), dtype=np.int64)
        self.all_ages = np.full((n_patients, n_onsets), PAD_AGE_DAYS, dtype=np.float64)
        for row in range(n_patients):
            filled = len(rows_tokens[row])
            self.all_disease_tokens[row, :filled] = rows_tokens[row]
            self.all_seq_points[row, :filled] = rows_points[row]
            self.all_ages[row, :filled] = rows_ages[row]
        self.all_sex = np.asarray(rows_male, dtype=bool)

    def _find_onsets(self, timeline_tokens, timeline_times):
        """
        Walk one timeline and record where its onsets are.

        Only a token's first occurrence counts, since the question is when a disease started. Its cut
        point is the boundary closing the moment it belongs to, so the cut falls after everything
        recorded alongside it.

        Args:
            timeline_tokens: Token IDs of one patient's timeline
            timeline_times: Age at each of those tokens, in years

        Returns:
            tuple: (onset token IDs, ages in days, cut points)
        """
        # The boundaries a timeline can be cut at, the last of which closes the final moment
        cut_locations = [position for position, token_id in enumerate(timeline_tokens)
                         if int(token_id) in self.separator_ids]
        cut_locations.append(len(timeline_tokens) - 1)

        onset_tokens, onset_ages, cut_points = [], [], []
        seen = set()
        cut_idx = 0
        for position, token_id in enumerate(timeline_tokens):
            token_id = int(token_id)
            if token_id in self.separator_ids:
                cut_idx += 1
            elif token_id in self.onset_ids and token_id not in seen:
                seen.add(token_id)
                onset_tokens.append(token_id)
                onset_ages.append(float(timeline_times[position]) * DAYS_PER_YEAR)
                cut_points.append(cut_locations[cut_idx])
        return onset_tokens, onset_ages, cut_points


def construct_target_tokens(disease_map, ref_dir, vocab):
    """
    Resolve the diseases to the tokens whose first occurrence marks their onset.

    Every code is cut to its 3-character category and looked up in the ICD-10-CM titles, because the
    title is how the vocabulary spells a category out. Codes the vocabulary lacks are dropped, and codes
    sharing a token are kept once. The names a token belongs to are carried along to label its result.

    Args:
        disease_map: Disease name to the ICD codes it is made of
        ref_dir: Directory containing the code system reference tables
        vocab: Vocabulary the timelines were encoded with

    Returns:
        tuple: (token IDs to evaluate, their token string, ICD categories and disease names)
    """
    code_to_name = load_icd_cm_code_to_name(ref_dir)
    disease_tokens, token_label, token_categories, token_diseases = [], {}, {}, {}
    for disease_name, codes in disease_map.items():
        resolved = 0
        for raw_code in codes:
            category = str(raw_code).replace(".", "").strip().upper()[:3]
            title = code_to_name.get(category)
            if title is None:
                logging.warning("%s%s: ICD %s has no ICD-10-CM title, dropped",
                                INDENT1, disease_name, category)
                continue
            token = f"ICD_{title}"
            token_id = vocab.encode(token)
            if np.isnan(token_id):
                logging.warning("%s%s: ICD %s (%s) is not in the vocabulary, dropped",
                                INDENT1, disease_name, category, token)
                continue
            token_id = int(token_id)
            if token_id not in token_label:
                disease_tokens.append(token_id)
                token_label[token_id] = token
                token_categories[token_id] = []
                token_diseases[token_id] = []
            if category not in token_categories[token_id]:
                token_categories[token_id].append(category)
            if disease_name not in token_diseases[token_id]:
                token_diseases[token_id].append(disease_name)
            resolved += 1
        if resolved == 0:
            logging.warning("%sDisease %r has no code the vocabulary can spell out, skipping",
                            INDENT1, disease_name)
    return disease_tokens, token_label, token_categories, token_diseases


def load_target_categories(target_tokens_path: str) -> tuple[list[str], list[str]]:
    """
    Read the ReClaim target tokens and cut each to the ICD category it names.

    Args:
        target_tokens_path: JSON list of target tokens

    Returns:
        tuple: (categories in file order, tokens no category could be read from)
    """
    with open(target_tokens_path) as f:
        tokens = json.load(f)
    categories, unparsed = [], []
    for token in tokens:
        match = TARGET_TOKEN_PATTERN.match(str(token).strip())
        if match is None:
            unparsed.append(str(token))
            continue
        categories.append(match.group(1).upper()[:3])
    return categories, unparsed


class ReclaimSequence:
    """
    One ReClaim eval_seq, read only for the block each of its seq points closes.

    ReClaim groups a patient's events into blocks of one calendar month and one visit kind, and closes
    every block with <ATT-n>, the whole number of months to the next block. The first block is the
    <AGE-n> anchor, placed in January of the year DOBYR + n. Summing the <ATT-n> values before a seq
    point therefore gives the exact month of its block, and the <VT-...> token opening the block gives
    its kind. No token of eval_seq reaches the model.
    """

    def __init__(self, eval_seq: str):
        """
        Args:
            eval_seq: The whitespace-separated ReClaim trajectory of one patient
        """
        self.tokens = eval_seq.split(" ")
        self.birth_year = self._anchor("<DOBYR-")
        age = self._anchor("<AGE-")
        self.first_month = None if self.birth_year is None or age is None \
            else (self.birth_year + age - 1970) * 12
        self.att_positions = [i for i, token in enumerate(self.tokens) if token.startswith("<ATT-")]
        self.months_passed = np.cumsum([int(self.tokens[i][5:-1]) for i in self.att_positions])
        self.ny_positions = [i for i, token in enumerate(self.tokens) if token == "<NY>"]

    def _anchor(self, prefix: str):
        """Read the value of the demographic anchor with this prefix, or None."""
        for token in self.tokens[:6]:
            if token.startswith(prefix) and token[len(prefix):-1].isdigit():
                return int(token[len(prefix):-1])
        return None

    def block(self, seq_point: int) -> tuple:
        """
        Read the block a seq point closes.

        Args:
            seq_point: Index into the sequence, which ReClaim scores at

        Returns:
            tuple: (month since January 1970, visit kind, None), or (None, None, why it cannot be read)
        """
        if self.first_month is None:
            return None, None, "sequence_without_age_anchor"
        if not 0 <= seq_point < len(self.tokens) or not self.tokens[seq_point].startswith("<ATT-"):
            return None, None, "seq_point_not_on_att"
        n_before = bisect.bisect_left(self.att_positions, seq_point)
        month = self.first_month + (int(self.months_passed[n_before - 1]) if n_before else 0)
        # The <NY> anchors count the years independently of the <ATT-n> sums, so the two must agree
        n_new_years = bisect.bisect_left(self.ny_positions, seq_point)
        if month // 12 != self.first_month // 12 + n_new_years:
            return None, None, "months_disagree_with_new_years"
        block_start = self.att_positions[n_before - 1] + 1 if n_before else 3
        kind = RECLAIM_VISIT_KINDS.get(self.tokens[block_start])
        if kind is None:
            return None, None, "block_without_visit"
        return month, kind, None


class EthosBlocks:
    """
    One ETHOS timeline, with the ReClaim block every event token falls in.

    The month is recovered from the age: an event sits at the middle of the birth year plus its age,
    rounded to the hour so that a visit at midnight on the first of a month stays in that month. The
    visit kind follows from the token. A drug comes only from a pharmacy visit, a token inside an
    inpatient stay belongs to it, and everything else is outpatient, which includes an emergency visit
    as it does in ReClaim. Separators, closing brackets and the end of the timeline are not events of
    any block, and a closing bracket is dated at discharge rather than at the visit start ReClaim uses.
    """

    def __init__(self, patient: dict, token_ids: dict):
        """
        Args:
            patient: One timeline, with its tokens, their ages and the birth year they count from
            token_ids: The separator, bracket and drug token IDs, as block_token_ids returns them
        """
        tokens, times = patient["tokens"], patient["times"]
        anchor = np.datetime64(f"{int(patient['age_reference']):04d}-07-01T00:00:00", "s")
        seconds = (np.round(times * SECONDS_IN_YEAR / 3600.0) * 3600).astype(np.int64)
        self.months = (anchor + seconds.astype("timedelta64[s]")).astype("datetime64[M]") \
            .astype(np.int64)
        self.kinds = np.full(len(tokens), OUTPATIENT, dtype=np.int64)
        self.is_event = np.ones(len(tokens), dtype=bool)
        self.is_separator = np.isin(tokens, list(token_ids["separators"]))
        in_stay = False
        for position, token_id in enumerate(tokens):
            token_id = int(token_id)
            if token_id in token_ids["drugs"]:
                self.kinds[position] = PHARMACY
            elif token_id == token_ids["inpatient_start"]:
                in_stay = True
                self.kinds[position] = INPATIENT
            elif token_id == token_ids["inpatient_end"]:
                in_stay = False
                self.is_event[position] = False
            elif token_id in token_ids["not_events"]:
                self.is_event[position] = False
            elif in_stay:
                self.kinds[position] = INPATIENT

        # Where each token last appears within the months eval_seq covers, which is all the checks on
        # a case or a control need to know
        self.last_position = {}
        for position in np.flatnonzero(self.months < FIRST_MONTH_AFTER_EVAL):
            self.last_position[int(tokens[position])] = int(position)

    def cut(self, month: int, kind: int) -> tuple:
        """
        Cut the timeline so that it holds everything ReClaim's prefix at this block holds.

        ReClaim's prefix holds every block up to and including this one. The cut falls at the first
        separator after the last event of those blocks, so the moment that event belongs to is kept
        whole, as the cuts of OnsetCohort are. That moment can hold an event of a later block of the
        same month, such as a drug dispensed on the day of the visit, which is reported as excess.

        Args:
            month: Months since January 1970 of the block to cut at
            kind: Rank of the block's visit kind within that month

        Returns:
            tuple: (cut point or None when no event precedes the block, whether the prefix holds excess)
        """
        within = self.is_event & ((self.months < month)
                                  | ((self.months == month) & (self.kinds <= kind)))
        if not within.any():
            return None, False
        last = int(np.flatnonzero(within)[-1])
        later_separators = np.flatnonzero(self.is_separator[last + 1:])
        cut = last + 1 + int(later_separators[0]) if len(later_separators) else len(self.months) - 1
        return cut, bool((self.is_event[:cut] & ~within[:cut]).any())


def block_token_ids(vocab) -> dict:
    """Collect the token IDs EthosBlocks tells the visit kinds and the non-events apart by."""
    def encoded(token):
        token_id = vocab.encode(token)
        return None if np.isnan(token_id) else int(token_id)

    separators = {int(token_id) for token_id in vocab.encode(SpecialToken.SEPARATOR_NAMES)}
    not_events = separators | {token_id for token_id in (encoded(SpecialToken.TIMELINE_END),
                                                         encoded("ED_ADMISSION_END"))
                               if token_id is not None}
    return {
        "separators": separators,
        "not_events": not_events,
        "drugs": {token_id for token, token_id in vocab.stoi.items() if str(token).startswith("ATC_")},
        "inpatient_start": encoded("INPATIENT_ADMISSION_START"),
        "inpatient_end": encoded("INPATIENT_ADMISSION_END"),
    }


def map_case_controls(case_control_path: str, traj_path: str, patients: list, vocab,
                      category_to_token: dict, target_categories: set) -> tuple[list, dict]:
    """
    Carry ReClaim's strata over to the ETHOS timelines.

    A ReClaim row id is a row of the trajectory parquet, which names the patient; the patient names the
    ETHOS timeline. A seq point becomes the ETHOS cut at the same block. Sex, age bracket and who is a
    case or a control are ReClaim's and are kept as given. A case whose target is missing after the
    cut, or a control whose timeline holds the target, is kept and counted, since the labels are the
    thing being matched.

    Args:
        case_control_path: ReClaim case-control parquet
        traj_path: ReClaim trajectory parquet its row ids index into
        patients: The per-patient timelines of the cohort
        vocab: Vocabulary the timelines were encoded with
        category_to_token: ICD category to the ETHOS token marking its onset
        target_categories: ICD categories to keep

    Returns:
        tuple: (strata in the shape the case-control code produces, counts of what the mapping did)
    """
    strata = pd.read_parquet(case_control_path, columns=[
        "sex", "age", "age_step", "disease_token_str", "case_ids", "control_ids",
        "case_seq_points", "control_seq_points"])
    traj = pd.read_parquet(traj_path, columns=["person_id", "eval_seq"])
    row_of_person = {int(patient["patient_id"]): row for row, patient in enumerate(patients)}
    token_ids = block_token_ids(vocab)
    sex_id = {"male": int(vocab.encode("SEX_M")), "female": int(vocab.encode("SEX_F"))}

    stats = Counter()
    stats["strata_in_file"] = len(strata)
    stats["trajectory_rows"] = len(traj)
    readers, cuts = {}, {}

    def reader(traj_row):
        """Pair a trajectory row with its ETHOS timeline, once per row."""
        if traj_row not in readers:
            row = None
            if 0 <= traj_row < len(traj):
                row = row_of_person.get(int(traj.person_id.iat[traj_row]))
            if row is None:
                problem = "patient_without_timeline" if 0 <= traj_row < len(traj) else "row_out_of_range"
                readers[traj_row] = (None, problem)
            else:
                readers[traj_row] = ((row, ReclaimSequence(traj.eval_seq.iat[traj_row]),
                                      EthosBlocks(patients[row], token_ids)), None)
        return readers[traj_row]

    mapped, categories_seen = [], set()
    for stratum in strata.itertuples(index=False):
        match = TARGET_TOKEN_PATTERN.match(str(stratum.disease_token_str).strip())
        category = match.group(1).upper()[:3] if match else None
        if category not in target_categories:
            stats["strata_not_a_target"] += 1
            continue
        categories_seen.add(category)
        if category not in category_to_token:
            stats["strata_target_not_in_vocabulary"] += 1
            continue
        target = category_to_token[category]

        out = {"sex": stratum.sex, "disease_token": target, "age": int(stratum.age),
               "age_step": int(stratum.age_step)}
        for label in ("case", "control"):
            rows, points = [], []
            for traj_row, seq_point in zip(getattr(stratum, f"{label}_ids"),
                                           getattr(stratum, f"{label}_seq_points")):
                found, problem = reader(int(traj_row))
                if found is None:
                    stats[f"{label}_{problem}"] += 1
                    continue
                row, sequence, blocks = found
                if int(patients[row]["context"][0]) != sex_id.get(stratum.sex):
                    stats[f"{label}_sex_disagrees"] += 1
                if sequence.birth_year != int(patients[row]["age_reference"]):
                    stats[f"{label}_birth_year_disagrees"] += 1

                key = (int(traj_row), int(seq_point))
                if key not in cuts:
                    month, kind, problem = sequence.block(int(seq_point))
                    cuts[key] = (None, False, problem) if problem else (*blocks.cut(month, kind), None)
                cut, excess, problem = cuts[key]
                if problem:
                    stats[f"{label}_{problem}"] += 1
                    continue
                if cut is None:
                    stats[f"{label}_nothing_before_cut"] += 1
                    continue

                last_target = blocks.last_position.get(target)
                if label == "case" and (last_target is None or last_target < cut):
                    stats["case_target_not_after_cut"] += 1
                if label == "control" and last_target is not None:
                    stats["control_has_target"] += 1
                if excess:
                    stats[f"{label}_prefix_holds_a_later_block"] += 1
                stats[f"{label}_mapped"] += 1
                rows.append(row)
                points.append(cut)
            out[f"{label}_ids"] = np.asarray(rows, dtype=np.int64)
            out[f"{label}_seq_points"] = np.asarray(points, dtype=np.int64)

        if len(out["case_ids"]) and len(out["control_ids"]):
            mapped.append(out)
            stats["strata_mapped"] += 1
        else:
            stats["strata_left_without_case_or_control"] += 1

    stats["target_categories_in_file"] = len(categories_seen)
    stats["prediction_points"] = sum(1 for cut in cuts.values() if cut[0] is not None)
    stats["prediction_points_holding_a_later_block"] = sum(1 for cut in cuts.values() if cut[1])
    return mapped, {name: int(count) for name, count in sorted(stats.items())}


def main():
    """Main execution function."""
    log_file = setup_logging(level=logging.INFO)
    logging.info("TRANSFORM PREDEFINED COHORT STARTING...")
    args = parse_args()
    traj_path = args.traj_parquet or os.path.join(os.path.dirname(args.case_control_ids),
                                                  "traj_test_pd.parquet")

    logging.info("\nConfiguration:")
    logging.info("%sTimelines file: %s", INDENT1, args.timelines)
    logging.info("%sVocabulary: %s", INDENT1, args.vocab)
    logging.info("%sReClaim case-control file: %s", INDENT1, args.case_control_ids)
    logging.info("%sReClaim trajectory file: %s", INDENT1, traj_path)
    logging.info("%sReClaim target tokens: %s", INDENT1, args.target_tokens)
    logging.info("%sOutput: %s", INDENT1, args.output)

    logging.info("********** 1 Reducing the timelines to case-control inputs **********")
    vocab = Vocabulary(args.vocab)
    # ReClaim's strata name their own patients, so every timeline has to stay available to them
    cohort = OnsetCohort(args.timelines, vocab, min_seq_len=0, min_onsets=0)
    logging.info("%sPatients that can be evaluated: %s", INDENT1, f"{len(cohort.patients):,}")
    logging.info("%sMale: %s | female: %s", INDENT1, f"{int(cohort.all_sex.sum()):,}",
                 f"{int((~cohort.all_sex).sum()):,}")

    logging.info("********** 2 Resolving the diseases to evaluate **********")
    # The diseases are ReClaim's target tokens and nothing else. This vocabulary names more ICD
    # categories than ReClaim evaluates, and sweeping all of them would average the area over a
    # different population of diseases than the other models are averaged over.
    target_categories, unparsed = load_target_categories(args.target_tokens)
    if unparsed:
        logging.warning("%s%d target tokens name no ICD category and are skipped", INDENT1,
                        len(unparsed))
    disease_map = {category: [category] for category in dict.fromkeys(target_categories)}
    disease_tokens, token_label, token_categories, token_diseases = construct_target_tokens(
        disease_map, args.ref_dir, vocab)
    if not disease_tokens:
        raise RuntimeError("None of the ICD codes could be resolved to a token of the vocabulary.")
    logging.info("%sTarget diseases requested: %s | target tokens to evaluate: %s", INDENT1,
                 f"{len(disease_map):,}", f"{len(disease_tokens):,}")
    # The two counts differ for two reasons, and the evaluation is averaged over the second, so say
    # which reason it was. A category the vocabulary cannot name is dropped and warned about above; a
    # category sharing a title with another resolves to the same token, and the two are scored once.
    shared = {token_id: categories for token_id, categories in token_categories.items()
              if len(categories) > 1}
    if shared:
        logging.warning("%s%d tokens stand for more than one requested category, and each is scored "
                        "once: %s", INDENT1, len(shared),
                        "; ".join(",".join(c) for c in list(shared.values())[:5]))
    if len(disease_tokens) <= MAX_DISEASES_TO_LIST:
        for token_id in disease_tokens:
            logging.info("%s%s: %s -> %s", INDENT2, ";".join(token_diseases[token_id]),
                         ";".join(token_categories[token_id]), token_label[token_id])

    logging.info("********** 3 Mapping ReClaim's case-control strata **********")
    category_to_token = {category: token_id for token_id, categories in token_categories.items()
                         for category in categories}
    strata, mapping_stats = map_case_controls(
        args.case_control_ids, traj_path, cohort.patients, vocab, category_to_token,
        set(disease_map))
    logging.info("%sMapping of the ReClaim strata:", INDENT1)
    for name, count in mapping_stats.items():
        logging.info("%s%-45s %s", INDENT2, name, f"{count:,}")
    logging.info("%sCase-control strata: %s", INDENT1, f"{len(strata):,}")
    if not strata:
        raise RuntimeError("No ReClaim stratum could be carried over to a timeline.")

    logging.info("********** 4 Writing the prepared cohort **********")
    prepared = {
        "case_control_ids": args.case_control_ids,
        "traj_parquet": traj_path,
        "target_tokens": args.target_tokens,
        "n_diseases_requested": len(disease_map),
        "n_tokens_evaluated": len(disease_tokens),
        "mapping": mapping_stats,
        # One entry per token, so evaluate.py can label a result without the reference tables
        "tokens": [{"token_id": token_id, "token": token_label[token_id],
                    "icd_codes": ";".join(token_categories[token_id]),
                    "disease": ";".join(token_diseases[token_id])}
                   for token_id in disease_tokens],
        "strata": [{name: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for name, value in stratum.items()} for stratum in strata],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        # Columns read off a parquet row arrive as numpy scalars, which the JSON encoder does not
        # know; item() hands it the plain Python value behind one.
        json.dump(prepared, f, default=lambda value: value.item())
    logging.info("%sWrote %s strata to %s", INDENT1, f"{len(strata):,}", args.output)

    logging.info("Log file: %s", log_file)
    logging.info("TRANSFORM PREDEFINED COHORT COMPLETED SUCCESSFULLY...")


if __name__ == "__main__":
    main()
