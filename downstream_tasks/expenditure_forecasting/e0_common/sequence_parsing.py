#!/usr/bin/env python3
"""
Shared token-parsing helpers for the expenditure-forecasting pipeline.

Sequence format:
  - Sequences in `seq_thru_cutoff` are already wrapped as `<sos> ... <eos>`;
    do NOT prepend `<sos>` and strip the trailing `<eos>` before splitting.
  - Year boundaries are bare `<NY>` tokens (no year suffix). The first observed
    year is anchored by `<AGE-XX>` instead of an `<NY>` token, so the i-th `<NY>`
    marks the start of observation year i+1.
  - `<COST-me>` encodes mantissa m x 10^e dollars, summed per month per payer;
    `<COST-00>` is zero/negative, `<COST-99>` is the clip token,
    `<COST-MISSING>` is null.
"""

from typing import Optional

VT_TYPES = ("inpatient", "outpatient", "pharmacy")

# The task has one target, next-year total expenditure.  The component
# (inpatient / outpatient / pharmacy) columns are still parsed by extract_outcome
# for the cohort files, but no model is trained on, predicts, or is scored on them.
GOLD_COST_COLS = ["next_annual_cost"]
PRED_COST_COLS = ["predicted_next_annual_cost"]


def parse_cost_value(token: str) -> int:
    """Parse a single COST token to an integer dollar amount.

    E.g., <COST-53> -> 5 * 10^3 = 5000.  <COST-MISSING> and malformed
    tokens parse to 0.
    """
    if token == "<COST-MISSING>":
        return 0
    if not (token.startswith("<COST-") and token.endswith(">")):
        return 0
    cost_code = token[6:-1]
    if len(cost_code) >= 2 and cost_code.isdigit():
        mantissa = int(cost_code[0])
        exponent = int(cost_code[1:])
        return mantissa * (10 ** exponent)
    return 0


def parse_age_value(sequence: str) -> Optional[int]:
    """Extract first age value from a sequence. Returns None if not found.

    Enrollees with no pre-cutoff events have demographics-only sequences
    without an <AGE-XX> token; those return None and are excluded upstream.
    """
    for token in sequence.strip().split():
        if token.startswith('<AGE-') and token.endswith('>'):
            try:
                return int(token[5:-1])
            except ValueError:
                return None
    return None


def strip_wrappers(sequence: str) -> str:
    """Remove a trailing <eos> (sequences are stored fully wrapped).

    The leading <sos> is kept: it is part of the model input.
    """
    seq = sequence.strip()
    if seq.endswith('<eos>'):
        seq = seq[: -len('<eos>')].strip()
    return seq


def extract_outcome(sequence: str, prefix: str = "") -> dict:
    """Sum COST tokens in a sequence up to the first <NY> (or <eos>) token.

    Costs are attributed to the most recent <VT-...> token (inpatient,
    outpatient, or pharmacy).  Returns total cost and per-VT breakdown,
    with column names optionally prefixed (e.g. prefix="predicted_").
    """
    total = 0
    costs_by_vt = {vt: 0 for vt in VT_TYPES}
    current_vt = None
    for token in sequence.strip().split():
        if token.startswith("<NY") or token == "<eos>":
            break
        if token.startswith("<VT-") and token.endswith(">"):
            vt_type = token[4:-1]
            if vt_type in costs_by_vt:
                current_vt = vt_type
        elif token.startswith("<COST-"):
            val = parse_cost_value(token)
            total += val
            if current_vt is not None:
                costs_by_vt[current_vt] += val
    return {
        f"{prefix}next_annual_cost": total,
        **{f"{prefix}next_annual_cost_{vt}": costs_by_vt[vt] for vt in VT_TYPES},
    }
