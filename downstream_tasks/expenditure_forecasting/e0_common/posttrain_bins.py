#!/usr/bin/env python3
"""
Dollar-discretization machinery for the posttrain design (final post-training
recipe): how continuous dollar amounts map onto the model's cost vocabulary,
and how token-level predictions are converted back into dollars.

Background
----------
The pretrained ReClaim vocabulary encodes every cost as a mantissa token
<COST-me> meaning m x 10^e dollars (e.g. <COST-23> = $2,000).  The set of
representable values -- $0, $1..$9, $10..$90, ..., $9x10^9 -- is a
log-spaced grid ("the mantissa lattice"): steps are one significant digit,
so relative precision is roughly constant across six orders of magnitude.

The posttrain design reuses this EXISTING vocabulary as its response bins:
an annual cost label is rounded onto the lattice and expressed with the
corresponding <COST-me> token, so the model predicts over tokens it already
saw billions of times during pretraining (no new vocabulary, no task head).

This module provides the two halves of that contract:

  encode_cost : dollars -> lattice        (build supervision labels / audits)
  BinSpec     : the frozen bin vocabulary + per-bin expected-dollar tables
                (train the ordinal loss / cash distributions out as dollars)

Both are shared through e0_common so the pair builder (e1), the trainer (e3)
and the readout (e4) can never disagree about what a bin means.
"""

import json
import numpy as np

from sequence_parsing import GOLD_COST_COLS


def encode_cost(v: np.ndarray) -> np.ndarray:
    """Round dollar amounts onto the mantissa lattice (returns DOLLARS).

    Vectorized numpy replica of the pretraining tokenizer's encode_cost
    (data_processing s1_construct_tokens.py), which produced the <COST-me>
    tokens from per-claim amounts with Spark's ROUND_HALF_UP.  Same rounding
    rules, different output type: this returns the quantized dollar VALUE
    (23847.0 -> 20000.0), not the token string -- downstream code that needs
    the token goes through BinSpec._mantissa_token.

    Rules (matching pretraining exactly):
      * <= 0 or NaN            -> 0
      * 0 < v <= 1             -> treated as $1 (the smallest positive bin)
      * round mantissa HALF_UP -> floor(m + 0.5); ties go up (2500 -> 3000)
      * carry                  -> a mantissa that rounds to 10 becomes
                                  1 x 10^(e+1)  (9500 -> 10000, never "10e3")
      * exponent clip at 9     -> anything above 9.5e9 saturates at 9x10^9

    Bit-identical replication matters: pair labels rounded with this operator
    land on the same lattice pretraining used, and the discretization audit
    (e5) uses it to quantize raw-claims reconstructions, so the audit tests
    the real pipeline rounding rather than an approximation.
    """
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    pos = v > 0
    if not pos.any():
        return out
    sv = np.where(v[pos] <= 1.0, 1.0, v[pos])   # sub-dollar positives -> $1
    b = np.floor(np.log10(sv))                   # exponent e
    m = sv / np.power(10.0, b)                   # mantissa in [1, 10)
    a = np.floor(m + 0.5)                        # Spark ROUND_HALF_UP
    carry = a >= 10                              # 9.5 <= m: promote exponent
    a = np.where(carry, 1.0, a)
    b = np.where(carry, b + 1, b)
    clip = b > 9                                 # tokenizer caps at <COST-99>
    a = np.where(clip, 9.0, a)
    b = np.where(clip, 9.0, b)
    out[pos] = a * np.power(10.0, b)
    return out


class BinSpec:
    """The frozen response-bin vocabulary of a trained posttrain model.

    Loads bin_artifact_mantissa.json (written once per pair set by
    e1_preprocessing/build_cost_posttrain_data.py from the supervision
    training split, never test data) and exposes three views every consumer
    must share:

      bin_ids    : the observed annual-cost lattice values, SORTED BY DOLLAR
                   VALUE (59 bins in [$0, $1M] under the censored spec).
                   The value ordering is load-bearing: the soft ordinal CE
                   loss (e3) smooths a Gaussian over these indices, so
                   "one bin off" must mean "adjacent in dollars".
      tokens     : the same bins spelled as <COST-me> tokens, aligned with
                   bin_ids -- the sub-vocabulary the model predicts over.
      ev_dollars : per target, the bin's own dollar value (<COST-23> ->
                   $2,000): the same nominal value the token carries in
                   pre-training.  The trainer's Huber term and both readouts
                   consume this table: predicted $ = sum_i p(bin_i) * value_i.
                   No fitted decoding table is used; the artifact's descriptive
                   per-bin means are read only for the bin list.

    Notes:
      * Bin values above $1M are capped by the consumer (EV_CAP in
        e3_posttrain/readout_instruct_{vllm,hf}.py), which makes the readout the
        censored expectation E[min(X, $1M)].
      * Checkpoints save a bin_artifact_ref.json pointing at their artifact,
        so readout always reuses the exact tables a model was trained with.
    """

    def __init__(self, artifact_path: str):
        with open(artifact_path) as f:
            art = json.load(f)
        cm_total = art["conditional_mean_dollars"]["next_annual_cost"]

        # Bin id == its quantized dollar value, so sorting ids numerically
        # yields the dollar ordering the ordinal loss depends on.
        ids = sorted(int(k) for k in cm_total.keys())
        self.bin_ids = ids
        self.tokens = [self._mantissa_token(q) for q in ids]

        # Every bin is valued at its own dollar value.
        self.ev_dollars = {}
        for tc in GOLD_COST_COLS:
            self.ev_dollars[tc] = np.array([self._nominal(i) for i in range(len(self.bin_ids))])

    @staticmethod
    def _mantissa_token(q: int) -> str:
        """Quantized dollars -> token spelling: 2000 -> <COST-23> (m=2, e=3).
        Inverse of sequence_parsing.parse_cost_value on lattice points; the
        epsilon guards exact powers of ten against float log10 error."""
        if q <= 0:
            return "<COST-00>"
        e = int(np.floor(np.log10(q) + 1e-9))
        m = int(round(q / 10 ** e))
        return f"<COST-{m}{e}>"

    def _nominal(self, idx: int) -> float:
        """The bin's own dollar value."""
        return float(self.bin_ids[idx])

    def __len__(self):
        return len(self.tokens)
