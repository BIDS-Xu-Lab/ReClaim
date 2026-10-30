"""Utility functions for baseline pipeline."""

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from lightgbm_baseline.data import canonicalize_dx_major_token


FEATURE_TOKEN_PREFIXES = ("<DX-MAJOR_", "<PROC-", "<RX-", "<ATT-")


def load_feature_vocab(tokens_file: str) -> Dict[str, int]:
    """Load the {token_string: token_id} feature vocabulary.

    `tokens_file` is either a JSON map or a Hugging Face tokenizer/checkpoint
    directory, in which case its DX-MAJOR / PROC / RX / ATT tokens are used.
    """
    if os.path.isdir(tokens_file):
        from transformers import AutoTokenizer

        vocab = AutoTokenizer.from_pretrained(tokens_file).get_vocab()
        return {t: int(i) for t, i in vocab.items() if t.startswith(FEATURE_TOKEN_PREFIXES)}
    with open(tokens_file) as f:
        return json.load(f)


def get_dx_major_tokens(tokens_file: str) -> List[str]:
    """Return sorted list of all <DX-MAJOR_*> token strings from the pre-saved vocab file."""
    vocab = load_feature_vocab(tokens_file)
    return sorted(t for t in vocab if t.startswith("<DX-MAJOR"))


def load_disease_token_id_map(
    disease_tokens: List[str],
    tokens_file: str,
) -> Dict[str, int]:
    """Return {token_string: token_id} for the given disease tokens."""
    full_map = load_feature_vocab(tokens_file)
    return {t: full_map.get(t, t) for t in disease_tokens}


def get_target_tokens(
    icd_code_map_file: Optional[str],
    tokens_file: str,
    all_tokens: List[str],
    all_diseases: bool = False,
    target_disease_token: str = "<DX-MAJOR_I10>",
    max_diseases: Optional[int] = None,
) -> List[str]:
    if icd_code_map_file:
        known = set(load_feature_vocab(tokens_file).keys())
        icd_code_map = pd.read_csv(icd_code_map_file)
        if "ICD code" not in icd_code_map.columns:
            raise ValueError(f'"ICD code" column not found in {icd_code_map_file}')
        icd_codes = icd_code_map["ICD code"].astype(str).tolist()
        selected_tokens = []
        for icd_code in icd_codes:
            token = canonicalize_dx_major_token(f"<DX-MAJOR_{icd_code}>")
            if token in known:
                selected_tokens.append(token)
        selected_tokens = list(dict.fromkeys(selected_tokens))
        selected_tokens = [t for t in selected_tokens if t in set(all_tokens)]
        return selected_tokens

    if all_diseases:
        selected_tokens = all_tokens
        if max_diseases is not None and max_diseases > 0:
            selected_tokens = selected_tokens[:max_diseases]
        return selected_tokens

    return [canonicalize_dx_major_token(target_disease_token)]
