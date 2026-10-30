"""
ETHOS Vocabulary

The token strings a checkpoint understands, the special tokens that punctuate a timeline, and the
injector that spells out the gap between two events.

The vocabulary is frozen: it was built when the model was trained, so a token string it does not
contain cannot be encoded. It also carries the quantile boundaries measured at training time, which
is why it is read back rather than rebuilt.

Adopted unchanged from the ETHOS repository, https://github.com/ipolharvard/ethos-paper, which is
MIT licensed. See LICENSE for the full terms and the copyright notice. This file joins the four
modules of that repository's ethos/tokenize package that this baseline needs, namely constants.py,
special_tokens.py, vocabulary.py and separators.py. Nothing is modified.

ETHOS stands for Enhanced Transformer for Health Outcome Simulation, and is described in Renc, Jia,
Samir, Was, Li, Bates and Sitek, "Zero shot health trajectory prediction using transformer",
npj Digital Medicine (2024), https://doi.org/10.1038/s41746-024-01235-0
"""

import pickle
from itertools import zip_longest
from typing import Any, Hashable, Iterable, Optional

import numpy as np
import pandas as pd
from torch import Tensor

TOKEN_DTYPE = np.uint16

TIME_DTYPE = np.float64

SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60

class SpecialToken:
    SEPARATORS = {
        k: v / (60 * 24 * 365.25)
        for k, v in {
            "_5m-15m": 5,
            "_15m-1h": 15,
            "_1h-2h": 60,
            "_2h-6h": 2 * 60,
            "_6h-12h": 6 * 60,
            "_12h-1d": 12 * 60,
            "_1d-3d": 24 * 60,
            "_3d-1w": 3 * 24 * 60,
            "_1w-2w": 7 * 24 * 60,
            "_2w-1mt": 2 * 7 * 24 * 60,
            "_1mt-3mt": 30 * 24 * 60,
            "_3mt-6mt": 3 * 30 * 24 * 60,
            "_=6mt": 6 * 30 * 24 * 60,
        }.items()
    }
    SEPARATOR_NAMES = list(SEPARATORS.keys())
    SEPARATOR_SIZES = np.fromiter(SEPARATORS.values(), dtype=TIME_DTYPE)
    TIMELINE_END = "_TIMELINE_END"
    DEATH = "_DEATH"

    # e.g., used for defining age of a person
    YEARS = {"_<5": 5, **{f"_{i - 5}-{i}y": i for i in range(10, 101, 5)}}
    YEAR_NAMES = list(YEARS.keys()) + ["_>100"]
    YEAR_BINS = list(YEARS.values())

    DECILES = [f"_Q{i}" for i in range(1, 11)]

    ALL = [*SEPARATOR_NAMES, TIMELINE_END, DEATH, *YEAR_NAMES, *DECILES]

    @staticmethod
    def get_longest_separator() -> (str, float):
        return SpecialToken.SEPARATOR_NAMES[-1], SpecialToken.SEPARATOR_SIZES[-1]

    @staticmethod
    def age_to_year_token(age: float) -> str:
        i = np.digitize(age, SpecialToken.YEAR_BINS)
        return SpecialToken.YEAR_NAMES[i]

    @staticmethod
    def convert_to_deciles(
        values: pd.Series,
        bins: Optional[Iterable] = None,
        scheme: str = "quantiles",
        ret_bins: bool = False,
    ) -> pd.Series | tuple[pd.Series, np.ndarray[float]]:
        labels = SpecialToken.DECILES
        if bins is None:
            if scheme == "equidistant":
                bins = np.linspace(values.min(), values.max(), len(labels) + 1)
            elif scheme == "quantiles":
                bins = values.quantile(q=np.linspace(0, 1, len(labels) + 1))
            else:
                raise ValueError(
                    f"Got invalid `scheme` {scheme}, available: 'equidistant' and 'quantiles'."
                )
        bins = np.unique(bins)
        if np.isnan(bins[-1]):
            bins = bins[:-1]
        discrete_values = pd.cut(
            values,
            bins=bins,
            labels=labels[: len(bins) - 1],
            include_lowest=True,
        )
        # Don't make these values categorical, because later we might want to map nans into
        # something else, and doing it with categorical-valued series is cumbersome.
        discrete_values = discrete_values.astype(object)
        if ret_bins:
            return discrete_values, bins
        return discrete_values

class QStorage:
    """Class to store quantiles and stokens for further processing using an existing vocab."""

    def __init__(self, storage: Optional[dict] = None):
        self._storage: dict[Hashable, Optional[np.ndarray]] = storage or {}

    def to_dict(self) -> dict:
        return self._storage

    def __bool__(self):
        return bool(self._storage)

    def values(self) -> set:
        # backward compatibility, the check is to be removed in the future
        return set(s for s in self._storage.keys() if isinstance(s, str))

    def register_values(self, records: Iterable[Hashable]):
        unique_values = np.unique(list(records))
        # nan is always put at the end, so we only check the last element
        if isinstance(unique_values[-1], float):
            unique_values = unique_values[:-1]
        self._storage.update(
            {record: None for record in unique_values if record not in self._storage}
        )

    def values_to_deciles(
        self, values: pd.Series, record_name: Hashable, scheme: str = "quantiles"
    ) -> pd.Series:
        bins = self._storage.get(record_name, None)
        if bins is not None:
            values.clip(bins[0], bins[-1], inplace=True)
        q_values, bins = SpecialToken.convert_to_deciles(values, bins, scheme, ret_bins=True)
        if record_name not in self._storage:
            self._storage[record_name] = bins
        return q_values

class Vocabulary:
    _stoi: dict[str, int]
    _itos: Optional[dict[int, str]]
    _q_storages: dict[Hashable, dict]
    _meta: dict[str, Any]

    def __init__(self, path=None):
        self._itos = None
        self._q_storages = {}
        self._meta = {}
        if path is not None:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._stoi = data["stoi"]
            self._itos = data["itos"]
            self._q_storages = data["q_storage"]
            self._meta = data["meta"]
        else:
            self._stoi = {t: i for i, t in enumerate(SpecialToken.ALL)}

    def tokenize(self, words: Iterable) -> np.ndarray[TOKEN_DTYPE]:
        new_words = np.asarray(np.copy(words))
        new_words = set(new_words[~pd.isna(new_words)])
        new_words = new_words.difference(self._stoi.keys())
        token_offset = len(self._stoi)
        new_stoi = dict(zip(new_words, np.arange(token_offset, token_offset + len(new_words))))
        self._stoi.update(new_stoi)
        if self._itos is not None:
            self._itos.update({v: k for k, v in new_stoi.items()})
        return self.encode(words)

    def encode(self, words: Iterable[str] | str) -> np.ndarray[np.float64] | np.float64:
        if isinstance(words, str):
            return self.stoi.get(words, np.nan)
        return np.fromiter(
            (self.stoi.get(w, np.nan) for w in words), count=len(words), dtype=np.float64
        )

    def decode(
        self, tokens: Iterable[TOKEN_DTYPE] | TOKEN_DTYPE
    ) -> np.ndarray[np.object_] | np.object_:
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        if not isinstance(tokens, Iterable):
            return self.itos.get(tokens, np.nan)
        return np.fromiter(
            (self.itos.get(t, np.nan) for t in tokens), count=len(tokens), dtype=np.object_
        )

    def __len__(self):
        return len(self.stoi)

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        if self._itos is None:
            self._itos = {v: k for k, v in self._stoi.items()}
        return self._itos

    @property
    def meta(self):
        return self._meta

    def add_meta(self, key, value):
        self._meta[key] = value

    def get_q_storage(self, name: Hashable) -> QStorage:
        return QStorage(self._q_storages.get(name))

    def add_q_storage(self, name: Hashable, q_storage: QStorage):
        self._q_storages[name] = q_storage.to_dict()

    def to_pickle(self, path):
        data = {
            "vocab_size": len(self._stoi),
            "stoi": self.stoi,
            "itos": self.itos,
            "q_storage": self._q_storages,
            "meta": self._meta,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def get_timeline_total_time(self, timeline: Iterable, decode=False) -> np.float64:
        sep_estimates = self.meta["separator_estimates"]["mean"]
        name, size = SpecialToken.get_longest_separator()
        sep_estimates[name] = size

        def get_sep_time(t):
            # not sure whether the second check should be implicit like this
            if decode and t < len(self):
                t = self.decode(t)
            return sep_estimates.get(t, np.nan)

        sep_times = np.vectorize(get_sep_time, otypes=[np.float64])(timeline)
        return np.nansum(sep_times)

class SeparatorInjector:
    def __init__(self, vocab: Vocabulary):
        sep_time_expectations = (
            SpecialToken.SEPARATOR_SIZES[1:] - SpecialToken.SEPARATOR_SIZES[:-1]
        ) / 2
        sep_time_expectations = [*sep_time_expectations, SpecialToken.SEPARATOR_SIZES[-1]]
        name_to_time = dict(zip(SpecialToken.SEPARATOR_NAMES, sep_time_expectations))
        # we start from 1 because 0 is the bin that times fall below the shortest separator
        self._sep_bin_to_token = dict(
            zip(range(1, len(SpecialToken.SEPARATOR_NAMES) + 1), SpecialToken.SEPARATOR_NAMES)
        )
        self._sep_token_to_time = {
            vocab.encode(sep_name): name_to_time[sep_name]
            for i, sep_name in enumerate(SpecialToken.SEPARATOR_NAMES, 1)
        }
        self._timeline_end_token = vocab.encode(SpecialToken.TIMELINE_END)

    def __call__(
        self,
        sorted_times: np.ndarray[float],
        sorted_event_tokens: Iterable[TOKEN_DTYPE],
    ) -> (np.ndarray[TIME_DTYPE], np.ndarray[TOKEN_DTYPE]):
        """The input must be sorted, adds also TIMELINE_END token."""
        deltas = sorted_times[1:] - sorted_times[:-1]
        # longest separator occurs multiple times if possible
        longest_sep, longest_sep_dur = SpecialToken.get_longest_separator()
        binned_deltas = np.digitize(deltas, SpecialToken.SEPARATOR_SIZES)
        separator_tokens = (
            [token] * round(deltas[i] / longest_sep_dur) if token == longest_sep else token
            for i, token in enumerate(binned_deltas)
        )
        tokens = np.fromiter(
            (
                self._conv_token_to_uint(token, self._timeline_end_token)
                for event_token, sep_token in zip_longest(sorted_event_tokens, separator_tokens)
                for token in (
                    event_token,
                    *(sep_token if isinstance(sep_token, list) else [sep_token]),
                )
                if token != 0  # 0 is the bin where the time falls below the shortest separator
            ),
            dtype=TOKEN_DTYPE,
        )

        times_iter = ShortMemoryIterator(sorted_times)
        times = np.fromiter(
            (
                self._conv_token_to_time(token, times_iter, self._timeline_end_token)
                for token in tokens
            ),
            dtype=TIME_DTYPE,
            count=len(tokens),
        )
        return times, tokens

    def _conv_token_to_uint(
        self, token: TOKEN_DTYPE, timeline_end_token: TOKEN_DTYPE
    ) -> TOKEN_DTYPE:
        if token in self._sep_bin_to_token:
            return token - 1
        elif token is None:
            return timeline_end_token
        return token

    def _conv_token_to_time(
        self, token: TOKEN_DTYPE, times_iter, timeline_end_token: TOKEN_DTYPE
    ) -> TIME_DTYPE:
        """For regular tokens, just returns their time, while separators are estimated."""
        if token in self._sep_token_to_time:
            return times_iter.prev + self._sep_token_to_time[token]
        elif token == timeline_end_token:
            return times_iter.prev
        return next(times_iter)

class ShortMemoryIterator:
    def __init__(self, iterable: Iterable):
        self.iterator = iter(iterable)
        self.prev = None

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.iterator)
        self.prev = value
        return value
