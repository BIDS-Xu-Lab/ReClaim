#!/usr/bin/env python3
"""
resolvers/common.py  --  the code-resolution interface.

`codesets.py` is vocab-neutral (drug generic names, ICD-9 code prefixes, CPT/HCPCS). A resolver turns
those study definitions into the concrete keys a data model matches on: for MarketScan, NDC codes
(drugs) and decimal-stripped ICD-9 prefixes (dx/proc), materialized by resolve_codes.

A ResolvedCodes object is the per-(source, cohort) lookup the adapter's match_* / drug_dispensings
primitives consume. Its methods return plain lists of string keys; the adapter knows how to match them.
"""

from abc import ABC, abstractmethod
from typing import List


class ResolvedCodes(ABC):
    """Per-(source, cohort) resolved code lookup. Values are match keys interpreted by the adapter."""

    @abstractmethod
    def dx(self, role: str) -> List[str]:
        """Resolved dx keys for a resolve_codes role: 'outcome_inpatient' | 'outcome_outpatient' |
        'prior_exclusion_dx'. [] if the role is absent."""

    @abstractmethod
    def predefined_dx(self, codes: List[str]) -> List[str]:
        """Resolve a baseline clinical flag's raw ICD-9 code list to this source's match keys."""

    @abstractmethod
    def drug(self, generics: List[str], single_ingredient_only: bool = False) -> List[str]:
        """Resolved drug keys (NDC for MarketScan) for the given generics.
        single_ingredient_only drops fixed-dose combinations (exposure arms; washout keeps them)."""

    def proc(self, role: str) -> List[str]:
        """Resolved proc keys (default: none)."""
        return []
