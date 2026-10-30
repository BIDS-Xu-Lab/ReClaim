#!/usr/bin/env python3
"""
datasources/registry.py  --  the data-source registry + adapter factory.

`--source <name>` on resolve_codes/build_cohort/build_long_table looks up this table.
"""

import os

DATASOURCES = {
    "marketscan": {
        "data_model": "marketscan",
        # strata: MDCR only (both controls, age >= 65). raw_root/vocab come from the CLI.
        "strata": ["MDCR"],
        "spark_profile": "bigmem",
    },
}


def get_source(name: str) -> dict:
    if name not in DATASOURCES:
        raise KeyError(f"Unknown data source '{name}'. Known: {sorted(DATASOURCES)}")
    return DATASOURCES[name]


def default_stratum(name: str) -> str:
    """The default analysis cell for a source."""
    return get_source(name)["strata"][0]


def make_adapter(source: str, spark, *, stratum=None, era_years=None, raw_roots=None,
                 codes_resolved_root="intermediate/codes_resolved", cfg=None, vocab_ndc=None,
                 restrict_ids=None):
    """Construct the adapter for `source`. MarketScan needs raw_roots + era_years + the payer
    stratum (+ vocab_ndc for build_long_table's drug tokens). Adapter classes are imported lazily."""
    entry = get_source(source)
    model = entry["data_model"]
    if model == "marketscan":
        from datasources.marketscan import MarketScanAdapter
        return MarketScanAdapter(spark, raw_roots=raw_roots, payer=stratum,
                                 era_years=era_years, codes_resolved_root=codes_resolved_root,
                                 vocab_ndc=vocab_ndc, restrict_ids=restrict_ids)
    raise ValueError(f"Unknown data_model '{model}' for source '{source}'")
