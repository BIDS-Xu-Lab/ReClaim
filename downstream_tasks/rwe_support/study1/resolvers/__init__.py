"""
resolvers -- code resolution dispatch.

Artifact layout: {codes_resolved_root}/{cohort_id}/...
"""
import os

from datasources.registry import get_source


def _resolved_dir(codes_resolved_root: str, source: str, cohort_id: str) -> str:
    return os.path.join(codes_resolved_root, cohort_id)


def load_resolved(codes_resolved_root: str, source: str, cohort_id: str):
    """Return the ResolvedCodes lookup for (source, cohort)."""
    model = get_source(source)["data_model"]
    if model == "marketscan":
        from resolvers.marketscan_resolver import MarketScanResolved
        return MarketScanResolved(codes_resolved_root, cohort_id)
    raise ValueError(f"Unknown data_model '{model}' for source '{source}'")


def run_resolver(source: str, cfg: dict, out_root: str, *, vocab_ndc=None):
    """resolve_codes dispatch: materialize the resolved-code artifacts for (source, cohort) under out_root."""
    model = get_source(source)["data_model"]
    out_dir = _resolved_dir(out_root, source, cfg["cohort_id"])
    if model == "marketscan":
        if not vocab_ndc:
            raise ValueError("--vocab_ndc is required for the marketscan source")
        from resolvers import marketscan_resolver
        marketscan_resolver.run(cfg, vocab_ndc, out_dir)
    else:
        raise ValueError(f"Unknown data_model '{model}' for source '{source}'")
    return out_dir
