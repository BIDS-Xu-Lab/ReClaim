#!/usr/bin/env python3
"""
resolve_codes.py  --  resolve study code sets once, up front (pandas, no Spark, no claims).

A one-time vocabulary lookup that depends only on the NDC->RxNorm dictionary, kept separate from
the cohort build so the exact code lists that filter the claims are a reviewable artifact on disk.

  (1) DRUGS: every generic name referenced by the cohort (exposure treatment/comparator, washout,
      exclusion sets, baseline drug flags, vaccine NDC generics) is resolved by EXACT,
      case-insensitive match on `rxnorm_ingredient_name`, collecting the matching `ndc_drug_code`s.
      Exact matching keeps "ibuprofen" from sweeping in combination products; any generic that
      resolves to zero NDCs is reported UNMATCHED and left out, never substring-guessed. An
      optional per-row `ingredient_name` alias handles RxNorm rollups spelled differently from the
      clinical generic (mefenamic acid -> mefenamate). "influenza vaccine" has no clean ingredient
      row and is resolved through the Redbook route instead (see resolvers/marketscan_resolver.py).

  (2) DIAGNOSIS / PROCEDURE codes need no dictionary lookup; they are centralized and normalized
      here: which outcome set is primary, required dx positions, care settings, the separate
      outpatient (physician-service) arm, and the exclusion / baseline-flag dx sets. Each code is
      normalized to its decimal-stripped prefix `match_key` (531.0 -> 5310), the key
      build_cohort/build_long_table prefix-match on.

OUTPUT (intermediate/codes_resolved/{cohort_id}/):
  drug_ndc_map.csv        generic_name, ndc_drug_code, n_ingredients
  resolution_summary.csv  generic_name, source, match_mode, ingredient_targets, n_ndc, status
  dx_proc_codes.csv       role, name, code, match_key, code_type, settings, position, meaning

All code CONTENT lives in codesets.py; there are no codeset CSVs.
Usage: python resolve_codes.py --config config/<cohort>.yaml --vocab_ndc <NDC->RxNorm csv>
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging

import yaml

from utils import setup_logging, time_execution
from resolvers import run_resolver
from datasources.registry import get_source

INDENT1, INDENT2 = "  ", "    "


def parse_args():
    p = argparse.ArgumentParser(description="Resolve study codes for a data source (resolve_codes)")
    p.add_argument("--config", required=True, help="Path to config/<cohort>.yaml")
    p.add_argument("--source", default="marketscan",
                   help="Data source name (datasources/registry.py).")
    p.add_argument("--vocab_ndc", default=None,
                   help="NDC->RxNorm mapping csv (REQUIRED for the marketscan source)")
    p.add_argument("--out_root", default="intermediate/codes_resolved",
                   help="Writes {out_root}/{cohort_id}/")
    return p.parse_args()


# =============================================================================
# main -- thin dispatcher: the per-model resolution lives in resolvers/<model>_resolver.py
# =============================================================================
@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("H0 RESOLVE CODES STARTING...")
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    cohort_id = cfg["cohort_id"]
    model = get_source(args.source)["data_model"]
    logging.info("Config: %s (cohort=%s, source=%s, model=%s)",
                 args.config, cohort_id, args.source, model)

    out_dir = run_resolver(args.source, cfg, args.out_root,
                           vocab_ndc=args.vocab_ndc)
    logging.info("H0 RESOLVE CODES COMPLETED (-> %s). Log: %s", out_dir, log_file)


if __name__ == "__main__":
    main()
