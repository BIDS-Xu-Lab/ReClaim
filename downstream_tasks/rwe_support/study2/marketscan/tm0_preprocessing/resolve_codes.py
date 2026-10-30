#!/usr/bin/env python3
"""
resolve_codes.py  --  resolve drug ingredient names -> NDC sets, once, up front.

Reads the NDC->RxNorm ingredient map (one row per NDC x ingredient) and materializes
reviewable artifacts under intermediate/codes_resolved/:

  exposure_ndc_map.csv   drug_class, ndc_drug_code   per-class exposure NDCs. An NDC belongs to
                         a class if ANY ingredient is in the class list (combination products
                         count toward their study-class component, as in the reference's name
                         matching); NDCs spanning >=2 STUDY classes are excluded from arms but
                         REMAIN in the washout set (prior use still blocks new-user status).
  washout_ndc.csv        all class NDCs + duals
  dual_class_ndc.csv     the ambiguous combos
  med_class_ndc_map.csv  26 baseline medication classes (substring match on ingredient name)
  resolution_summary.csv per-class/ingredient counts

Light step (pandas only, no Spark). Usage: python tm0_preprocessing/resolve_codes.py --vocab_ndc <csv>
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
import argparse
import logging

import pandas as pd

import codesets_tte as cs
from utils import setup_logging, time_execution

DEFAULT_VOCAB = os.environ.get("VOCAB_NDC", "")   # NDC->RxNorm-ingredient map; set in env.sh


def parse_args():
    p = argparse.ArgumentParser(description="TTE code resolution (drug ingredient -> NDC)")
    p.add_argument("--vocab_ndc", default=DEFAULT_VOCAB)
    p.add_argument("--out_dir", default="intermediate/codes_resolved")
    return p.parse_args()


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("RESOLVE CODES STARTING...")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vocab = pd.read_csv(args.vocab_ndc, dtype=str,
                        usecols=["ndc_drug_code", "rxnorm_ingredient_name"])
    vocab["ing"] = vocab["rxnorm_ingredient_name"].str.lower().str.strip()
    vocab = vocab.dropna(subset=["ndc_drug_code", "ing"]).drop_duplicates()
    logging.info("Vocab: %s NDC x ingredient rows, %s distinct NDC",
                 f"{len(vocab):,}", f"{vocab.ndc_drug_code.nunique():,}")

    # ---- exposure classes: exact ingredient match ----
    rows, summary = [], []
    for cls, ings in cs.EXPOSURE_INGREDIENTS.items():
        hit = vocab[vocab.ing.isin([i.lower() for i in ings])]
        for ing in ings:
            n = (hit.ing == ing.lower()).sum()
            summary.append({"set": cls, "ingredient": ing, "n_ndc": int(n),
                            "status": "ok" if n else "UNMATCHED"})
            if n == 0:
                logging.warning("  UNMATCHED ingredient: %s (%s)", ing, cls)
        for ndc in hit.ndc_drug_code.unique():
            rows.append({"drug_class": cls, "ndc_drug_code": ndc})
    expo = pd.DataFrame(rows).drop_duplicates()

    # dual-STUDY-class NDCs -> ambiguous arm assignment; drop from arms, keep in washout
    per_ndc = expo.groupby("ndc_drug_code")["drug_class"].agg(lambda s: sorted(set(s)))
    dual = per_ndc[per_ndc.map(len) >= 2]
    dual_df = pd.DataFrame({"ndc_drug_code": dual.index,
                            "classes": ["|".join(c) for c in dual.values]})
    washout = pd.DataFrame({"ndc_drug_code": sorted(expo.ndc_drug_code.unique())})
    expo_clean = expo[~expo.ndc_drug_code.isin(dual.index)]
    logging.info("Exposure NDCs: %d total, %d dual-class excluded from arms (kept in washout)",
                 expo.ndc_drug_code.nunique(), len(dual_df))
    for cls, g in expo_clean.groupby("drug_class"):
        logging.info("  %-8s %d NDCs", cls, len(g))

    # ---- 26 medication classes: substring match on ingredient name ----
    med = cs.med_classes()
    med_rows = []
    for mcls, terms in med.items():
        pat = "|".join(pd.Series(terms).str.replace(r"([.^$*+?()\[\]{}|\\])", r"\\\1", regex=True))
        hit = vocab[vocab.ing.str.contains(pat, na=False, regex=True)]
        med_rows.append(pd.DataFrame({"med_class": mcls,
                                      "ndc_drug_code": hit.ndc_drug_code.unique()}))
        summary.append({"set": "med_class", "ingredient": mcls,
                        "n_ndc": int(hit.ndc_drug_code.nunique()),
                        "status": "ok" if len(hit) else "UNMATCHED"})
        logging.info("  med class %-28s %6d NDCs (%d ingredients)", mcls,
                     hit.ndc_drug_code.nunique(), hit.ing.nunique())
    med_df = pd.concat(med_rows, ignore_index=True).drop_duplicates()

    expo_clean.to_csv(os.path.join(args.out_dir, "exposure_ndc_map.csv"), index=False)
    washout.to_csv(os.path.join(args.out_dir, "washout_ndc.csv"), index=False)
    dual_df.to_csv(os.path.join(args.out_dir, "dual_class_ndc.csv"), index=False)
    med_df.to_csv(os.path.join(args.out_dir, "med_class_ndc_map.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "resolution_summary.csv"), index=False)
    logging.info("RESOLVE CODES COMPLETED -> %s. Log: %s", args.out_dir, log_file)


if __name__ == "__main__":
    main()
