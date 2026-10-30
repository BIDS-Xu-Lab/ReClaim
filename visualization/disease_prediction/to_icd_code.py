#!/usr/bin/env python3
"""Re-key an AUC table from tokenizer ids to 3-character ICD-10 codes.

`test_case_controls.py` writes `disease_token` as an id of the evaluated
model's tokenizer. Ids are not comparable across checkpoints, so all plotting
scripts join on the ICD-10 code instead. Works for both `auc_results.csv` and
`auc_results_unpooled.csv`.

    python to_icd_code.py --tokenizer /path/to/checkpoint \
        --input auc_results.csv --output auc_results.icd.csv

Tables that already carry a `disease_token_str` column (e.g. the LightGBM
baseline output) need neither: the code is parsed from the string. For tables
written in another id space, pass `--id_map` with a two-column csv
(`disease_token,icd_code`) instead of `--tokenizer`.
"""

import argparse
import re

import pandas as pd

DX_MAJOR_RE = re.compile(r"<DX-MAJOR_([A-Z0-9]{3})>")


def id_to_code_from_tokenizer(path: str) -> dict[int, str]:
    from transformers import AutoTokenizer

    vocab = AutoTokenizer.from_pretrained(path).get_vocab()
    return {int(i): m.group(1) for tok, i in vocab.items() if (m := DX_MAJOR_RE.fullmatch(tok))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tokenizer", help="Checkpoint/tokenizer that produced the disease_token ids")
    src.add_argument("--id_map", help="csv with columns disease_token,icd_code")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "disease_token_str" in df.columns and not (args.tokenizer or args.id_map):
        codes = df["disease_token_str"].astype(str).str.extract(DX_MAJOR_RE)[0]
        df = df.drop(columns="disease_token_str")
    else:
        if args.tokenizer:
            id_to_code = id_to_code_from_tokenizer(args.tokenizer)
        elif args.id_map:
            m = pd.read_csv(args.id_map)
            id_to_code = dict(zip(m["disease_token"].astype(int), m["icd_code"].astype(str)))
        else:
            parser.error("--tokenizer or --id_map is required when the table has no disease_token_str column")
        codes = df["disease_token"].astype(int).map(id_to_code)
    n_missing = int(codes.isna().sum())
    if n_missing:
        raise SystemExit(
            f"{n_missing}/{len(df)} disease_token ids do not decode to <DX-MAJOR_*> tokens under "
            "this tokenizer; the table was probably written by a different checkpoint."
        )
    df.insert(0, "icd_code", codes)
    df.drop(columns="disease_token").to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
