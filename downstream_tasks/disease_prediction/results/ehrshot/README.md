# EHRSHOT disease-onset AUC results

Per-disease AUCs on EHRSHOT for five models, all scored on the **same**
case/control cohort built by [`prepare_disease_case_controls_fast.py`](../../prepare_disease_case_controls_fast.py)
(`OFFSET=365.25`, `AGE_GROUP_RANGE=20,100,10`). Every file has one row per
3-character ICD-10 code; the 1,024 codes and the per-code `n_samples`,
`n_diseased` and `n_healthy` are identical across all five files.

| File | Model | Mean AUC |
|------|-------|----------|
| `reclaim_l.csv` | ReClaim-L (Qwen3-L, post-trained on MarketScan) | 0.7527 |
| `llama3.1.csv` | Llama-3.1 architecture, post-trained with the ReClaim pipeline | 0.7237 |
| `gpt2.csv` | GPT-2 architecture, post-trained with the ReClaim pipeline | 0.7164 |
| `delphi.csv` | Delphi trained on MarketScan | 0.6893 |
| `lightgbm.csv` | LightGBM on bag-of-codes features, trained on MarketScan | 0.6466 |

Columns:

| Column | Meaning |
|--------|---------|
| `icd_code` | 3-character ICD-10 code of the endpoint. |
| `auc` | Mean AUC over the `(sex, age bracket)` strata. |
| `n_samples` | Number of strata averaged. |
| `n_diseased`, `n_healthy` | Cases and controls summed over strata. |

The models were evaluated with different tokenizers, so the raw
`disease_token` ids written by the pipeline were converted to ICD-10 codes
([`to_icd_code.py`](../../../../visualization/disease_prediction/to_icd_code.py)) before release.

These are the current (revised) results; the AUCs in the first arXiv
version of the paper were computed with an earlier ReClaim checkpoint and
differ slightly.

Regenerate the comparison figures from the repository root:

```bash
bash visualization/disease_prediction/run_ehrshot_figures.sh
```

The figures and paired-comparison summary are written to
[`../../output/figures/ehrshot/`](../../output/figures/ehrshot/).
