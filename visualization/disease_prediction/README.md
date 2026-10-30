# Disease-Onset Figures

Plotting scripts for the disease-onset prediction figures of the ReClaim
paper, plus the EHRSHOT results needed to reproduce the external-validation
panels. MarketScan and YNHH results are not distributed; the scripts take
them as inputs.

## Inputs: AUC tables keyed by ICD-10 code

All scripts read `auc_results.csv` / `auc_results_unpooled.csv` files (from
`../../downstream_tasks/disease_prediction/test_case_controls.py`, or any method with the same columns) after
converting `disease_token` to the 3-character ICD-10 code:

```bash
python to_icd_code.py --tokenizer /path/to/evaluated/checkpoint \
    --input auc_results.csv --output auc_results.icd.csv
```

Token ids are specific to a checkpoint's tokenizer. Two checkpoints (e.g. a
pre-trained and a post-trained model, or ReClaim and a baseline) can give
the same disease different ids, so joining raw ids pairs unrelated diseases.
All scripts therefore require an `icd_code` column and compare methods only
on the codes they share. For methods that do not use a Hugging Face
tokenizer, pass `--id_map` with a `disease_token,icd_code` csv instead.

The baselines need less: `../../downstream_tasks/disease_prediction/delphi_baseline/evaluate.py` already writes an
`icd_code` column, and LightGBM tables carry `disease_token_str`, which
`to_icd_code.py --input ... --output ...` parses without a tokenizer.

Required columns: `icd_code`, `auc`; `n_diseased`, `n_healthy` for the
rare-disease subset and the case-count scatter; `sex`, `age`, `auc_delong`
for unpooled tables.

## Figures

Run from this directory; outputs go to `--out_dir` (default
`../../downstream_tasks/disease_prediction/output/figures/`) as
`.png` and `.pdf`, with a `*_summary.csv` of the plotted statistics.

| Paper panel | Script | Inputs |
|-------------|--------|--------|
| Fig. 2a-d, 2e-h | `plot_method_comparison.py` | ReClaim, Delphi, LightGBM on MarketScan retrospective / prospective |
| Fig. 3a-c, 3e-g | `plot_method_comparison.py --no_rare` | same, on EHRSHOT / YNHH |
| Fig. 3d, 3h | `plot_dataset_shift.py` | ReClaim on the external dataset and on MarketScan retrospective |
| Fig. 4a-d | `plot_auc_stratified.py` | ReClaim pooled + unpooled tables, MarketScan retrospective |
| Fig. 4e | `plot_followup_auc.py` | ReClaim evaluated at several `OFFSET` horizons |
| Supplementary model-scaling panel | `plot_model_scaling.py` | ReClaim-S/M/L at the same training stage |
| Supplementary post-training panels | `plot_pretrain_posttrain.py` | ReClaim-L pre-trained and post-trained |

What each script draws:

- **`plot_method_comparison.py`** — box plots with per-disease points and
  mean diamonds on the diseases shared by the three methods, plus a rare
  subset (case fraction < 0.0005 and > 10 cases, from ReClaim's counts);
  per-disease scatter of ReClaim against each baseline with the identity
  line, mean paired difference ± 95% CI and the fraction above the diagonal.
- **`plot_dataset_shift.py`** — Bland-Altman plot of the per-disease AUC
  change between two datasets, colored by ICD-10 chapter, with the mean
  difference and 95% limits of agreement.
- **`plot_auc_stratified.py`** — AUC vs number of cases (log scale, diseases
  with > 10 cases) colored by chapter, a chapter legend, per-chapter box
  plots, and per-age-bracket / per-sex distributions of chapter mean AUCs.
- **`plot_followup_auc.py`** — box plots per horizon on the shared diseases,
  points colored by chapter.
- **`plot_model_scaling.py`** — mean AUC ± 95% CI per model size with paired
  deltas.
- **`plot_pretrain_posttrain.py`** — box plots and per-disease paired
  scatter of pre- vs post-training AUC.

Shared style, loading and drawing code is in `common.py`; chapter labels
and colors come from `../../downstream_tasks/disease_prediction/data/icd_code_map_with_category.csv`.

## Examples

EHRSHOT (bundled results, Fig. 3a-c; written to `../../downstream_tasks/disease_prediction/output/figures/ehrshot/` by default):

```bash
bash run_ehrshot_figures.sh
```

MarketScan retrospective (Fig. 2a-d), with your own converted results:

```bash
python plot_method_comparison.py \
    --reclaim reclaim_retro.icd.csv --delphi delphi_retro.icd.csv --lightgbm lightgbm_retro.icd.csv \
    --dataset Retrospective --stem retrospective
```

Prediction horizons (Fig. 4e), labels in months:

```bash
python plot_followup_auc.py --input 1=auc_30d.icd.csv --input 6=auc_180d.icd.csv \
    --input 12=auc_365d.icd.csv --input 60=auc_1825d.icd.csv
```

Model scaling (supplementary figure), smallest model first:

```bash
python plot_model_scaling.py --input ReClaim-S=s.icd.csv --input ReClaim-M=m.icd.csv --input ReClaim-L=l.icd.csv
```

## Bundled results

[`../../downstream_tasks/disease_prediction/results/ehrshot/`](../../downstream_tasks/disease_prediction/results/ehrshot/) holds per-disease AUCs on EHRSHOT for ReClaim-L, Delphi,
LightGBM and two architecture baselines (GPT-2, Llama-3.1), all on the same
1,024-disease cohort. See [`downstream_tasks/disease_prediction/results/ehrshot/README.md`](../../downstream_tasks/disease_prediction/results/ehrshot/README.md).

## Dependencies

`matplotlib`, `numpy`, `pandas`; `transformers` for `to_icd_code.py --tokenizer`.
