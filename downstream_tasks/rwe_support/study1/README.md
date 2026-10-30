# study1: known-effect validation

Study 1: a known-effect replication of Schneeweiss et al., *Epidemiology* 2009 on MarketScan Medicare-supplemental
(MDCR) claims, asking whether a foundation-model embedding can replace hdPS covariates in the propensity score.

## Design

| | `positive_control_celecoxib_gi` | `negative_control_flu_hipfx` |
|---|---|---|
| Contrast | celecoxib vs nonselective NSAID -> GI bleed | influenza vaccine vs none -> hip fracture |
| Truth | protective (masked by confounding by indication) | null |
| Cohort | new users, 2011-2014, 180-day follow-up (cell `MDCR`) | landmark Oct 1 2011, Jan-Apr follow-up (cell `MDCR_s2011`) |
| Outcome model | logistic, PS deciles | Cox, PS deciles |

- Specifications (one PS each, same cohort): `crude`, `demo`, `hdps` (demo + k = 500, unpenalised logistic), and
  `delphi-only`, `reclaim-{s,m,l}-only` (age + sex + embedding, ridge logistic).
- PCA: `<model>-pca<k>-only`, k = 60, 120, 500 (`b2_reduce_embeddings.sh`).
- Doubly robust: cross-fitted AIPW for the effect in the treated, PS `hdps`/`delphi-only`/`reclaim-s-only` x
  outcome models Delphi/ReClaim-S/M/L (`b3_2_dr.sh`).
- Population: all eligible MDCR enrollees with a ReClaim token sequence (the tokenizability gate in `b0_2`,
  token store `$MS_PROCESSED_ROOT/$DATA_VERSION/$MS_RUN_NAME` from `env.sh`).

## Run

```bash
B01=$(sbatch --parsable b0_preprocessing/b0_1_resolve_codes.sh)
B02=$(sbatch --parsable --dependency=afterok:$B01 b0_preprocessing/b0_2_build_cohorts.sh)
B11=$(sbatch --parsable --dependency=afterok:$B02 b1_baseline/b1_1_select_covariates.sh)
BP=$(MODELS=none ENABLE_DELPHI=1 sbatch --parsable --cpus-per-task=32 --mem=500G --time=10:00:00 --dependency=afterok:$B02 b2_embeddings/b2_1_embeddings.sh)
EMB=""; for M in reclaim-s reclaim-m reclaim-l; do
  EMB="$EMB:$(MODELS=$M ENABLE_DELPHI=0 sbatch --parsable --time=16:00:00 --dependency=afterok:$BP b2_embeddings/b2_1_embeddings.sh)"; done
B31=$(ENABLE_EMBEDDINGS=1 sbatch --parsable --dependency=afterok:$B11:$BP$EMB b3_estimate/b3_1_estimate.sh)
sbatch --dependency=afterok:$B31 b3_estimate/b3_3_summary.sh
sbatch --dependency=afterok:$B31 b3_estimate/b3_2_dr.sh

# PCA (after the embeddings)
PCA=$(sbatch --parsable b2_embeddings/b2_reduce_embeddings.sh)
SPECS="reclaim-s-pca60-only reclaim-s-pca120-only reclaim-s-pca500-only" sbatch --dependency=afterok:$PCA b3_estimate/b3_1_estimate.sh
```

Knobs: `COHORTS` (all wrappers); `MODELS`, `ENABLE_DELPHI`, `BACKEND=vllm|hf` (`b2_1`); `SPECS`,
`BASELINE_SPECS`, `EMBED_SPECS`, `ENABLE_EMBEDDINGS` (`b3_1`); `DR_COHORTS` (`b3_2`, `b3_2b_estimate_dr.sh`
re-estimates from cached fits). Existing embeddings are reused.

## Outputs

```
results/mdcr/<cohort>/<cell>/<spec>/decile_adjusted/estimate.json
results/mdcr/<cohort>/<cell>/dr_aipw/dr_estimators_grid.csv
results/mdcr/_summary/supplementary_table.csv, cohort_characteristics.csv, forest plots
```
