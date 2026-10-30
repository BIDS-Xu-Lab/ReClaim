# study2/ynhh: target trial emulation on the YNHH EHR

Study 2 on the YNHH OMOP EHR: the same target trial emulation of Tang et al. 2026 as `study2/marketscan`, with the
reference's cohort construction implemented on OMOP tables.

## Design

- Index 2019-01-01 to 2024-09-30; two emulations (vs SGLT2i, vs DPP4i); follow-up to first outcome, death or
  2024-09-30.
- Diagnoses are matched as ICD-10 prefixes on source values (`ROUTE=source`, cell `ynhh_src`; `ROUTE=standard`
  uses standard concepts instead).
- Specifications, estimators and negative-control calibration as in `study2/marketscan`.

## Code sets

`lookup_table/` (the PennCIL code sets: `outcome_cov_icd.csv`, `cluster_master.csv`, `huilin_med.csv`,
`glp_nco_icd_0225.csv`, `glp_nco_reference24.txt`) is not included in the repository; it comes from the
[PennCIL lab at the University of Pennsylvania](https://penncil.med.upenn.edu/people/) and may be provided by
request. Place it at `study2/ynhh/lookup_table/`.

## Run

```bash
bash submit_all.sh            # whole chain with dependencies (EMBED=0 skips the GPU stage)

# or stage by stage
sbatch ty0_preprocessing/ty0_1_resolve_codes.sh
sbatch ty0_preprocessing/ty0_2_build_cohorts.sh
sbatch ty1_baseline/ty1_1_select_covariates.sh
sbatch ty2_embeddings/ty2_1_embeddings.sh                 # GPU
ENABLE_EMBEDDINGS=1 sbatch ty3_estimate/ty3_1_estimate.sh
sbatch ty3_estimate/ty3_2_summary.sh
sbatch ty3_estimate/ty3_3_expert_balance.sh               # balance of every spec on the reference covariates
```

`env.sh` also needs `YNHH_OMOP_ROOT` (one parquet directory per OMOP table) and, without internet access,
`CRAN_REPO`.

## Outputs

```
results/ynhh_src/<comparison>/<spec>/<estimator>/estimate.json, balance_smd.csv, nco_estimates.csv
results/ynhh_src_vs_{sglt2,dpp4}/table1.csv
results/ynhh_src/<comparison>/<spec>/{siptw_cox,psm_cox}/balance_expert_smd.csv
results/_summary_ynhh_src/summary_outcomes.csv, summary_diagnostics.csv, summary_expert_balance.csv
```
