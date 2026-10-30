# study2/marketscan: target trial emulation on MarketScan claims

Study 2 on MarketScan claims: target trial emulation of Tang et al. 2026 (GLP-1RA vs SGLT2i and vs DPP4i, six
psychiatric outcomes), built with the reference's own cohort construction (Supplement Fig. S1, Table S1).

## Design

- Population: CCAE + MDCR + MDCD, restricted to the held-out (val + test) split of ReClaim pre-training; index
  2019-01-01 to 2024-09-30. Two emulations, each with its own GLP-1RA arm.
- Claims adaptations: pharmacy coverage required; observability = one medical claim in the prior year;
  disenrollment censors (no death date in claims).
- Specifications (one PS each): `crude`, `demo`, `paper` (the reference's Table S5 covariates buildable in claims),
  `hdps` (demo + k = 500), `reclaim-{s,m,l}-only` (age + sex + embedding, ridge).
- Estimators: stabilized IPTW Cox (primary; weights truncated at p1/p99), 1:1 PSM Cox (caliper 0.2 SD, plus a
  sweep 0.05 to 1), PS-decile stratified Cox; calibration against the reference's 24 negative-control outcomes.

## Code sets

`lookup_table/` (the PennCIL code sets: `outcome_cov_icd.csv`, `cluster_master.csv`, `huilin_med.csv`,
`glp_nco_icd_0225.csv`, `glp_nco_reference24.txt`) is not included in the repository; it comes from the
[PennCIL lab at the University of Pennsylvania](https://penncil.med.upenn.edu/people/) and may be provided by
request. Place it at `study2/marketscan/lookup_table/`.

## Run

```bash
sbatch tm0_preprocessing/tm0_1_preprocessing.sh   # codes, cohorts (union + two emulations), hdPS long table
sbatch tm1_baseline/tm1_1_select_covariates.sh    # hdPS per emulation
sbatch tm2_embeddings/tm2_1_embeddings.sh         # ReClaim-S/M/L embeddings (GPU)
sbatch tm3_estimate/tm3_1_estimate.sh             # specification x estimator per emulation
sbatch tm3_estimate/tm3_2_summary.sh              # summary tables and figures
sbatch tm3_estimate/tm3_3_expert_balance.sh       # balance of every spec on the reference covariates (Table S5)
```

Knobs: `MODELS`, `BACKEND=vllm|hf` (tm2); `SPECS`, `ENABLE_EMBEDDINGS`, `CALIPER_SWEEP` (tm3).

## Outputs

```
results/<comparison>/<spec>/<estimator>/estimate.json, balance_smd.csv, nco_estimates.csv
results/<comparison>/<spec>/{siptw_cox,psm_cox}/balance_expert_smd.csv
results/_summary/summary_outcomes.csv, summary_diagnostics.csv, summary_expert_balance.csv
```
