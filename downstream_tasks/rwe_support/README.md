# RWE Support

This folder contains the support pipeline for the real-world evidence (RWE)
analysis in ReClaim. The task evaluates whether embeddings of patient trajectory
sequences improve confounding adjustment in a target trial emulation of
second-line antidiabetic medications.

The analysis uses a held-out MarketScan RWE cohort processed with the v6
ReClaim data pipeline. Unlike the prediction tasks, this RWE case study uses
full longitudinal trajectories without a 2023 temporal split.

## Task

Following Tang et al. (2025), "Association of GLP-1 receptor agonist use with
psychiatric outcomes in adults with type 2 diabetes: a target trial emulation",
we conduct this target trial emulation with minor modifications for pairwise
comparisons among three antidiabetic medication classes:

- GLP-1 receptor agonists
- SGLT-2 inhibitors
- DPP-4 inhibitors

For each comparison, propensity scores are estimated with:

- Traditional covariates only (`v6-no`)
- Traditional covariates plus ReClaim embeddings (`v6-s`, `v6-m`, `v6-l`)
- Traditional covariates plus Delphi embeddings (`delphi`)

Residual systematic bias is evaluated with negative control outcomes (NCOs) and
summarized by expected absolute systematic error (EASE). Primary outcome effect
estimates are also empirically calibrated using the NCO-derived error model.

## Local Inputs

Before running, prepare these local inputs:

- v6 ReClaim MarketScan processed data. Set `MARKETSCAN_SPLIT_DIR` to the
  explicit split folder used for this analysis, usually
  `<PROCESSED_MARKETSCAN_DATA_ROOT>/v6/marketscan_val`. For `TEST_SPLIT=both`,
  use a comma-separated pair of split folders.
- NDC to RxNorm ingredient mapping from the vocabulary mapping pipeline,
  generated under `preprocessing/vocab_mapping/mapping/`
- Local `lookup_table/` code-set asset for cohort definitions, outcomes,
  covariate clusters, medication groups, and NCOs. This asset is not included
  in the repository; it comes from the
  [PennCIL lab at the University of Pennsylvania](https://penncil.med.upenn.edu/people/)
  and may be provided by request.
  Expected files:
  - `outcome_cov_icd.csv`: ICD code sets for cohort eligibility, primary
    outcomes, and selected baseline covariates
  - `cluster_master.csv`: ICD diagnosis clusters used to create baseline
    medical-condition covariates
  - `huilin_med.csv`: medication class keyword lists used to create baseline
    medication covariates
  - `glp_nco_icd_0225.csv`: negative control outcome definitions and ICD codes
- ReClaim model checkpoints for `v6-s`, `v6-m`, and `v6-l`
- Delphi checkpoint and disease map for Delphi embedding extraction

Generated cohort files, embeddings, logs, Slurm outputs, and evaluation results
are local artifacts written under `intermediate/`, `logs/`, `slurm_job_logs/`,
and `results/`. The local `lookup_table/` asset should also remain outside Git.

## Pipeline

Run commands from this folder on a Slurm node.

### Step 1: Cohort Construction

```bash
sbatch s1_data_preprocess.sh
```

This script runs three Python stages:

- `p1_data_preparation.py`: identifies candidate GLP-1, SGLT-2, and DPP-4 users
  from ReClaim pharmacy tokens between 2018 and 2024, maps NDCs to drug names,
  and creates diagnosis, medication, visit, and demographic helper tables.
- `p2_cohort_extract.py`: defines the new-user cohort, assigns the index date
  as the first qualifying study-drug prescription on or after January 1, 2019,
  applies eligibility criteria, and constructs covariates, primary outcomes,
  and NCOs.
- `p3_construct_prevseq.py`: constructs each patient's pre-index ReClaim token
  sequence from events before the index date.

Key eligibility criteria in `p2_cohort_extract.py`:

- Age 18 or older at index date
- Baseline type 2 diabetes diagnosis
- No baseline type 1 diabetes or end-stage renal disease
- At least one baseline healthcare encounter
- No GLP-1, SGLT-2, or DPP-4 use during the 365-day baseline window
- No ambiguous treatment assignment from multiple study-drug classes on the
  same earliest prescription date

Main outputs:

- `intermediate/v6/selected_cohort/`
- `intermediate/v6/df_processed/`
- `intermediate/v6/final_cohort/cohort.csv`
- `intermediate/v6/final_cohort/cohort_with_pre_entry_seq.csv`

### Step 2: Embedding Generation

```bash
sbatch s2_generate_embeddings.sh
```

This script creates analysis-ready cohort CSV files for all model settings:

- `cohort_with_v6-no_embeddings.csv`: cohort without representation columns
- `cohort_with_v6-s_embeddings.csv`: ReClaim-S embeddings
- `cohort_with_v6-m_embeddings.csv`: ReClaim-M embeddings
- `cohort_with_v6-l_embeddings.csv`: ReClaim-L embeddings
- `cohort_with_delphi_embeddings.csv`: Delphi embeddings

ReClaim embeddings are extracted from `pre_entry_seq` using last-token pooling
from the final hidden layer. Sequences are left-truncated to the 4,096-token
context window by default, keeping the most recent pre-index events. Embeddings
are L2-normalized and written as `rep_0`, `rep_1`, etc.

Delphi embeddings are generated by converting ReClaim diagnosis-major tokens to
Delphi disease IDs, converting v6 relative `<ATT-*>` month intervals to days,
and applying Delphi's shorter 48-token context window with recent-event
selection. Delphi embeddings are also last-token pooled and L2-normalized.

### Step 3: Propensity Score and EASE Evaluation

```bash
sbatch s3_evaluation.sh
```

This script evaluates each model setting across three comparisons:

- GLP-1 vs DPP-4
- GLP-1 vs SGLT-2
- SGLT-2 vs DPP-4

`p5_ps_ease_eval.R` estimates propensity scores using LASSO-penalized logistic
regression, performs 1:1 nearest-neighbor matching with ATT estimand, and fits
Poisson log-link models with HC0 robust standard errors for NCOs and primary
outcomes.

NCOs are outcomes expected to have no causal relationship with the study drugs
but to share similar sources of bias. Their post-matching treatment-effect
estimates are used to fit an empirical null distribution, and EASE summarizes
the expected absolute residual systematic error on the log rate-ratio scale.
The same empirical error model is then used to calibrate primary outcome
confidence intervals.

For each run, outputs are written under:

```text
results/v6/<treatment>_vs_<control>/<model>_embeddings/
```

Typical outputs include:

- `shared_results.json`
- `primary_outcome_results.json`
- `ease.png`
- `equipoise.png`
- `nco_forest.png`
