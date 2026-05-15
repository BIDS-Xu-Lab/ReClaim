# EHR AUC Evaluation

Stratified AUC evaluation of a causal-LM-style health-trajectory model on
electronic-health-record (EHR) sequences.

For every major-disease ICD token in the model's vocabulary, the pipeline
constructs an age-stratified case/control cohort, scores each patient with
the model, and reports per-disease AUC (with DeLong variance) aggregated
across age brackets and sex.

## Repository layout

```
.
├── prepare_eval_data_new.py            # Stage 1 - build per-patient eval sequences
├── prepare_disease_case_controls_fast.py
│                                       # Stage 2 - assemble case/control IDs per disease
├── test_case_controls.py               # Stage 3 - run inference and compute AUC
├── scripts/
│   └── eval_ehr.sh                     # End-to-end driver (env-var configured)
├── utils/
│   ├── data_ops.py                     # Sequence parsing and case/control matching
│   └── evaluate_auc.py                 # DeLong AUC and variance
└── data/
    └── icd_code_map_with_category.csv  # ICD codes evaluated (optional filter)
```

## Input data contract

The pipeline consumes a parquet file of patient trajectories. Required
columns:

| Column                   | Type            | Description                                                                 |
|--------------------------|-----------------|-----------------------------------------------------------------------------|
| `seq`                    | `str`           | Whitespace-separated trajectory of model tokens for the patient.            |
| `token_count_thru_2022`  | `int`           | Index in `seq` marking the end of the in-sample (history) region.           |
| `total_token_count`      | `int`           | Index in `seq` marking the end of the full (history + outcome) region.     |
| `enrollee_id` *or* `person_id` | `str` / `int` | Patient identifier.                                                       |

Token grammar expected by the parser (from `utils/data_ops.py`):

- `<AGE_k>`     — patient age, in years.
- `<SEX-k>`     — patient sex.
- `<ATT-k>`     — gap between visits, in units of `--att_type` (`day`,
                  `week`, or `month`).
- `<NY>`        — new-year marker.
- `<VT-*>`      — visit-type marker (inpatient / outpatient / pharmaceutical).
- `<DX-MAJOR_*>`— major ICD code; these are the prediction targets.

## Pipeline stages

### Stage 1 — `prepare_eval_data_new.py`

Reads raw trajectory parquet shards, for each patient finds:

- `disease_points`               — token positions of first-occurrence
                                   `<DX-MAJOR_*>` tokens within the history
                                   region.
- `disease_points_ages`          — patient age (in days) at each disease
                                   point, computed by walking `<NY>` /
                                   `<ATT-k>` tokens forward from the
                                   `<AGE_k>` token.
- `disease_points_visit_location`— token index of the `<ATT-k>` opening
                                   the visit in which the disease appears.
- `eval_seq`                     — `seq` truncated to the history region.

Patients with fewer than two disease points are dropped. With `--longitudinal`
the parser walks the full sequence up to `total_token_count`; otherwise it
stops at `token_count_thru_2022`.

Writes `traj_test_pd.parquet`.

### Stage 2 — `prepare_disease_case_controls_fast.py`

For every major-disease token `D` (optionally restricted by
`--icd_code_map_file`), and for each (sex, age-bracket) cell:

- **Cases**    — patients whose first occurrence of `D` falls in the bracket.
- **Controls** — patients who never develop `D` (across the whole trajectory)
                 and have a valid prediction point in the same age bracket.
                 The prediction point for each control is the latest token
                 at least `--offset` days before the bracket boundary.

The expensive O(n·m²) pre-computation of valid prediction indices is done
once per sex via `_precompute_disease_case_control_shared` and reused
across all diseases.

Writes `disease_case_control_ids.parquet`, a row per
`(disease, sex, age-bracket)` cell containing case IDs, control IDs, and
the token position at which each patient should be scored.

### Stage 3 — `test_case_controls.py`

1. Collects all unique `(patient_row, seq_point)` pairs that appear in
   the case/control table and deduplicates them.
2. Runs the model once per unique pair (multi-GPU via
   `torch.multiprocessing`), retaining only the logits over the
   disease-token subspace to keep memory bounded.
3. For each `(disease, sex, age-bracket)` cell, looks up cached logits
   for the cell's cases and controls, then computes AUC and variance
   with DeLong's method (`utils.evaluate_auc.get_auc_delong_var`).
4. Aggregates per-bracket AUCs to a single per-disease AUC by averaging
   bracket means and summing their DeLong variances divided by n².

Writes:

- `auc_results.csv`             — one row per disease, columns
                                  `auc`, `n_samples`, `n_diseased`, `n_healthy`.
- `auc_results_unpooled.csv`    — one row per `(disease, sex, age-bracket)`.
- `logits_cache.pkl`            — cached logits, reusable with `--skip_logits`.

## Running the pipeline

The driver script is configured entirely through environment variables.
There are no hard-coded paths.

```bash
export SOURCE_DATA_DIR=/path/to/raw/trajectories
export EVAL_DATA_DIR=/path/to/eval/workdir
export MODEL_PATH=/path/to/hf/checkpoint

bash scripts/eval_ehr.sh
```

Required:

| Variable          | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `SOURCE_DATA_DIR` | Directory containing raw trajectory parquet shards.     |
| `EVAL_DATA_DIR`   | Output / working directory for all intermediate files.  |
| `MODEL_PATH`      | Hugging Face causal-LM checkpoint to evaluate.          |

Useful optional overrides (defaults in `scripts/eval_ehr.sh`):

| Variable            | Default      | Notes                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------|
| `ICD_CODE_MAP_FILE` | `./data/icd_code_map_with_category.csv` | Restrict evaluation to listed ICD codes.        |
| `ATT_TYPE`          | `month`      | Unit for `<ATT-k>` tokens (`day` / `week` / `month`).                |
| `SEQ_LEN`           | `4096`       | Model context length.                                                 |
| `OFFSET`            | `365.25`     | Prediction lead time (days) before the outcome event.                 |
| `AGE_GROUP_RANGE`   | `20,100,10`  | Age brackets as `start,end,step` years.                              |
| `AGE_INDEX`         | `3`          | Position of `<AGE_k>` in the trajectory.                              |
| `SEX_INDEX`         | `1`          | Position of `<SEX-k>` in the trajectory.                              |
| `DEMO_END_INDEX`    | `3`          | First token after the demographics prefix.                            |
| `NUM_GPUS`          | `4`          | GPUs used during Stage 3 inference.                                  |
| `AUC_WORKERS`       | `12`         | Threads for AUC computation.                                          |
| `BATCH_SIZE`        | `8`          | Batch size for Stage 2 tokenization.                                  |
| `EVAL_BATCH_SIZE`   | `8`          | Per-GPU batch size for Stage 3 inference.                             |

Stages 1 and 2 are skipped automatically if their output already exists,
so re-running the script is cheap when only the model changes.

## Reading the results

`auc_results.csv` has one row per disease:

| Column                | Meaning                                              |
|-----------------------|------------------------------------------------------|
| `disease_token`       | Vocabulary token, e.g. `<DX-MAJOR_E11>`.            |
| `auc`                 | Mean AUC across age brackets and sex.                |
| `n_samples`           | Number of `(sex, age-bracket)` cells aggregated.    |
| `n_diseased`          | Total cases used.                                    |
| `n_healthy`           | Total controls used.                                 |

`auc_results_unpooled.csv` adds `sex` and `age` columns and reports the
per-bracket AUC, useful for plotting calibration by age.

## Dependencies

- `torch`, `transformers`
- `pandas`, `numpy`, `scipy`
- `tqdm`

Install with `uv pip install torch transformers pandas numpy scipy tqdm`,
or add them to your project's `pyproject.toml` with
`uv add torch transformers pandas numpy scipy tqdm`.
