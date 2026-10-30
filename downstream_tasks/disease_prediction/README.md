# Disease Prediction (EHR)

Stratified AUC evaluation of a ReClaim-style causal language model on
tokenized electronic-health-record (EHR) trajectories.

For every major-disease ICD token (`<DX-MAJOR_*>`) in the model vocabulary,
the pipeline builds a sex- and age-stratified case/control cohort, scores
each patient with the model's next-token logit for that disease, computes a
DeLong AUC per stratum, and averages the strata into one AUC per disease.

- [`EVALUATION.md`](EVALUATION.md) — step-by-step description of the
  evaluation protocol (disease points, prediction points, case/control
  rules, scoring, pooling) with a worked synthetic example.
- [`examples/synthetic/`](examples/synthetic/) — a fully synthetic cohort
  and tiny model that exercise the whole pipeline on CPU in a few minutes.
- [`visualization/disease_prediction/`](../../visualization/disease_prediction/) (repo root) — scripts for the disease-onset figures
  of the paper; per-disease EHRSHOT results for ReClaim and baselines are in
  [`results/ehrshot/`](results/ehrshot/). Rendered figures and summaries are in
  [`output/`](output/README.md).
- Baselines scored on the same cohort: [`lightgbm_baseline/`](lightgbm_baseline/)
  (training and evaluation), [`delphi_baseline/`](delphi_baseline/)
  (evaluation of a trained Delphi checkpoint) and
  [`ethos_baseline/`](ethos_baseline/) (evaluation of the upstream ETHOS
  checkpoint on EHRSHOT). See [Baselines](#baselines).

## Layout

```
disease_prediction/
├── prepare_eval_data_new.py               # Stage 1 - disease points / ages / visit locations
├── prepare_disease_case_controls_fast.py  # Stage 2 - case/control IDs per (disease, sex, age)
├── test_case_controls.py                  # Stage 3 - model inference + AUC
├── run_lightgbm_baseline.py               # LightGBM baseline CLI (prepare / features / fit / eval)
├── scripts/
│   ├── eval_ehr.sh                        # End-to-end driver (env-var configured)
│   ├── run_lightgbm.sh                    # LightGBM baseline: train + evaluate
│   ├── eval_delphi.sh                     # Delphi baseline: evaluate a checkpoint
│   └── eval_ethos.sh                      # ETHOS baseline: build timelines + evaluate
├── lightgbm_baseline/                     # LightGBM bag-of-codes baseline package
├── delphi_baseline/                       # Delphi evaluation on the shared cohort
├── ethos_baseline/                        # ETHOS evaluation on the shared cohort
├── utils/
│   ├── data_ops.py                        # Sequence parsing and case/control matching
│   └── evaluate_auc.py                    # DeLong AUC and variance
├── data/
│   └── icd_code_map_with_category.csv     # ICD-10 majors evaluated (optional filter)
├── results/ehrshot/                       # Released per-disease AUC tables for five models
├── output/figures/ehrshot/                 # Rendered figures and paired-comparison summary
├── examples/synthetic/                    # Synthetic data + tiny model, CPU demo
├── EVALUATION.md                          # Protocol details and worked example
└── README.md
```

All commands below assume the working directory is
`downstream_tasks/disease_prediction/`.

## Input data contract

A directory of trajectory parquet shards (searched recursively), one row per
patient, as produced by the EHR preprocessing under `preprocessing/`:

| Column                  | Type    | Description                                                          |
|-------------------------|---------|----------------------------------------------------------------------|
| `person_id`             | `int` / `str` | Patient identifier.                                            |
| `seq`                   | `str`   | Whitespace-separated token trajectory.                               |
| `token_count_thru_2022` | `int`   | Index in `seq` where the history (pre-cutoff) region ends.           |
| `total_token_count`     | `int`   | Index in `seq` where the full trajectory ends.                       |

Token grammar expected by the parser (positions are the defaults):

```
<sos> <SEX-k> <DOBYR-yyyy> <AGE-a> <ATT-m> <VT-...> <DX-PRINCIPAL> <DX-MAJOR_E11> <DX-MINOR_9> ... <NY> <ATT-m> ...
  0      1         2          3     └──────────────── one visit ────────────────────────────┘
```

- `<SEX-1>` / `<SEX-2>` at `SEX_INDEX` (`<SEX-1>` is treated as male).
- `<AGE-a>` at `AGE_INDEX`: age in years at the start of the trajectory.
- `<ATT-m>` opens every visit; `m` is the time since the previous visit within
  the current calendar year, in units of `ATT_TYPE` (`day` / `week` /
  `month`, 1 / 7 / 30 days).
- `<NY>` marks a new calendar year and resets the within-year clock.
- `<DX-MAJOR_*>` are the prediction targets (ICD-10 3-character codes).

The tokenizer must be word-level (one whitespace token = one id, no special
tokens added on encode), with `<unk>` at id 0 and `<eos>` at id 2, which the
pipeline uses as the pad token. This is how ReClaim tokenizers are built
under `pre-training/`.

Disease instruction-data preparation and post-training live in
[`post-training/`](post-training/README.md). The shared trainer and configuration
remain in `pre-training/qwen_training/`.

## Pipeline stages

| Stage | Script | Output | Needs |
|-------|--------|--------|-------|
| 1 | `prepare_eval_data_new.py` | `traj_test_pd.parquet` | CPU |
| 2 | `prepare_disease_case_controls_fast.py` | `{OFFSET}-{AGE_GROUP_RANGE}-disease_case_control_ids.parquet` | CPU, tokenizer only |
| 3 | `test_case_controls.py` | `auc_results.csv`, `auc_results_unpooled.csv`, `logits_cache.pkl` | model, GPU(s) recommended |

1. **Stage 1** walks each trajectory and records, for every first occurrence
   of a `<DX-MAJOR_*>` token, its token position (`disease_points`), the
   patient's age in days at that point (`disease_points_ages`), and the
   position where that visit ends (`disease_points_visit_location`). Patients
   with fewer than two disease points are dropped.
2. **Stage 2** turns those arrays into, for each `(disease, sex, age bracket)`
   cell, a list of case rows, control rows, and the token position at which
   each should be scored. It stores both the integer `disease_token` and its
   string form `disease_token_str`.
3. **Stage 3** truncates each patient at their scoring position, appends
   `<NY> <INSTRUCT-DX>`, and reads the model's logit for the target disease
   at the last position. Each unique `(patient, position)` is run once and
   its disease-token logits are cached. AUCs are computed per cell with
   DeLong's method and averaged per disease.

The exact rules for disease points, prediction points, cases and controls are
in [`EVALUATION.md`](EVALUATION.md).

### Robustness to tokenizer mismatch

Stage 2 IDs are integer token ids of the tokenizer used to build them.
Stage 3 re-resolves every `disease_token` through the *evaluated* model's
tokenizer using `disease_token_str`, then checks that a sample of ids decodes
to disease-like tokens and aborts otherwise. One cohort file can therefore be
reused across checkpoints whose vocabularies are ordered differently; an old
cohort file without `disease_token_str` is used as-is (and must match the
model's vocabulary).

## Running the pipeline

The driver is configured entirely through environment variables:

```bash
export SOURCE_DATA_DIR=/path/to/ehr/trajectories
export EVAL_DATA_DIR=/path/to/eval/workdir
export MODEL_PATH=/path/to/hf/checkpoint

bash scripts/eval_ehr.sh
```

Stages 1 and 2 are skipped when their outputs already exist, so evaluating
another checkpoint on the same cohort only re-runs stage 3. To split the
CPU-heavy and GPU stages across jobs:

```bash
STAGE=gen  bash scripts/eval_ehr.sh    # stages 1-2 on a CPU / large-memory node
STAGE=eval bash scripts/eval_ehr.sh    # stage 3 on a GPU node
```

Required:

| Variable          | Meaning                                                 |
|-------------------|---------------------------------------------------------|
| `SOURCE_DATA_DIR` | Directory containing trajectory parquet shards.         |
| `EVAL_DATA_DIR`   | Output / working directory for intermediate files.      |
| `MODEL_PATH`      | Hugging Face causal-LM checkpoint to evaluate.          |

Optional (defaults in `scripts/eval_ehr.sh`):

| Variable             | Default       | Notes                                                             |
|----------------------|---------------|-------------------------------------------------------------------|
| `STAGE`              | `all`         | `all`, `gen` (stages 1-2) or `eval` (stage 3).                    |
| `PYTHON`             | `python`      | Interpreter used for all stages.                                  |
| `CC_PREP_MODEL_PATH` | `$MODEL_PATH` | Checkpoint whose tokenizer builds the stage-2 cohort.             |
| `EVAL_OUTPUT_DIR`    | `$EVAL_DATA_DIR/auc_case_control_eval_{OFFSET}-{AGE_GROUP_RANGE}` | Stage-3 output directory; set one per model. |
| `ICD_CODE_MAP_FILE`  | `./data/icd_code_map_with_category.csv` | Restrict to listed ICD codes; empty = all `<DX-MAJOR_*>` tokens. |
| `ATT_TYPE`           | `month`       | Unit of `<ATT-k>` tokens.                                         |
| `SEQ_LEN`            | `4096`        | Model context length (stage 2 tokenization and stage 3 truncation). |
| `OFFSET`             | `365.25`      | Minimum lead time (days) between prediction point and onset.      |
| `AGE_GROUP_RANGE`    | `20,100,10`   | Age brackets as `start,end,step` years.                           |
| `AGE_INDEX`          | `3`           | Position of `<AGE-a>`.                                            |
| `SEX_INDEX`          | `1`           | Position of `<SEX-k>`.                                            |
| `DEMO_END_INDEX`     | `3`           | First position scanned for disease tokens.                        |
| `NUM_THREADS`        | `48`          | Stage-2 threads (one disease per task).                           |
| `NUM_GPUS`           | `4`           | Stage-3 GPUs; one process per GPU. CPU is used if none is visible. |
| `EVAL_BATCH_SIZE`    | `16`          | Stage-3 per-GPU batch size.                                       |
| `AUC_WORKERS`        | `12`          | Stage-3 threads for AUC computation.                              |

`prepare_eval_data_new.py` additionally accepts `--sample_size` and
`--patient_ids_file` / `--id_column` to evaluate a subset, and
`prepare_disease_case_controls_fast.py` accepts `--padding_length` (maximum
disease points kept per patient, default 600) to bound stage-2 memory.

## Outputs

`auc_results.csv` — one row per disease:

| Column          | Meaning                                                   |
|-----------------|-----------------------------------------------------------|
| `disease_token` | Token id under the evaluated model's tokenizer.           |
| `auc`           | Mean AUC across the `(sex, age bracket)` cells.           |
| `n_samples`     | Number of cells averaged.                                 |
| `n_diseased`    | Total cases across cells.                                 |
| `n_healthy`     | Total controls across cells.                              |

`auc_results_unpooled.csv` — one row per `(disease, sex, age bracket)` with
`auc_delong`, `n_diseased`, `n_healthy`.

`logits_cache.pkl` / `logits.pkl` — cached logits; rerun stage 3 with
`--skip_logits` to recompute AUCs without inference.

Map ids back to codes with the model tokenizer, e.g.
`tokenizer.convert_ids_to_tokens(df.disease_token.tolist())`.

## Quick start on synthetic data

```bash
bash examples/synthetic/run_synthetic_example.sh ./outputs/synthetic_example
```

This simulates 3,000 EHR trajectories with planted risk factors, trains a
tiny Qwen3 model on a separate synthetic cohort, and runs all three stages on
CPU. See [`EVALUATION.md`](EVALUATION.md#worked-synthetic-example) for a walk
through of the intermediate files.

## Baselines

Every baseline reuses stages 1-2: they are scored on the same case/control ids
file and score every case and control at the same truncation point, so
per-disease AUCs and case/control counts line up with ReClaim's exactly. The
LightGBM and Delphi baselines read `traj_test_pd.parquet` as their input;
ETHOS reads it only to find each patient and truncation point in timelines of
its own, because its vocabulary is not ReClaim's.

| Baseline | Driver | What is included |
|----------|--------|------------------|
| LightGBM on bag-of-codes features | `scripts/run_lightgbm.sh` | Training (per-disease models on a training split) and evaluation. See [`lightgbm_baseline/README.md`](lightgbm_baseline/README.md). |
| Delphi | `scripts/eval_delphi.sh` | Evaluation of an existing Delphi checkpoint; training uses the original Delphi code. See [`delphi_baseline/README.md`](delphi_baseline/README.md). |
| ETHOS | `scripts/eval_ethos.sh` | Evaluation of the upstream MIMIC-pretrained ETHOS checkpoint; training is the upstream project's. ETHOS builds its timelines from the EHRSHOT OMOP tables, since its vocabulary is not ReClaim's, and is scored on the same cohort and the same disease list. See [`ethos_baseline/README.md`](ethos_baseline/README.md). |

`examples/synthetic/run_synthetic_lightgbm.sh` trains and evaluates the
LightGBM baseline on the synthetic cohort after `run_synthetic_example.sh`.

## Dependencies

`torch`, `transformers` (>= 4.51 for Qwen3), `tokenizers`, `pandas`,
`pyarrow`, `numpy`, `joblib`, `tqdm`; additionally `lightgbm`, `scikit-learn`
and `scipy` for the LightGBM baseline, `pyspark` with a Java runtime for the
Delphi baseline, and the same plus `h5py` for the ETHOS baseline.

```bash
uv pip install torch transformers tokenizers pandas pyarrow numpy joblib tqdm
```
