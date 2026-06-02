# Expenditure Forecasting Baseline Models

This folder trains gradient boosting tree regression baselines for the
expenditure forecasting task, including XGBoost and LightGBM implementations.
The baselines use v6 MarketScan ReClaim trajectories and predict next annual
cost from engineered demographic, prior-cost, DX, and RX features.

The baseline setup is based on prior work on machine-learning prediction of
high-need, high-cost patients using clinical and claims data:

Osawa, I. et al. [Machine-learning-based prediction models for high-need
high-cost patients using nationwide clinical and claims data](https://www.nature.com/articles/s41746-020-00354-8).
npj Digital Medicine, 2020.

The broader task and evaluation workflow are documented in `../README.md`.

## Workflow

Run from this folder on a Slurm node:

```bash
cd downstream_tasks/expenditure_forecasting/baseline
sbatch baseline_process.sh
sbatch baseline_modelling.sh
```

Before running, update the configuration block in each shell script:

- `PROCESSED_MARKETSCAN_DATA_ROOT`: root containing v6 MarketScan ReClaim trajectory
  folders
- `COST_TASK_ROOT`: root for expenditure forecasting baseline data, models, and
  evaluation artifacts
- `CONDA_WORK_ROOT`: optional root for conda env/package caches
- `RECLAIM_DATA_DIR`: v6 MarketScan trajectory folder; defaults to
  `${PROCESSED_MARKETSCAN_DATA_ROOT}/v6/marketscan_full_combined`
- `PROCESSED_DATA_DIR`: task-specific processed data output folder. Stage 1
  writes train/validation `preprocessed_input.parquet` files here, and Stage 2
  writes baseline feature parquet files here.
- `VOCAB_PATH`: ReClaim vocabulary parquet folder used by Stage 2 to choose the
  DX major-code and RX tokens included as tabular features.
- `OUTPUT_DIR`: trained baseline model output folder. Stage 3 writes XGBoost
  and LightGBM models, feature importance files, and training metrics here.
- `TOKEN_FEATURE_TYPES`: controls how DX/RX token features are encoded;
  `binary` records presence/absence, while `count` records token frequency.
- `HISTORY_LENGTH`: controls how much historical context is used for prior-cost
  and clinical token features; `1year` uses the most recent calendar-year
  window, while `all` uses all available context after demographics.

## Stage 1: Prediction Points

`baseline_process.sh` first runs
`preprocess_find_prediction_point_for_train.py`.

This script:

- reads training trajectories from `final_data/train`
- samples 5,000,000 patients by default
- filters patients to age 16 or older using the first `<AGE-*>` token
- selects a random middle `<NY>` token as the prediction point
- keeps the selected `<NY>` token in `context_seq`
- defines `future_seq` as the tokens after the selected `<NY>`
- sums future `<COST-*>` tokens up to the next `<NY>` as the next annual cost
- splits eligible examples into 90% train and 10% validation

The split is created for baseline model training only. It does not create the
`retrospective` or `prospective` section folders used by the test/evaluation scripts.

Expected outputs:

```text
<processed_data_dir>/train/preprocessed_input.parquet
<processed_data_dir>/val/preprocessed_input.parquet
```

Each output includes:

- `enrollee_id`
- `context_seq`
- `future_seq`
- `next_annual_cost`
- `next_annual_cost_inpatient`
- `next_annual_cost_outpatient`
- `next_annual_cost_pharmacy`

## Stage 2: Baseline Features

`baseline_process.sh` then runs `preprocess_baseline_features.py` for both
`train` and `val`, and for each token feature type.

Following the reference paper, baseline features are selected from demographics,
prior expenditure, diagnosis history, and medication history.

The feature builder creates:

- `sex`
- `age`, computed from the `<AGE-*>` anchor plus later `<NY>` tokens in
  `context_seq`
- `prev_cost`, summed from prior `<COST-*>` tokens in the selected history
  window
- DX major-code features from `<DX-MAJOR_*>` vocabulary tokens
- RX features from vocabulary RX tokens

Feature settings:

- `history_length=1year`: use tokens from the second-to-last `<NY>` through the
  end of the context
- `history_length=all`: use tokens after `<DOBYR-*>` through the end of the
  context
- `token_feature_type=binary`: token presence/absence
- `token_feature_type=count`: token counts

Expected outputs:

```text
<processed_data_dir>/train/baseline_features_<history_length>_<token_feature_type>.parquet
<processed_data_dir>/val/baseline_features_<history_length>_<token_feature_type>.parquet
```

## Stage 3: Model Training

`baseline_modelling.sh` trains both model families:

- `regressor_xgboost.py`
- `regressor_lightgbm.py`

For each feature setting, both scripts train one regressor per continuous cost
target:

- `next_annual_cost`
- `next_annual_cost_inpatient`
- `next_annual_cost_outpatient`
- `next_annual_cost_pharmacy`

Both model families use fixed hyperparameters with early stopping on the
validation split.

Expected model outputs:

```text
<output_dir>/<history_length>/<token_feature_type>/xgboost/<target>/xgb_best_model.json
<output_dir>/<history_length>/<token_feature_type>/xgboost/<target>/feature_importance.csv
<output_dir>/<history_length>/<token_feature_type>/xgboost/<target>/training_metrics.csv

<output_dir>/<history_length>/<token_feature_type>/lightgbm/<target>/lgbm_best_model.txt
<output_dir>/<history_length>/<token_feature_type>/lightgbm/<target>/feature_importance.csv
<output_dir>/<history_length>/<token_feature_type>/lightgbm/<target>/training_metrics.csv
```

These trained models are later used by `../s3_cost_prediction.sh` to generate
test-set baseline predictions.
