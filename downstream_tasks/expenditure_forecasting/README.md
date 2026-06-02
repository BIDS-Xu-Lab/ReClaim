# Expenditure Forecasting

This folder contains the expenditure forecasting task built on top of v6
MarketScan ReClaim trajectories. The task asks models to predict the next
annual healthcare cost from a tokenized patient history.

The scripts assume the v6 monthly sequence format described in
`../../preprocessing/versions/v6_marketscan/README.md`.

## Task Definition

The task predicts an enrollee's healthcare expenditure in the next calendar
year from the preceding context segment. Each patient trajectory is split into a
context sequence and a future sequence. Tokens through the prediction-point
`<NY>` form `context_seq`; tokens after it form `future_seq`.

The outcome is total expenditure in the next calendar year, computed by summing
`<COST-*>` tokens in `future_seq` until the next `<NY>` token. The same
calendar-year rule is applied within visit types to create inpatient,
outpatient, and pharmacy expenditure targets.

Two evaluation settings are created:

- `retrospective`: randomly chooses a middle `<NY>` token within `seq_thru_2022`.
  The context includes the selected `<NY>` token, and the outcome is the next
  annual cost after that point.
- `prospective`: uses the 2023 boundary as the prediction point. The context
  contains history through 2022 plus the first 2023 `<NY>` token, and the
  outcome is the next annual cost after that point.

Patients are filtered to age 16 or older using the first `<AGE-*>` token in
`seq_thru_2022`.

## Workflow

Run scripts from this folder unless noted otherwise. The provided shell scripts
are designed for a Slurm environment; update each script's configuration block
before submitting.

Typical evaluation run:

```bash
cd downstream_tasks/expenditure_forecasting
sbatch s1_preprocess_test.sh
sbatch s2_generate_vllm.sh
sbatch s3_cost_prediction.sh
sbatch s4_eval_cost.sh
```

### Step 1: Prepare Test Prediction Points

`s1_preprocess_test.sh` runs:

- `preprocess_find_prediction_point.py`
- `preprocess_baseline_features.py`

It creates `preprocessed_input.parquet` for both `retrospective` and `prospective`,
then creates baseline feature parquet files needed for evaluating trained
baseline models. See `baseline/README.md` for baseline feature definitions and
model training.

Expected output layout:

```text
<processed_data_dir>/<split>/retrospective/preprocessed_input.parquet
<processed_data_dir>/<split>/retrospective/baseline_features_1year_binary.parquet
<processed_data_dir>/<split>/retrospective/baseline_features_1year_count.parquet
<processed_data_dir>/<split>/prospective/preprocessed_input.parquet
<processed_data_dir>/<split>/prospective/baseline_features_1year_binary.parquet
<processed_data_dir>/<split>/prospective/baseline_features_1year_count.parquet
```

### Step 2: Generate ReClaim Forecasts

`s2_generate_vllm.sh` runs `s2_generate_vllm.py` with VLLM as the backend for
fast, high-throughput inference.

The script reads each section's `preprocessed_input.parquet`, uses `context_seq`
as the prompt, and writes generated future sequences. Here, `<section>` matches
the script variable and is either `retrospective` or `prospective`.

Expected output:

```text
<processed_data_dir>/<split>/<section>/model_gen/<model>/generated_data_<N>.parquet
```

### Step 3: Convert Model Outputs to Cost Predictions

`s3_cost_prediction.sh` combines two prediction paths:

- XGBoost and LightGBM baselines read baseline feature parquet files, run the
  trained regressors, and write predicted expenditure columns to
  `cost_predictions.csv`.
- ReClaim generations are parsed by `cost_agg_reclaim.py`, averaged across
  generations, and written as `cost_predictions_<N>.csv`.

Expected output:

```text
<processed_data_dir>/<split>/<section>/model_gen/<model>/cost_predictions.csv
<processed_data_dir>/<split>/<section>/model_gen/<model>/cost_predictions_<N>.csv
```

### Step 4: Evaluate Cost Predictions

`s4_eval_cost.sh` runs `eval_cost_main.py`.

The evaluation reports both regression metrics for continuous cost prediction
and classification metrics after binning costs with different cutoff settings.

Expected output:

```text
<result_dir>/<split>/<section>/regression/regression_metrics.csv
<result_dir>/<split>/<section>/classification/<cutoffs>/classification_metrics.csv
<result_dir>/<split>/<section>/classification/<cutoffs>/classification_reports.txt
```

## Baseline Model Training

Baseline training lives in `baseline/`. See `baseline/README.md` for the
train/validation preprocessing, feature construction, and XGBoost/LightGBM
training workflow.
