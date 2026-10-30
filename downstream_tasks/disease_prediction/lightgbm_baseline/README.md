# LightGBM Baseline

A per-disease gradient-boosted classifier on bag-of-codes features, scored on
exactly the same case/control cohort as ReClaim (`../scripts/eval_ehr.sh`
stages 1-2), so its AUCs are directly comparable.

Entry points: `../run_lightgbm_baseline.py` (phased CLI) and
`../scripts/run_lightgbm.sh` (driver). Run from `downstream_tasks/disease_prediction/`.

## Method

For each target disease *D* (a `<DX-MAJOR_*>` token):

1. **Training cohort** (`--phase prepare`). Training trajectories are parsed
   with the same stage-1 code as the evaluation (`utils/data_ops.py`). A
   patient whose first *D* has a prediction point at least `OFFSET` days
   earlier is a case at that point; every other patient with a valid
   prediction point is a control at their first one. Rows whose prediction
   age falls outside `AGE_GROUP_RANGE` are dropped.
2. **Validation cohort.** Rows of a case/control ids file (built by
   `prepare_disease_case_controls_fast.py` on held-out data) for *D*, matched
   by `disease_token_str`. It is used for early stopping only.
3. **Features** (`--phase features`). For each row the history up to the
   prediction point is reduced to
   - counts of every `<DX-MAJOR_*>`, `<PROC-*>`, `<RX-*>` and `<ATT-*>` token
     in the feature vocabulary (scikit-learn `CountVectorizer`, one column per
     token), and
   - 7 dense features: numbers of diagnosis, procedure, medication and visit
     tokens, history length, age at the prediction point, male indicator.

   `BASELINE_FEATURE_MODE=relfreq` instead L1-normalizes the counts per
   patient; the default `raw` was used for the reported results, and models
   must be scored with the mode they were trained with.
4. **Model** (`--phase fit`). `LGBMClassifier(objective="binary",
   max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)` with
   up to 10,000 trees and early stopping after 100 rounds without validation
   AUC improvement. One model per disease, saved as
   `MODEL_DIR/per_disease/<DX-MAJOR_X>/lgb_model.txt`.
5. **Evaluation** (`--phase eval --inference_only`). The saved models score
   every case and control of the evaluation cohort at the same truncation
   points ReClaim uses; AUCs are computed per `(sex, age bracket)` with
   DeLong's method and averaged per disease, as in `../EVALUATION.md`.

## Running

```bash
# 1. Build the validation and evaluation cohorts (stages 1-2, CPU only)
SOURCE_DATA_DIR=/path/to/val/trajectories  EVAL_DATA_DIR=/path/to/val_cohort  \
MODEL_PATH=/path/to/reclaim/checkpoint STAGE=gen bash scripts/eval_ehr.sh
SOURCE_DATA_DIR=/path/to/test/trajectories EVAL_DATA_DIR=/path/to/test_cohort \
MODEL_PATH=/path/to/reclaim/checkpoint STAGE=gen bash scripts/eval_ehr.sh

# 2. Train, then score the test cohort
MODEL_DIR=/path/to/lightgbm_models \
FEATURE_VOCAB=/path/to/reclaim/checkpoint \
TRAIN_DATA_DIR=/path/to/train/trajectories \
VAL_DATA_DIR=/path/to/val_cohort \
EVAL_DATA_DIR=/path/to/test_cohort \
bash scripts/run_lightgbm.sh
```

`STAGE=train` or `STAGE=eval` runs one half; to score an external dataset
with trained models, rerun `STAGE=eval` with another `EVAL_DATA_DIR` and
`EVAL_OUTPUT_DIR`. `FEATURE_VOCAB` is any tokenizer directory (or a JSON
`{token: id}` map) whose DX-MAJOR / PROC / RX / ATT tokens define the feature
columns; keep it fixed between training and evaluation.

All phases cache their outputs under `MODEL_DIR` (`cache/`, `cohorts/`,
`features/`) and `--resume` skips completed diseases. For million-patient
training sets, raise `NUM_WORKERS`, `COHORT_WORKERS` and `FEATURE_WORKERS`
as memory allows; feature workers are separate processes and dominate memory
use. `TRAIN_SAMPLE_SIZE` subsamples training patients.

## Outputs

`EVAL_OUTPUT_DIR/auc_results.csv` and `auc_results_unpooled.csv` have the
same columns as the ReClaim outputs plus `disease_token_str`, so
`../../../visualization/disease_prediction/to_icd_code.py --input ... --output ...` converts them
without a tokenizer. `disease_stats.json` and `skipped_diseases.json` list
per-disease status (e.g. `not_in_cohort`, `single_class_train`, `model_not_found`).

`examples/synthetic/run_synthetic_lightgbm.sh` runs the whole pipeline on
synthetic data in a few minutes on CPU.
