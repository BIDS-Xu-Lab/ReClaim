# Delphi Baseline (evaluation only)

Scores a trained [Delphi](https://github.com/gerstung-lab/Delphi) model on the
**same** case/control cohort, patients and prediction points used for ReClaim,
so the two methods' AUCs are directly comparable.

Delphi is described in Shmatko et al., *Learning the natural history of human
disease with generative transformers*, Nature (2025). `model.py` and the
DeLong AUC routines come from the Delphi repository and keep its MIT license (see
`LICENSE`; the license covers code only, Delphi model weights are distributed
by the original authors under CC BY-NC-ND 4.0). The remaining scripts adapt
Delphi's evaluation to ReClaim's predefined cohorts.

**Training is not included.** You need a Delphi checkpoint directory
containing `ckpt.pt` (with `model_args` and `model`), trained with the
original Delphi code on data converted the same way as in `preprocess.py`.

## Contents

| File | Purpose |
|------|---------|
| `preprocess.py` | Convert tokenized trajectories (`eval_seq`) to Delphi's binary format (`test.bin`, one `(patient, age_in_days, token)` uint32 triple per event) and a `pid_mapping.tsv`. |
| `transform_predefined_cohort.py` | Map the ReClaim cohort (`*-disease_case_control_ids.parquet`) to Delphi patient indices, Delphi disease tokens and Delphi prefix endpoints. |
| `evaluate.py` | Run the checkpoint on every unique `(patient, prefix)` pair and compute DeLong AUC per `(disease, sex, age bracket)`, pooled per disease. |
| `evaluate_auc.py` | Prefix inference and AUC computation used by `evaluate.py`. |
| `model.py`, `utils.py` | Delphi architecture and the patient-index helper. |
| `nature_disease_map.csv` | ICD-10 code → Delphi disease id (1,208 three-character ICD-10 codes). |
| `delphi_labels_chapters_colours_icd.csv` | Delphi token labels (code, name, ICD-10 chapter). |

## How the cohort is mapped

Delphi sees only diagnoses (first occurrence of each 3-character ICD-10 code,
with age in days), sex, and nothing else. For each cohort entry:

- **Disease.** The endpoint is read from the cohort's `disease_token_str`
  (`<DX-MAJOR_E11>` → `E11`) and looked up in `nature_disease_map.csv`, so the
  cohort can come from any ReClaim tokenizer. Endpoints that are not Delphi
  diseases are dropped. Cohort files without `disease_token_str` are rejected;
  rebuild them with `../prepare_disease_case_controls_fast.py`.
- **Patient.** The cohort row index points into `traj_test_pd.parquet`; its
  `person_id` is mapped to the Delphi patient index through `pid_mapping.tsv`.
- **Prediction point.** ReClaim scores `eval_seq.split(" ")[:seq_point]`. The
  same disease-extraction logic as `preprocess.py` is replayed on the raw
  sequence, and the prefix endpoint is the last Delphi record strictly before
  `seq_point`. Each patient's replayed events are checked against `test.bin`
  and mismatches are dropped and counted.

The model's next-token logit for the target disease at the end of the prefix
is the score, and AUCs are computed and pooled exactly as in
`../test_case_controls.py`.

## Running

From `downstream_tasks/disease_prediction/`, after `scripts/eval_ehr.sh` has
built `traj_test_pd.parquet` and the case/control parquet in `EVAL_DATA_DIR`:

```bash
export EVAL_DATA_DIR=/path/to/eval/workdir
export DELPHI_CKPT_DIR=/path/to/delphi/checkpoint     # contains ckpt.pt

bash scripts/eval_delphi.sh                 # all stages
STAGE=gen  bash scripts/eval_delphi.sh      # stages 1-2 only (CPU)
STAGE=eval bash scripts/eval_delphi.sh      # stage 3 only (GPU if available)
```

See the header of `scripts/eval_delphi.sh` for optional variables
(`CASE_CONTROL_IDS_FILE`, `DELPHI_WORK_DIR`, `EVAL_OUTPUT_DIR`, `PID_COL`,
`ATT_TIME_UNIT`, ...). Intermediate files and logs go to
`DELPHI_WORK_DIR` (default `$EVAL_DATA_DIR/delphi`).

## Outputs

In `EVAL_OUTPUT_DIR`:

| File | Content |
|------|---------|
| `auc_results.csv` | One row per disease: `icd_code, auc, n_samples, n_diseased, n_healthy`. Same columns as ReClaim's `auc_results.csv`, but keyed by ICD-10 code, so it can be passed directly to `visualization/disease_prediction/` (repo root). |
| `auc_results_unpooled.csv` | One row per `(disease, sex, age bracket)`: `icd_code, sex, age, auc_delong, n_diseased, n_healthy`. |
| `auc_<ckpt>_<data>_offset_<d>_predefined.csv` | Detailed table with the Delphi token, DeLong variance, disease name and ICD-10 chapter. |

`DELPHI_WORK_DIR/predefined_cohort_debug.json` records how many cases and
controls were dropped and why (e.g. endpoints outside Delphi's vocabulary).

## Dependencies

`torch`, `numpy`, `pandas`, `pyarrow`, `tqdm`, and `pyspark` with a Java
runtime for `preprocess.py` (`java` on `PATH` or `JAVA_HOME` set).
