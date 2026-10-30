# ETHOS Baseline (evaluation only)

Scores a frozen, MIMIC-pretrained [ETHOS](https://github.com/ipolharvard/ethos-paper)
model on the **same** case/control cohort, patients and prediction points used
for ReClaim, so the two methods' AUCs are directly comparable.

ETHOS, Enhanced Transformer for Health Outcome Simulation, is described in Renc
et al., *Zero shot health trajectory prediction using transformer*, npj Digital
Medicine (2024). `model.py` and `vocabulary.py` are adopted unchanged from the
ETHOS repository and keep its MIT license (see `LICENSE`; the license covers
code only, and the ETHOS checkpoint and vocabulary are distributed by the
original authors). Both files name what they took in their header. The
remaining scripts turn EHRSHOT into the timelines that model reads, and adapt
its scoring to ReClaim's predefined cohorts; they copy no ETHOS code, but
`preprocess.py` follows the ETHOS tokenizer's rules, which its header records.

**Training is not included, and nothing of ETHOS's own is redistributed here.**
Clone the [ETHOS repository](https://github.com/ipolharvard/ethos-paper) and
point three variables at it:

| Variable | What it is | Where it comes from |
|----------|------------|---------------------|
| `ETHOS_MODEL` | The checkpoint to score | The ETHOS authors |
| `ETHOS_VOCAB` | The vocabulary that checkpoint was trained with | The ETHOS authors |
| `ETHOS_REF_DIR` | The code system reference tables, `ethos/data/` of a checkout | The ETHOS repository |

The vocabulary must be the one that checkpoint was trained against, since it
decides which token strings exist. `ETHOS_REF_DIR` supplies four files, which
decide how a code is spelled out: `icd10cm-order-Jan-2021.csv.gz` for the
ICD-10-CM titles, `atc_coding.csv.gz` for the drug group names, and
`icd_cm_9_to_10_mapping.csv.gz` with `icd_pcs_9_to_10_mapping.csv.gz` for the
ICD-9 translations. Taking them from the repository rather than from a copy is
what keeps a code reaching the token the model was trained on.

ETHOS reads its own vocabulary rather than ReClaim's, so it cannot be run on
ReClaim's trajectories. It is given the EHRSHOT OMOP tables and builds its own
timelines from them. The cohort is what the two share, and that is what the
comparison rests on.

## Contents

| File | Purpose |
|------|---------|
| `preprocess.py` | Convert the EHRSHOT OMOP tables to ETHOS timelines (`timelines.hdf5`), in four steps: the RxNorm → ATC crosswalk, the cohort filter, the tokens, and the timelines. |
| `transform_predefined_cohort.py` | Map the ReClaim cohort (`*-disease_case_control_ids.parquet`) to ETHOS patients, ETHOS disease tokens and ETHOS cut points. |
| `evaluate.py` | Run the checkpoint on every unique `(patient, cut point)` pair and compute DeLong AUC per `(disease, sex, age bracket)`, pooled per disease. |
| `evaluate_auc.py` | DeLong AUC and the case/control cache helpers used by `evaluate.py`. |
| `model.py`, `vocabulary.py` | ETHOS architecture with its checkpoint loader, and the vocabulary with its special tokens and separator injector. |
| `utils.py` | Logging, timing, OMOP table paths and the two file readers. |
| `target_tokens.json` | The 1,208 `<DX-MAJOR_*>` tokens ReClaim evaluates. This is the disease selection, and it is the same file the ReClaim evaluation uses. |

## How the cohort is mapped

ETHOS sees diagnoses, procedures, medications, stay brackets and sex, each
placed by the age of the patient. Who is a case, who is a control and which
prediction point either is scored at are **not** decided here: they are read
from ReClaim's case/control parquet, so every model is asked about the same
patients with the same labels. For each cohort entry:

- **Disease.** The endpoints are the 1,208 tokens of `target_tokens.json`, cut
  to the ICD-10 category they name (`<DX-MAJOR_E11>` → `E11`) and looked up in
  the ICD-10-CM titles, because the title is how the vocabulary spells a
  category out. This vocabulary names 1,733 categories, which is more than
  ReClaim evaluates; sweeping all of them would average the AUC over a
  different population of diseases, so the selection is fixed. The run logs
  `Target diseases requested` against `target tokens to evaluate`, and warns
  separately about each reason the second is smaller: a category the vocabulary
  cannot name, and two categories sharing a title and therefore a token.
- **Patient.** The cohort row index points into `traj_test_pd.parquet`; its
  `person_id` is the OMOP `person_id` the timelines are keyed by.
- **Prediction point.** ReClaim groups a patient's events into blocks of one
  calendar month and one visit kind, and scores at the `<ATT-n>` that closes a
  block. The same month and visit kind are located in the ETHOS timeline, and
  the timeline is cut at the separator closing that block. Each patient's
  mapped sex and birth year are checked against the timeline, and strata that
  cannot be carried over are dropped and counted.

The model's next-token logit for the target disease at the end of the cut
timeline is the score, and AUCs are computed and pooled exactly as in
`../test_case_controls.py`.

## Running

From `downstream_tasks/disease_prediction/`, after `scripts/eval_ehr.sh` has
built `traj_test_pd.parquet` and the case/control parquet in `EVAL_DATA_DIR`:

```bash
export EHRSHOT_OMOP_DIR=/path/to/ehrshot/omop     # OMOP CDM parquet tables
export ETHOS_VOCAB=/path/to/vocab.pkl             # upstream ETHOS vocabulary
export ETHOS_MODEL=/path/to/checkpoint.pt         # upstream ETHOS checkpoint
export ETHOS_REF_DIR=/path/to/ethos-paper/ethos/data   # reference tables
export EVAL_DATA_DIR=/path/to/eval/workdir

bash scripts/eval_ethos.sh                 # all stages
STAGE=gen  bash scripts/eval_ethos.sh      # stages 1-2 only (CPU, stage 1 needs Java)
STAGE=eval bash scripts/eval_ethos.sh      # stage 3 only (GPU if available)
```

See the header of `scripts/eval_ethos.sh` for optional variables
(`CASE_CONTROL_IDS_FILE`, `ETHOS_WORK_DIR`, `EVAL_OUTPUT_DIR`, `TARGET_TOKENS`,
`BATCH_SIZE`, ...). Timelines, the prepared cohort and logs go to
`ETHOS_WORK_DIR` (default `$EVAL_DATA_DIR/ethos`).

EHRSHOT requires its own data-use agreement.

**OMOP tables used:**

| Table | Used for |
|-------|----------|
| `person` | Sex, birth year, age filtering |
| `visit_occurrence` | Visit type, dates, stay brackets, visit span filtering |
| `payer_plan_period` | Dating the first event when a patient was insured before their first visit. Never tokenized, and the stage runs without the table |
| `condition_occurrence` | Diagnosis tokens |
| `procedure_occurrence` | Procedure tokens |
| `drug_exposure` | Medication tokens |
| `concept`, `concept_relationship` | Resolving a record to the code written down, and the RxNorm → ATC crosswalk |

## Outputs

`ETHOS_WORK_DIR` holds person-level files and so points outside the repository.
`person_context/`, `excluded_personids/` and `all_claim_tokens/` are Spark
intermediates, `timelines.hdf5` holds one timeline per patient, and
`predefined_cohort.json` names the patients of every stratum. Only
`ehrshot_rxnorm_to_atc.csv.gz`, the code crosswalk, carries no patient.

In `EVAL_OUTPUT_DIR`:

| File | Content |
|------|---------|
| `auc_results.csv` | One row per target disease, pooled over its strata, with `icd_codes` naming the ICD-10 category. Pass it through `visualization/disease_prediction/to_icd_code.py` to get the column layout of [`../results/ehrshot/`](../results/ehrshot/). |
| `auc_results_unpooled.csv` | One row per `(target disease, sex, age bracket)`, with the DeLong variance. |
| `onset_summary.json` | Patients, strata, prediction points, the mean and median AUC over the target diseases, and how many ReClaim strata could not be carried over. |

Both tables and the summary cover the target diseases and nothing else:
`evaluate.py` refuses to run if a stratum names a token outside
`target_tokens.json`. The mean is reported with its denominator,
`n_target_diseases_with_auc`, which falls below 1,208 for the two reasons given
above. Under `mapping`, the summary also records how many cases and controls
were dropped on the way from ReClaim's strata to the timelines, and why.

## Dependencies

`torch`, `numpy`, `pandas`, `pyarrow`, `h5py`, `tqdm`, and `pyspark` with a
Java runtime for `preprocess.py` (`java` on `PATH` or `JAVA_HOME` set). The two
evaluation scripts need no Spark.
