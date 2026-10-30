# Disease-Onset Evaluation Protocol

This document specifies how `scripts/eval_ehr.sh` turns tokenized EHR
trajectories into per-disease AUCs, and walks through every intermediate
artefact on a fully synthetic cohort (`examples/synthetic/`). No real patient
data appears in this document.

- [1. Task](#1-task)
- [2. Stage 1: disease points](#2-stage-1-disease-points)
- [3. Stage 2: prediction points, cases and controls](#3-stage-2-prediction-points-cases-and-controls)
- [4. Stage 3: scoring and AUC](#4-stage-3-scoring-and-auc)
- [5. Worked synthetic example](#worked-synthetic-example)
- [6. Pitfalls](#6-pitfalls)

## 1. Task

For a disease *D* (a `<DX-MAJOR_*>` ICD-10 3-character code), the model sees
a patient's history up to a *prediction point* and must rank patients who
will receive a first diagnosis of *D* at least `OFFSET` days (default one
year) later above patients who never receive *D*. Ranking is done within
strata of sex and age at the prediction point (10-year brackets from 20 to
100 by default), so that AUC is not inflated by the model simply learning
that older patients get more disease. Stratum AUCs are averaged into one AUC
per disease.

## 2. Stage 1: disease points

`prepare_eval_data_new.py` → `traj_test_pd.parquet`

For each patient the parser scans tokens from `DEMO_END_INDEX` to the end of
the trajectory (`total_token_count`, because the driver passes
`--longitudinal`) and maintains a clock:

- the starting age is `<AGE-a>` at `AGE_INDEX`, i.e. `a × 365.25` days;
- `<NY>` adds one year and resets the within-year offset;
- `<ATT-m>` adds `m × {1, 7, 30}` days (for `ATT_TYPE` `day` / `week` /
  `month`) to the within-year offset and opens a new visit.

Every **first** occurrence of a `<DX-MAJOR_*>` token becomes a *disease
point*. Three aligned arrays are stored per patient:

| Column | Meaning |
|--------|---------|
| `disease_points` | Token index of the first occurrence. |
| `disease_points_ages` | Age in days at that token. |
| `disease_points_visit_location` | Index of the `<ATT-*>` that opens the *next* visit (or the last token). Truncating the sequence at this index keeps the whole visit containing the disease point and nothing after it. |

`eval_seq` holds the (truncated) token sequence used by stage 3. Patients with
fewer than two disease points are dropped because they cannot supply both a
prediction point and a later event.

## 3. Stage 2: prediction points, cases and controls

`prepare_disease_case_controls_fast.py` →
`{OFFSET}-{AGE_GROUP_RANGE}-disease_case_control_ids.parquet`

Let patient *p* have disease points *0 … M−1* with ages *t₀ ≤ t₁ ≤ …*.

**Prediction point for point *j* (*j* ≥ 1).**
`pred(j)` is the latest disease point *i* with `tᵢ < tⱼ − OFFSET`; if none
exists, point *j* has no valid prediction point. The model will be shown the
sequence up to the end of the visit containing point `pred(j)`, so the
history always stops at least `OFFSET` days before the event at *j*.

**Age bracket.** Point *j* belongs to bracket `[A, A+step)` if the age at its
prediction point, `t_pred(j) / 365.25`, falls in that bracket. Sex is read
from `SEX_INDEX` (`<SEX-1>` = male, otherwise female).

For each disease *D* and each `(sex, bracket)` cell:

- **Case.** Patient *p* is a case if their first occurrence of *D* is point
  *j* ≥ 1 and `pred(j)` exists and lies in the bracket. A patient whose first
  disease point is *D* itself (*j* = 0) has no history and is not a case.
- **Control.** Patient *p* is a control if *D* never appears anywhere in the
  trajectory (including point 0, and including the follow-up after the
  history cutoff) and some point *j* ≥ 1 has a prediction point in the
  bracket. Of the eligible points the first one is used. Requiring a later
  point *j* means every control has at least `OFFSET` days of observed
  follow-up after their prediction point.
- Each patient contributes at most one prediction point per cell, and a cell
  is kept only if it has at least one case and one control.

Each output row is one `(disease, sex, bracket)` cell:

| Column | Meaning |
|--------|---------|
| `disease_token`, `disease_token_str` | Target disease as integer id (of the stage-2 tokenizer) and as string. |
| `sex`, `age`, `age_step` | Cell definition (`age` is the bracket start, in years). |
| `case_ids`, `control_ids` | Row indices into `traj_test_pd.parquet`. |
| `case_seq_points`, `control_seq_points` | Truncation index into `eval_seq` for each patient. |

The O(n·m²) prediction-point search is disease-independent, so it is
computed once per sex and shared by all diseases.

## 4. Stage 3: scoring and AUC

`test_case_controls.py` → `auc_results.csv`, `auc_results_unpooled.csv`

1. `disease_token` is re-resolved from `disease_token_str` under the evaluated
   model's tokenizer, and a sample of ids is checked to decode to disease
   tokens (see `README.md`).
2. For each unique `(row, seq_point)` the prompt is

   ```
   eval_seq.split(" ")[:seq_point] + " <NY> <INSTRUCT-DX>"
   ```

   which matches the post-training format
   (`<history> <INSTRUCT-DX> <future new diseases> <eos>`, see
   `downstream_tasks/disease_prediction/post-training/process_posttrain_data/`).
3. The score of a patient for disease *D* is the model's raw next-token
   logit of `<DX-MAJOR_D>` at the last prompt position. Only the logits of
   the evaluated disease tokens are kept, and each prompt is run once no
   matter how many cells it appears in.
4. For each cell, AUC is computed with DeLong's method with cases as the
   positive class (`auc_results_unpooled.csv`, column `auc_delong`).
5. The disease-level `auc` in `auc_results.csv` is the unweighted mean over
   its cells; `n_samples` is the number of cells and `n_diseased` /
   `n_healthy` are summed over cells.

<a id="worked-synthetic-example"></a>
## 5. Worked synthetic example

### 5.1 Running it

```bash
cd downstream_tasks/disease_prediction
bash examples/synthetic/run_synthetic_example.sh ./outputs/synthetic_example
```

`make_synthetic_example.py` simulates patients over 8 calendar years (6 of
history, 2 of follow-up) from an annual-hazard model with age effects and
planted risk factors, e.g.

| Disease | Main planted risk factors |
|---------|---------------------------|
| `E11` type 2 diabetes | `R73` hyperglycaemia/prediabetes (strong), `E66` obesity, `E78` hyperlipidaemia |
| `I25` chronic ischaemic heart disease | `I10` hypertension, `E78`, `E11`, age |
| `N18` chronic kidney disease | `E11`, `I10`, age |
| `J06`, `M54`, `R07`, `Z00` | none (acute filler codes on random visits) |

It writes 3,000 evaluation trajectories, then trains a 2-layer Qwen3 model
(hidden size 64, 294-token word-level vocabulary) for 400 steps on 6,000
separately simulated patients in the post-training format. The driver then
runs stages 1–3 on CPU with `ICD_CODE_MAP_FILE=""` (all 14 majors) and
`SEQ_LEN=1024`. The whole run takes about 4 minutes on 4 CPU cores. Numbers
below are from seed 0 and may differ slightly across library versions.

### 5.2 One patient through stages 1–2

Row 2326 of `traj_test_pd.parquet` (female, age 67 at the start), with token
indices:

```
 0 <sos>  1 <SEX-2>  2 <DOBYR-1948>  3 <AGE-67>
 4 <ATT-0>  5 <VT-outpatient>  6 <DX-PRINCIPAL>  7 <DX-MAJOR_J06>  8 <DX-MINOR_4>  9 <RX-007> 10 <COST-HIGH>
11 <ATT-8> 12 <VT-outpatient> 13 <DX-PRINCIPAL> 14 <DX-MAJOR_R07> 15 <DX-MINOR_7> 16 <COST-MID>
17 <NY>
18 <ATT-5> ... 21 <DX-MAJOR_R07> ...
24 <ATT-5> ... 27 <DX-MAJOR_M54> ...
30 <NY>
31 <ATT-1> ... 34 <DX-MAJOR_Z00> ...
38 <ATT-3> ... 41 <DX-MAJOR_M54> ...  44 <ATT-3> ...  50 <ATT-4> ...
56 <NY>
57 <ATT-10> ... 60 <DX-MAJOR_J06> ...
63 <ATT-1> 64 <VT-outpatient> 65 <DX-PRINCIPAL> 66 <DX-MAJOR_R73> 67 <DX-MINOR_1> 68 <RX-008> 69 <COST-LOW>
70 <NY>
71 <ATT-0> ... 74 <DX-MAJOR_R73> ...
77 <NY>
78 <ATT-5> ... 81 <DX-MAJOR_R73> ...
84 <ATT-6> 85 <VT-outpatient> 86 <DX-PRINCIPAL> 87 <DX-MAJOR_E11> 88 <DX-MINOR_8> 89 <DX-SECONDARY> 90 <DX-MAJOR_R73> ...
95 <NY> ... 103 <NY> ... 110 <eos>
```

Stage 1 output for this row:

| point *j* | token | `disease_points` | age (years) | `disease_points_visit_location` |
|---|---|---|---|---|
| 0 | `<DX-MAJOR_J06>` | 7  | 67.00 | 11 |
| 1 | `<DX-MAJOR_R07>` | 14 | 67.66 | 18 |
| 2 | `<DX-MAJOR_M54>` | 27 | 68.82 | 31 |
| 3 | `<DX-MAJOR_Z00>` | 34 | 69.08 | 38 |
| 4 | `<DX-MAJOR_R73>` | 66 | 70.90 | 71 |
| 5 | `<DX-MAJOR_E11>` | 87 | 72.90 | 96 |

Age of point 1: 67 years + `<ATT-8>` × 30 days = 67.66. Repeated codes
(`R07` at 21, `R73` at 74/81/90) are not disease points.

**As a case for `E11`.** The first `E11` is point 5 at 72.90 years. The
latest point before 72.90 − 1 = 71.90 years is point 4 (`R73`, 70.90), so the
prediction age is 70.90 → cell `(female, 70)`, and `case_seq_points = 71`.
The prompt given to the model is tokens 0–70 (ending with the `R73` visit and
the following `<NY>`) plus `<NY> <INSTRUCT-DX>`; the `E11` diagnosis two
years later is not visible.

**As a control for `I10`.** She never has `I10`. For bracket 60: point 1
(67.66) has no earlier point more than a year before it; point 2 (68.82) has
`pred = 1` at 67.66 years → eligible, `control_seq_points = 18`. For bracket
70 the first eligible point is 5 with `pred = 4` at 70.90 → sequence point 71.
She therefore appears as a control in both `(female, 60)` and
`(female, 70)` for `I10`, at different prediction points.

She is also a case for `R73` in `(female, 60)` (event at 70.90, prediction
point 3 at 69.08 → sequence point 38). She is never a control for `R73` or
`E11`, since those codes appear in her trajectory.

### 5.3 One cell through stage 3

The `(E11, female, 70)` row of the case/control file:

| field | value |
|---|---|
| `disease_token_str` | `<DX-MAJOR_E11>` |
| `case_ids` / `control_ids` | 71 / 275 patients |
| mean `E11` logit, cases vs controls | 3.429 vs 3.106 |
| row 2326's logit at sequence point 71 | 3.748 (above 85% of the cell's controls) |
| `auc_delong` (± DeLong s.e.) | 0.673 ± 0.036 |

`E11` has 14 such cells (2 sexes × 7 brackets from 20 to 80); its pooled AUC
is their unweighted mean.

### 5.4 Disease-level results

`auc_results.csv`, with ids mapped back to codes:

| disease | `auc` | cells | cases | controls |
|---|---|---|---|---|
| `I25` | 0.784 | 14 | 771 | 3118 |
| `N18` | 0.709 | 12 | 470 | 3385 |
| `E11` | 0.698 | 14 | 537 | 3503 |
| `K21` | 0.672 | 14 | 533 | 3458 |
| `I10` | 0.670 | 14 | 1080 | 2361 |
| `E78` | 0.662 | 14 | 979 | 2574 |
| `R73` | 0.661 | 14 | 520 | 3466 |
| `F32` | 0.657 | 14 | 265 | 3935 |
| `E66` | 0.629 | 14 | 367 | 3741 |
| `J45` | 0.622 | 14 | 178 | 4114 |
| `M54` | 0.607 | 14 | 931 | 59 |
| `R07` | 0.601 | 11 | 795 | 39 |
| `J06` | 0.587 | 11 | 819 | 45 |
| `Z00` | 0.581 | 13 | 937 | 57 |

Mean over the 14 diseases: 0.653.

Things to notice:

- Diseases with the strongest planted risk factors (`I25`, `N18`, `E11`)
  rank highest; the acute filler codes rank lowest.
- `F32`/`J45` have no planted risk factor yet score above 0.5. A case's
  prediction point is the last eligible point before the event, while a
  control's is the first eligible point in the bracket, so case prompts
  contain more history: for `F32` the prediction point lies on average
  2.20 calendar years into the trajectory for cases vs 1.76 for controls
  (`E11`: 2.75 vs 1.73). Any score that varies with history length can
  exploit this. It is a property of the protocol, so the same effect exists
  on real data.
- Almost everyone eventually receives the filler codes, so they have very
  few controls; their AUCs rest on a handful of controls per cell. Inspect
  `n_healthy` and the unpooled file before reading much into such diseases.
- Cells with very few cases (e.g. `E11` in males aged 20–29 has 2 cases)
  enter the unweighted mean with the same weight as large cells.

As a baseline, `make_synthetic_example.py --train_steps 0` keeps the model at
random initialisation. Scored on the same cohort (reuse `EVAL_DATA_DIR`, set a
new `EVAL_OUTPUT_DIR`, `STAGE=eval`), it gives a mean AUC of 0.513, with
individual diseases between 0.38 and 0.57. This spread likely reflects
small cells and the history-length difference above; the prompts themselves
always end at least `OFFSET` days before a case's event.

## 6. Pitfalls

- **Tokenizer id spaces.** Integer token ids are only meaningful under the
  tokenizer that produced them. Keep `disease_token_str` in cohort files, and
  if stage 3 aborts on the decode sanity check, rebuild the cohort or check
  that `MODEL_PATH` is the intended checkpoint.
- **Subsampling.** `case_ids` / `control_ids` are row indices into the stage-1
  file. Subsample in stage 1 (`--sample_size`, `--patient_ids_file`) and
  rebuild stage 2; `test_case_controls.py --sample_frac` keeps only the first
  rows of the data file and is only valid with a cohort built on that same
  subset.
- **Word-level tokenization.** Stage 2 reads disease ids at token positions
  computed on whitespace tokens, so the tokenizer must map one whitespace
  token to exactly one id and add no special tokens.
- **Stale intermediates.** Stages 1–2 are skipped when their outputs exist.
  Delete them (or use a new `EVAL_DATA_DIR`) after changing `OFFSET`,
  `AGE_GROUP_RANGE`, the ICD filter or the source data. Stage 3 writes to
  `EVAL_OUTPUT_DIR`; set a distinct one per checkpoint.
