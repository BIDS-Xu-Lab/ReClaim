# ReClaim Data Preprocessing (v6 EHRSHOT)

This folder preprocesses the original EHRSHOT OMOP CDM data into ReClaim-compatible monthly token sequences for external validation.

EHRSHOT is a public benchmark for evaluating EHR foundation models using longitudinal structured EHR data from Stanford Medicine. We use EHRSHOT as an external EHR validation set; this pipeline starts from the original EHRSHOT OMOP-formatted tables and converts them into the same monthly ReClaim token format used for MarketScan disease-onset evaluation. See the [EHRSHOT website](https://som-shahlab.github.io/ehrshot-website/) for the dataset and benchmark description.

## Shared Behavior

This pipeline intentionally follows the main [v6 MarketScan README](../v6_marketscan/README.md) for:

- v6 monthly timestamping and `<ATT-N>` temporal tokens
- token format, category ordering, anchor tokens, and sequence construction
- ICD-9-CM to ICD-10-CM diagnosis mapping
- procedure and drug combination-token logic
- 2022/2023 split fields used for prospective evaluation
- vocabulary finalization and sanity-check principles

This README only documents what is different for EHRSHOT.

## Pipeline Stages

| Stage | Script | EHRSHOT-specific role |
|-------|--------|-----------------------|
| S0 | `s0_data_filtering.py` | Filter EHRSHOT patients and collapse OMOP payer periods |
| S1 | `s1_construct_tokens.py` | Convert OMOP records into ReClaim tokens |
| S2 | `s2_construct_seqences.py` | Build monthly patient trajectories using the shared v6 sequence logic |
| S3 | `s3_finalize_vocab.py` | Add EHRSHOT-observed tokens into a ReClaim-compatible vocabulary |
| S5 | `s5_sanity_check.py` | Run trajectory sanity checks |

There is no S4 train/validation/test split because EHRSHOT is an external validation cohort, not a pretraining corpus.

## EHRSHOT-Specific Processing

| Area | Handling |
|------|----------|
| Source data | Original EHRSHOT OMOP CDM parquet tables under `OMOP_DIR` |
| Patient ID | Uses OMOP `person_id` rather than MarketScan `enrollee_id` |
| Date window | Keeps records from `2008-01-01` through `2025-03-31` |
| Filtering | Excludes patients with less than 30 days between first and last visit, or age at first event outside 10-110 |
| Enrollment | Uses `payer_plan_period`; drops enrollment periods under 30 days; merges overlapping or adjacent periods within 1 month per `person_id` and `PAYER` |
| Payer fields | Maps OMOP `payer_concept_id` to ReClaim/MarketScan-like `PAYER`, `PLANTYP`, and `CAP` using the domain-expert annotated `payer_concept_annotated.csv` |
| Location | Assumes `EGEOLOC-62` (California) as a proxy for Stanford Medicine, which EHRSHOT data are based on |
| Visit type | Uses OMOP `visit_concept_id`: inpatient `262`, `9201`; outpatient `9202`, `581477`, `9203`; pharmacy `9202`, `581477`, `9203`, `581458` with a valid drug source concept |
| Diagnosis | Uses `condition_occurrence` joined to `concept`; keeps ICD-10-CM and maps ICD-9-CM using `icd9_to_icd10_mapping.csv` |
| Procedure | Uses `procedure_occurrence` joined to `concept`; maps CPT4, ICD10PCS, and ICD9CM to SNOMED using `proc_to_snomed_mapping.csv`; all procedures are marked principal because EHRSHOT lacks a procedure principal indicator |
| Drug | Uses `drug_exposure` joined to `concept`; maps RxNorm drug concepts to RxNorm ingredients using `rxnorm_drug_to_rxnorm_ingredient_mapping.csv` |
| Cost | Always emits `<COST-MISSING>` because EHRSHOT does not provide claim cost |
| Discharge status | Always emits `<DS-MISSING>` because discharge status is not available in the source fields used here |
| Length of stay | Derived from inpatient `visit_start_DATE` and `visit_end_DATE`: `LS-0` for stays under 7 days, `LS-1` for 7 days or longer |
| Vocabulary | Includes observed EHRSHOT tokens plus the structured token ranges needed for compatibility with the MarketScan-trained ReClaim model |

## Expected Inputs

Prepare these local inputs before running:

- `EHRSHOT_DATA_ROOT`: root containing the original EHRSHOT data; `OMOP_DIR` is `${EHRSHOT_DATA_ROOT}/ehrshot_omop`
- `PROCESSED_EHRSHOT_DATA_ROOT`: root for generated outputs; `EXTERNAL_VAL_DATA_DIR` is `${PROCESSED_EHRSHOT_DATA_ROOT}/v6/ehrshot`
- `VOCAB_MAPPING_DIR`: repo-relative by default (`../../vocab_mapping/mapping`)
- `MARKETSCAN_DICTIONARY_DIR`: local MarketScan dictionary folder used for compatible `PLANTYP` and `EGEOLOC` vocabularies

**OMOP tables under `OMOP_DIR`:**

| Table | Used for |
|-------|----------|
| `person` | Demographics and age filtering |
| `payer_plan_period` | Enrollment start/end and payer/plan fields |
| `visit_occurrence` | Visit type, timestamps, length of stay, and visit-duration filtering |
| `condition_occurrence` | Diagnosis tokens |
| `procedure_occurrence` | Procedure tokens |
| `drug_exposure` | Medication tokens |
| `concept` | OMOP vocabulary lookup |

**Mapping and dictionary files:**

- `VOCAB_MAPPING_DIR/icd9_to_icd10_mapping.csv`
- `VOCAB_MAPPING_DIR/proc_to_snomed_mapping.csv`
- `VOCAB_MAPPING_DIR/rxnorm_drug_to_rxnorm_ingredient_mapping.csv`
- `VOCAB_MAPPING_DIR/payer_concept_annotated.csv`
- `MARKETSCAN_DICTIONARY_DIR/PLANTYP.csv`
- `MARKETSCAN_DICTIONARY_DIR/EGEOLOC.csv`

The mapping files are generated or documented by [`../../vocab_mapping/README.md`](../../vocab_mapping/README.md).

## Outputs

```text
EXTERNAL_VAL_DATA_DIR/
|-- ehrshot_payer_plan_period_collapsed/
|-- excluded_personids/
|-- all_demographic_tokens/
|-- all_enrollment_tokens/
|-- all_claim_tokens/
|-- all_anchor_tokens/
|-- duration/
|-- demo_seq/
|-- event_seq_per_timestamp/
|-- trajectory/
|-- all_att_tokens/
`-- vocab/
```
