# ReClaim Data Preprocessing (v6 YNHH)

This folder preprocesses Yale New Haven Health (YNHH) OMOP CDM data into ReClaim-compatible monthly token sequences for external validation.

YNHH is a private hospital-based EHR cohort used as an external validation set. This pipeline starts from the original YNHH OMOP-formatted tables and converts them into the same monthly ReClaim token format used for MarketScan disease-onset evaluation.

## Shared Behavior

This pipeline intentionally follows the main [v6 MarketScan README](../v6_marketscan/README.md) for:

- v6 monthly timestamping and `<ATT-N>` temporal tokens
- token format, category ordering, anchor tokens, and sequence construction
- ICD-9-CM to ICD-10-CM diagnosis mapping
- procedure and drug combination-token logic
- 2022/2023 split fields used for prospective evaluation
- vocabulary finalization and sanity-check principles

This README only documents what is different for YNHH.

## Pipeline Stages

| Stage | Script | YNHH-specific role |
|-------|--------|--------------------|
| S0 | `s0_data_filtering.py` | Filter YNHH patients, collapse OMOP payer periods, and sample 110,000 eligible patients so the final cohort retains at least about 100,000 patients |
| S1 | `s1_construct_tokens.py` | Convert YNHH OMOP records into ReClaim tokens |
| S2 | `s2_construct_seqences.py` | Build monthly patient trajectories using the shared v6 sequence logic |
| S3 | `s3_finalize_vocab.py` | Add YNHH-observed tokens into a ReClaim-compatible vocabulary |
| S5 | `s5_sanity_check.py` | Run trajectory sanity checks |

There is no S4 train/validation/test split because YNHH is an external validation cohort, not a pretraining corpus.

## YNHH-Specific Processing

| Area | Handling |
|------|----------|
| Source data | Original YNHH OMOP CDM parquet tables under `OMOP_DIR` |
| Patient ID | Uses OMOP `person_id` rather than MarketScan `enrollee_id` |
| Date window | Keeps records from `2008-01-01` through `2025-03-31`; missing payer-period end dates are filled with `2025-03-31` |
| Filtering | Requires at least 180 enrollment days, at least 30 days between first and last visit, age at first event from 10 to 110, and at least one enrollment period |
| Sampling | S0 samples 110,000 eligible patients without replacement using `SAMPLE_SIZE` and `seed`, saves `sampled_eligible_personids`, and downstream stages process only that sampled cohort. The target is set slightly above 100,000 so the final validation set retains at least about 100,000 patients after downstream processing. |
| Enrollment | Uses `payer_plan_period`; drops enrollment periods under 30 days; merges overlapping or adjacent periods within 1 month per `person_id` and `PAYER` |
| Payer fields | Maps OMOP `payer_concept_id` to ReClaim/MarketScan-like `PAYER`, `PLANTYP`, and `CAP` using the domain-expert annotated `payer_concept_annotated.csv`; missing mapped values become `MISSING` |
| Location | Assumes `EGEOLOC-04` as the proxy location for Yale New Haven Health |
| Visit type | Uses OMOP `visit_concept_id`: inpatient `262`, `9201`; outpatient `9202`, `581477`, `9203`; pharmacy `9202`, `581477`, `9203`, `581458` with a valid drug source concept |
| Diagnosis | Uses `condition_occurrence` joined to `concept`; keeps ICD-10-CM and maps ICD-9-CM using `icd9_to_icd10_mapping.csv` |
| Procedure | Uses `procedure_occurrence` joined to `concept`; keeps SNOMED procedures as-is; maps CPT4, ICD10PCS, and ICD9CM to SNOMED using `proc_to_snomed_mapping.csv`; excludes HCPCS-like source codes; all procedures are marked principal because the source fields used here do not provide a procedure principal indicator |
| Drug | Uses `drug_exposure` joined to `concept`; maps both RxNorm drugs and NDCs to RxNorm ingredients using `rxnorm_drug_to_rxnorm_ingredient_mapping.csv` and `ndc_drug_to_rxnorm_ingredient_mapping.csv` |
| Cost | Always emits `<COST-MISSING>` because claim cost is not available in the source fields used here |
| Discharge status | Always emits `<DS-MISSING>` because discharge status is not available in the source fields used here |
| Length of stay | Derived from inpatient `visit_start_datetime` and `visit_end_datetime`: `LS-0` for stays under 7 days, `LS-1` for 7 days or longer |
| Vocabulary | Includes observed YNHH tokens plus the structured token ranges needed for compatibility with the MarketScan-trained ReClaim model |

## Expected Inputs

Prepare these local inputs before running:

- `YNHH_DATA_ROOT`: root containing the original YNHH data; `OMOP_DIR` is `${YNHH_DATA_ROOT}/ynhh_omop`
- `PROCESSED_YNHH_DATA_ROOT`: root for generated outputs; `EXTERNAL_VAL_DATA_DIR` is `${PROCESSED_YNHH_DATA_ROOT}/v6/ynhh`
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
- `VOCAB_MAPPING_DIR/ndc_drug_to_rxnorm_ingredient_mapping.csv`
- `VOCAB_MAPPING_DIR/payer_concept_annotated.csv`
- `MARKETSCAN_DICTIONARY_DIR/PLANTYP.csv`
- `MARKETSCAN_DICTIONARY_DIR/EGEOLOC.csv`

The mapping files are generated or documented by [`../../vocab_mapping/README.md`](../../vocab_mapping/README.md).

## Outputs

```text
EXTERNAL_VAL_DATA_DIR/
|-- ynhh_payer_plan_period_collapsed/
|-- excluded_personids/
|-- sampled_eligible_personids/
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
