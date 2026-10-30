# ReClaim Data Preprocessing: OMOP Pipeline for EHRSHOT

A PySpark-based data preprocessing pipeline for transforming OMOP CDM healthcare data into tokenized sequences compatible with the ReClaim foundation model. It applies the tokenization scheme of the MarketScan pipeline ([`../../pipeline_marketscan`](../../pipeline_marketscan/README.md)) to the EHRSHOT OMOP CDM export; the source-specific adaptations are listed in [Differences from the MarketScan Pipeline](#differences-from-the-marketscan-pipeline).

**Key Features:**
- Processes OMOP CDM data (developed on EHRSHOT)
- Monthly timestamp granularity (same as the MarketScan pipeline)
- Vocabulary-level code mapping and standardization
- Temporal anchor tokens (AGE, NY) for year tracking
- Cutoff-relative sequence splitting via `--training_cutoff_year` (no hardcoded 2022/2023 columns)

**Timestamp Granularity:** All events occurring in the same **month** are grouped together. The timestamp is set to the **first day of the month** using `F.trunc(date, "month")`, and sequence assembly keeps ordered event groups within that month (see `payer_order` below).

## Overview

| Stage | Script | Description |
|-------|--------|-------------|
| S0 | `s0_data_filtering.py` | Filter patients by quality criteria |
| S1 | `s1_construct_tokens.py` | Convert OMOP data to standardized tokens |
| S2 | `s2_construct_seqences.py` | Build temporal sequences per patient |
| S3 | `s3_finalize_vocab.py` | Create complete vocabulary |
| S5 | `s5_sanity_check.py` | Run sanity checks on trajectory data |

## Pipeline Stages
---

### Stage 0: Data Filtering (`s0_data_filtering.py`)

Filters patients based on data quality and eligibility criteria from OMOP CDM tables.

**Filtering Criteria:**
- Minimum total enrollment days (default: 180 days)
- Minimum EHR visit duration (default: 30 days between first and last visit)
- Age at first event: 10-110 years
- Enrollment periods < 30 days are dropped before merging

**Enrollment Period Processing:**
- Overlapping/adjacent enrollment periods within 1 month are merged per (person_id, PAYER)
- `PLANTYP` and `CAP` are taken from the first period (by earliest start date) when merging
- Payer mapping: C → CCAE, R → Medicare, D → Medicaid

**Outputs:**
- `external_val_data/ehrshot_payer_plan_period_collapsed/` - Merged enrollment periods
- `external_val_data/excluded_personids/` - Excluded patient IDs

---

### Stage 1: Construct Tokens (`s1_construct_tokens.py`)

Converts raw OMOP data into standardized tokens with vocabulary mapping.

**Token Structure:**

```
<sos> [Demographic] [Events separated by ATT] <eos>
```

| Component | Description |
|:----------|:------------|
| **Demographic** | Static patient characteristics (appear once at start): `SEX`, `DOBYR` |
| **Events** | Temporal events grouped by timestamp unit (month), separated by `<ATT-N>` tokens |
| **ATT** | Artificial Time Token: `<ATT-N>` indicates N **months** until next event (appended after each timestamp's tokens). |

**Event Types & Ordering (within same timestamp unit):**

Following the MarketScan pipeline, every token carries a `payer_order` and Stage 2 orders event groups within a month by `payer_order` **first**, then `event_type`. In EHRSHOT, enrollment tokens take their true payer order (CCAE=0, MDCR=1, MDCD/MISSING=2), so when several payers' enrollment events fall in the same month each payer's block (`ERLST` → `PLANTYP` → `CAP` → `EGEOLOC`) is emitted coherently instead of interleaving categories across payers. Clinical/visit tokens come from a single EHR stream and are not payer-attributed, so all claim tokens (and anchor tokens) use `payer_order = 0`. Same-month groups are separated by `<ATT-0>`.

Within a single event group, tokens are ordered by `event_type`, then by `category_priority` within each type:

| `event_type` | Event Type | Token Order (by `category_priority`) |
|:------------:|:-----------|:-------------------------------------|
| 0 | **Anchor** | `AGE` → `NY` |
| 1 | **Enrollment Start** | `ERLST` → `PLANTYP` → `CAP` → `EGEOLOC` |
| 2 | **Outpatient Visit** | `VT` → `DX` → `PROC` → `COST` |
| 3 | **Pharmacy Visit** | `VT` → `RX` → `COST` |
| 4 | **Inpatient Visit** | `VT` → `DX` → `PROC` → `DS` → `LS` → `COST` |
| 5 | **Enrollment End** | `ERLED` |

**Within DX/PROC categories:**
- `PRINCIPAL` marker → principal codes → `SECONDARY` marker → secondary codes
- For PROC: hierarchical tokens ordered by `comb_order` (e.g., `COMBSTART` → codes → `COMBEND`)

**Principal/Secondary Determination:**
- For DX: Based on `condition_status_concept_id == 32902` (primary diagnosis)
- For PROC: All procedures are marked as principal in EHRSHOT (no principal indicator available)

**Anchor Token Logic:**
- `AGE`: Placed at Jan 1 of first event year. Value = first_event_year - year_of_birth
- `NY`: Placed at Jan 1 of each subsequent year (marks year boundaries)


**Token Categories:**

| Group | Category | Description | Value Range | Example |
|:------|:---------|:------------|:------------|:--------|
| Demographic | `SEX` | Patient sex | 1, 2, MISSING | `<SEX-1>` |
| | `DOBYR` | Date of birth year | 1900-2014 | `<DOBYR-1965>` |
| Enrollment | `ERLST` | Enrollment start (payer) | CCAE, MDCR, MDCD, MISSING | `<ERLST-CCAE>` |
| | `PLANTYP` | Plan type (follows ERLST) | 1-9, MISSING | `<PLANTYP-5>` |
| | `CAP` | Capitation: 0=FFS, 1=capitated | 0, 1, MISSING | `<CAP-0>` |
| | `EGEOLOC` | Geographic location | 62 (fixed for EHRSHOT) | `<EGEOLOC-62>` |
| | `ERLED` | Enrollment end (payer) | CCAE, MDCR, MDCD, MISSING | `<ERLED-MDCR>` |
| Anchor | `AGE` | Age at first event year | 10-110 | `<AGE-65>` |
| | `NY` | New year marker | (single token) | `<NY>` |
| ATT | `ATT` | **Months** until next event | 0-12 | `<ATT-3>` |
| Visit | `VT` | Visit type | inpatient, outpatient, pharmacy | `<VT-inpatient>` |
| Clinical | `DX` | Diagnosis (ICD-10-CM split) | MAJOR_*, MINOR_*, SUFFIX_*, PRINCIPAL, SECONDARY, NOMAP, MISSING | `<DX-MAJOR_E11>` |
| | `PROC` | Procedure (SNOMED) | SNOMED codes, COMBSTART, COMBEND, CPT4*, PRINCIPAL, SECONDARY, NOMAP, MISSING | `<PROC-387713003>` |
| | `RX` | Medication (RxNorm ingredient) | RxNorm IDs, COMBSTART, COMBEND, NOMAP, MISSING | `<RX-161>` |
| Inpatient | `DS` | Discharge status | MISSING (not available in EHRSHOT) | `<DS-MISSING>` |
| | `LS` | Length of stay | 0 (<7d), 1 (≥7d), MISSING | `<LS-0>` |
| | `COST` | Aggregated cost | MISSING (not available in EHRSHOT) | `<COST-MISSING>` |
| Instruct | `INSTRUCT` | Task instruction tokens for fine-tuning | COST, DX | `<INSTRUCT-COST>` |

**Vocabulary Mappings:**

| Mapping | Source → Target | Approach |
|---------|-----------------|----------|
| **Diagnosis** | ICD-9-CM → ICD-10-CM | One-to-many preserved as separate rows; ICD-10-CM codes kept as-is |
| **Procedure** | CPT4/ICD9CM/ICD10PCS → SNOMED | Hierarchical with COMBSTART/COMBEND grouping; unmapped CPT4 preserved with `CPT4` prefix |
| **Drug** | RxNorm Drug → RxNorm Ingredient | Hierarchical with COMBSTART/COMBEND grouping; unmapped codes get `NOMAP` token |

**Diagnosis Mapping Logic (DXVER handling):**
- `vocabulary_id == 'ICD10CM'` (DXVER = "0"): Keep original code, source_vocab = ICD10CM
- `vocabulary_id == 'ICD9CM'` (DXVER = "9") with mapping found: Use target ICD-10-CM, source_vocab = ICD9CM
- `vocabulary_id == 'ICD9CM'` (DXVER = "9") without mapping: **`NOMAP`** (the raw ICD-9 code is not carried forward)
- DXVER is derived from the OMOP `vocabulary_id`, so it is never NULL in EHRSHOT; the MarketScan date-based NULL-DXVER disambiguation branches are kept in the code only for parity with the main pipeline and never fire

**Mapped ICD-10 DX diagnostics (`mapped_dx_icd10`):** during claim-token construction, S1 accumulates per-code counts of the mapped ICD-10 DX values (before MAJOR/MINOR/SUFFIX splitting) across the inpatient and outpatient runs, then unions + sums them into a single `mapped_dx_icd10` parquet dataset sorted by frequency.

**ICD-10-CM Token Splitting:**
- MAJOR: First 3 characters (category code) → `<DX-MAJOR_E11>`
- MINOR: Characters 4+ excluding trailing letters → `<DX-MINOR_65>`
- SUFFIX: Trailing 1-3 uppercase letters → `<DX-SUFFIX_A>`

**Hierarchical Mapping for Procedure Codes:**

When a source code maps to multiple target codes (one-to-many), the targets are grouped using `<PROC-COMBSTART>` and `<PROC-COMBEND>` tokens:

*Procedure Example (CPT4 → SNOMED):*
```
Source: <PROC-74301> (CPT4 code for imaging procedure)
Target: <PROC-COMBSTART> <PROC-16278002> <PROC-71093009> <PROC-COMBEND>
        └─ grouped SNOMED concepts representing the procedure
```

*Unmapped CPT4 Fallback:*
```
Source: <PROC-99213> (CPT4 with no SNOMED mapping)
Target: <PROC-CPT499213>
        └─ preserves original CPT4 code with prefix
```

**Hierarchical Mapping for Drug Codes:**

When a drug maps to multiple ingredients (combination drugs), the targets are grouped using `<RX-COMBSTART>` and `<RX-COMBEND>` tokens:

*Drug Example (RxNorm Drug → RxNorm Ingredient):*
```
Source: <RX-1000000> (RxNorm drug with multiple ingredients)
Target: <RX-COMBSTART> <RX-17767> <RX-321064> <RX-5487> <RX-COMBEND>
        └─ multiple active ingredients in the drug
```

**EHRSHOT-Specific Notes:**
- `COST` is always `MISSING` (no cost data available in EHRSHOT OMOP)
- `DS` (discharge status) is always `MISSING` (not available)
- `EGEOLOC` is fixed at `62` for all patients
- Drug vocabulary filtering: Only RxNorm codes are processed
- Procedure vocabulary filtering: CPT4, ICD10PCS, ICD9CM (HCPCS excluded)

**Aggregation Within Month:**

When multiple events occur within the same month, the following rules apply:

| Token Type | Aggregation Rule |
|------------|------------------|
| **COST** | Always `MISSING` (not available in EHRSHOT) |
| **VT** | First-by-date |
| **DX, PROC, RX** | Deduplicated by keeping first occurrence (by original_date), then sorted by original_date |
| **LS, DS** (inpatient) | All distinct values kept (multiple tokens if different) |

**Output parquet files:**
- `external_val_data/all_demographic_tokens/`
- `external_val_data/all_enrollment_tokens/`
- `external_val_data/all_claim_tokens/`
- `external_val_data/all_anchor_tokens/`
- `external_val_data/mapped_dx_icd10/`
- `external_val_data/duration/`

---

### Stage 2: Construct Sequences (`s2_construct_seqences.py`)

Assembles tokens into ordered patient trajectories.

**Sequence Structure:**
```
<sos> <SEX-X> <DOBYR-XXXX> <AGE-XX> [events...] <eos>
```

**Event Ordering (within same month timestamp):**

Tokens within the same month are sorted hierarchically:

1. **`payer_order`** - Payer-coherent event groups first (CCAE=0, MDCR=1, MDCD/MISSING=2 for enrollment tokens, always 0 for claim/anchor tokens)
2. **`event_type`** - Group by visit type
3. **`category_priority`** - Within each visit, order by category
4. **`sub_order`** - Within DX/PROC, order by principal/secondary
5. **`original_date`** - Within same category, order by actual event date
6. **`original_token`** / **`comb_order`** - Final tie-breakers

The same `(timestamp, payer_order, event_type)` order is used to compute `<ATT-N>` gaps, so the inter-month gap attaches to the true last group of the month and `<ATT-0>` separates intra-month groups.

| Priority | Category |
|----------|----------|
| 0 | `AGE` (anchor tokens) |
| 1 | `NY` (new year marker) |
| 2 | `ERLST` (enrollment start) |
| 3 | `PLANTYP` |
| 4 | `CAP` |
| 5 | `EGEOLOC` |
| 6 | `VT` (visit type) |
| 7 | `DX` (diagnoses: PRINCIPAL marker → codes → SECONDARY marker → codes) |
| 8 | `PROC` (procedures: PRINCIPAL marker → codes → SECONDARY marker → codes) |
| 9 | `RX` (medications) |
| 10 | `DS` (discharge status) |
| 11 | `LS` (length of stay) |
| 12 | `COST` |
| 13 | `ERLED` (enrollment end) |

**Temporal Tokens:**
- `<ATT-N>`: Appended after each timestamp's events to indicate **N months** until next event

**Training Cutoff Split:**
- The split point is controlled by the required `--training_cutoff_year` argument (set in `run_ehrshot.sh`, default `2023`, matching the MarketScan pipeline). Events with `year >= training_cutoff_year` are held out; column names are cutoff-relative.
- `seq_thru_cutoff`: Self-contained sequence covering all events through the end of (`training_cutoff_year` - 1), wrapped as `<sos> [demographics] [events] <eos>`. The events portion includes the trailing ATT that bridges to the first from-cutoff event.
- `event_seq_from_cutoff`: All events from January 1 of `training_cutoff_year` onwards; always starts with a `<NY>` token.
- `first_token_index_from_cutoff`: 0-based index pointing to the first from-cutoff `<NY>` token for prediction tasks
  - `NULL` if no from-cutoff data OR no thru-cutoff data (need historical context to predict)
  - Formula: `token_count_thru_cutoff` (since `token_count_thru_cutoff` already includes `<sos>` + 2 demo tokens)

**Trajectory Filtering:**
- S2 does not filter trajectories. Missing DOBYR is excluded in S1 (persons without `year_of_birth` get no demographic tokens and are dropped by the inner join with `demo_seq`), missing AGE cannot occur, and the 10-110 age range is already enforced in S0.

**Outputs:**
- `external_val_data/demo_seq/`
- `external_val_data/event_seq_per_timestamp/`
- `external_val_data/trajectory/`
- `external_val_data/all_att_tokens/`

**Trajectory Output Schema:**

| Column | Description |
|--------|-------------|
| `person_id` | Patient identifier from OMOP |
| `seq_thru_cutoff` | Self-contained pretraining-style sequence: `<sos>` + demographic tokens + event tokens through (`training_cutoff_year` - 1) + `<eos>` |
| `event_seq_from_cutoff` | Event sequence starting from `training_cutoff_year` (starts with `<NY>` token if available) |
| `seq` | Full trajectory string from `<sos>` to `<eos>` |
| `first_token_index_from_cutoff` | 0-based index pointing to the first from-cutoff `<NY>` token. `NULL` if: (1) no events from the cutoff onward, or (2) no historical events before the cutoff |
| `total_token_count` | Total tokens: `<sos>` + 2 demo tokens + all event tokens + `<eos>` |
| `token_count_thru_cutoff` | Tokens from start through (`training_cutoff_year` - 1): `<sos>` + 2 demo tokens + thru-cutoff event tokens (no `<eos>`) |
| `event_token_count_from_cutoff` | Event tokens from the cutoff year only (excludes `<sos>`, `<eos>`, demo tokens, and pre-cutoff events) |
| `total_duration` | Days between first and last event |

---

### Stage 3: Finalize Vocabulary (`s3_finalize_vocab.py`)

Creates the complete vocabulary by combining observed tokens with theoretically possible tokens.

**Vocabulary Sources:**
1. **From Data**: All unique tokens observed in the dataset
2. **By Design**: All theoretically possible values for structured fields

**Vocabulary Ranges By Design:**
| Category | Range |
|----------|-------|
| `DOBYR` | 1900-2014 |
| `AGE` | 10-110 |
| `SEX` | 1, 2, MISSING |
| `PLANTYP` | From MarketScan dictionary (+ MISSING) |
| `EGEOLOC` | From MarketScan dictionary (+ MISSING) |
| `CAP` | 0, 1, MISSING |
| `DS` | 0-4, MISSING |
| `LS` | 0, 1, MISSING |
| `COST` | 00-99, MISSING |
| `ATT` | 0-12 months |

**Instruct Tokens for Downstream Tasks:**

Special tokens added for instruction-tuning:

| Token | Purpose |
|-------|---------|
| `<INSTRUCT-COST>` | Instruction token for cost prediction tasks |
| `<INSTRUCT-DX>` | Instruction token for diagnosis prediction tasks |

**Outputs:**
- `external_val_data/vocab/` - Final vocabulary with token frequencies

---

### Stage 5: Sanity Check (`s5_sanity_check.py`)

Performs validation checks on the trajectory data.

**Checks Performed:**

1. **Missing Token Check**: Verify presence of required tokens (`<DOBYR>`, `<SEX>`, `<AGE>`)

2. **Temporal Information Check**:
   - Extract DOBYR + AGE to get `start_year_seq`
   - Count `<NY>` tokens to get `num_years_passed_seq`
   - Compare with `floor(total_duration/365)`
   - Report delta distribution and cases where delta > 1 or > 2 years

3. **Split Point Checks** (for rows with `first_token_index_from_cutoff`; the expected end year is derived from the required `--training_cutoff_year` argument):
   - Token at split index is `<NY>`
   - First token of `event_seq_from_cutoff` is `<NY>`
   - Last token (before the trailing `<eos>`) of `seq_thru_cutoff` is `<ATT-*>`
   - DOBYR + AGE + count(`<NY>` in seq_thru_cutoff) = `training_cutoff_year` - 1

---

## Usage

### Prerequisites

The conda environment from [`../../environment.yml`](../../environment.yml) (Python 3.11, PySpark), a Java runtime for local-mode Spark (`run_ehrshot.sh` loads the cluster module `SPARK_MODULE`), and the mapping files from [`../../vocab_mapping`](../../vocab_mapping/README.md).

### Running the Pipeline

All paths come from `../../env.sh` (copy [`../../env.sh.example`](../../env.sh.example)): `EHRSHOT_OMOP_DIR`, `VOCAB_MAPPING_DIR`, `MS_DICTIONARY_DIR`, and the output location `$MS_PROCESSED_ROOT/$DATA_VERSION/$EHRSHOT_RUN_NAME`. The driver runs S0, S1, S2, S3 and S5 in order and stops at the first failed stage:

```bash
cd pipeline_omop/ehrshot
sbatch run_ehrshot.sh
```

Or run each stage individually (`--training_cutoff_year` is required for s2 and s5):

```bash
# Stage 0: Filter patients (reads ../payer_concept_annotated.csv)
python s0_data_filtering.py --omop_dir /path/to/omop --external_val_data_dir external_val_data

# Stage 1: Construct tokens
python s1_construct_tokens.py --omop_dir /path/to/omop --mapping_dir /path/to/mapping --external_val_data_dir external_val_data

# Stage 2: Construct sequences
python s2_construct_seqences.py --external_val_data_dir external_val_data --training_cutoff_year 2023

# Stage 3: Finalize vocabulary
python s3_finalize_vocab.py --external_val_data_dir external_val_data --dictionary_dir /path/to/MSN_LOOKUP

# Stage 5: Sanity check
python s5_sanity_check.py --external_val_data_dir external_val_data --training_cutoff_year 2023
```

---

## Data Sources

### Input Tables (OMOP CDM Format)

| Table | Description |
|-------|-------------|
| `person` | Patient demographics (person_id, gender_concept_id, year_of_birth) |
| `payer_plan_period` | Insurance enrollment periods |
| `visit_occurrence` | Healthcare visits (inpatient, outpatient) |
| `condition_occurrence` | Diagnosis records |
| `procedure_occurrence` | Procedure records |
| `drug_exposure` | Medication records |
| `concept` | Vocabulary concepts for code lookups |

### Visit Concept ID Mapping

| Visit Type | visit_concept_id |
|------------|------------------|
| Inpatient | 262, 9201 |
| Outpatient | 9202, 581477, 9203 |
| Pharmacy | 9202, 581477, 9203, 581458 |

### Mapping Files

In `mapping_dir` (generated by [`../../vocab_mapping`](../../vocab_mapping/README.md)), read by S1:
- `icd9_to_icd10_mapping.csv` - ICD-9 to ICD-10-CM crosswalk
- `proc_to_snomed_mapping.csv` - Procedure to SNOMED-CT mapping
- `rxnorm_drug_to_rxnorm_ingredient_mapping.csv` - RxNorm drug to RxNorm ingredient mapping

In `pipeline_omop/` (shared with the YNHH pipeline), read by S0:
- `payer_concept_annotated.csv` - OMOP `payer_concept_id` to MarketScan `PAYER` / `PLANTYP` / `CAP`, with the reason for each assignment

---

## Output Directory Structure

```
external_val_data/
├── ehrshot_payer_plan_period_collapsed/  # S0 output
├── excluded_personids/                    # S0 output
├── all_demographic_tokens/                # S1 output
├── all_enrollment_tokens/                 # S1 output
├── all_claim_tokens/                      # S1 output
├── all_anchor_tokens/                     # S1 output
├── mapped_dx_icd10/                       # S1 output
├── duration/                              # S1 output
├── demo_seq/                              # S2 output
├── event_seq_per_timestamp/               # S2 output
├── trajectory/                            # S2 output
├── all_att_tokens/                        # S2 output
└── vocab/                                 # S3 output
```

---

## Logging

Logs are saved to `logs/` directory with timestamps:
```
logs/s0_data_filtering_20241229_143052.log
logs/s1_construct_tokens_20241229_144523.log
logs/s2_construct_seqences_20241229_145012.log
logs/s3_finalize_vocab_20241229_150234.log
logs/s5_sanity_check_20241229_151023.log
```

---

## Sequence Example

A synthetic sequence (line breaks added after each `<ATT-N>` for readability). COST is always `MISSING` and EGEOLOC is fixed at `62` for an OMOP source.

```
<sos> <SEX-2> <DOBYR-1961> <AGE-51> <ATT-0>
<ERLST-CCAE> <PLANTYP-6> <CAP-0> <EGEOLOC-62> <ATT-4>
<VT-outpatient> <DX-PRINCIPAL> <DX-MAJOR_E11> <DX-MINOR_9> <DX-SECONDARY> <DX-MAJOR_E78> <DX-MINOR_5> <PROC-PRINCIPAL> <PROC-CPT499213> <COST-MISSING> <ATT-0>
<VT-pharmacy> <RX-6809> <COST-MISSING> <ATT-8>
<NY> <ATT-3>
<VT-outpatient> <DX-PRINCIPAL> <DX-MAJOR_E11> <DX-MINOR_9> <COST-MISSING> <ATT-9>
<NY> ... <eos>
```

`<AGE-51>` anchors January of the first observed year (2012). Enrollment starts the same month (`<ATT-0>`), an outpatient visit and a pharmacy fill follow in May (`<ATT-4>`, then `<ATT-0>`), `<ATT-8>` reaches the next new-year marker, and `<ATT-3>` an April visit.

---

## Differences from the MarketScan Pipeline

| Aspect | MarketScan | OMOP (EHRSHOT) |
|--------|---------------|------------|
| Data Format | MarketScan proprietary | OMOP CDM |
| Patient ID | `enrollee_id` | `person_id` |
| COST | Aggregated per month (per payer) | Always `MISSING` |
| DS (discharge status) | Available | Always `MISSING` |
| EGEOLOC | From data | Fixed at `62` |
| Drug Mapping | NDC → RxNorm Ingredient | RxNorm Drug → RxNorm Ingredient |
| RX Hierarchical | Uses COMBSTART/COMBEND | Uses COMBSTART/COMBEND (aligned) |
| Principal Indicator (PROC) | Available | All marked as principal |
| Train/Test Split (S4) | Included | Not applicable |
| Payer Values | CCAE, MDCR, MDCD | CCAE, MDCR, MDCD, MISSING |
| `payer_order` on claims | True payer of the claim stream (CCAE=0, MDCR=1, MDCD=2) | Always `0` (single EHR stream, not payer-attributed) |
| `payer_order` on enrollment | CCAE=0, MDCR=1, MDCD=2 | CCAE=0, MDCR=1, MDCD/MISSING=2 (aligned) |
| DXVER | From claim data; can be NULL (date/structure fallback used) | Derived from OMOP `vocabulary_id`; never NULL |
| Training cutoff | `--training_cutoff_year` (default 2023) | `--training_cutoff_year` (aligned, default 2023) |
| Sanity check sampling (S5) | Samples up to 100k rows | Checks all trajectories (small dataset) |
| Date window | All years present in the raw files | Fixed 2008-01-01 to 2025-03-31 in S0/S1 |
| Intermediate materialization (S2) | Sorted tokens written to `_intermediate_sorted_tokens/` | Not needed (small dataset) |
