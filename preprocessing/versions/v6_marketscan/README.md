# ReClaim Data Preprocessing (v6)

A PySpark-based data preprocessing pipeline for transforming MarketScan healthcare claims data into tokenized sequences for foundation model training. The provided Slurm script is configured for running Spark jobs on the Yale Center for Research Computing (YCRC) cluster.

Version 6 (`v6`) uses monthly temporal resolution. Events are timestamped at the month level, and Artificial Time Tokens (`<ATT-N>`) represent elapsed months from 0 to 12.

**Timestamp Granularity:** All events occurring in the same **month** share a monthly timestamp. The timestamp is set to the **first day of the month** using `F.trunc(date, "month")`, and sequence assembly keeps ordered event groups within that month.

## Overview

| Stage | Script | Description |
|-------|--------|-------------|
| S0 | `s0_data_filtering.py` | Filter enrollees by quality criteria |
| S1 | `s1_construct_tokens.py` | Convert claims to standardized tokens |
| S2 | `s2_construct_seqences.py` | Build temporal sequences per patient |
| S3 | `s3_finalize_vocab.py` | Create complete vocabulary |
| S4 | `s4_train_test_split.py` | Split into train/val/test sets |
| S5 | `s5_sanity_check.py` | Run structural and temporal sanity checks |

## Pipeline Stages
---

### Stage 0: Data Filtering (`s0_data_filtering.py`)

Filters enrollees based on data quality and eligibility criteria.

**CCAE + MDCR (Commercial + Medicare) Filtering:**
- `EIDFLAG == 1`: Valid enrollee ID assignment
- `RX == 1`: Drug data availability
- Minimum total member days (default: 180 days)
- Minimum claims duration (default: 30 days between first and last claim records)
- Age at first event: 10-110 years

**MDCD (Medicaid) Filtering:**
- `DRUGCOVG == 1`: Drug coverage availability
- Minimum total member days
- Minimum claims duration
- `MEDICARE != 1`: Exclude Medicare dual-eligible (avoid data leakage)
- Age at first event: 10-110 years

**Outputs:**
- `processed_data/excluded_enrollids_ccae_mdcr/` - Excluded CCAE/MDCR enrollee IDs
- `processed_data/excluded_enrollids_mdcd/` - Excluded MDCD enrollee IDs

---

### Stage 1: Construct Tokens (`s1_construct_tokens.py`)

Converts raw claims data into standardized tokens with vocabulary mapping.

**Token Format:**

Most tokens follow `<CATEGORY-VALUE>`, where `CATEGORY` identifies the token type and `VALUE` stores the encoded claim, enrollment, demographic, or temporal value. Examples: `<SEX-1>`, `<DX-MAJOR_E11>`, `<PROC-CPT499213>`, `<ATT-3>`. Sentinel and boundary tokens such as `<sos>`, `<eos>`, and `<NY>` do not use this format.

**Token Structure:**

```
<sos> [Demographic] [Events separated by ATT] <eos>
```

| Component | Description |
|:----------|:------------|
| **Demographic** | Static patient characteristics (appear once at start): `SEX`, `DOBYR` |
| **Events** | Temporal event groups ordered by month and event type, separated by `<ATT-N>` tokens |
| **ATT** | Artificial Time Token: `<ATT-N>` indicates N **months** until the next event group. `<ATT-0>` can separate multiple event groups in the same month. **In v6, timestamp unit = month (monthly granularity).** |

**Event Types & Ordering (within same timestamp unit):**

Events within the same timestamp unit are ordered by `event_type`, then by `category_priority` within each type:

| `event_type` ↓ | Event Type | Token Order (by `category_priority`) |
|:------------:|:-----------|:-------------------------------------|
| 0 | **Anchor** | `AGE` → `NY` |
| 1 | **Enrollment Start** | `ERLST` → `PLANTYP` → `CAP` → `EGEOLOC` |
| 2 | **Outpatient Visit** | `VT` → `DX` → `PROC` → `COST` |
| 3 | **Pharmacy Visit** | `VT` → `RX` → `COST` |
| 4 | **Inpatient Visit** | `VT` → `DX` → `PROC` → `DS` → `LS` → `COST` |
| 5 | **Enrollment End** | `ERLED` |

**Within DX/PROC categories:**
- `PRINCIPAL` marker → principal codes → `SECONDARY` marker → secondary codes
- Hierarchical procedure/drug tokens ordered by `comb_order` (e.g., `PROC-COMBSTART` → codes → `PROC-COMBEND`)

**Anchor Token Logic:**
- `AGE`: Placed at Jan 1 of first event year. Value = first_event_year - DOBYR
- `NY`: Placed at Jan 1 of each subsequent year (marks year boundaries)
- Example: `<DOBYR-1980>` + `<AGE-30>` means first event in 2010; next `<NY>` marks 2011


**Token Categories:**

| Group | Category | Description | Value Range | Example |
|:------|:---------|:------------|:------------|:--------|
| Demographic | `SEX` | Patient sex | 1, 2, MISSING | `<SEX-1>` |
| | `DOBYR` | Date of birth year | 1900-2014 | `<DOBYR-1965>` |
| Enrollment | `ERLST` | Enrollment start (payer) | CCAE, MDCR, MDCD, MISSING | `<ERLST-CCAE>` |
| | `PLANTYP` | Plan type (follows ERLST) | 1-9, MISSING | `<PLANTYP-5>` |
| | `CAP` | Capitation: 0=FFS, 1=capitated | 0, 1, MISSING | `<CAP-0>` |
| | `EGEOLOC` | Geographic location | MSN lookup, MISSING | `<EGEOLOC-42>` |
| | `ERLED` | Enrollment end (payer) | CCAE, MDCR, MDCD, MISSING | `<ERLED-MDCR>` |
| Anchor | `AGE` | Age at first event year | 10-110 | `<AGE-65>` |
| | `NY` | New year marker | (single token) | `<NY>` |
| ATT | `ATT` | Time until next event group | 0-12 | `<ATT-3>` |
| Visit | `VT` | Visit type | inpatient, outpatient, pharmacy, MISSING | `<VT-inpatient>` |
| Clinical | `DX` | Diagnosis (ICD-10-CM split) | MAJOR_*, MINOR_*, SUFFIX_*, PRINCIPAL, SECONDARY, NOMAP, MISSING | `<DX-MAJOR_E11>` |
| | `PROC` | Procedure (SNOMED plus CPT4 fallback) | SNOMED codes, CPT4 fallback tokens, COMBSTART, COMBEND, PRINCIPAL, SECONDARY, NOMAP, MISSING | `<PROC-387713003>` |
| | `RX` | Medication (RxNorm ingredient) | RxNorm IDs, COMBSTART, COMBEND, NOMAP, MISSING | `<RX-161>` |
| Inpatient | `DS` | Discharge status | 0-4, MISSING | `<DS-3>` |
| | `LS` | Length of stay | 0 (<7d), 1 (≥7d), MISSING | `<LS-0>` |
| | `COST` | Aggregated cost | 00-99, MISSING | `<COST-52>` |
| Instruct | `INSTRUCT` | Task instruction tokens for fine-tuning | COST, DX | `<INSTRUCT-COST>` |

**Vocabulary Mappings:**

For detailed mapping methodology and source files, see [`../../vocab_mapping/README.md`](../../vocab_mapping/README.md).

| Mapping | Source → Target | Approach |
|---------|-----------------|----------|
| **Diagnosis** | ICD-9-CM → ICD-10-CM | One-to-many preserved as separate rows |
| **Procedure** | CPT4/ICD9CM/ICD10PCS → SNOMED | Hierarchical with COMBSTART/COMBEND grouping |
| **Drug** | NDC → RxNorm Ingredient | Two-step hierarchical with COMBSTART/COMBEND grouping |

**Hierarchical Mapping for Procedure and Drug Codes:**

Diagnosis codes are handled differently from procedure and drug codes. ICD-9-CM diagnosis codes are mapped to ICD-10-CM using GEMs and one-to-many targets are preserved as separate diagnosis token options. Procedure and drug mappings can represent grouped target concepts, so one-to-many targets are wrapped with domain-specific combination tokens.

When a procedure or drug source code maps to multiple target codes, the targets are grouped using `<PROC-COMBSTART>`/`<PROC-COMBEND>` or `<RX-COMBSTART>`/`<RX-COMBEND>` tokens. Each source code maps to exactly one unique combination (targets are ordered deterministically by design):

*Procedure Example (CPT4 → SNOMED):*
```
Source: <PROC-74301> (CPT4 code for imaging procedure)
Target: <PROC-COMBSTART> <PROC-16278002> <PROC-71093009> <PROC-COMBEND>
        └─ grouped SNOMED concepts representing the procedure (based on the relationship used to map)
```

*Drug Example (NDC → RxNorm Ingredient):*
```
Source: <NDCNUM-00069015001> (NDC for combination drug)
Target: <RX-COMBSTART> <RX-235496> <RX-6832> <RX-COMBEND>
        └─ multiple active ingredients in the drug
```

*Unmapped CPT4 Fallback:*
```
Source: <PROC-99213> (CPT4 with no SNOMED mapping)
Target: <PROC-CPT499213>
        └─ preserves original CPT4 code with prefix
```

This approach ensures:
- Multiple related concepts are kept together as a unit
- Cross-vocabulary harmonization (CPT4, ICD codes → unified SNOMED)
- Unmapped CPT4 codes are retained with a prefixed fallback token, preserving information from a major MarketScan outpatient procedure vocabulary

**Principal/Secondary Markers:**
- `<DX-PRINCIPAL>` / `<DX-SECONDARY>` markers precede diagnosis codes indicating if the following diagnosis code is principal or secondary
- `<PROC-PRINCIPAL>` / `<PROC-SECONDARY>` markers precede procedure codes indicating if the following procedure code is principal or secondary

**v6 Aggregation Within Month:**

When multiple events occur within the same month, the following rules apply:

| Token Type | Aggregation Rule |
|------------|------------------|
| **COST** | Sum of all costs within the month |
| **VT** | First-by-date  |
| **DX, PROC, RX** | Deduplicated by keeping first occurrence (by original_date), then sorted by original_date |
| **LS, DS** (inpatient) | All distinct values kept (multiple tokens if different) |

**Cost Discretization:** Costs are summed within each monthly event group, then the summed cost is rounded to one significant digit and encoded as two digits: the first digit is the rounded mantissa and the second digit is the order-of-magnitude exponent. The mantissa is rounded to the nearest whole number with HALF_UP tie handling using PySpark `F.round(m)`. If the rounded mantissa becomes 10, the code carries to `1` and increases the exponent by 1. For example, 940 is rounded to 900, represented as `9 x 10^2`, and encoded as `<COST-92>`; 950 is rounded to 1000, represented as `1 x 10^3`, and encoded as `<COST-13>`. Null values become `<COST-MISSING>`, zero or negative values become `<COST-00>`, and very large values are clipped to `<COST-99>`.

**Output parquet files:**
- `processed_data/all_demographic_tokens/*.parquet`
- `processed_data/all_enrollment_tokens/*.parquet`
- `processed_data/all_claim_tokens/*.parquet`
- `processed_data/all_anchor_tokens/*.parquet`
- `processed_data/duration/*.parquet`

---

### Stage 2: Construct Sequences (`s2_construct_seqences.py`)

Assembles tokens into ordered patient trajectories.

**Sequence Structure:**
```
<sos> <SEX-X> <DOBYR-XXXX> <AGE-XX> [events...] <eos>
```

**Sequence Ordering:**

Stage 2 first sorts event groups by their assigned monthly `timestamp`. Within each timestamp, it applies the Stage 1 event ordering (`event_type`, `category_priority`, and `sub_order`) to serialize anchor, enrollment, visit, clinical, cost, and enrollment-end tokens. `original_date`, `original_token`, and `comb_order` are then used as deterministic tie-breakers so repeated or grouped tokens have stable within-month order.

**Temporal Tokens:**
- `<ATT-N>`: Appended after each monthly event group to indicate **N months** until the next event group (uses `lead` for cleaner 2023 split)
- Year boundaries are marked by `<AGE-X>` for the first observed year and `<NY>` for each subsequent year

**Synthetic Sequence Example:**

This synthetic sequence starts with static demographic tokens, followed by monthly event groups. Each event group is followed by an `<ATT-N>` token, where `N` is the number of months until the next event group.

```
<sos> <SEX-1> <DOBYR-1974> <AGE-44> <ATT-0>
<ERLST-CCAE> <PLANTYP-6> <CAP-0> <EGEOLOC-04> <ATT-12>
<NY> <ATT-12>
<NY> <ATT-10>
<VT-outpatient> <DX-PRINCIPAL> <DX-MAJOR_R07> <DX-MINOR_9> <DX-MAJOR_E11> <DX-MINOR_9> <PROC-PRINCIPAL> ... <COST-32> <ATT-0>
<VT-pharmacy> <RX-6809> <COST-51> <ATT-2>
<NY> ...
```

Here, `<AGE-44>` anchors the sequence in January of the first observed year (2018). The first `<ATT-0>` means the next event group, enrollment start, occurs in the same month. The first `<ATT-12>` moves from January 2018 to the January 2019 new-year marker, and the second `<ATT-12>` moves to the January 2020 new-year marker, indicating that no claim events occurred during those intervals. `<ATT-10>` then moves to a November 2020 outpatient visit, and the next `<ATT-0>` shows that the pharmacy fill occurs in that same month.

**2023 Split:**
- The ReClaim pretraining corpus uses records through December 31, 2022; 2023+ events are separated for prospective evaluation.
- Sequences are split at 2022/2023 boundary
- `seq_thru_2022`: Events before 2023 (includes ATT to first 2023 event)
- `event_seq_from_2023`: Events from 2023 onwards (always starts with `<NY>` token)
- `first_token_index_from_2023`: 0-based index pointing to the `<NY>` token of 2023 for prediction tasks
  - `NULL` if no 2023 data OR no 2022 data (need historical context to predict)
  - Formula: `token_count_thru_2022` (since token_count_thru_2022 already includes `<sos>` + 2 demo tokens)

**Outputs:**
- `processed_data/demo_seq/`
- `processed_data/event_seq_per_timestamp/`
- `processed_data/trajectory/`
- `processed_data/all_att_tokens/`

---

### Stage 3: Finalize Vocabulary (`s3_finalize_vocab.py`)

Creates the complete vocabulary by combining observed tokens with theoretically possible tokens.

**Vocabulary Sources:**
1. **From Data**: All unique tokens observed in the intermediate token tables generated by Stages 1-2: `all_demographic_tokens`, `all_enrollment_tokens`, `all_claim_tokens`, `all_anchor_tokens`, and `all_att_tokens`. Clinical tokens (`DX`, `PROC`, and `RX`) are added based on the mapped codes observed in the processed data.
2. **By Design**: All theoretically possible values for structured fields

**Vocabulary Ranges By design:**
| Category | Range |
|----------|-------|
| `DOBYR` | 1900-2014 |
| `AGE` | 10-110 |
| `SEX` | From MarketScan dictionary |
| `PLANTYP` | From MarketScan dictionary |
| `EGEOLOC` | From MarketScan dictionary |
| `CAP` | From MarketScan dictionary |
| `DS` | From MarketScan dictionary |
| `LS` | From MarketScan dictionary |
| `COST` | 00-99, MISSING |
| `ATT` | 0-12 |

**Instruct Tokens for Downstream Tasks:**

Special tokens added for instruction-tuning and task-specific fine-tuning:

| Token | Purpose |
|-------|---------|
| `<INSTRUCT-COST>` | Instruction token for cost prediction tasks |
| `<INSTRUCT-DX>` | Instruction token for diagnosis prediction tasks |

These tokens are included in the vocabulary but not present in the pre-training sequences. They can be inserted during fine-tuning to signal the model to perform specific prediction tasks.

**Outputs:**
- `processed_data/vocab/` - Final vocabulary with token frequencies

---

### Stage 4: Train/Val/Test Split (`s4_train_test_split.py`)

Splits trajectories into train/validation/test sets.

**Split Strategy:**
- **Larger subsets / full datasets (>10M patients)**: Fixed 1M validation, 1M test, rest for training
- **Smaller subsets for fast iteration (≤10M patients)**: 80% train, 10% validation, 10% test

**Hybrid Approach:**
1. HuggingFace datasets for deterministic splitting (by enrollee_id)
2. Spark for efficient parallel I/O

**Outputs:**
- `processed_data/split_assignments/train_split_ids.parquet` - Training enrollee IDs
- `processed_data/split_assignments/val_split_ids.parquet` - Validation enrollee IDs
- `processed_data/split_assignments/test_split_ids.parquet` - Test enrollee IDs
- `processed_data/split_assignments/overall_split/` - Combined split labels
- `final_data/train/` - Training set
- `final_data/val/` - Validation set
- `final_data/test/` - Test set

---

### Stage 5: Sanity Check (`s5_sanity_check.py`)

Runs automated checks on the trajectory output.

**Checks:**
- Required token presence: `<DOBYR>`, `<SEX>`, and `<AGE>`
- Temporal consistency between `DOBYR`, `AGE`, `<NY>` markers, and `total_duration`
- 2022/2023 split boundary correctness when `first_token_index_from_2023` is available

**Outputs:**
- Log output only; this stage does not create model input files

---

## Running the Pipeline

The full MarketScan corpus is large: the study uses CCAE, MDCR, and MDCD data with more than 200 million enrollees and 43.8 billion claims records, and the local raw Parquet layout is approximately 1.7 TB. For the full dataset processing, it is recommended to use Spark on a distributed server with Slurm allocation.

This workflow is intended to run on the Yale Center for Research Computing (YCRC) Slurm environment. The provided Slurm script loads `Spark/3.5.4-foss-2022b-Scala-2.13`, activates a Python 3.11 environment, starts a Spark cluster inside the Slurm allocation, runs stages S0-S5 in order, and writes Slurm output to `slurm_job_logs/`.

Before running, prepare the local inputs and update the configuration block in `spark_preproces_v6.sh`:
- `RAW_MARKETSCAN_DATA_ROOT`: folder containing raw MarketScan CCAE, MDCR, and MDCD Parquet files from 2008-2024. Expected tables are enrollment detail (`t`), inpatient admissions (`i`), outpatient services (`o`), and pharmacy claims (`d`) for CCAE/MDCR and MDCD. `BASE_DIR` is derived from this root.
- `PROCESSED_MARKETSCAN_DATA_ROOT`: root for generated outputs. `INPUT_FOLDER_PATH` is `${PROCESSED_MARKETSCAN_DATA_ROOT}/v6/marketscan_val` and contains `processed_data/` and `final_data/`.
- `VOCAB_MAPPING_DIR`: folder containing vocabulary mapping files generated from [`../../vocab_mapping/README.md`](../../vocab_mapping/README.md), usually `../../vocab_mapping/mapping` relative to this repository. Expected files are `icd9_to_icd10_mapping.csv`, `proc_to_snomed_mapping.csv`, and `ndc_drug_to_rxnorm_ingredient_mapping.csv`.
- `MARKETSCAN_DICTIONARY_DIR`: local MarketScan dictionary folder containing `PLANTYP.csv` and `EGEOLOC.csv`; `DICTIONARY_DIR` is derived from this local root
- `PAYER_LIST` and `TABLE_LIST`: default to `CCAE MDCR MDCD` and `t i o d`; the pipeline does not use the MarketScan `a` table
- `CONDA_ENV_NAME` or `CONDA_ENV_PATH`: Python environment containing PySpark, HuggingFace `datasets`, and `pandas`; defaults to the environment name `reclaim`

Run from this folder on a YCRC node:

```bash
cd preprocessing/versions/v6_marketscan
sbatch spark_preproces_v6.sh
```

---

## Output Directory Structure

```
INPUT_FOLDER_PATH/
├── processed_data/
│   ├── excluded_enrollids_ccae_mdcr/    # S0 output
│   ├── excluded_enrollids_mdcd/         # S0 output
│   ├── all_demographic_tokens/          # S1 output
│   ├── all_enrollment_tokens/           # S1 output
│   ├── all_claim_tokens/                # S1 output
│   ├── all_anchor_tokens/               # S1 output
│   ├── duration/                        # S1 output
│   ├── demo_seq/                        # S2 output
│   ├── event_seq_per_timestamp/         # S2 output
│   ├── _intermediate_sorted_tokens/      # S2 internal materialization
│   ├── trajectory/                      # S2 output
│   ├── all_att_tokens/                  # S2 output
│   ├── vocab/                           # S3 output
│   └── split_assignments/               # S4 split IDs and combined labels
└── final_data/
    ├── train/                           # S4 output
    ├── val/                             # S4 output
    └── test/                            # S4 output
```
