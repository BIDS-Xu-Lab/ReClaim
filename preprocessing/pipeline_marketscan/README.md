# ReClaim Data Preprocessing: MarketScan Pipeline

A PySpark-based data preprocessing pipeline for transforming MarketScan healthcare claims data into tokenized sequences for foundation model training. The provided Slurm script is configured for running Spark jobs on the Yale Center for Research Computing (YCRC) cluster.

The pipeline uses monthly temporal resolution. Events are timestamped at the month level, and Artificial Time Tokens (`<ATT-N>`) represent elapsed months from 0 to 12.

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
| **ATT** | Artificial Time Token: `<ATT-N>` indicates N **months** until the next event group. `<ATT-0>` can separate multiple event groups in the same month. |

**Event Types & Ordering (within same timestamp unit):**

CCAE and MDCR are linkable (same person, shared `CR` enrollee_id), so a coverage transition can place claims from both payers in the same month. To avoid collapsing two payers into one event group, each event group carries a `payer_order` (CCAE=0, MDCR=1; anchor tokens use `0` and still sort first because they are the only `event_type = 0` tokens; MDCD is independent and uses a distinct `2`). Within a timestamp, groups are ordered by `payer_order` **first**, then `event_type`, so during a transition the CCAE chapter (including `ERLED-CCAE`) is emitted before the MDCR chapter (starting with `ERLST-MDCR`). Same-type payer groups in one month are separated by `<ATT-0>`.

Within a single payer's event group, tokens are ordered by `event_type`, then by `category_priority` within each type:

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

For detailed mapping methodology and source files, see [`../vocab_mapping/README.md`](../vocab_mapping/README.md).

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

**Aggregation Within Month:**

Aggregation happens **per payer** (the aggregation key includes `payer_order`). CCAE and MDCR claims that fall in the same month are therefore kept as two separate event groups (CCAE first, then MDCR, separated by `<ATT-0>`) rather than merged; costs are summed within each payer's group, not across payers. Within a single payer's group, the following rules apply:

| Token Type | Aggregation Rule |
|------------|------------------|
| **COST** | Sum of all costs within the month (per payer) |
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

Stage 2 first sorts event groups by their assigned monthly `timestamp`, then by `payer_order` and `event_type`. Within each group it applies the Stage 1 event ordering (`category_priority` and `sub_order`) to serialize anchor, enrollment, visit, clinical, cost, and enrollment-end tokens. `original_date`, `original_token`, and `comb_order` are then used as deterministic tie-breakers so repeated or grouped tokens have stable within-month order. The same `(timestamp, payer_order, event_type)` order is used to compute the `<ATT-N>` gaps, so the inter-month gap deterministically attaches to the last group of the month and `<ATT-0>` separates groups within the month.

**Temporal Tokens:**
- `<ATT-N>`: Appended after each monthly event group to indicate **N months** until the next event group (uses `lead` for a cleaner cutoff split)
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

**Training Cutoff Split:**
- The split point is controlled by the `--training_cutoff_year` argument (set in `run_marketscan.sh`, default `2023`). Events with `year >= training_cutoff_year` are held out for prospective evaluation; everything before is the pretraining corpus. Column names are cutoff-relative (not hardcoded to a specific year). With the default cutoff of `2023`, the boundary falls at the 2022/2023 transition.
- `seq_thru_cutoff`: Self-contained pretraining sequence covering all events from the start of follow-up **through the end of (`training_cutoff_year` - 1)**, i.e. December 31 of the year before the cutoff. With the default cutoff `2023`, this means everything up to and including December 31, **2022**. It is wrapped as `<sos> [demographics] [events through Dec 31 of cutoff-1] <eos>` (the events portion includes the trailing ATT that bridges to the first event in `training_cutoff_year`). Enrollees whose first event is at/after the cutoff have no pre-cutoff events, so their `seq_thru_cutoff` is demographics-only (`<sos> [demographics] <eos>`); these enrollees are retained (S0 does not drop late-starting enrollees).
- `event_seq_from_cutoff`: All events from January 1 of `training_cutoff_year` onwards (with the default cutoff `2023`, this is **2023-01-01 and later**); always starts with a `<NY>` token.
- `first_token_index_from_cutoff`: 0-based index pointing to the first from-cutoff `<NY>` token for prediction tasks
  - `NULL` if no from-cutoff data OR no thru-cutoff data (need historical context to predict)
  - Formula: `token_count_thru_cutoff` (since `token_count_thru_cutoff` already includes `<sos>` + 2 demo tokens)

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
- training-cutoff split boundary correctness when `first_token_index_from_cutoff` is available

**Outputs:**
- Log output only; this stage does not create model input files

---

## Running the Pipeline

The full MarketScan corpus is large: the study uses CCAE, MDCR, and MDCD data with more than 200 million enrollees and 43.8 billion claims records, and the local raw Parquet layout is approximately 1.7 TB. For the full dataset processing, it is recommended to use Spark on a distributed server with Slurm allocation.

`run_marketscan.sh` is written for a Slurm cluster whose Spark module provides `spark-start` (the Yale Center for Research Computing setup). It loads the module named by `SPARK_MODULE`, activates the conda environment `RECLAIM_ENV` (Python 3.11 with PySpark, HuggingFace `datasets` and `pandas`; see [`../environment.yml`](../environment.yml)), starts a Spark standalone cluster inside the allocation, runs S0-S5 in order and stops at the first failed stage. Slurm output goes to `slurm_job_logs/`, per-stage logs to `logs/`.

All paths come from `../env.sh` (copy [`../env.sh.example`](../env.sh.example)):
- `MS_RAW_DIR`: raw MarketScan Parquet files (2008-2024), one folder per payer (`CCAE/`, `MDCR/`, `MDCD/`), with enrollment detail (`t`), inpatient admissions (`i`), outpatient services (`o`) and pharmacy claims (`d`). The pipeline does not use the `a` table.
- `MS_PROCESSED_ROOT`, `DATA_VERSION`, `MS_RUN_NAME`: outputs are written to `$MS_PROCESSED_ROOT/$DATA_VERSION/$MS_RUN_NAME/` (`processed_data/` and `final_data/`).
- `VOCAB_MAPPING_DIR`: mapping files generated by [`../vocab_mapping`](../vocab_mapping/README.md) (default `../vocab_mapping/mapping`): `icd9_to_icd10_mapping.csv`, `proc_to_snomed_mapping.csv` and `ndc_drug_to_rxnorm_ingredient_mapping.csv`.
- `MS_DICTIONARY_DIR`: the MarketScan dictionary folder (`MSN_LOOKUP`); S3 reads `PLANTYP.csv` and `EGEOLOC.csv`.

The stage parameters (minimum member days 180, minimum claims duration 30 days, training cutoff year 2023, executor sizing) are set in the driver itself.

```bash
cd pipeline_marketscan
sbatch run_marketscan.sh
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
