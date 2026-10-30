# ReClaim data preprocessing

Code that turns MarketScan claims (and an external OMOP CDM source) into the token sequences ReClaim is trained
and evaluated on.

| Module | What it does | Data |
|---|---|---|
| `vocab_mapping/` | one-time code standardisation: ICD-9-CM to ICD-10-CM (CMS GEMs), CPT4/ICD9CM/ICD10PCS to SNOMED and NDC to RxNorm ingredient (OHDSI vocabulary) | CMS GEMs, OHDSI Athena vocabulary |
| `pipeline_marketscan/` | MarketScan claims to token sequences and the train/val/test split, stages `s0`-`s5` on a Spark cluster | MarketScan CCAE, MDCR, MDCD |
| `pipeline_omop/ehrshot/` | the same tokenisation for the EHRSHOT OMOP CDM export (stages `s0`-`s3`, `s5`; no split) | EHRSHOT (OMOP CDM) |
| `pipeline_omop/ynhh/` | the same tokenisation for the YNHH OMOP CDM export, full cohort or a sampled subset (no split) | YNHH EHR (OMOP CDM) |

Stages: `s0` enrollee filtering, `s1` token construction with the vocabulary mappings, `s2` monthly sequences with
`<ATT-N>` time tokens and the training-cutoff split columns, `s3` vocabulary, `s4` train/val/test split (MarketScan only),
`s5` sanity checks. Each module README documents the token format and every stage.

## Setup

```bash
conda env create -f environment.yml && conda activate reclaim-preprocessing
cp env.sh.example env.sh                         # data roots, conda env, Spark module, output location
```

## Run

```bash
(cd vocab_mapping && ./create_mappings.sh)       # once; writes vocab_mapping/mapping/
(cd pipeline_marketscan && sbatch run_marketscan.sh)    # -> $MS_PROCESSED_ROOT/$DATA_VERSION/$MS_RUN_NAME
(cd pipeline_omop/ehrshot && sbatch run_ehrshot.sh)     # -> $MS_PROCESSED_ROOT/$DATA_VERSION/$EHRSHOT_RUN_NAME
(cd pipeline_omop/ynhh && sbatch run_ynhh_full.sh)      # -> $MS_PROCESSED_ROOT/$DATA_VERSION/$YNHH_FULL_RUN_NAME
```

The drivers are written for a Slurm cluster with a Spark module that provides `spark-start` (Yale Center for Research
Computing); `pipeline_marketscan/` needs a multi-node allocation for the full corpus, `pipeline_omop/ehrshot/` runs local-mode Spark
on one node, and `pipeline_omop/ynhh/` either (`run_ynhh_full.sh` multi-node, `run_ynhh_subset.sh` one node).
`pipeline_omop/payer_concept_annotated.csv` maps OMOP payer concepts to MarketScan payer and plan type for both OMOP sources.

## Data

No data, vocabularies, mappings or outputs are distributed. MarketScan data and its dictionary (`MSN_LOOKUP`) need a
Merative license, OMOP sources their own data use agreement, and the OHDSI vocabulary is downloaded from Athena (CPT4
needs a UMLS license); `vocab_mapping/README.md` lists the sources. The generated mappings (`vocab_mapping/mapping/`),
run logs and `env.sh` are gitignored.
