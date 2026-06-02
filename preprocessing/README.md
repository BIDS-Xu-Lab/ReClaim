# ReClaim Data Processing

This folder contains scripts for generating vocabulary mappings and preprocessing MarketScan/EHR data into ReClaim-compatible token sequences.

## Contents

| Path | Description |
|------|-------------|
| [`vocab_mapping/`](vocab_mapping/README.md) | Creates diagnosis, procedure, drug, and payer mapping files used by the preprocessing pipelines |
| [`versions/v6_marketscan/`](versions/v6_marketscan/README.md) | Main v6 monthly MarketScan pipeline for CCAE, MDCR, and MDCD claims data |
| [`versions/v6_ehrshot/`](versions/v6_ehrshot/README.md) | EHRSHOT external-validation pipeline from original OMOP CDM data |
| [`versions/v6_ynhh/`](versions/v6_ynhh/README.md) | YNHH external-validation pipeline from original OMOP CDM data |

## Notes

- `v6` uses month-level timestamps and `<ATT-N>` tokens for elapsed months between event groups.
- `v6_marketscan` is the reference pipeline for shared tokenization, sequence construction, vocabulary finalization, and sanity checks.
- EHRSHOT and YNHH reuse the shared v6 logic and only document dataset-specific differences in their own READMEs.
- Raw data, MarketScan dictionary files, generated mappings, processed outputs, and logs are local artifacts and are not included in this repository.

## Typical Order

1. Run vocabulary mapping first to generate the mapping files required by downstream preprocessing:

```bash
cd preprocessing/vocab_mapping
./create_mappings.sh
```

2. Configure local root paths in the relevant pipeline shell script.

3. Run the desired pipeline, for example:

```bash
cd preprocessing/versions/v6_marketscan
sbatch spark_preproces_v6.sh
```

See each linked README for required inputs and pipeline-specific details.
