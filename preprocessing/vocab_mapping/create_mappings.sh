#!/bin/bash
# Run all vocabulary mapping scripts
# This script uses relative paths from preprocessing/vocab_mapping.
# It generates final mapping files in mapping/ plus intermediate audit files.

set -e  # Exit on error

echo "============================================"
echo "Vocabulary Mapping Pipeline"
echo "============================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
GEMS_FILE="standard_vocab/diagnosis_gems_2018/2018_I9gem.txt"
OHDSI_VOCAB_DIR="standard_vocab/ohdsi_vocab_v20250827"
OUTPUT_DIR="mapping"

echo "Configuration:"
echo "  GEMS_FILE: $GEMS_FILE"
echo "  OHDSI_VOCAB_DIR: $OHDSI_VOCAB_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

echo "Step 1/3: Creating ICD-9 to ICD-10 diagnosis mapping..."
echo "----------------------------------------"
python create_dx_mapping.py \
  --gems_file "$GEMS_FILE" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "Step 2/3: Creating procedure codes to SNOMED mapping..."
echo "----------------------------------------"
python create_proc_mapping.py \
  --ohdsi_vocab_dir "$OHDSI_VOCAB_DIR" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "Step 3/3: Creating NDC to RxNorm ingredient mapping..."
echo "----------------------------------------"
python create_rx_mapping.py \
  --ohdsi_vocab_dir "$OHDSI_VOCAB_DIR" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "All mappings completed successfully!"
echo "============================================"
echo ""
echo "Final mapping files used by MarketScan preprocessing:"
echo "  - $OUTPUT_DIR/icd9_to_icd10_mapping.csv"
echo "  - $OUTPUT_DIR/proc_to_snomed_mapping.csv"
echo "  - $OUTPUT_DIR/ndc_drug_to_rxnorm_ingredient_mapping.csv"
echo ""
echo "Additional mapping files used for external validation or audit:"
echo "  - $OUTPUT_DIR/rxnorm_drug_to_rxnorm_ingredient_mapping.csv"
echo "  - $OUTPUT_DIR/cpt_to_snomed_procedure_mapping.csv"
echo "  - $OUTPUT_DIR/icd9cm_to_snomed_procedure_mapping.csv"
echo "  - $OUTPUT_DIR/icd10pcs_to_snomed_procedure_mapping.csv"
echo "  - $OUTPUT_DIR/ndc_drug_to_rxnorm_drug_mapping.csv"
echo "  - $OUTPUT_DIR/*_original.csv"
echo ""
echo "Logs available in: logs/"
echo ""
