#!/bin/bash
# ReClaim YNHH External Validation Pipeline
# Runs all preprocessing stages sequentially

set -e  # Exit on any error

echo "=============================================="
echo "ReClaim YNHH External Validation Pipeline"
echo "=============================================="
echo ""

# Activate conda environment
echo "Activating conda environment: reclaim"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reclaim

# Configuration - modify these paths as needed
VERSION=6
YNHH_DATA_ROOT="${YNHH_DATA_ROOT:-/path/to/ynhh_data_root}"
PROCESSED_YNHH_DATA_ROOT="${PROCESSED_YNHH_DATA_ROOT:-/path/to/processed_ynhh_data_root}"
MARKETSCAN_DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR:-/path/to/marketscan_dictionary_dir}"

OMOP_DIR="${YNHH_DATA_ROOT}/ynhh_omop"
VOCAB_MAPPING_DIR="../../vocab_mapping/mapping"
DICTIONARY_DIR="${MARKETSCAN_DICTIONARY_DIR}"
EXTERNAL_VAL_DATA_DIR="${PROCESSED_YNHH_DATA_ROOT}/v${VERSION}/ynhh"
SAMPLE_SIZE=110000


echo "Configuration:"
echo "  YNHH_DATA_ROOT: $YNHH_DATA_ROOT"
echo "  PROCESSED_YNHH_DATA_ROOT: $PROCESSED_YNHH_DATA_ROOT"
echo "  OMOP_DIR: $OMOP_DIR"
echo "  VOCAB_MAPPING_DIR: $VOCAB_MAPPING_DIR"
echo "  MARKETSCAN_DICTIONARY_DIR: $MARKETSCAN_DICTIONARY_DIR"
echo "  DICTIONARY_DIR: $DICTIONARY_DIR"
echo "  EXTERNAL_VAL_DATA_DIR: $EXTERNAL_VAL_DATA_DIR"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Stage 0: Data Filtering
echo "=============================================="
echo "Stage 0: Data Filtering"
echo "=============================================="
python3 s0_data_filtering.py \
    --omop_dir ${OMOP_DIR} \
    --mapping_dir ${VOCAB_MAPPING_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --sample_size ${SAMPLE_SIZE}
echo "Stage 0 completed!"
echo ""

# Stage 1: Construct Tokens
echo "=============================================="
echo "Stage 1: Construct Tokens"
echo "=============================================="
python3 s1_construct_tokens.py \
    --omop_dir ${OMOP_DIR} \
    --mapping_dir ${VOCAB_MAPPING_DIR} \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 1 completed!"
echo ""

# Stage 2: Construct Sequences
echo "=============================================="
echo "Stage 2: Construct Sequences"
echo "=============================================="
python3 s2_construct_seqences.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 2 completed!"
echo ""

# Stage 3: Finalize Vocabulary
echo "=============================================="
echo "Stage 3: Finalize Vocabulary"
echo "=============================================="
python3 s3_finalize_vocab.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR} \
    --dictionary_dir ${DICTIONARY_DIR}
echo "Stage 3 completed!"
echo ""

# Stage 5: Sanity Check
echo "=============================================="
echo "Stage 5: Sanity Check"
echo "=============================================="
python3 s5_sanity_check.py \
    --external_val_data_dir ${EXTERNAL_VAL_DATA_DIR}
echo "Stage 5 completed!"
echo ""

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Output directory: ${EXTERNAL_VAL_DATA_DIR}"
echo "Logs saved to: logs/"
