#!/bin/bash

set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# defaults
DATA_SOURCE="vf"
DATA_SIZE="full_combined" #(1000k or marketscan_full_combined)
MAX_SEQLEN=4096
INPUT_FOLDER="***->[REPLACE WITH YOUR PATH]<-***/reclaim_data/${DATA_SOURCE}/marketscan_${DATA_SIZE}"

SAVE_BASE="***->[REPLACE WITH YOUR PATH]<-***/tokenizers"
TOK_FOLDER="${SAVE_BASE}/hf_tokenizer_full_wrapped_${DATA_SOURCE}_${DATA_SIZE}"

python "${SCRIPT_DIR}/build_hf_tokenizer.py" \
    --input_folder_path "$INPUT_FOLDER" \
    --save_path "$SAVE_BASE" \
    --data_version "$DATA_SOURCE" \
    --data_size "$DATA_SIZE"

python "${SCRIPT_DIR}/tokenzie_raw_data.py" \
    --input_folder_path "$INPUT_FOLDER" \
    --tokenizer_path "$TOK_FOLDER" \
    --max_seq_len ${MAX_SEQLEN}
