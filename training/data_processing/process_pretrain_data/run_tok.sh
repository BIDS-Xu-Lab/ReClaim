#!/bin/bash

# defaults
DATA_SOURCE="v6"
DATA_SIZE="full_combined"
MAX_SEQLEN=4096
INPUT_FOLDER="***->[REPLACE WITH YOUR PATH]<-***/reclaim_data/${DATA_SOURCE}/marketscan_${DATA_SIZE}"
TOK_FOLDER="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/tokenizers/hf_tokenizer_full_wrapped_${DATA_SOURCE}_${DATA_SIZE}"

python ***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/src_data_processing/process_pretrain_data/build_hf_tokenizer.py \
    --data_version ${DATA_SOURCE} \
    --data_size ${DATA_SIZE}

python ***->[REPLACE WITH YOUR PATH]<-***/src_data_processing/process_pretrain_data/tokenzie_raw_data.py \
    --input_folder_path "$INPUT_FOLDER" \
    --tokenizer_path "$TOK_FOLDER" \
    --max_seq_len ${MAX_SEQLEN}