#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

data_version="vf"
data_source="full_combined"
data_size=100000

demo_end_index=3
min_att_tokens=2
min_ny_tokens=0

split_by="random"
instruct_type="disease"
instruct_token="<INSTRUCT-DX>"
instruct_position="after"

data_dir="***->[REPLACE WITH YOUR PATH]<-***/reclaim_data/${data_version}/marketscan_${data_source}"
tok_dir="***->[REPLACE WITH YOUR PATH]<-***/tokenizers/hf_tokenizer_full_wrapped_${data_version}_full_combined"

traj_path="${data_dir}/processed_data/trajectory"
split_path="${data_dir}/processed_data/split_assignments/overall_split"

base_dir="***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/${data_version}_marketscan_${data_source}_seqlen_4096_prospect/data/post_training_data/${instruct_type}"
raw_output_dir="${base_dir}/raw/raw_size_${data_size}_pos_${instruct_position}_split_${split_by}"
token_folder="tokens"

echo "Preparing post_training data for ${instruct_token}..."
python "${SCRIPT_DIR}/prepare_disease_instruct_data.py" \
    --data_dir "$data_dir" \
    --output_dir "$raw_output_dir" \
    --traj_file "$traj_path" \
    --split_file "$split_path" \
    --split_name "train" \
    --demo_end_index "$demo_end_index" \
    --split_by "$split_by" \
    --instruct_token "$instruct_token" \
    --instruct_position "$instruct_position" \
    --data_size "$data_size" \
    --min_ny_tokens "$min_ny_tokens" \
    --min_att_tokens "$min_att_tokens" \
    --seed 1024

echo "Tokenizing post_training data..."
python "${SCRIPT_DIR}/tokenize_post_train_data.py" \
    --tok_dir "$tok_dir" \
    --base_dir "$base_dir" \
    --file_dir "$raw_output_dir" \
    --token_folder "$token_folder" \
    --data_version "$data_version" \
    --data_source "$data_source" \
    --data_size "$data_size" \
    --split_by "$split_by" \
    --instruct_position "$instruct_position"
