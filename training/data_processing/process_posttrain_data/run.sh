#!/bin/bash
data_version="v6"
data_source="full_combined"
data_size=100000

demo_end_index=3
min_att_tokens=2
min_ny_tokens=0

task="disease"
instruct_type="split"
split_by="random"
instruct_token="<INSTRUCT-DX>"
instruct_position="after"

data_dir=***->[REPLACE WITH YOUR PATH]<-***/${data_version}/marketscan_${data_source}
traj_path="${data_dir}/processed_data/trajectory"
split_path="${data_dir}/processed_data/split_assignments/overall_split"
base_dir=***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/${data_version}_marketscan_${data_source}_seqlen_4096/data/post_training_data/${task}
raw_output_dir=${base_dir}/raw/raw_size_${data_size}_${instruct_type}_pos_${instruct_position}_split_${split_by}

echo "Preparing post_training data for ${instruct_token}..."
python prepare_disease_instruct_data.py \
    --data_dir "$data_dir" \
    --output_dir "$raw_output_dir" \
    --traj_file "$traj_path" \
    --split_file "$split_path" \
    --split_name train \
    --demo_end_index "$demo_end_index" \
    --instruct_type "$instruct_type" \
    --split_by "$split_by" \
    --instruct_token "$instruct_token" \
    --instruct_position "$instruct_position" \
    --data_size "$data_size" \
    --min_ny_tokens "$min_ny_tokens" \
    --min_att_tokens "$min_att_tokens"

tokenizer_dir=***->[REPLACE WITH YOUR PATH]<-***

echo "Tokenizing post_training data..."
python tokenize_post_train_data.py \
    --base_dir "$base_dir" \
    --file_dir "$raw_output_dir" \
    --data_version "$data_version" \
    --data_source "$data_source" \
    --data_size "$data_size" \
    --tokenizer_dir "$tokenizer_dir" \
    --instruct_type "$instruct_type" \
    --split_by "$split_by" \
    --instruct_position "$instruct_position"