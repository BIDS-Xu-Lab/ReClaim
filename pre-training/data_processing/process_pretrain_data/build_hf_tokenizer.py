import os
import re
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, decoders, trainers
import collections

parser = argparse.ArgumentParser(description="Preprocess text data for pretraining")
parser.add_argument("--input_folder_path", type=str, required=True, help="Path to the input folder containing corpus and tokenizer")
parser.add_argument("--save_path", type=str, required=True, help="Path to the tokenizer")
parser.add_argument("--data_version", type=str, default="v4", help="data_version")
parser.add_argument("--data_size", type=str, default="1000k", help="data_size")
args = parser.parse_args()

vocab_df = pd.read_parquet(f'{args.input_folder_path}/processed_data/vocab')

# drop duplicates
# vocab_df = vocab_df.drop_duplicates(subset=["token"])

vocab_list = vocab_df["token"].to_list()
vocab_list = [str(t) for t in vocab_list]

# chceck duplicates token in the vocab list
duplicates = [item for item, count in collections.Counter(vocab_list).items() if count > 1]
print("Duplicated tokens: ", duplicates)

for token in duplicates:
    print(vocab_df[vocab_df["token"] == token])

special_tokens = ["<unk>", "<sos>", "<eos>", "<pad>"]

token_list = special_tokens + vocab_list
print(len(token_list))

# add <DX-Instruct> token if not already in the list
# if "<Instruct-DX>" not in token_list:   
#     print("Adding <DX-Instruct> token")
#     token_list.append("<DX-Instruct>")

vocab = {tok: idx for idx, tok in enumerate(token_list)}

tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))

tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, 
                                    unk_token="<unk>", 
                                    bos_token="<sos>", 
                                    eos_token="<eos>", 
                                    pad_token="<pad>")

print(hf_tok.encode("<sos> <DX-MAJOR_E11> <DX-MINOR_9> <DX-ICD10_E78.2> <INSTRUCT-DX> <INSTRUCT-COST> <eos>"))

hf_tok.save_pretrained(f"{args.save_path}/hf_tokenizer_full_wrapped_{args.data_version}_{args.data_size}")
