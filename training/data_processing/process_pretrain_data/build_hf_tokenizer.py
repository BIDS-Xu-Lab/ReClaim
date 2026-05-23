import os
import re
import json
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, decoders, trainers
import collections

import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_version", type=str, default="v4")
    p.add_argument("--data_size", type=str, default="1000k")
    return p.parse_args()

args = parse_args()
data_version = args.data_version
data_size = args.data_size

vocab_df = pd.read_parquet(f'***->[REPLACE WITH YOUR PATH]<-***/reclaim_data/{data_version}/marketscan_{data_size}/processed_data/vocab')

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

vocab = {tok: idx for idx, tok in enumerate(token_list)}

tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))

tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, 
                                    unk_token="<unk>", 
                                    bos_token="<sos>", 
                                    eos_token="<eos>", 
                                    pad_token="<pad>")

print(hf_tok.encode("<sos> <DX-MAJOR_E11> <DX-MINOR_9> <DX-ICD10_E78.2> <INSTRUCT-DX> <INSTRUCT-COST> <eos>"))

hf_tok.save_pretrained(f"/***->[REPLACE WITH YOUR PATH]<-***/ReClaim_Pretraining/tokenizers/hf_tokenizer_full_wrapped_{data_version}_{data_size}")