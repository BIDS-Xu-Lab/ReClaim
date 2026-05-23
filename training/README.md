# ReClaim Training

Pre-training and post-training of Qwen-3 causal language models on MarketScan claims trajectories. Pre-training learns a foundation model from tokenized patient sequences; post-training fine-tunes it on instruction-formatted tasks (e.g., disease prediction).

## Repository layout

```
qwen_training            # Model training (pre-train + post-train)
├── config
│   ├── ds_config.json           # DeepSpeed ZeRO-2 config
│   ├── pre_train_config.json    # Pre-training hyperparameters
│   └── post_train_config.json   # Post-training hyperparameters
├── loggers
│   └── loggers.py               # CSV / lightweight-save / loss-spike callbacks
├── models
│   ├── init_model.py            # Qwen3 config builder (S / M / L)
│   └── data_collator.py         # Padding collator for post-training
├── train.py                     # Main entry: pre_training or post_training
└── train.sh                     # torchrun launcher

data_processing              # Data preparation
├── process_pretrain_data
│   ├── build_hf_tokenizer.py    # Build WordLevel HF tokenizer from vocab parquet
│   ├── tokenzie_raw_data.py     # Tokenize + pack sequences into max_seq_len blocks
│   └── run_tok.sh
└── process_posttrain_data
    ├── data_ops.py                       # Sequence splitting / instruction helpers
    ├── prepare_disease_instruct_data.py  # Spark job: build <INSTRUCT-DX> sequences
    ├── tokenize_post_train_data.py       # Tokenize + build labels (loss only after split)
    └── run.sh
```

## Pipeline

1. **Tokenizer** — `build_hf_tokenizer.py` wraps the MarketScan vocab in a HuggingFace `PreTrainedTokenizerFast` with `<unk>/<sos>/<eos>/<pad>` specials.
2. **Pre-training data** — `tokenzie_raw_data.py` tokenizes raw `seq` text and packs into fixed-length blocks (default 4096).
3. **Pre-training** — `train.py --training_mode pre_training` initializes a Qwen3 model from scratch (`qwen3-S/M/L`) and trains with causal LM + z-loss regularization.
4. **Post-training data** — `prepare_disease_instruct_data.py` filters trajectories (sex / DOBYR / ATT counts), splits each sequence at a visit / year / random point, inserts an `<INSTRUCT-DX>` token, and appends the held-out major-disease tokens as targets. `tokenize_post_train_data.py` tokenizes and masks labels so loss is computed only past the split point.
5. **Post-training** — `train.py --training_mode post_training` loads the pretrained checkpoint and fine-tunes on the instruction dataset.

## Running

Edit the `***->[REPLACE WITH YOUR PATH]<-***` placeholders in the shell scripts, then:

```bash
# Build tokenizer + tokenize pre-training corpus
bash data_processing/process_pretrain_data/run_tok.sh

# Build + tokenize post-training instruction data
bash data_processing/process_posttrain_data/run.sh

# Train (set TRAINING_MODE=pre_training or post_training in the script)
bash qwen_training/train.sh
```
