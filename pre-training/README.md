# ReClaim Pre-training

Build a WordLevel Hugging Face tokenizer, tokenize prospective patient sequences,
and pre-train a Qwen3 causal language model. This directory replaces `training/`.
Disease instruction-data preparation and the post-training launcher now live in
[`downstream_tasks/disease_prediction/post-training/`](../downstream_tasks/disease_prediction/post-training/README.md).

## Layout

```text
pre-training/
├── data_processing/process_pretrain_data/
│   ├── build_hf_tokenizer.py
│   ├── tokenzie_raw_data.py
│   └── run_tok.sh
└── qwen_training/
    ├── config/                 # DeepSpeed and both training-mode configurations
    ├── files/                  # ICD mapping used by disease post-training
    ├── loggers/
    ├── models/
    ├── train_zloss.py          # Shared pre_training / post_training entry point
    └── train.sh                # Pre-training Slurm launcher
```

The Hopper `src_all_qwen_training/` contents are in `qwen_training/`, except
`post_train.sh`, which is in the disease-prediction post-training directory.
The Hopper pre-training launcher retains its name, `train.sh`; it replaces the
old `pre_train.sh`.

## Configure local paths

Replace every `***->[REPLACE WITH YOUR PATH]<-***` in the files you will run with
your own absolute path. Keep the surrounding quotes. The placeholder identifies
a configurable path prefix; retain or adjust the suffix to match your storage.
Repository-internal file references are resolved from the checkout or script
location and contain no personal paths.

| File | Path settings |
|------|---------------|
| `data_processing/process_pretrain_data/run_tok.sh` | `INPUT_FOLDER`: processed corpus; `SAVE_BASE`: tokenizer output directory. |
| `data_processing/process_pretrain_data/tokenzie_raw_data.py` | `base_dir`: tokenized dataset and tokenizer-copy output. |
| `qwen_training/train_zloss.py` | `base` and `post_train_base`: use the same `ReClaim_TrainingData` root as data preparation. |
| `qwen_training/train.sh` | `REPO_ROOT`: the absolute path to your repository checkout. |

The corpus directory must contain `processed_data/vocab`, `final_data/train/`,
and `final_data/val/`. Tokenization reads `seq_thru_cutoff`. Keep the corpus
directory naming (`<data_version>/marketscan_<data_size>`), `DATA_SOURCE`, and the
generated `<data_version>_marketscan_<data_size>_seqlen_4096_prospect` directory
consistent across tokenization and training.

## Run

From the repository root, after configuring the paths:

```bash
bash pre-training/data_processing/process_pretrain_data/run_tok.sh
mkdir -p logs/pre-train
sbatch pre-training/qwen_training/train.sh
```

The launcher expects Slurm, eight GPUs per node, and the indicated CUDA/GCC
modules. Adjust the resource directives, modules, and network settings for your
cluster before submission. Slurm log directories must exist before `sbatch`.
Python dependencies include PyTorch, Transformers with Qwen3 support, DeepSpeed,
Datasets, Tokenizers, pandas, and PyArrow.

Training configurations are in `qwen_training/config/`. Model checkpoints,
tokenizers, logs, and training summaries are written under the configured data
root. Post-training reuses this trainer and its configurations; see the
[post-training instructions](../downstream_tasks/disease_prediction/post-training/README.md).
