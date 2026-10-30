# Disease Prediction Post-training

Prepare disease instruction examples and fine-tune a pre-trained ReClaim model.
These files come from the Hopper `src_data_processing/process_posttrain_data/`
and `src_all_qwen_training/post_train.sh`.

```text
post-training/
├── process_posttrain_data/
│   ├── data_ops.py
│   ├── prepare_disease_instruct_data.py
│   ├── tokenize_post_train_data.py
│   └── run.sh
└── post_train.sh
```

The shared trainer, model helpers, callbacks, ICD mapping, and DeepSpeed and
training configurations live in
[`pre-training/qwen_training/`](../../../pre-training/qwen_training/).

## Configure local paths

Replace every `***->[REPLACE WITH YOUR PATH]<-***` in the files you will run with
your own absolute path. Keep the surrounding quotes and adjust any suffix to
match your storage.

| File | Path settings |
|------|---------------|
| `process_posttrain_data/run.sh` | `data_dir`: processed corpus; `tok_dir`: tokenizer; `base_dir`: instruction-data output. |
| `process_posttrain_data/prepare_disease_instruct_data.py` | `--java_home` default: your Java installation for Spark. |
| `post_train.sh` | `REPO_ROOT`: repository checkout; `PRE_TRAIN_MODEL`: pre-trained model checkpoint including its tokenizer. |
| `../../../pre-training/qwen_training/train_zloss.py` | `base` and `post_train_base`: the same `ReClaim_TrainingData` root used for tokenization and instruction-data output. |

The ICD map is resolved automatically from
`pre-training/qwen_training/files/icd_code_map_with_category.csv`.
Keep `data_version`, `data_source`, `data_size`, and `split_by` in `run.sh`
consistent with `DATA_VERSION`, `DATA_SOURCE`, `POST_TRAIN_DATA_BASE`,
`POST_TRAIN_SIZE`, and `SPLIT` in `post_train.sh`.

## Run

From the repository root, after configuring the paths:

```bash
bash downstream_tasks/disease_prediction/post-training/process_posttrain_data/run.sh
mkdir -p logs/post-train
sbatch downstream_tasks/disease_prediction/post-training/post_train.sh
```

Data preparation requires PySpark/Java, pandas, PyArrow, Datasets, and
Transformers. It reads `processed_data/trajectory` and
`processed_data/split_assignments/overall_split`, selects the training split,
and produces instruction examples using `seq_thru_cutoff`. Tokenization masks
the prompt and supervises only the completion. The default maximum length is
4096; keep it consistent with the trainer configuration.

The launcher uses the shared `train_zloss.py --training_mode post_training` and
`config/post_train_config.json`. It expects Slurm and eight GPUs; adjust resource
directives and CUDA modules for your cluster. Create the Slurm log directory
before submission. For evaluation after training, see the
[disease prediction README](../README.md).
