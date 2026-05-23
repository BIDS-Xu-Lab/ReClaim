#!/usr/bin/env python3
"""
Pre-train and post-train Qwen-3 on MarketScan dataset
"""

import json
import os, datetime
import argparse
import torch
from pathlib import Path 
from datasets import load_from_disk

from transformers import (Qwen3ForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, 
                          Trainer, TrainingArguments)

from loggers.loggers import CSVLoggerCallback, LightweightSaveCallBack

from models.init_model import create_model_config
from models.data_collator import SimplePadCollator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_source", type=str, default="input")
    p.add_argument("--post_train_data_source", type=str, default="input")
    p.add_argument("--pretrained_model", type=str, default=" ")
    p.add_argument("--output_folder_name", type=str, default="output")
    p.add_argument("--training_mode", type=str, default="pre_training", 
                                        choices=["pre_training", "post_training"],
                                        help="Training mode to run.")
    p.add_argument("--model_type", type=str, default='qwen3-S',
                                    choices=['qwen3-L', 'qwen3-M', 'qwen3-S', 'gpt-2'],
                                    help="Model type to train.")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--deepspeed", type=str, default="ds_config.json")
    p.add_argument("--pre_train_config", type=str, default="pre_train_config.json")
    p.add_argument("--post_train_config", type=str, default="post_train_config.json")
    p.add_argument("--post_train_task", type=str, default="disease")
    p.add_argument("--resume_from_checkpoint", type=str, default="false")

    return p.parse_args()

args = parse_args()

# 0. Set up paths
base = f"***->[REPLACE WITH YOUR PATH]<-***/ReClaim_TrainingData/{args.data_source}"
data_base = f"{base}/data"
save_base = f"{base}/model"
log_dir = str(f"{save_base}/{args.output_folder_name}/{args.model_type}/training_logs")
light_model_dir = f"{save_base}/{args.output_folder_name}/{args.model_type}/staged_models"
final_out = f"{save_base}/{args.output_folder_name}/{args.model_type}/final_models"
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(light_model_dir).mkdir(parents=True, exist_ok=True)
Path(final_out).mkdir(parents=True, exist_ok=True)

# 1. Load tokenizer, datasets, and training configurations
if args.training_mode == "pre_training":
    print("="*20)
    print("Pre-training: loading tokenizer, datasets, and training configurations")
    print("="*20)

    train_ds = load_from_disk(f"{data_base}/pre_training_data/train_tokens")
    eval_ds = load_from_disk(f"{data_base}/pre_training_data/val_tokens")

    print(f"Pre-training set size: {len(train_ds)}")
    print(f"Validation set size: {len(eval_ds)}")

    with open(args.pre_train_config) as f:
        training_config = json.load(f)

elif args.training_mode == "post_training":
    print("="*20)
    print("Post-training: loading tokenizer, datasets, and training configurations")
    print("="*20)

    train_ds = load_from_disk(f"{data_base}/post_training_data/{args.post_train_task}/tokens/{args.post_train_data_source}")
    print(f"Post-training set size: {len(train_ds)}")

    with open(args.post_train_config) as f:
        training_config = json.load(f)

# save training config file to model folder
with open(f"{save_base}/{args.output_folder_name}/{args.model_type}/training_config.json", "w") as f:
    json.dump(training_config, f)
    
# 2. Create model configuration based on model type
if args.training_mode == "pre_training":
    print("="*20)
    print("Pre-training: creating model configuration")
    print("="*20)

    tokenizer_path = f"{base}/tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{base}/tokenizer")
    print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id,
            "eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)

    cfg = create_model_config(
        args.model_type, 
        tokenizer, 
        training_config["max_seq_len"],
    )
    model = AutoModelForCausalLM.from_config(cfg)
    print(cfg)
    print(f"Number of parameters in the model: {model.num_parameters()/1e6:.2f}M")

elif args.training_mode == "post_training":
    print("="*20)
    print("Post-training: loading model from checkpoint")
    print("="*20)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{args.pretrained_model}")
    print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id,
            "eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    print(f"Number of parameters in the model: {model.num_parameters()/1e6:.2f}M")

# 3. Set up training arguments
training_args = TrainingArguments(
    output_dir = str(f"{save_base}/{args.output_folder_name}/{args.model_type}/checkpoints_{args.training_mode}"), 
    overwrite_output_dir = True,
    num_train_epochs = training_config["num_train_epochs"],
    per_device_train_batch_size = training_config["train_micro_batch_size_per_gpu"],
    gradient_accumulation_steps = training_config["gradient_accumulation_steps"],
    max_grad_norm = training_config["gradient_clipping"],
    # LOGGING AND SAVING
    logging_steps = 2,
    save_steps = training_config["ckpt_save_steps"],
    save_strategy="steps",
    save_total_limit = 1,
    save_on_each_node = False,
    # VALIDATION CONFIGS
    eval_strategy = "steps" if args.training_mode == "pre_training" else "no",
    eval_steps = training_config["eval_steps"],
    # TRAINING CONFIGS
    optim = "adamw_torch",
    learning_rate = training_config["lr"],
    lr_scheduler_type = "cosine",
    weight_decay = training_config["weight_decay"],
    warmup_steps = training_config["warmup_steps"],
    dataloader_num_workers = 8,
    remove_unused_columns = False,
    # DEEPSPEED
    bf16 = True,
    torch_compile = False,
    deepspeed = args.deepspeed,
    ddp_find_unused_parameters = False
)

class ZLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.get("loss")
        logits = outputs.get("logits")

        z_loss = 0.0
        if logits is not None:
            labels = inputs.get("labels")
            logits_float = logits.float()
            log_z = torch.logsumexp(logits_float, dim=-1)

            if labels is not None:
                mask = (labels != -100)
                valid_log_z = log_z[mask]

                if valid_log_z.numel() > 0:
                    z_loss = 1e-4 * torch.mean(valid_log_z ** 2)

                else:
                    z_loss = 0.0

            else:
                z_loss = 0.0

        total_loss = ce_loss + z_loss

        return (total_loss, outputs) if return_outputs else total_loss

# 4. Set up callbacks
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

CSVCallback = CSVLoggerCallback(f"{log_dir}/{args.training_mode}_log_{timestamp}.csv")
StageCallback = LightweightSaveCallBack(
                                        output_dir=os.path.join(light_model_dir, args.training_mode),
                                        save_every_steps=training_config["light_weightsave_steps"]
                                        )

# 5. Initialize trainer for pre-training
if args.training_mode == "pre_training":
    collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)

elif args.training_mode == "post_training":
    collator = SimplePadCollator(
        pad_token_id=tokenizer.pad_token_id, 
        pad_to_multiple_of=8
    )

trainer = ZLossTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_ds,
    eval_dataset=eval_ds if args.training_mode == "pre_training" else None,
    tokenizer=tokenizer,
    data_collator=collator
)

StageCallback.trainer = trainer
trainer.add_callback(CSVCallback)
trainer.add_callback(StageCallback)

if args.resume_from_checkpoint == "false":
    trainer.train()
else:
    trainer.train(resume_from_checkpoint=True)

if args.training_mode == "pre_training":
    trainer.save_model(os.path.join(final_out, args.training_mode))
    tokenizer.save_pretrained(os.path.join(final_out, args.training_mode))
elif args.training_mode == "post_training":
    trainer.save_model(os.path.join(final_out, args.training_mode, args.post_train_data_source))
    tokenizer.save_pretrained(os.path.join(final_out, args.training_mode, args.post_train_data_source))

print(f"{args.training_mode} completed. Saved model to {final_out}")
print(f"Tokenizer saved to {final_out}")