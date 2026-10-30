#!/usr/bin/env python3
"""
Post-training (final design, "posttrain"): token-native distributional cost readout
over the existing mantissa <COST-me> bins.

Fine-tunes a pretrained ReClaim model on instruct pairs
(context -> <INSTRUCT-COST> tok_total <eos>)   [one response token]
built by build_cost_posttrain_data.py.

Loss (response positions only; prompt and <INSTRUCT-COST> are masked):
  - bin positions: soft ordinal cross-entropy over the bin sub-vocabulary
      target = Gaussian over bin indices (sigma bins), HL-Gauss style
    + ev_weight * Huber( E_softmax[log1p dollars] - log1p(true dollars) )
      The expectation is over the bins' own dollar values (BinSpec.ev_dollars).
  - <eos> position: standard cross-entropy.

Vocabulary: the response reuses the existing <COST-me> tokens (bin id =
quantized dollars, value-ordered); no vocabulary change, no task head.
Full fine-tune of the backbone; torchrun DDP when launched with >1 GPU.
"""

import os
import json
import math
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, disable_caching
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

disable_caching()

from utils import setup_logging
from sequence_parsing import GOLD_COST_COLS, parse_cost_value
from posttrain_bins import BinSpec

RESPONSE_SLOTS = len(GOLD_COST_COLS) + 1  # bin token(s) + <eos>; 1 + 1 = 2


def parse_args():
    p = argparse.ArgumentParser(description="posttrain cost-instruct post-training")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--pairs_dir", type=str, required=True,
                   help="Dir with train_pairs.parquet / val_pairs.parquet")
    p.add_argument("--bin_artifact", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Ordinal label-smoothing width in bin indices")
    p.add_argument("--ev_weight", type=float, default=0.5,
                   help="Weight of the expected-value auxiliary loss")
    p.add_argument("--z_loss_weight", type=float, default=1e-4,
                   help="z-loss on supervised positions; 1e-4 matches pre-training "
                        "and the disease post-training (sensitivity: negligible "
                        "effect; kept for framework consistency; 0 = off)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--max_train_pairs", type=int, default=None,
                   help="Subset for smoke tests")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint in output_dir if present")
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================

def build_dataset(pairs_path, tokenizer, seq_len, max_pairs=None):
    ds = load_dataset("parquet", data_files=pairs_path, split="train")
    if max_pairs is not None and len(ds) > max_pairs:
        ds = ds.select(range(max_pairs))
    logging.info("  Loaded %d pairs from %s", len(ds), pairs_path)

    max_prompt = seq_len - (RESPONSE_SLOTS + 1)  # + <INSTRUCT-COST>

    def encode(ex):
        prompt_ids = tokenizer(ex["context_seq"] + " <INSTRUCT-COST>",
                               add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_prompt + 1:
            prompt_ids = prompt_ids[-(max_prompt + 1):]
        # response after the instruct token: 4 bin tokens + <eos>
        resp_text = " ".join(ex["response_seq"].split()[1:])  # drop <INSTRUCT-COST>
        resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
        assert len(resp_ids) == RESPONSE_SLOTS, \
            f"response tokenised to {len(resp_ids)} ids: {resp_text}"
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "dollar_labels": [math.log1p(min(ex[tc], 1_000_000)) for tc in GOLD_COST_COLS],
        }

    keep = {"input_ids", "labels", "dollar_labels"}
    ds = ds.map(encode, num_proc=max(1, (os.cpu_count() or 8) // 2),
                remove_columns=[c for c in ds.column_names if c not in keep],
                desc="Tokenising pairs")
    return ds


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels, dollars = [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = maxlen - n
            # left padding keeps response positions at the sequence end
            input_ids.append([self.pad_id] * pad + f["input_ids"])
            attn.append([0] * pad + [1] * n)
            labels.append([-100] * pad + f["labels"])
            dollars.append(f["dollar_labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "dollar_labels": torch.tensor(dollars, dtype=torch.float),
        }


# =============================================================================
# Loss
# =============================================================================

class DistributionalTrainer(Trainer):
    """Soft ordinal CE + EV auxiliary on the bin position(s) (one: total); CE on <eos>."""

    def __init__(self, *args, bin_token_ids=None, ev_log_dollars=None,
                 sigma=1.0, ev_weight=0.5, z_loss_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._bin_ids = bin_token_ids            # (K,) long
        self._ev_log = ev_log_dollars            # (S, K) float, log1p dollars; S = 1
        self._sigma = sigma
        self._ev_weight = ev_weight
        self._z_weight = z_loss_weight
        # Our loss is a per-microbatch MEAN and ignores num_items_in_batch.
        # The backbone accepts loss kwargs, so without this flag the Trainer skips
        # dividing by gradient_accumulation_steps (it assumes the loss was
        # num_items-normalized) -> gradients ga x too large, i.e. effective
        # lr = nominal x ga.
        self.model_accepts_loss_kwargs = False
        K = len(bin_token_ids)
        idx = torch.arange(K, dtype=torch.float)
        # (K, K) soft target rows: row k = Gaussian over indices centred at k
        soft = torch.exp(-0.5 * ((idx[None, :] - idx[:, None]) / sigma) ** 2)
        self._soft_targets = soft / soft.sum(dim=1, keepdim=True)
        self._tok2bin = None                     # lazy vocab->bin index map

    def _bin_index_of(self, token_ids, vocab_size, device):
        if self._tok2bin is None or self._tok2bin.device != device:
            m = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
            m[self._bin_ids.to(device)] = torch.arange(len(self._bin_ids), device=device)
            self._tok2bin = m
        return self._tok2bin[token_ids]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        dollar_labels = inputs.pop("dollar_labels")          # (B, S) log1p; S = 1
        labels = inputs.pop("labels")                        # (B, T)
        outputs = model(**inputs)
        logits = outputs.logits                              # (B, T, V)
        B, T, V = logits.shape
        device = logits.device

        # next-token shift
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        sup = shift_labels != -100                           # (B, T-1)
        bin_of = self._bin_index_of(shift_labels.clamp(min=0), V, device)
        is_bin = sup & (bin_of >= 0)
        is_plain = sup & (bin_of < 0)                        # <eos>

        loss = torch.zeros((), device=device)

        # ---- z-loss on supervised positions (parity check with the disease
        # post-training, which regularizes the full-vocab log-partition) ----
        if self._z_weight > 0 and sup.any():
            log_z = torch.logsumexp(shift_logits[sup].float(), dim=-1)
            loss = loss + self._z_weight * (log_z ** 2).mean()

        # ---- plain CE on <eos> positions --------------------------------
        if is_plain.any():
            loss = loss + F.cross_entropy(
                shift_logits[is_plain], shift_labels[is_plain])

        # ---- bin positions ----------------------------------------------
        if is_bin.any():
            bin_ids = self._bin_ids.to(device)
            sub_logits = shift_logits[is_bin][:, bin_ids]    # (N, K)
            log_probs = F.log_softmax(sub_logits, dim=-1)
            tgt = self._soft_targets.to(device)[bin_of[is_bin]]
            soft_ce = -(tgt * log_probs).sum(dim=-1).mean()
            loss = loss + soft_ce

            if self._ev_weight > 0:
                # position within the response selects the EV vector; always slot 0
                resp_pos = torch.zeros_like(shift_labels)
                csum = torch.cumsum(is_bin.long(), dim=1)
                resp_pos = (csum - 1).clamp(min=0)           # 0-indexed bin slot
                pos_sel = resp_pos[is_bin]                   # (N,)
                ev_log = self._ev_log.to(device)             # (S, K); (1, K)
                probs = log_probs.exp()
                ev_pred = (probs * ev_log[pos_sel]).sum(dim=-1)   # (N,)
                # true log1p dollars, aligned by (batch row, slot)
                rows = is_bin.nonzero(as_tuple=True)[0]
                ev_true = dollar_labels[rows, pos_sel]
                ev_loss = F.huber_loss(ev_pred, ev_true)
                loss = loss + self._ev_weight * ev_loss

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    args = parse_args()
    set_seed(args.seed)
    logging.info("COST INSTRUCT POST-TRAINING STARTING...")
    logging.info("  Args: %s", vars(args))
    if torch.cuda.is_available():   # the GPU model is part of the compute statement; the logs are its only record
        logging.info("  Device: %s x%d", torch.cuda.get_device_name(0), int(os.environ.get("WORLD_SIZE", 1)))

    binspec = BinSpec(args.bin_artifact)
    logging.info("  Bins: %d mantissa bins", len(binspec))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)

    # ---- vocabulary: all bin tokens must already exist ----------------------
    missing = [t for t in binspec.tokens if tokenizer.convert_tokens_to_ids(t) is None]
    assert not missing, f"tokens missing from vocab: {missing[:5]}"

    bin_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in binspec.tokens], dtype=torch.long)
    ev_log = torch.tensor(
        np.stack([np.log1p(binspec.ev_dollars[tc]) for tc in GOLD_COST_COLS]),
        dtype=torch.float)

    # ---- data ------------------------------------------------------------
    train_ds = build_dataset(os.path.join(args.pairs_dir, "train_pairs.parquet"),
                             tokenizer, args.seq_len, args.max_train_pairs)
    val_ds = build_dataset(os.path.join(args.pairs_dir, "val_pairs.parquet"),
                           tokenizer, args.seq_len,
                           None if args.max_train_pairs is None
                           else max(64, args.max_train_pairs // 10))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<eos>")
    collator = Collator(pad_id)

    # ---- trainer ---------------------------------------------------------
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=args.bf16,
        seed=args.seed,
        report_to=[], disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = DistributionalTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        bin_token_ids=bin_token_ids, ev_log_dollars=ev_log,
        sigma=args.sigma, ev_weight=args.ev_weight,
        z_loss_weight=args.z_loss_weight,
    )

    import glob as _glob
    has_ckpt = bool(_glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=True if (args.resume and has_ckpt) else None)

    # ---- save final -------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    # Provenance: the exact resolved recipe this checkpoint was trained with
    # (args also cover any override, so every model documents its own recipe).
    with open(os.path.join(final_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args),
                   "effective_batch": args.per_device_batch * args.grad_accum
                                      * int(os.environ.get("WORLD_SIZE", 1)),
                   "transformers_note": "model_accepts_loss_kwargs=False "
                       "(gradient-accumulation scaling)"}, f, indent=1)
    with open(os.path.join(final_dir, "bin_artifact_ref.json"), "w") as f:
        json.dump({"bin_artifact": os.path.abspath(args.bin_artifact),
                   "tokens": binspec.tokens}, f, indent=1)
    logging.info("  Saved final model to: %s", final_dir)

    logging.info("COST INSTRUCT POST-TRAINING COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
