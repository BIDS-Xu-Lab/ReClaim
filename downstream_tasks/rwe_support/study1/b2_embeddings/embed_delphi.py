#!/usr/bin/env python3
"""
embed_delphi.py  --  Delphi patient embeddings (rep_*).

The Delphi counterpart to embed_reclaim.py, with the same contract on both ends:
  IN   a prepare_sequences `cohort_with_pre_entry_seq*.parquet`  (person_id + pre_entry_seq)
  OUT  `person_id + rep_0..rep_{d-1}`, parquet
so ladder_driver.sh/estimate_ladder consume a Delphi spec exactly as they consume a ReClaim one.

ALIGNED WITH embed_reclaim:
  * pooling       : last token of the final hidden state, then L2 normalisation. Delphi is also a
                    causal LM, so only the last position has seen the whole history.
  * truncation    : keep the most recent events. The embedding is read out at the final token, so
                    it represents the patient as of the last event kept; keeping the oldest events
                    would anchor it at a point that varies with history length, i.e. with
                    healthcare utilisation, itself correlated with treatment. Note the naming
                    trap: Delphi's CLI calls keep-recent `--select right`, HuggingFace calls the
                    same behaviour `truncation_side="left"`. The input builder here slices the last
                    min(len, block_size) records directly and is not exposed as a flag.
  * null handling : a null pre_entry_seq is an error, never silently dropped (prepare_sequences
                    emits a header-only sequence for patients with no in-window events, so a null
                    means a truncated file; dropping rows would let a partial file reach
                    estimate_ladder, which zero-fills missing rep_*).
  * output        : parquet, float32, person_id first.

WHERE DELPHI NECESSARILY DIFFERS:
  * tokens read   : only `<DX-MAJOR_*>` diagnoses (mapped through the disease map) plus the
                    `<SEX-*>`/`<AGE-*>` header; drugs, costs and visit types are not consumed.
  * context       : block_size 48 first-occurrence diagnosis events, vs ReClaim's 4096 tokens.
                    The realised truncation rate is logged.
  * extra input   : Delphi.forward(idx, age) takes ages in days alongside tokens (sinusoidal
                    AgeEncoding). Ages are reconstructed by walking `<ATT-N>` intervals from the
                    `<AGE-*>` header.
  * token data    : the benchmark feeds Delphi the same prevseq ReClaim uses. Everything this
                    parser consumes (age header position, ATT in months/relative, DX-MAJOR codes)
                    matches the tokens Delphi was trained on, and the DX vocabulary is fixed by
                    the disease map.

No vLLM backend: Delphi is a bespoke nn.Module loaded from a raw .pt whose forward takes a second
`age` tensor that a pooling runner cannot pass. HF/PyTorch only.

MODEL CODE IS VENDORED from the upstream Delphi embedding script, unchanged except for three fixes,
each marked and explained in place. All three are cases where the upstream code is wrong for this
use (embedding a patient's recent history) and fail silently, producing finite, unit-norm,
correctly-shaped vectors:
  1. get_delphi_embeddings pooled the last token as if sequences were left-aligned, but the inputs
     are right-aligned, so the embedding was read from a padding position that has attended to
     nothing. It now reads the final position.
  2. PARSE_TOKEN_CAP removes an inherited 4096-token parse cap that discarded the most RECENT
     tokens before the keep-recent window was applied.
  3. Upstream builds the model input with get_batch, a training batcher whose x = tokens[:, :-1]
     excludes the most recent event by construction (it is the y target). main() builds the
     inference input directly; get_batch is retained unchanged but unused, so the vendored block
     stays comparable with its source.
These fixes mean this script does not reproduce the upstream script's embeddings, deliberately:
the fidelity that matters is to the model, not to that script.

Usage: python embed_delphi.py --model_path <ckpt.pt> --disease_map <csv> --input <prevseq.parquet>
           --output <parquet> [--att_convention relative|absolute]
"""

import os
import sys
import math
import inspect    # used by the vendored Delphi.configure_optimizers
import builtins
import argparse
import logging
import warnings   # used by the vendored Delphi.generate
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import setup_logging, time_execution

INDENT1, INDENT2 = "  ", "    "

# DEVIATION #2 from the vendored source (see the module docstring).
# Upstream parses only the first `max_length=4096` tokens of a sequence. That cap keeps the OLDEST
# tokens and discards the most recent ones (a right-truncation); the keep-recent window then keeps
# the last 48 events of whatever survived, so a long history would be embedded from mid-history
# events while claiming to represent the patient's most recent state. Delphi's real context limit
# is block_size (48 events), enforced by the input builder, and parsing is pure Python over a token
# list, so lifting the cap costs a little CPU and nothing on the model.
PARSE_TOKEN_CAP = 10 ** 9

# =================================================================================================
# VENDORED from the upstream Delphi embedding script -- model definition + data prep. Unchanged.
# =================================================================================================
def new_gelu(x):
    """GELU (tanh approximation)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y, att

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask) 
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att

class AgeEncoding(nn.Module):

    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(x /365.25 * self.div_term)
        y = self.linear(y)
        
        return y
    
@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])

class Delphi(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wae = AgeEncoding(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1-self.config.token_dropout) 
        x = x + age_emb
        x = self.transformer.drop(x)
        
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0),1,idx.size(1),1)
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1)))
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0

        
        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            logits = self.lm_head(x)

            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[...,ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1 
            for k in ignored_tokens:
                pass_tokens &= (targets != k)

            if pass_tokens.any():
                max_t = int(targets[pass_tokens].max().item())
                min_t = int(targets[pass_tokens].min().item())
                if max_t >= logits.size(-1) or min_t < 0:
                    raise ValueError(f"Target index out of range: min {min_t}, max {max_t}, vocab_size {logits.size(-1)}")

            
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)
            
            lse = torch.logsumexp(logits,-1)
            lse = - torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, idx.size(1), device=device, dtype=torch.float32)
                                           .view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2)))
            ldt = - torch.log(dt + self.config.t_min).view(-1)
            
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_dt = torch.mean(loss_dt[pass_tokens]) 
            
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
            
        else:
            logits = self.lm_head(x[:, :, :])
            loss = None

        return logits, loss, att

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            
    def adjust_block_size(self, block_size):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Split parameters into decayed / non-decayed groups and return the AdamW optimizer."""

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.remove('lm_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, termination_tokens=None, top_k=None):
        """Autoregressive trajectory sampling (upstream; unused by the embedding driver)."""
        if termination_tokens is None:
            warnings.warn('When using a custem dataset, consider changing the `termination_tokens` argument.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            logits, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:,self.config.ignore_tokens] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:,None]
            age_next = age[...,[-1]] + t_next[0][:,None]
            
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            if torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break
        
        pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)

        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([logits[:,j].scatter_(1, fill[:,:j+1], -torch.inf) for j in range(fill.shape[1])]).transpose(0,1)

        return idx, age, logits


def load_disease_mapping_from_csv(csv_file_path):
    """
    Load the disease mapping csv (columns: disease, icd_code, disease_id) -> {major code: id}.
    The csv's disease_id starts at 13 but Delphi requires the first id to be 12, so ids are shifted by -1.
    """
    logging.info("Loading disease mapping from CSV: %s", csv_file_path)
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Disease mapping CSV file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)
    
    required_cols = ['disease', 'disease_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    unique_mappings = df[['disease', 'disease_id']].drop_duplicates()
    
    mapping = {}
    for _, row in unique_mappings.iterrows():
        disease_code = row['disease']
        original_id = int(row['disease_id'])
        adjusted_id = original_id - 1
        mapping[disease_code] = adjusted_id
    
    logging.info("Loaded %d diseases from CSV successfully!", len(mapping))
    if mapping:
        logging.info("ID range: %d to %d", min(mapping.values()), max(mapping.values()))
        sample_items = list(mapping.items())[:5]
        logging.info("Sample mappings: %s", sample_items)
    
    return mapping

# Track unknown diseases for reporting
unknown_diseases = set()

def parse_disease_id(token_str, mapping):
    """<DX-MAJOR_A00> -> mapping['A00'], or -1 for any other token."""
    if not token_str.startswith('<DX-MAJOR_'):
        return -1
    
    major_code = token_str[10:-1]  # Remove '<DX-MAJOR_' prefix and '>' suffix
    result = mapping.get(major_code, -1)
    
    if result == -1:
        unknown_diseases.add(major_code)

    return result


def get_prediction_points_data(
        seq, 
        disease_mapping, 
        sex_mapping, 
        demo_end_index=4, 
        age_index=3, 
        sex_index=1, 
        max_length=PARSE_TOKEN_CAP,   # DEVIATION #2: upstream default is 4096
        first_occurrence=True,
        att_time_unit="days", 
        att_time_mode="relative"
    ):
    """
    Parse one token string into Delphi prediction points: walks <ATT-N> intervals from the <AGE-*>
    header (att_time_unit days|weeks|months; att_time_mode relative|absolute) and records each
    <DX-MAJOR_*> token (first occurrence only by default).

    Returns (prediction_points, prediction_points_ages, prediction_points_disease_ids, sex_id).
    """
    tokens = seq.split(" ")

    prediction_points = []
    prediction_points_ages = []
    prediction_points_disease_ids = []

    sex_id = 0
    if len(tokens) > sex_index:
        token = tokens[sex_index]
        if token in sex_mapping:
            sex_id = sex_mapping[token]
        else:
            pass
    else:
        pass

    if len(tokens) <= age_index:
        return [], [], [], sex_id
    
    age_token = tokens[age_index]
    try:
        if age_token.startswith('<AGE-') and age_token.endswith('>'):
            day_age = int(age_token[5:-1]) * 365.25
        else:
            raise ValueError("Format mismatch")
    except (ValueError, IndexError):
        return [], [], [], sex_id

    days_passed = 0
    disease_tokens = set()
    
    if att_time_unit == "days":
        time_multiplier = 1
    elif att_time_unit == "weeks":
        time_multiplier = 7
    elif att_time_unit == "months":
        time_multiplier = 30
    else:
        time_multiplier = 1
    
    length = builtins.min(len(tokens) - 2, max_length - 2)
    i = demo_end_index

    while i < length:
        current_token = tokens[i]

        if current_token.startswith('<ATT-'):
            try:
                att_value = int(current_token[5:-1]) if current_token != "<ATT-0>" else 0
                att_days = att_value * time_multiplier
                
                if att_time_mode == "relative":
                    days_passed += att_days
                else:
                    days_passed = att_days
            except ValueError:
                pass

        elif current_token.startswith('<DX-MAJOR_'):
            disease_id = parse_disease_id(current_token, disease_mapping)

            should_record = False

            if first_occurrence:
                if current_token not in disease_tokens:
                    should_record = True
                    disease_tokens.add(current_token)
            else:
                should_record = True
            
            if should_record:
                prediction_points.append(i)
                prediction_points_ages.append(day_age + days_passed)
                prediction_points_disease_ids.append(disease_id)

        i += 1
    
    return prediction_points, prediction_points_ages, prediction_points_disease_ids, sex_id


def process_all_patients(traj_df, pid_col, seq_col, att_time_unit, att_time_mode, disease_mapping, sex_mapping,
                         demo_end_index=4, age_index=3, sex_index=1):
    """
    Parse every patient's sequence into long records [patient_id, patient_age, token_id]
    (sex record at age 0 first, then first-occurrence diagnoses). Returns (DataFrame, pid_map).
    """
    unique_pids = traj_df[pid_col].unique()
    sorted_pids = sorted(unique_pids)
    pid_map = {pid: idx for idx, pid in enumerate(sorted_pids)}

    all_records = []
    invalid_token_count = 0  

    logging.info("Processing %d patients from trajectory dataframe...", len(sorted_pids))

    pids = traj_df[pid_col].values
    seqs = traj_df[seq_col].values
    
    for original_pid, seq in tqdm(zip(pids, seqs), total=len(pids), desc="Processing patients", file=sys.stdout):
        new_pid_idx = pid_map[original_pid]

        locs, ages, dids, sex_id = get_prediction_points_data(
            seq,
            disease_mapping=disease_mapping,
            sex_mapping=sex_mapping,
            demo_end_index=demo_end_index,
            age_index=age_index,
            sex_index=sex_index,
            first_occurrence=True,
            att_time_unit=att_time_unit,
            att_time_mode=att_time_mode,
        )

        all_records.append({
            'patient_id': new_pid_idx,
            'patient_age': 0,
            'token_id': sex_id,
        })

        for j in range(len(locs)):
            token_id = dids[j]
            if token_id == -1:
                invalid_token_count += 1
                continue
            all_records.append({
                'patient_id': new_pid_idx,
                'patient_age': ages[j],
                'token_id': token_id,
            })
    
    final_df = pd.DataFrame(all_records)
    
    logging.info("=== Processing Summary ===")
    logging.info("Total records created: %d", len(final_df))
    logging.info("Invalid tokens skipped: %d", invalid_token_count)

    if not final_df.empty:
        patient_counts = final_df.groupby('patient_id').size()
        

        logging.info("Total patients (no filtering applied): %d", len(patient_counts))

        token_min = final_df['token_id'].min()
        token_max = final_df['token_id'].max()
        logging.info("=== Token ID Validation ===")
        logging.info("Token ID range: %d to %d", token_min, token_max)
        logging.info("Recommended vocab_size: %d", token_max + 2)
        
        if (final_df['token_id'] < 0).any():
            raise ValueError("ERROR: Found negative token IDs in final data!")

    return final_df, pid_map



def preprocess_cohort_to_delphi_format(
    cohort_df: pd.DataFrame,
    disease_mapping: dict,
    att_convention: str = 'relative',
    seq_col: str = 'pre_entry_seq',
    pid_col: str = 'patient_id',
) -> np.ndarray:
    """
    Convert a cohort dataframe with token sequences to the Delphi array format:
    numpy array of shape (N, 3) with columns [patient_id, patient_age, token_id].
    att_convention selects the ATT time unit/mode via ATT_CONVENTIONS.
    """
    convention = ATT_CONVENTIONS[att_convention]
    att_time_unit = convention['att_time_unit']
    att_time_mode = convention['att_time_mode']
    
    logging.info("Preprocessing cohort to Delphi format...")
    logging.info("ATT convention: %s (time_unit=%s, time_mode=%s)",
                 att_convention, att_time_unit, att_time_mode)
    
    if pid_col not in cohort_df.columns:
        cohort_df = cohort_df.copy()
        cohort_df[pid_col] = range(len(cohort_df))
        logging.info("Using row index as patient_id")
    
    processed_df, pid_map = process_all_patients(
        traj_df=cohort_df,
        pid_col=pid_col,
        seq_col=seq_col,
        att_time_unit=att_time_unit,
        att_time_mode=att_time_mode,
        disease_mapping=disease_mapping,
        sex_mapping=SEX_MAPPING,
    )
    
    data = processed_df[['patient_id', 'patient_age', 'token_id']].values.astype(np.float32)
    
    logging.info("Preprocessed data shape: %s", data.shape)
    logging.info("Unique patients: %d", len(np.unique(data[:, 0])))
    
    return data


# =============================================================================
# Embedding Extraction
# =============================================================================


def get_p2i(data):
    """
    Patient -> (start_row, n_rows) index into `data`, built by scanning column 0 for value changes
    (vendored unchanged). Order follows the input cohort's row order, and a patient whose rows are
    not contiguous (a duplicated person_id) is split in two; main() asserts len(p2i) == len(cohort).
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


# get_batch below is vendored unchanged but is NOT called by this driver. It is a TRAINING
# batcher: it returns the teacher-forcing pair x = tokens[:, :-1], y = tokens[:, 1:], so the most
# recent event is excluded from x by construction (it is the prediction target). main() builds the
# inference input directly instead (DEVIATION #3). Retained so the vendored block stays comparable
# with its source and the upstream behaviour stays inspectable.
def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False, 
              no_event_token_rate=5, cut_batch=False):
    """
    Upstream training batcher: packs sequences, optionally inserts "no event" tokens, and returns
    the teacher-forcing pair (x, a, y, b) = (input tokens, input ages, target tokens, target ages).
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values

    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    tokens = tokens + 1

    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


# =============================================================================
# Data preprocessing (vendored)
# =============================================================================

SEX_MAPPING = {
    '<SEX-2>': 1,
    '<SEX-1>': 2,
}

# ATT (time-gap) token conventions: gaps in months, relative to the previous event or absolute.
ATT_CONVENTIONS = {
    'relative': {
        'att_time_unit': 'months',
        'att_time_mode': 'relative',
        'time_multiplier': 30,
    },
    'absolute': {
        'att_time_unit': 'months',
        'att_time_mode': 'absolute',
        'time_multiplier': 30,
    },
}



def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms


def get_delphi_embeddings(
    model: Delphi,
    idx: torch.Tensor,
    age: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Last-token pooled Delphi embeddings: runs the forward pass up to the final LayerNorm and
    reads the final position (for a causal model only the last token has seen the whole sequence).
    """
    idx = idx.to(device)
    age = age.to(device)

    with torch.no_grad():
        # mirrors model.forward exactly, including the unconditional (1 - token_dropout) scaling
        tok_emb = model.transformer.wte(idx)
        age_emb = model.transformer.wae(age.unsqueeze(-1))
        x = model.transformer.token_drop(tok_emb) * (1 - model.config.token_dropout)
        x = x + age_emb
        x = model.transformer.drop(x)

        attn_mask = (idx > 0).view(idx.size(0), 1, 1, idx.size(1)) * \
                    (idx > 0).view(idx.size(0), 1, idx.size(1), 1)
        attn_mask = attn_mask * (torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0)
        attn_mask = attn_mask + (idx == 0).view(idx.size(0), 1, 1, idx.size(1)) * \
                    torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask * (torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0)

        for block in model.transformer.h:
            x, _ = block(x, attn_mask)

        x = model.transformer.ln_f(x)

        # Last-token pooling. DEVIATION #1 from the vendored source (see the module docstring):
        # upstream pools at `(idx > 0).sum(dim=1) - 1`, which assumes LEFT-aligned tokens with
        # padding on the right, but the inputs are RIGHT-aligned, so for most patients that index
        # lands inside the leading padding. The attention mask above gives every pad position a
        # diagonal-only row, so a pad's hidden state has seen no patient history. With
        # right-aligned inputs the last real token is always the final column, so the read-out is
        # position -1, matching embed_reclaim (HF left-truncation puts the most recent token last).
        embeddings = x[:, -1, :]

        # Safety net: an all-pad row would be pooled at a padding position. The input builder makes
        # this unreachable (every patient has at least the sex record), so this firing means the
        # input construction is broken.
        n_empty = int((idx > 0).sum(dim=1).eq(0).sum())
        if n_empty:
            logging.warning("%s%d/%d patients in this batch have an ALL-PAD input row -- their "
                            "embedding encodes nothing; the input builder should make this "
                            "impossible", INDENT2, n_empty, idx.size(0))

    return embeddings


# =================================================================================================
# BENCHMARK LAYER -- mirrors embed_reclaim.py
# =================================================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Generate Delphi rep_* embeddings for an prepare_sequences prevseq file",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="delphi", choices=["delphi"],
                   help="Model id. Drives logging and the output filename tag.")
    p.add_argument("--model_path", required=True, help="Delphi checkpoint (.pt)")
    p.add_argument("--disease_map", required=True, help="nature_disease_map.csv")
    p.add_argument("--input", required=True,
                   help="prepare_sequences cohort_with_pre_entry_seq*.parquet")
    p.add_argument("--output", required=True, help="Output .parquet path")
    p.add_argument("--batch_size", type=int, default=256)
    # No --padding flag: that was get_batch's training-time "no event" token augmentation; the
    # driver builds the inference input directly (DEVIATION #3) with no augmentation.
    p.add_argument("--block_size", type=int, default=48,
                   help="Delphi context window in EVENTS (first-occurrence diagnoses)")
    p.add_argument("--att_convention", default="relative", choices=sorted(ATT_CONVENTIONS),
                   help="ATT time-gap convention of the token data the INPUT prevseq was built from "
                        "(the benchmark's tokens use relative months). A mismatch does not crash -- it "
                        "silently shifts every age the model sees.")
    p.add_argument("--limit", type=int, default=None, help="Embed only the first N rows (smoke tests)")
    p.add_argument("--keep_seq_col", action="store_true")
    return p.parse_args()


def load_sequences(path, limit=None):
    """Read the prepare_sequences prevseq file. Every cohort row must carry a sequence (see the
    null-handling note in the module docstring). Identical contract to embed_reclaim."""
    if not path.endswith(".parquet"):
        raise ValueError(f"expected an prepare_sequences cohort_with_pre_entry_seq*.parquet, got: {path}")
    df = pd.read_parquet(path)
    if "pre_entry_seq" not in df.columns:
        raise ValueError(f"{path} has no pre_entry_seq column (got {list(df.columns)[:8]}). "
                         "Expected an prepare_sequences cohort_with_pre_entry_seq*.csv.")
    if limit:
        df = df.head(limit).copy()
        logging.info("%s--limit %d: smoke-test subset", INDENT2, limit)
    n_null = int(df["pre_entry_seq"].isna().sum())
    if n_null:
        raise ValueError(f"{n_null:,} of {len(df):,} rows have a null pre_entry_seq. prepare_sequences emits a "
                         "header-only sequence for patients with no in-window events, so nulls mean "
                         "the file is truncated or corrupt. Refusing to proceed.")
    return df, df["pre_entry_seq"].astype(str).tolist()


def load_delphi(model_path, block_size, device):
    """Load the checkpoint into the vendored Delphi class."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    margs = ckpt["model_args"]
    cfg = DelphiConfig(**margs)
    model = Delphi(cfg)
    sd = ckpt["model"]
    for k in list(sd.keys()):                       # strip the compile/DDP prefix if present
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    logging.info("%sDelphi: n_layer=%d n_head=%d n_embd=%d block_size=%d vocab=%d",
                 INDENT2, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size, cfg.vocab_size)
    if block_size > cfg.block_size:
        raise ValueError(f"--block_size {block_size} exceeds the checkpoint's {cfg.block_size}")
    return model, cfg


@time_execution
def main():
    args = parse_args()
    # Validate the output path before any compute: the write is the last step.
    if not args.output.endswith(".parquet"):
        raise ValueError(f"--output must be a .parquet path, got: {args.output}")
    log_file = setup_logging(level=logging.INFO)
    logging.info("H5c DELPHI EMBEDDINGS STARTING (model=%s)", args.model)
    for f in (args.model_path, args.disease_map):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Not found: {f}")

    df, seqs = load_sequences(args.input, args.limit)
    logging.info("%s%s patients from %s", INDENT2, f"{len(seqs):,}", args.input)

    # The parser reads the age from a FIXED position (age_index=3) and, if the token there does not
    # match `<AGE-...>`, returns NO events for that patient: not an error, just an embedding built
    # from nothing. prepare_sequences puts it there (`<sos> <SEX-..> <DOBYR-..>` then the age
    # token), but the parser does not verify that layout, so check it explicitly.
    tok3 = [s.split(" ") for s in seqs]
    # Header-only sequences (no events) are legitimate: the assembly below emits the sex token at
    # age 0 for every patient, so they yield a valid demographics-only embedding. The guard fires
    # only when a sequence HAS event tokens but the one at the fixed age position is not <AGE-*>,
    # i.e. real layout drift, under which the parser would silently drop every event.
    header_only = [i for i, t in enumerate(tok3) if len(t) <= 3]
    bad_age = [i for i, t in enumerate(tok3) if len(t) > 3 and not t[3].startswith("<AGE-")]
    if header_only:
        logging.info("%s%d of %d sequences are header-only (no pre-index events) -> "
                     "demographics-only Delphi embedding", INDENT2, len(header_only), len(seqs))
    if bad_age:
        raise RuntimeError(
            f"{len(bad_age)} of {len(seqs)} sequences HAVE event tokens but no <AGE-*> at index 3 "
            f"(first offender row {bad_age[0]}: {' '.join(tok3[bad_age[0]][:6])!r}). The Delphi "
            "parser reads the age from that fixed position and silently returns zero events when it "
            "is missing, so this would produce content-free embeddings rather than fail.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("%sdevice=%s%s", INDENT2, device,
                 f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    disease_mapping = load_disease_mapping_from_csv(args.disease_map)
    logging.info("%sdisease map: %d codes", INDENT2, len(disease_mapping))

    # seq -> long array of [patient_id, patient_age, token_id], first-occurrence diagnoses only.
    work = df[["person_id", "pre_entry_seq"]].copy() if "person_id" in df.columns else df.copy()
    data = preprocess_cohort_to_delphi_format(
        cohort_df=work, disease_mapping=disease_mapping,
        att_convention=args.att_convention, seq_col="pre_entry_seq", pid_col="person_id")

    # p2i indexes `data` in cohort row order (see get_p2i), so the embedding matrix is positionally
    # aligned with `df` and can be concatenated below without a merge key. A single dropped or
    # duplicated patient would shift every subsequent row's embedding onto the wrong person, so
    # assert the alignment instead of tolerating drops.
    p2i = get_p2i(data)
    if len(p2i) != len(df):
        raise RuntimeError(
            f"row-alignment broken: {len(p2i)} patients in the Delphi array vs {len(df)} cohort rows. "
            "A patient was dropped or a person_id is duplicated; positional concat would misalign "
            "every embedding after the first offender.")
    logging.info("%s%s patients -> %s Delphi records", INDENT2, f"{len(p2i):,}", f"{len(data):,}")
    n_capped = int((p2i[:, 1] > args.block_size).sum())
    logging.info("%s%s patients (%.1f%%) exceed the %d-event window; oldest events are dropped",
                 INDENT2, f"{n_capped:,}", 100.0 * n_capped / len(p2i), args.block_size)

    model, cfg = load_delphi(args.model_path, args.block_size, device)

    # Batched forward, keeping the most recent events (== HF truncation_side='left').
    #
    # DEVIATION #3: the input is built here, not by the vendored get_batch, whose teacher-forcing
    # x = tokens[:, :-1] excludes the most recent event by construction. Each patient's last
    # min(len, block_size) records are laid out with get_batch's own conventions (+1 token shift so
    # 0 is the pad id; pads left; pad age = -10000 = get_batch's mask_time), so every event up to
    # and including the most recent is kept. Rows in `data` are already chronological per patient
    # (ages are a cumulative sum), so no re-sort is needed, and slicing [start, start+len) per
    # patient makes cross-patient bleed impossible.
    n_pat = len(p2i)
    n_batches = (n_pat + args.batch_size - 1) // args.batch_size
    logging.info("%sembedding %s patients in %s batches of %d (keep-recent, last %d events)",
                 INDENT2, f"{n_pat:,}", f"{n_batches:,}", args.batch_size, args.block_size)
    bs = args.block_size
    chunks = []
    for b in tqdm(range(n_batches), desc="Delphi batches", file=sys.stdout):
        ix = range(b * args.batch_size, min((b + 1) * args.batch_size, n_pat))
        idx = torch.zeros((len(ix), bs), dtype=torch.long)
        age = torch.full((len(ix), bs), -10000.0)
        for r, i in enumerate(ix):
            s, l = int(p2i[i, 0]), int(p2i[i, 1])
            k = min(l, bs)
            rows = data[s + l - k : s + l]                                    # the k most recent
            idx[r, bs - k:] = torch.from_numpy(rows[:, 2].astype(np.int64)) + 1
            age[r, bs - k:] = torch.from_numpy(rows[:, 1].astype(np.float32))
        with torch.no_grad():
            out = get_delphi_embeddings(model, idx.to(device), age.to(device), device)
        chunks.append(out.float().cpu().numpy())
        if device.type == "cuda" and b % 50 == 0:
            torch.cuda.empty_cache()

    emb = normalize_embeddings(np.concatenate(chunks, axis=0).astype(np.float32))
    if len(emb) != len(df):
        raise RuntimeError(f"embedding count {len(emb)} != cohort rows {len(df)}")
    logging.info("%sembeddings %s", INDENT2, emb.shape)
    norms = np.linalg.norm(emb, axis=1)
    logging.info("%snorms min=%.6f max=%.6f mean=%.6f", INDENT2, norms.min(), norms.max(), norms.mean())

    out = df if args.keep_seq_col else df.drop(columns=["pre_entry_seq"])
    reps = pd.DataFrame(emb, columns=[f"rep_{i}" for i in range(emb.shape[1])], index=out.index)
    out = pd.concat([out, reps], axis=1)
    if len(out) != len(df):
        raise RuntimeError(f"row count changed: {len(df)} -> {len(out)}")
    logging.info("%swriting %s patients x %s cols -> %s",
                 INDENT2, f"{len(out):,}", f"{out.shape[1]:,}", args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_parquet(args.output, compression="snappy", index=False)
    logging.info("H5c COMPLETED. Log: %s", log_file)


if __name__ == "__main__":
    main()
