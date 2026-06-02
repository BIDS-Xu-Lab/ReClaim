#!/usr/bin/env python3
"""
Extract patient embeddings from Delphi foundation model.

This script processes clinical token sequences and generates embeddings using
a pre-trained Delphi model with last-token pooling (appropriate for causal LMs).

Embeddings are L2 normalized to unit vectors, making dot product equivalent to
cosine similarity. This is optimal for patient similarity search and propensity
score matching augmentation.

Data Versions:
- v6: ATT token time unit = months, time mode = relative (interval from previous token)
- v7: ATT token time unit = months, time mode = absolute (interval from initial age)

This is a self-contained script that includes:
- Delphi model definition
- Data preprocessing utilities
- Embedding extraction functions
"""

import os
import sys
import math
import builtins
import argparse
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import setup_logging, time_execution


# =============================================================================
# Delphi Model Definition (from model.py)
# =============================================================================

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = False #hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
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
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term) #* (1-self.div_term)
        y[..., 1::2] = torch.cos(x /365.25 * self.div_term) #* (1-self.div_term)
        y = self.linear(y)
        
        #x = self.wae[:x.size(0)]
        return y #self.dropout(x)
    
@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
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
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            #wae = nn.Linear(1, config.n_embd, bias=True), ##nn.Embedding(config.block_size, config.n_embd),
            wae = AgeEncoding(config),
            #mlp = MLP(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
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
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        age_emb = self.transformer.wae(age.unsqueeze(-1)) # age embeddings of shape (b, t, n_embd)
        #age_emb = self.transformer.mlp(age_emb)
        x = self.transformer.token_drop(tok_emb) * (1-self.config.token_dropout) 
        x = x + age_emb
        x = self.transformer.drop(x)
        
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0),1,idx.size(1),1)  # Do not attend to padded positions
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) # Mask co-occuring tokens
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), device=device)) > 0 # Except for padding
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0

        
        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            # next token cross entropy loss, padding masked
            logits = self.lm_head(x)

            # if we are given some desired targets also calculate the loss
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[...,ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1 
            for k in ignored_tokens: # and gender
                pass_tokens &= (targets != k)

            # Sanity check: ensure target class indices are within logits vocab range.
            if pass_tokens.any():
                # move to cpu scalar to trigger a clear Python exception instead of a CUDA assert
                max_t = int(targets[pass_tokens].max().item())
                min_t = int(targets[pass_tokens].min().item())
                if max_t >= logits.size(-1) or min_t < 0:
                    raise ValueError(f"Target index out of range: min {min_t}, max {max_t}, vocab_size {logits.size(-1)}")

            #age_min = age.gather(1,(((idx >=4) * (idx <=12)) + 0).argmax(1)[:,None])
            #logits[...,-1][age <= age_min] = -100. #-float('Inf') ## Death can only occur after age_min
            
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)
            
            # time to next event loss, padding masked
            lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
            lse = - torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, idx.size(1), device=device, dtype=torch.float32)
                                           .view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2)))  # Use time from last untied token
            ldt = - torch.log(dt + self.config.t_min).view(-1)
            
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1))) ## Exponential log-likelihood (real statistics, TM)
            loss_dt = torch.mean(loss_dt[pass_tokens]) 
            
            # Both losses combined
            # loss = loss_ce + loss_dt
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
            
            #loss += 5.0 * F.mse_loss(lse.view(-1)*(ldt != 0), ldt) ## Adds MSE for log time difference to next observed event
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            
    def adjust_block_size(self, block_size):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, termination_tokens=None, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Selected parameters:
        --------------------

        termination_tokens: list[int] -  a list of tokens that indicate the and of the trajectory.
        Usually it is the "Death" token, but could be several tokens e.g. to indicate different
        death reasons.
        top_k: None, does nothing
        """
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
            
            # sample from exponential distributions for each disease using the inverse CDF method, then take min
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:,None] # the index of the min sampled time
            age_next = age[...,[-1]] + t_next[0][:,None] # the value of the min sampled time
            
            # append sampled index to the running sequence and continue
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


# =============================================================================
# Data Utilities (from delphi_utils.py)
# =============================================================================

def get_p2i(data):
    """
    Get the patient to index mapping.
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


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False, 
              no_event_token_rate=5, cut_batch=False):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'center', 'right', 'smart_random', 'smart_right'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

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

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            #TODO maybe use the same shift for all lifestyle tokens in the trajectory?
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # insert a "no event" token every 5 years on average
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

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # cut to maintain the block size
    #TODO it would be better to use the strategy defined by the "select" parameter
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


# =============================================================================
# Data Preprocessing (from final_delphi_data_preprocessing.py)
# =============================================================================

# Fixed mappings
SEX_MAPPING = {
    '<SEX-2>': 1,  # Female
    '<SEX-1>': 2,  # Male
}

# Data version configurations
DATA_VERSION_CONFIG = {
    'v6': {
        'att_time_unit': 'months',
        'att_time_mode': 'relative',
        'time_multiplier': 30,  # months to days
    },
    'v7': {
        'att_time_unit': 'months',
        'att_time_mode': 'absolute',
        'time_multiplier': 30,  # months to days
    },
}


def load_disease_mapping_from_csv(csv_file_path):
    """
    Load and parse disease mapping from a CSV file (nature_disease_map.csv format).
    
    The CSV should have columns: disease, icd_code, disease_id
    Uses 'disease' column as the major code and 'disease_id' column for mapping.
    
    Note: The original CSV has disease_id starting from 13, but Delphi requires
    the first disease ID to be 12. Therefore, all IDs are shifted by -1.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary mapping disease codes (e.g., 'A00') to integer IDs (starting from 12)
    """
    logging.info("Loading disease mapping from CSV: %s", csv_file_path)
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Disease mapping CSV file not found: {csv_file_path}")

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Validate required columns
    required_cols = ['disease', 'disease_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Get unique disease to disease_id mappings
    # Since multiple icd_codes map to the same disease, we take unique pairs
    unique_mappings = df[['disease', 'disease_id']].drop_duplicates()
    
    # Create mapping dictionary with ID shifted by -1 (13 -> 12, 14 -> 13, etc.)
    mapping = {}
    for _, row in unique_mappings.iterrows():
        disease_code = row['disease']
        original_id = int(row['disease_id'])
        # Shift by -1 so that A00 maps to 12 instead of 13
        adjusted_id = original_id - 1
        mapping[disease_code] = adjusted_id
    
    logging.info("Loaded %d diseases from CSV successfully!", len(mapping))
    if mapping:
        logging.info("ID range: %d to %d", min(mapping.values()), max(mapping.values()))
        # Log first few mappings for verification
        sample_items = list(mapping.items())[:5]
        logging.info("Sample mappings: %s", sample_items)
    
    return mapping

# Track unknown diseases for reporting
unknown_diseases = set()

def parse_disease_id(token_str, mapping):
    """
    Parse disease token and return disease ID.
    
    Token format: <DX-MAJOR_A00> -> extract 'A00'
    Prefix '<DX-MAJOR_' has length 10
    """
    if not token_str.startswith('<DX-MAJOR_'):
        return -1
    
    # Extract major code: <DX-MAJOR_A00> -> A00
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
        max_length=4096, 
        first_occurrence=True, 
        att_time_unit="days", 
        att_time_mode="relative"
    ):
    """
    Parses a sequence string into model input format (prediction points).
    
    Args:
        seq: Space-separated token string
        disease_mapping: Dictionary mapping disease codes to IDs
        sex_mapping: Dictionary mapping sex tokens to IDs
        demo_end_index: Index after demographic tokens (start of events)
        age_index: Index of age token
        sex_index: Index of sex token
        max_length: Maximum sequence length to process
        first_occurrence: Whether to only record the first occurrence of a disease
        att_time_unit: Time unit for ATT tokens - 'days', 'weeks', or 'months'
        att_time_mode: Time mode for ATT tokens - 'relative' (interval from previous) or 'absolute' (interval from initial age)
        
    Returns:
        tuple: (prediction_points, prediction_points_ages, prediction_points_disease_ids, sex_id)
    """
    tokens = seq.split(" ")

    prediction_points = []
    prediction_points_ages = []
    prediction_points_disease_ids = []

    # Parse sex
    sex_id = 0
    if len(tokens) > sex_index:
        token = tokens[sex_index]
        if token in sex_mapping:
            sex_id = sex_mapping[token]
        else:
            pass
    else:
        pass

    # Parse age (format: <AGE-XX>)
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
    
    # Define time unit multiplier to convert to days
    if att_time_unit == "days":
        time_multiplier = 1
    elif att_time_unit == "weeks":
        time_multiplier = 7
    elif att_time_unit == "months":
        time_multiplier = 30
    else:
        # Default to days if unknown unit
        time_multiplier = 1
    
    length = builtins.min(len(tokens) - 2, max_length - 2)
    i = demo_end_index

    while i < length:
        current_token = tokens[i]

        if current_token.startswith('<ATT-'):
            try:
                att_value = int(current_token[5:-1]) if current_token != "<ATT-0>" else 0
                # Convert ATT value to days based on the time unit
                att_days = att_value * time_multiplier
                
                # Apply time mode: relative (cumulative) vs absolute (direct)
                if att_time_mode == "relative":
                    # Relative mode: ATT value is interval from previous time point
                    days_passed += att_days
                else:
                    # Absolute mode: ATT value is interval from initial age token
                    days_passed = att_days
            except ValueError:
                pass

        # Updated check for MAJOR format prefix consistency
        elif current_token.startswith('<DX-MAJOR_'):
            # Token format: <DX-MAJOR_A00>
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
    Process all patients in the dataframe to generate model inputs.
    Using zip iteration for robustness against dynamic column names.
    
    Args:
        traj_df: Pandas DataFrame containing patient data
        pid_col: Column name for Patient ID
        seq_col: Column name for Sequence
        att_time_unit: Time unit for ATT tokens ('days', 'weeks', 'months')
        att_time_mode: Time mode for ATT tokens ('relative', 'absolute')
        disease_mapping: Dict for disease IDs
        sex_mapping: Dict for sex IDs
        demo_end_index: Index where demographic tokens end
        age_index: Index of age token
        sex_index: Index of sex token
        
    Returns:
        tuple: (Processed DataFrame, Patient ID mapping dictionary)
    """
    unique_pids = traj_df[pid_col].unique()
    sorted_pids = sorted(unique_pids)
    pid_map = {pid: idx for idx, pid in enumerate(sorted_pids)}

    all_records = []
    invalid_token_count = 0  

    logging.info("Processing %d patients from trajectory dataframe...", len(sorted_pids))

    # Using zip for performance and robustness
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
        
        # Commented out: Now keeping all patients including those with < 2 records
        # valid_pids = patient_counts[patient_counts >= 2].index
        # final_df = final_df[final_df['patient_id'].isin(valid_pids)]

        logging.info("Total patients (no filtering applied): %d", len(patient_counts))
        # logging.info("Patients after filtering (>= 2 records): %d", len(valid_pids))
        # logging.info("Patients removed: %d", len(patient_counts) - len(valid_pids))

        # Validate token IDs
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
    data_version: str = 'v6',
    seq_col: str = 'pre_entry_seq',
    pid_col: str = 'patient_id',
) -> np.ndarray:
    """
    Convert cohort dataframe with token sequences to Delphi binary format.
    
    This function processes the cohort dataframe containing pre_entry_seq
    (tokenized clinical sequences) and converts it to the format expected
    by Delphi's get_batch function: numpy array with columns
    [patient_id, patient_age, token_id].
    
    Args:
        cohort_df: DataFrame with patient sequences
        disease_mapping: Dict mapping disease codes to IDs
        data_version: 'v6' or 'v7' for time unit/mode configuration
        seq_col: Column name containing token sequences
        pid_col: Column name for patient IDs (will use row index if not present)
        
    Returns:
        numpy array of shape (N, 3) with columns [patient_id, patient_age, token_id]
    """
    # Get time configuration from data version
    version_config = DATA_VERSION_CONFIG.get(data_version, DATA_VERSION_CONFIG['v6'])
    att_time_unit = version_config['att_time_unit']
    att_time_mode = version_config['att_time_mode']
    
    logging.info("Preprocessing cohort to Delphi format...")
    logging.info("Data version: %s (time_unit=%s, time_mode=%s)", 
                 data_version, att_time_unit, att_time_mode)
    
    # Create temporary patient ID column if not present
    if pid_col not in cohort_df.columns:
        # Use row index as patient ID
        cohort_df = cohort_df.copy()
        cohort_df[pid_col] = range(len(cohort_df))
        logging.info("Using row index as patient_id")
    
    # Process all patients
    processed_df, pid_map = process_all_patients(
        traj_df=cohort_df,
        pid_col=pid_col,
        seq_col=seq_col,
        att_time_unit=att_time_unit,
        att_time_mode=att_time_mode,
        disease_mapping=disease_mapping,
        sex_mapping=SEX_MAPPING,
    )
    
    # Convert to numpy array in Delphi format: [patient_id, patient_age, token_id]
    data = processed_df[['patient_id', 'patient_age', 'token_id']].values.astype(np.float32)
    
    logging.info("Preprocessed data shape: %s", data.shape)
    logging.info("Unique patients: %d", len(np.unique(data[:, 0])))
    
    return data


# =============================================================================
# Embedding Extraction
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract patient embeddings from Delphi foundation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_version', type=str, default='v6',
                        choices=['v6', 'v7'],
                        help='Data version: v6 (relative time) or v7 (absolute time)')
    parser.add_argument('--model_path', type=str,
                        default=os.environ.get("DELPHI_MODEL_PATH", "/path/to/delphi_model_checkpoint.pt"),
                        help='Path to Delphi model checkpoint')
    parser.add_argument('--disease_map_path', type=str,
                        default=os.environ.get("DELPHI_DISEASE_MAP", "/path/to/delphi_disease_map.csv"),
                        help='Path to disease mapping CSV file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--block_size', type=int, default=48,
                        help='Maximum sequence length (Delphi context window)')
    parser.add_argument('--padding', type=str, default='none',
                        choices=['none', 'random', 'regular'],
                        help='Padding strategy for sequences')
    parser.add_argument('--select', type=str, default='right',
                        choices=['left', 'right', 'random'],
                        help='Sequence selection: right keeps recent events')
    parser.add_argument('--random_seed', type=int, default=66,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize embeddings to unit vectors.
    
    After normalization, dot product equals cosine similarity,
    which is useful for similarity search and clustering.
    """
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
    Extract embeddings from Delphi model using last-token pooling.
    
    This function manually executes the forward pass up to the final LayerNorm,
    then applies last-token pooling to get sequence-level representations.
    
    For causal models like Delphi, only the last token has seen the entire sequence,
    making it the appropriate choice for sequence-level representation.
    """
    idx = idx.to(device)
    age = age.to(device)

    with torch.no_grad():
        # Token embedding + Age embedding (matches model.forward exactly)
        tok_emb = model.transformer.wte(idx)
        age_emb = model.transformer.wae(age.unsqueeze(-1))
        # Apply token_drop + unconditional scaling to match original forward pass:
        #   x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        # In eval mode token_drop is identity, but the (1 - token_dropout) scaling
        # is always applied. Currently token_dropout=0.0 so this is 1.0, but we
        # keep it for correctness if a model with token_dropout > 0 is ever used.
        x = model.transformer.token_drop(tok_emb) * (1 - model.config.token_dropout)
        x = x + age_emb
        x = model.transformer.drop(x)

        # Build attention mask
        attn_mask = (idx > 0).view(idx.size(0), 1, 1, idx.size(1)) * \
                    (idx > 0).view(idx.size(0), 1, idx.size(1), 1)
        attn_mask = attn_mask * (torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0)
        attn_mask = attn_mask + (idx == 0).view(idx.size(0), 1, 1, idx.size(1)) * \
                    torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask * (torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0)

        # Pass through all Transformer blocks
        for block in model.transformer.h:
            x, _ = block(x, attn_mask)

        # Final LayerNorm
        x = model.transformer.ln_f(x)

        # Last-token pooling
        attention_mask = (idx > 0).long()
        seq_lengths = attention_mask.sum(dim=1)
        last_token_indices = torch.clamp(seq_lengths - 1, min=0)

        batch_size = x.size(0)
        embeddings = x[
            torch.arange(batch_size, device=device),
            last_token_indices,
            :
        ]

    return embeddings


@time_execution
def add_delphi_embeddings_to_cohort(
    cohort_df: pd.DataFrame,
    output_file: str,
    model_path: str,
    disease_map_path: str,
    data_version: str = 'v6',
    batch_size: int = 256,
    block_size: int = 48,
    padding: str = 'none',
    select: str = 'right',
    seq_col: str = 'pre_entry_seq',
) -> pd.DataFrame:
    """
    Add Delphi embedding columns to the cohort dataframe.
    
    Uses last-token pooling appropriate for causal/decoder models like Delphi,
    where only the final token has seen the entire sequence context.
    
    Embeddings are L2 normalized to unit vectors, making dot product equivalent
    to cosine similarity. This is optimal for patient similarity search and
    propensity score matching augmentation.
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)

    if device.type == 'cuda':
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        logging.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Filter out rows with null sequences
    cohort_df_filtered = cohort_df.dropna(subset=[seq_col]).reset_index(drop=True)
    null_count = len(cohort_df) - len(cohort_df_filtered)
    if null_count > 0:
        logging.warning("Filtered out %d patients with null %s", null_count, seq_col)
    logging.info("Processing %d patients with valid sequences", len(cohort_df_filtered))

    # Load disease mapping
    logging.info("Loading disease mapping from %s", disease_map_path)
    disease_mapping = load_disease_mapping_from_csv(disease_map_path)
    logging.info("Loaded %d disease mappings", len(disease_mapping))

    # Preprocess cohort to Delphi format
    data = preprocess_cohort_to_delphi_format(
        cohort_df=cohort_df_filtered,
        disease_mapping=disease_mapping,
        data_version=data_version,
        seq_col=seq_col,
    )

    # Get patient to index mapping
    p2i = get_p2i(data)
    num_patients = len(p2i)
    logging.info("Total patients in preprocessed data: %d", num_patients)

    # Load Delphi model
    logging.info("Loading Delphi model from %s", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']

    logging.info("Model config: n_layer=%d, n_head=%d, n_embd=%d, block_size=%d, vocab_size=%d",
                 model_args['n_layer'], model_args['n_head'], model_args['n_embd'],
                 model_args['block_size'], model_args['vocab_size'])

    conf = DelphiConfig(**model_args)
    model = Delphi(conf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    n_embd = conf.n_embd
    logging.info("Model loaded successfully. Embedding dimension: %d", n_embd)

    # Process in batches
    logging.info("Generating embeddings...")
    logging.info("Settings: batch_size=%d, block_size=%d, padding='%s', select='%s'",
                 batch_size, block_size, padding, select)

    all_embeddings = []
    
    num_batches = (num_patients + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches", file=sys.stdout):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_patients)
        batch_patient_indices = list(range(start_idx, end_idx))

        batch_data = get_batch(
            ix=batch_patient_indices,
            data=data,
            p2i=p2i,
            select=select,
            block_size=block_size,
            device='cpu',
            padding=padding,
        )

        idx = batch_data[0].to(device)
        age = batch_data[1].to(device).float()

        batch_embeddings = get_delphi_embeddings(model, idx, age, device)
        batch_embeddings = batch_embeddings.float().cpu().numpy()
        all_embeddings.append(batch_embeddings)

        if device.type == 'cuda' and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Concatenate and normalize
    embeddings_array = np.vstack(all_embeddings)
    logging.info("Generated embeddings shape: %s", embeddings_array.shape)

    embeddings_array = normalize_embeddings(embeddings_array)
    logging.info("Normalized embeddings to unit vectors (L2 norm = 1)")

    norms = np.linalg.norm(embeddings_array, axis=1)
    logging.info("Embedding norms - min: %.6f, max: %.6f, mean: %.6f",
                 norms.min(), norms.max(), norms.mean())

    # Create embedding dataframe
    embedding_df = pd.DataFrame(
        embeddings_array,
        columns=[f'rep_{i}' for i in range(embeddings_array.shape[1])]
    )

    # Concatenate with original data
    cohort_df_with_embedding = pd.concat([
        cohort_df_filtered.reset_index(drop=True),
        embedding_df.reset_index(drop=True),
    ], axis=1)

    logging.info("Final dataframe: %d patients x %d columns",
                 cohort_df_with_embedding.shape[0], cohort_df_with_embedding.shape[1])

    # Save results
    logging.info("Saving to %s", output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    cohort_df_with_embedding.to_csv(output_file, index=False)
    logging.info("Save complete!")

    return cohort_df_with_embedding


@time_execution
def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info("=" * 60)
    logging.info("Delphi Patient Embedding Extraction Script")
    logging.info("=" * 60)
    logging.info("Log file: %s", log_file)

    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Construct file paths (same convention as ReClaim)
    working_dir = os.getcwd()
    version = int(args.data_version[1:])  # e.g., "v6" -> 6

    final_cohort_dir = os.path.join(working_dir, 'intermediate', f'v{version}', 'final_cohort')
    input_file = os.path.join(final_cohort_dir, "cohort_with_pre_entry_seq.csv")
    output_file = os.path.join(final_cohort_dir, "cohort_with_delphi_embeddings.csv")

    logging.info("Working directory: %s", working_dir)
    logging.info("Data version: %s (version=%d)", args.data_version, version)
    logging.info("  - Time unit: %s", DATA_VERSION_CONFIG[args.data_version]['att_time_unit'])
    logging.info("  - Time mode: %s", DATA_VERSION_CONFIG[args.data_version]['att_time_mode'])
    logging.info("Model path: %s", args.model_path)
    logging.info("Disease map: %s", args.disease_map_path)
    logging.info("Input file: %s", input_file)
    logging.info("Output file: %s", output_file)
    logging.info("Batch size: %d", args.batch_size)
    logging.info("Block size: %d", args.block_size)
    logging.info("Padding: %s", args.padding)
    logging.info("Select: %s", args.select)

    # Check input files exist
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.disease_map_path):
        raise FileNotFoundError(f"Disease mapping not found: {args.disease_map_path}")

    # Load cohort data
    logging.info("Loading cohort data...")
    cohort_df = pd.read_csv(input_file, low_memory=False)
    logging.info("Loaded cohort: %d patients x %d columns", cohort_df.shape[0], cohort_df.shape[1])

    # Extract embeddings
    add_delphi_embeddings_to_cohort(
        cohort_df=cohort_df,
        output_file=output_file,
        model_path=args.model_path,
        disease_map_path=args.disease_map_path,
        data_version=args.data_version,
        batch_size=args.batch_size,
        block_size=args.block_size,
        padding=args.padding,
        select=args.select,
    )

    logging.info("=" * 60)
    logging.info("Embedding extraction complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
