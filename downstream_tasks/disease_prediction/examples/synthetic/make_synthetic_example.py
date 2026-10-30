#!/usr/bin/env python3
"""
Build a fully synthetic, CPU-sized fixture for the disease-onset evaluation.

Writes into --out_dir:
  trajectories/part-0.parquet  - synthetic EHR trajectories (evaluation input)
  model/                       - tiny Qwen3 causal LM + word-level tokenizer

No real patient data is used. Patients are simulated from a small hazard
model with planted risk factors (e.g. prediabetes/obesity -> type 2 diabetes,
hypertension + hyperlipidemia -> ischaemic heart disease), so a model that
learns the structure should score AUC > 0.5 on those endpoints. The tiny model
is trained for a few hundred steps on a separate synthetic training cohort in
the post-training format: `<history> <NY> <INSTRUCT-DX> <future new majors> <eos>`.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

# code -> (annual base hazard, age slope per decade over 40, {risk factor: log-hazard ratio})
DISEASES = {
    "I10": (0.030, 0.45, {"E66": 0.8}),
    "E78": (0.030, 0.35, {"E66": 0.5}),
    "E66": (0.020, 0.00, {}),
    "R73": (0.015, 0.30, {"E66": 1.0}),
    "E11": (0.006, 0.35, {"R73": 2.2, "E66": 1.2, "E78": 0.5}),
    "I25": (0.004, 0.60, {"I10": 1.3, "E78": 1.2, "E11": 0.8}),
    "N18": (0.003, 0.55, {"E11": 1.6, "I10": 1.2}),
    "J45": (0.010, -0.10, {}),
    "F32": (0.015, 0.00, {}),
    "K21": (0.020, 0.15, {"E66": 0.6}),
}
ACUTE = ["J06", "M54", "R07", "Z00"]  # non-chronic codes used to fill visits
N_YEARS = 8
HISTORY_YEARS = 6  # years before the nominal "thru_2022" cutoff
START_YEAR = 2015


def build_vocab():
    specials = ["<unk>", "<sos>", "<eos>", "<pad>"]
    tokens = ["<SEX-1>", "<SEX-2>", "<NY>", "<INSTRUCT-DX>"]
    tokens += ["<VT-outpatient>", "<VT-inpatient>", "<DX-PRINCIPAL>", "<DX-SECONDARY>"]
    tokens += [f"<DOBYR-{y}>" for y in range(1900, 2011)]
    tokens += [f"<AGE-{a}>" for a in range(0, 111)]
    tokens += [f"<ATT-{k}>" for k in range(0, 13)]
    tokens += [f"<DX-MAJOR_{c}>" for c in list(DISEASES) + ACUTE]
    tokens += [f"<DX-MINOR_{d}>" for d in range(10)]
    tokens += [f"<RX-{i:03d}>" for i in range(20)]
    tokens += [f"<COST-{b}>" for b in ["LOW", "MID", "HIGH"]]
    return {t: i for i, t in enumerate(specials + tokens)}


def simulate_patient(rng):
    """Return (token list, index of last history token) for one synthetic patient."""
    sex = int(rng.integers(1, 3))
    age0 = int(rng.integers(25, 81))
    tokens = ["<sos>", f"<SEX-{sex}>", f"<DOBYR-{START_YEAR - age0}>", f"<AGE-{age0}>"]
    has = set()
    history_end = None
    for year in range(N_YEARS):
        if year > 0:
            tokens.append("<NY>")
        if year == HISTORY_YEARS:
            history_end = len(tokens) - 1
        age = age0 + year
        new = []
        for code, (base, age_slope, risks) in DISEASES.items():
            if code in has:
                continue
            log_h = np.log(base) + age_slope * (age - 40) / 10
            log_h += sum(w for r, w in risks.items() if r in has)
            if rng.random() < 1 - np.exp(-np.exp(log_h)):
                new.append(code)
        months = np.sort(rng.integers(0, 12, size=int(rng.poisson(2.5)) + 1))
        new_at = {code: int(rng.integers(len(months))) for code in new}
        prev = 0
        for v, month in enumerate(months):
            codes = [c for c, at in new_at.items() if at == v]
            codes += [c for c in sorted(has) if rng.random() < 0.5]  # chronic codes recur
            if not codes or rng.random() < 0.3:
                codes.append(str(rng.choice(ACUTE)))
            vt = "<VT-inpatient>" if rng.random() < 0.05 else "<VT-outpatient>"
            tokens += [f"<ATT-{int(month - prev)}>", vt, "<DX-PRINCIPAL>"]
            for j, c in enumerate(codes):
                if j == 1:
                    tokens.append("<DX-SECONDARY>")
                tokens += [f"<DX-MAJOR_{c}>", f"<DX-MINOR_{int(rng.integers(10))}>"]
            if rng.random() < 0.4:
                tokens.append(f"<RX-{int(rng.integers(20)):03d}>")
            tokens.append(str(rng.choice(["<COST-LOW>", "<COST-MID>", "<COST-HIGH>"])))
            prev = month
        has.update(new)
    tokens.append("<eos>")
    return tokens, history_end


def make_trajectories(n, rng, id_offset=0):
    rows = []
    for i in range(n):
        tokens, history_end = simulate_patient(rng)
        rows.append(
            {
                "person_id": id_offset + i,
                "seq": " ".join(tokens),
                "token_count_thru_2022": history_end,
                "total_token_count": len(tokens),
            }
        )
    return pd.DataFrame(rows)


def instruct_example(tokens, rng):
    """Post-training style sample: history up to a visit boundary, then the new future majors."""
    visit_starts = [i for i, t in enumerate(tokens) if t.startswith("<ATT-")]
    split = int(rng.choice(visit_starts[1:])) if len(visit_starts) > 1 else visit_starts[0]
    seen = {t for t in tokens[:split] if t.startswith("<DX-MAJOR")}
    future = []
    for t in tokens[split:]:
        if t.startswith("<DX-MAJOR") and t not in seen:
            seen.add(t)
            future.append(t)
    prompt = tokens[:split] + ["<NY>", "<INSTRUCT-DX>"]
    return prompt, future + ["<eos>"]


def train_tiny_model(vocab, train_df, steps, seed, max_len=1024):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=max_len,
        bos_token_id=vocab["<sos>"],
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    if steps <= 0:
        return model
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    seqs = [s.split(" ") for s in train_df["seq"]]
    model.train()
    for step in range(steps):
        batch = [instruct_example(seqs[i], rng) for i in rng.integers(len(seqs), size=32)]
        length = min(max_len, max(len(p) + len(t) for p, t in batch))
        input_ids = torch.full((len(batch), length), vocab["<pad>"])
        labels = torch.full((len(batch), length), -100)
        attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
        for b, (prompt, target) in enumerate(batch):
            ids = [vocab[t] for t in prompt + target][:length]
            input_ids[b, : len(ids)] = torch.tensor(ids)
            attention_mask[b, : len(ids)] = 1
            # Loss only on the targets after <INSTRUCT-DX>.
            labels[b, len(prompt) : len(ids)] = torch.tensor(ids[len(prompt) :])
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"[train] step {step:4d}  loss {loss.item():.3f}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_eval", type=int, default=3000, help="Evaluation patients")
    parser.add_argument("--n_train", type=int, default=6000, help="Patients used to train the tiny model")
    parser.add_argument("--train_steps", type=int, default=400, help="0 = keep random init (AUC ~ 0.5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab = build_vocab()

    traj_dir = os.path.join(args.out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    eval_df = make_trajectories(args.n_eval, rng)
    eval_df.to_parquet(os.path.join(traj_dir, "part-0.parquet"), index=False)
    print(f"[data] wrote {len(eval_df)} evaluation trajectories to {traj_dir}")

    train_df = make_trajectories(args.n_train, rng, id_offset=10**6)
    model = train_tiny_model(vocab, train_df, args.train_steps, args.seed)

    model_dir = os.path.join(args.out_dir, "model")
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "synthetic_diseases.json"), "w") as f:
        json.dump({c: {"base": b, "age_slope": s, "risks": r} for c, (b, s, r) in DISEASES.items()}, f, indent=2)
    print(f"[model] wrote tokenizer ({len(vocab)} tokens) and model to {model_dir}")


if __name__ == "__main__":
    main()
