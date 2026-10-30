#!/usr/bin/env python3
"""Tabulate every model_gen/<model>/timing.json AND timing_<N>.json into
  results/<split>/inference_cost.csv             one row per (model, section) + an evaluation_total row per model
  results/<split>/inference_cost_comparison.csv  pretrain-MC anchor vs post-train readout, per backbone <size><ver>
Fields per row: engine, gpu, n_gpus, patients, wall_s, gpu_s, gpu_s_per_patient, gpu_h,
tokens_generated_per_patient, tokens_generated_per_trajectory, trajectories, prompt_tokens, engine_load_s, generate_s.
Every path (MC generator, vLLM readout, HF readout) defines wall the same way — engine/model load + generation,
per section — so rows are comparable wherever engine, GPU and n_gpus match; the comparison table says so
explicitly (hardware_matched) and reports the MC / post-train ratios in GPU-s per patient and generated tokens.

Naming: a Monte-Carlo record (trajectories > 1) is filed as <model>-<trajectories>, e.g.
reclaim-l-pretrain-64, the same id the regression / classification / AUC rows use for that N, so every N of an
anchor coexists.  generate_vllm.py writes timing_<N>.json per N; a plain timing.json of a Monte-Carlo run is read as
well (duplicates of the same (model, section) resolve to the most recent 'recorded').  Post-train readouts
(trajectories = 1) write timing.json and keep their name."""
import os, sys, json, glob, re, argparse
import pandas as pd

p = argparse.ArgumentParser(); p.add_argument("--data_root", required=True); p.add_argument("--result_root", required=True)
p.add_argument("--from_csv", default=None, help="Rebuild from the SECTION rows of an existing inference_cost*.csv instead of "
               "timing files")
p.add_argument("--gen_num", type=int, default=None, help="Restrict the comparison table to MC anchors with this many trajectories")
a = p.parse_args()
if a.from_csv:
    df = pd.read_csv(a.from_csv); df = df[df["section"] != "evaluation_total"].copy()
else:
    rows = []
    files = sorted(glob.glob(os.path.join(a.data_root, "*", "model_gen", "*", "timing.json"))) + \
            sorted(glob.glob(os.path.join(a.data_root, "*", "model_gen", "*", "timing_*.json")))
    for tj in files:
        for r in json.load(open(tj)):
            r = {k: v for k, v in r.items() if k not in ("engine_cfg", "sampling")}
            k = r.get("trajectories")
            if k is not None and int(k) > 1:                    # MC anchor: file under its N
                mfile = re.search(r"timing_(\d+)\.json$", tj)
                assert mfile is None or int(mfile.group(1)) == int(k), f"{tj}: filename N != trajectories {k}"
                r["model"] = f'{r["model"]}-{int(k)}'
            rows.append(r)
    if not rows:
        print("no timing files found under", a.data_root); sys.exit(0)
    df = pd.DataFrame(rows)
    if "recorded" in df:
        df = df.sort_values("recorded").drop_duplicates(subset=["model", "section"], keep="last")
for c in ("trajectories", "tokens_generated_per_trajectory", "prompt_tokens", "engine_load_s", "generate_s"):
    if c not in df: df[c] = None
df["_tok"] = df["tokens_generated_per_patient"] * df["patients"]          # patient-weighted means below
df["_tok_traj"] = df["tokens_generated_per_trajectory"] * df["patients"]
tot = (df.groupby(["model", "engine", "gpu", "n_gpus"], as_index=False)
         .agg(patients=("patients", "sum"), wall_s=("wall_s", "sum"), gpu_s=("gpu_s", "sum"), gpu_h=("gpu_h", "sum"),
              _tok=("_tok", "sum"), _tok_traj=("_tok_traj", "sum"),
              trajectories=("trajectories", "max"), prompt_tokens=("prompt_tokens", "sum"),
              engine_load_s=("engine_load_s", "sum"), generate_s=("generate_s", "sum")))
tot["tokens_generated_per_patient"] = tot["_tok"] / tot["patients"]
tot["tokens_generated_per_trajectory"] = tot["_tok_traj"] / tot["patients"]
tot = tot.drop(columns=["_tok", "_tok_traj"]); df = df.drop(columns=["_tok", "_tok_traj"])
tot["gpu_s_per_patient"] = tot["gpu_s"] / tot["patients"]; tot["section"] = "evaluation_total"
out = pd.concat([df, tot], ignore_index=True).sort_values(["model", "section"])
os.makedirs(a.result_root, exist_ok=True)
path = os.path.join(a.result_root, "inference_cost.csv"); out.to_csv(path, index=False)
print(f"wrote {path}: {len(df)} section rows, {len(tot)} evaluation totals")
print(tot[["model", "engine", "gpu", "n_gpus", "patients", "gpu_s_per_patient", "gpu_h", "tokens_generated_per_patient"]].to_string(index=False))

# ---- pretrain MC vs post-train, per backbone: reclaim-<sv>-pretrain-<N>  vs  reclaim-<sv>-instruct-* ----
sv = lambda m: (re.match(r"reclaim-([sml])-", m) or [None, None])[1]
tot["backbone"] = tot["model"].map(sv)
mc = tot[tot["model"].str.contains(r"-pretrain-\d+$")]
if a.gen_num is not None:
    mc = mc[mc["trajectories"].astype(int) == a.gen_num]
pt = tot[tot["model"].str.contains("-posttrain")]
cmp_rows = []
for _, q in pt.iterrows():
    for _, m in mc[mc["backbone"] == q["backbone"]].iterrows():
        cmp_rows.append(dict(
            backbone=q["backbone"], posttrain_model=q["model"], mc_model=m["model"],
            posttrain_engine=q["engine"], mc_engine=m["engine"],
            posttrain_gpu=f'{q["gpu"]} x{q["n_gpus"]}', mc_gpu=f'{m["gpu"]} x{m["n_gpus"]}',
            hardware_matched=bool(q["gpu"] == m["gpu"] and q["n_gpus"] == m["n_gpus"]),
            posttrain_gpu_s_per_patient=round(q["gpu_s_per_patient"], 5), mc_gpu_s_per_patient=round(m["gpu_s_per_patient"], 5),
            gpu_s_ratio_mc_over_posttrain=round(m["gpu_s_per_patient"] / q["gpu_s_per_patient"], 1),
            posttrain_gpu_h=round(q["gpu_h"], 3), mc_gpu_h=round(m["gpu_h"], 3),
            posttrain_tokens_per_patient=q["tokens_generated_per_patient"], mc_tokens_per_patient=m["tokens_generated_per_patient"],
            token_ratio_mc_over_posttrain=round(m["tokens_generated_per_patient"] / q["tokens_generated_per_patient"], 1)
                if q["tokens_generated_per_patient"] and m["tokens_generated_per_patient"] else None,
            mc_trajectories=int(m["trajectories"])))
if cmp_rows:
    cpath = os.path.join(a.result_root, "inference_cost_comparison.csv")
    cdf = pd.DataFrame(cmp_rows).sort_values(["backbone", "posttrain_model", "mc_trajectories"]); cdf.to_csv(cpath, index=False)
    print(f"wrote {cpath}: {len(cdf)} pretrain-vs-posttrain pairs")
    print(cdf[["posttrain_model", "mc_model", "hardware_matched", "gpu_s_ratio_mc_over_posttrain", "token_ratio_mc_over_posttrain",
               "posttrain_gpu_h", "mc_gpu_h"]].to_string(index=False))
else:
    print("no pretrain/posttrain pair on the same backbone yet -> inference_cost_comparison.csv not written")
