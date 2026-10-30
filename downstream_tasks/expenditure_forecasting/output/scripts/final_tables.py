#!/usr/bin/env python3
"""Resolve the reported roster into the paper's expenditure tables (output/tables/).

The roster is fixed in output/scripts/roster.sh and passed here verbatim; every id is looked up
exactly in the metric tables written by e4_eval (no pattern matching), and a roster model without
a row is reported as missing, never substituted.

Writes (output/tables/):
  benchmark.csv / .tex            both evaluations x roster: n, R2, MAE, RMSE, HNHC F1 / precision /
                                  recall, ROC-AUC, PR-AUC
  benchmark_by_payer.csv / .tex   the same within CCAE+MDCR and MDCD
  inference_cost.csv / .tex       cost of the Monte-Carlo anchor and the post-trained readout, with ratios
  ROSTER.md                       the ids behind the tables, the files read, any missing cell"""
import os, re, sys, argparse, datetime
import numpy as np, pandas as pd

SEC = ["retrospective", "prospective"]
PAYERS = {"CR": "CCAE+MDCR", "MD": "MDCD"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", required=True)
    p.add_argument("--models", nargs="+", required=True, help="ordered roster (FINAL_TABLE_MODELS)")
    p.add_argument("--gen_num", type=int, required=True, help="MC trajectories of the reported anchor (FINAL_N_GENERATION)")
    p.add_argument("--lightgbm_window", default="1year", help="window of the reported LightGBM pair; other windows are ablation rows")
    p.add_argument("--cutoff_hnhc", default="30000")
    p.add_argument("--tables_dir", required=True, help="output/tables")
    return p.parse_args()


def describe(m, a):
    """(role, size, stage, table label) for a roster id."""
    if m == "trainmean":   return "anchor", "", "", "Train-mean anchor"
    if m == "persistence": return "anchor", "", "", "Persistence (last-year cost)"
    g = re.match(r"^lightgbm-(reg|cls)-(\w+)$", m)
    if g:
        w = "1-yr window" if g.group(2) == "1year" else "all history"
        role = "lightgbm" if g.group(2) == a.lightgbm_window else "lightgbm_ablation"
        return role, "", "", f"LightGBM {'regressor' if g.group(1) == 'reg' else 'classifier'}, {w}"
    g = re.match(r"^reclaim-([sml])-pretrain-(\d+)$", m)
    if g:
        assert int(g.group(2)) == a.gen_num, f"{m}: N != FINAL_N_GENERATION {a.gen_num}"
        return "pretrain", g.group(1).upper(), "pre-training", f"ReClaim-{g.group(1).upper()}, pre-training (MC, {a.gen_num} trajectories)"
    g = re.match(r"^reclaim-([sml])-posttrain$", m)
    if g:
        return "posttrain", g.group(1).upper(), "post-training", f"ReClaim-{g.group(1).upper()}, post-training"
    raise SystemExit(f"roster id not understood: {m}")


def read(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    keys = [c for c in ("source", "model", "cost_type") if c in df.columns]
    return df.drop_duplicates(subset=keys, keep="last")


def cell(df, model, source, col):
    if df is None: return np.nan
    r = df[(df["model"] == model) & (df["source"] == source)]
    if "cost_type" in df.columns: r = r[r["cost_type"] == "total"]
    return float(r[col].values[0]) if len(r) and pd.notna(r[col].values[0]) else np.nan


def main():
    a = parse_args(); out = a.tables_dir; os.makedirs(out, exist_ok=True)
    roster = [(m,) + describe(m, a) for m in a.models]
    print("Reported roster (output/scripts/roster.sh):"); [print(f"  {r[1]:10s} {r[2]:2s} {r[0]}") for r in roster]
    files, missing, rows, payer_rows = [], [], [], []
    for sec in SEC:
        base = os.path.join(a.result_root, sec)
        reg = read(os.path.join(base, "regression", "regression_metrics.csv"))
        cls = read(os.path.join(base, "classification", a.cutoff_hnhc, "classification_metrics.csv"))
        auc = read(os.path.join(base, "binary_auc", a.cutoff_hnhc, "auc_metrics.csv"))
        files += [os.path.join(base, f) for f in ("regression/regression_metrics.csv",
                  f"classification/{a.cutoff_hnhc}/classification_metrics.csv", f"binary_auc/{a.cutoff_hnhc}/auc_metrics.csv")]
        for m, role, size, stage, label in roster:
            for src, tag in [("total", None)] + list(PAYERS.items()):
                n = cell(reg, m, src, "n_valid")
                if np.isnan(n): n = cell(auc, m, src, "n_samples")
                rec = dict(section=sec, model=m, role=role, size=size, stage=stage, label=label, n=n,
                           R2=cell(reg, m, src, "R2"), MAE=cell(reg, m, src, "MAE"), RMSE=cell(reg, m, src, "RMSE"),
                           F1_hnhc=cell(cls, m, src, "f1-binary"), prec_hnhc=cell(cls, m, src, "precision-binary"),
                           rec_hnhc=cell(cls, m, src, "recall-binary"),
                           ROC_AUC=cell(auc, m, src, "roc_auc"), PR_AUC=cell(auc, m, src, "pr_auc"))
                if src == "total":
                    rows.append(rec)
                    # structural blanks are expected; anything else on the roster is a missing cell
                    want = {"anchor": ["R2", "MAE"], "lightgbm": ["R2", "MAE"] if "-reg-" in m else ["ROC_AUC", "PR_AUC"],
                            "lightgbm_ablation": ["R2", "MAE"] if "-reg-" in m else ["ROC_AUC", "PR_AUC"],
                            "pretrain": ["R2", "MAE", "ROC_AUC", "PR_AUC"], "posttrain": ["R2", "MAE", "ROC_AUC", "PR_AUC"]}[role]
                    for c in want:
                        if np.isnan(rec[c]): missing.append(f"{sec}: {m} has no {c}")
                else:
                    rec = dict(rec, payer=tag); payer_rows.append(rec)
    S = pd.DataFrame(rows); P = pd.DataFrame(payer_rows)
    S.to_csv(os.path.join(out, "benchmark.csv"), index=False); P.to_csv(os.path.join(out, "benchmark_by_payer.csv"), index=False)

    # ---- inference cost: evaluation totals of the roster + MC / post-train ratios per size ----
    cpath = os.path.join(a.result_root, "inference_cost.csv"); files.append(cpath)
    C = None
    if os.path.exists(cpath):
        c = pd.read_csv(cpath); c = c[c["section"] == "evaluation_total"].set_index("model")
        crows = []
        for m, role, size, stage, label in roster:
            if role not in ("pretrain", "posttrain"): continue
            if m not in c.index: missing.append(f"inference cost: no evaluation_total row for {m}"); continue
            r = c.loc[m]
            crows.append(dict(model=m, role=role, size=size, stage=stage, engine=r["engine"], gpu=r["gpu"], n_gpus=int(r["n_gpus"]),
                              patients=int(r["patients"]), trajectories=int(r["trajectories"]) if pd.notna(r["trajectories"]) else 1,
                              tokens_generated_per_patient=float(r["tokens_generated_per_patient"]),
                              gpu_s_per_patient=float(r["gpu_s_per_patient"]), gpu_h=float(r["gpu_h"])))
        C = pd.DataFrame(crows)
        if len(C):
            for size in sorted(C["size"].unique()):
                pre, post = C[(C["size"] == size) & (C.role == "pretrain")], C[(C["size"] == size) & (C.role == "posttrain")]
                if len(pre) and len(post):
                    C.loc[post.index, "gpu_s_ratio_mc_over_posttrain"] = float(pre["gpu_s_per_patient"].iloc[0] / post["gpu_s_per_patient"].iloc[0])
                    C.loc[post.index, "token_ratio_mc_over_posttrain"] = float(pre["tokens_generated_per_patient"].iloc[0] / post["tokens_generated_per_patient"].iloc[0])
                    C.loc[post.index, "hardware_matched"] = bool(pre["gpu"].iloc[0] == post["gpu"].iloc[0] and pre["n_gpus"].iloc[0] == post["n_gpus"].iloc[0])
            C.to_csv(os.path.join(out, "inference_cost.csv"), index=False)
    else:
        missing.append("inference cost: results/<split>/inference_cost.csv not found")

    # ---- LaTeX ----
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"; f0 = lambda v: "--" if pd.isna(v) else f"{v:,.0f}"
    prev = {sec: S[(S.section == sec) & (S.model.str.startswith("lightgbm-cls-"))] for sec in SEC}
    def n_of(sec): v = S[(S.section == sec)]["n"].dropna(); return int(v.max()) if len(v) else 0
    gpu_txt = ""
    if C is not None and len(C):
        g = C.iloc[0]; gpu_txt = f"{g['n_gpus']} $\\times$ {g['gpu']} (data-parallel)"
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{\textcolor{blue}{Complete benchmark results for next-year healthcare expenditure forecasting}. Held-out cohort, "
             f"retrospective (n = {n_of('retrospective'):,}) and prospective (n = {n_of('prospective'):,}) evaluations. Regression: total next-year "
             r"expenditure censored at \$1,000,000 (R$^2$, MAE and RMSE in dollars). HNHC identification: next-year total $\geq$ \$" + f"{int(a.cutoff_hnhc):,}"
             r"; ROC-AUC and PR-AUC are threshold-free (the classifier's output, the post-trained model's bin mass above the cutoff, or the fraction of "
             f"Monte-Carlo trajectories above it); F1, precision and recall use predicted dollars $\\geq$ the cutoff for every dollar-emitting model and the "
             r"classifier's argmax. A classifier emits no dollars; the train-mean anchor never flags HNHC. Pre-training = "
             f"{a.gen_num}-trajectory Monte-Carlo readout of the pre-trained backbone; post-training = the post-trained model (100,000 pairs; one response token, the next-year total).}}",
             r"\label{tab:expenditure_full_results}   % requires \usepackage{booktabs}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\hline",
             r" & \multicolumn{3}{c}{\textbf{Regression (total expenditure)}} & \multicolumn{5}{c}{\textbf{HNHC identification ($\geq$ \$" + f"{int(a.cutoff_hnhc):,}" + r")}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}",
             r"\textbf{Model} & \textbf{R$^2$} & \textbf{MAE} & \textbf{RMSE} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} \\", r"\hline"]
    for sec in SEC:
        lines.append(r"\multicolumn{9}{@{}l}{\textit{" + sec.capitalize() + r" evaluation}} \\")
        for _, r in S[S.section == sec].iterrows():
            lab = r["label"]; bold = r["role"] == "posttrain" and r["size"] == "L"
            lab = r"\textbf{" + lab + "}" if bold else lab
            lines.append(f"{lab} & {f3(r.R2)} & {f0(r.MAE)} & {f0(r.RMSE)} & {f3(r.ROC_AUC)} & {f3(r.PR_AUC)} & {f3(r.F1_hnhc)} & {f3(r.prec_hnhc)} & {f3(r.rec_hnhc)} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "benchmark.tex"), "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{\textcolor{blue}{Sensitivity analysis: next-year total expenditure forecasting stratified by payer}. Held-out cohort split by the enrollee's "
             r"payer database: CCAE+MDCR (commercial and Medicare supplemental) and MDCD (Medicaid). Regression on the censored total (R$^2$, MAE in dollars); "
             r"HNHC identification (next-year total $\geq$ \$" + f"{int(a.cutoff_hnhc):,}" + r"; ROC-AUC, PR-AUC). Model definitions as in Table~\ref{tab:expenditure_full_results}.}",
             r"\label{tab:expenditure_by_payer}   % requires \usepackage{booktabs}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\hline",
             r" & \multicolumn{4}{c}{\textbf{CCAE+MDCR}} & \multicolumn{4}{c}{\textbf{MDCD}} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"\textbf{Model} & \textbf{R$^2$} & \textbf{MAE} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{R$^2$} & \textbf{MAE} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\", r"\hline"]
    for sec in SEC:
        ns = {t: P[(P.section == sec) & (P.payer == t)]["n"].dropna() for t in PAYERS.values()}
        lines.append(r"\multicolumn{9}{@{}l}{\textit{" + sec.capitalize() + " evaluation" +
                     "".join(f"; {t} n = {int(v.max()):,}" for t, v in ns.items() if len(v)) + r"}} \\")
        for m, role, size, stage, label in roster:
            cr = P[(P.section == sec) & (P.model == m) & (P.payer == "CCAE+MDCR")].iloc[0]
            md = P[(P.section == sec) & (P.model == m) & (P.payer == "MDCD")].iloc[0]
            lab = r"\textbf{" + label + "}" if (role == "posttrain" and size == "L") else label
            lines.append(f"{lab} & {f3(cr.R2)} & {f0(cr.MAE)} & {f3(cr.ROC_AUC)} & {f3(cr.PR_AUC)} & {f3(md.R2)} & {f0(md.MAE)} & {f3(md.ROC_AUC)} & {f3(md.PR_AUC)} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "benchmark_by_payer.tex"), "w").write("\n".join(lines) + "\n")

    if C is not None and len(C):
        tot_pat = int(C["patients"].max())
        lines = [r"\begin{table}[H]", r"\centering",
                 r"\caption{\textcolor{blue}{Inference cost of pre-trained versus post-trained ReClaim on the expenditure forecasting task}. GPU-hours to produce "
                 f"annual total-expenditure forecasts for the full held-out expenditure cohort ({tot_pat:,} enrollees), recorded by the evaluation pipeline for the runs "
                 f"reported in this paper on identical hardware and software: {gpu_txt}, the same vLLM engine configuration for both readouts, one job at a time, "
                 r"wall-clock time $\times$ GPUs including engine start-up. Pre-train is the Monte-Carlo readout of the pre-trained model, which draws "
                 f"{a.gen_num} future-claims trajectories per enrollee and aggregates their costs; post-train is the expenditure-post-trained model, which decodes one "
                 r"cost token greedily. Mean generated tokens per enrollee = total tokens generated by the engine divided by the number of enrollees. Ratio = pre-train divided by post-train within each block.}",
                 r"\label{tab:inference_cost}", r"\small", r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{@{}lrrr@{}}", r"\hline",
                 r"\textbf{Backbone} & \textbf{Pre-train} & \textbf{Post-train} & \textbf{Ratio} \\", r"\hline",
                 r"\multicolumn{4}{@{}l}{\textit{Mean generated tokens per enrollee}} \\"]
        def pair(size):
            pre = C[(C["size"] == size) & (C.role == "pretrain")]; post = C[(C["size"] == size) & (C.role == "posttrain")]
            return (pre.iloc[0] if len(pre) else None), (post.iloc[0] if len(post) else None)
        for size in ["S", "M", "L"]:
            pre, post = pair(size)
            if pre is None or post is None: continue
            lines.append(f"ReClaim-{size} & {pre.tokens_generated_per_patient:,.0f} & {post.tokens_generated_per_patient:,.0f} & "
                         f"{pre.tokens_generated_per_patient / post.tokens_generated_per_patient:,.0f}$\\times$ \\\\")
        lines += [r"\hline", r"\multicolumn{4}{@{}l}{\textit{GPU-hours per evaluation}} \\"]
        for size in ["S", "M", "L"]:
            pre, post = pair(size)
            if pre is None or post is None: continue
            lines.append(f"ReClaim-{size} & {pre.gpu_h:.2f} & {post.gpu_h:.2f} & {pre.gpu_s_per_patient / post.gpu_s_per_patient:.0f}$\\times$ \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(out, "inference_cost.tex"), "w").write("\n".join(lines) + "\n")

    # ---- ROSTER.md: ids + provenance + missing ----
    md = ["# Reported roster", "",
          "Built by `output/scripts/final_tables.py` from `output/scripts/roster.sh`. Every id is looked up exactly.", "",
          f"* Monte-Carlo anchor trajectories (FINAL_N_GENERATION): **{a.gen_num}**",
          f"* LightGBM window shown in the figure (FINAL_LIGHTGBM_WINDOW): **{a.lightgbm_window}**", f"* HNHC cutoff: **${int(a.cutoff_hnhc):,}**", "", "## Models", "",
          "| role | size | model id | table label |", "|---|---|---|---|"] + [f"| {r[1]} | {r[2] or '—'} | `{r[0]}` | {r[4]} |" for r in roster] + \
         ["", "## Files read", ""] + [f"* `{os.path.relpath(f)}`" for f in files] + ["", "## Missing cells", ""] + \
         ([f"* {m}" for m in missing] if missing else ["none: every roster cell is present"])
    open(os.path.join(out, "ROSTER.md"), "w").write("\n".join(md) + "\n")
    print(S[["section", "model", "n", "R2", "MAE", "ROC_AUC", "PR_AUC"]].to_string(index=False))
    if missing: print("\nMISSING (" + str(len(missing)) + "):\n  " + "\n  ".join(missing))
    print("wrote", out)


if __name__ == "__main__":
    main()
