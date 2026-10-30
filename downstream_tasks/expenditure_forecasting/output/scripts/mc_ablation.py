#!/usr/bin/env python3
"""Monte-Carlo trajectory-count ablation of the pre-trained anchor: an extraction over the metric
tables written by e4_eval (no predictions, no gold). Every trajectory count N of an anchor is an
independent model id, reclaim-<size>-pretrain-<N>, in
  <section>/regression/regression_metrics.csv                    R2, MAE, RMSE
  <section>/classification/<cutoff>/classification_metrics.csv   F1 / precision / recall at predicted $ >= cutoff
  <section>/binary_auc/<cutoff>/auc_metrics.csv                  ROC-AUC, PR-AUC; score = fraction of the N
                                                                 trajectories at or above the cutoff
  inference_cost.csv                                             GPU-s per enrollee per N
Writes
  <tables_dir>/mc_ablation.csv / .tex     section, size, N, metrics, GPU-s per enrollee (every size and N)
  <figure_dir>/mc_ablation_figure.{pdf,png}   a retrospective / b prospective x (R2 | ROC-AUC | PR-AUC)
                                          against N on a linear axis; by default ReClaim-L and
                                          N = 1, 8, 16, 32, 64 (--plot_sizes, --plot_ns); dashed navy =
                                          LightGBM from <tables_dir>/benchmark.csv.
N = 1 is greedy decoding of a single trajectory (generate_vllm.py), not a Monte-Carlo estimate.
Style: 180 mm, 5-7 pt Helvetica/Arial, Type 42, no tight bbox."""
import os, re, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEC = ["retrospective", "prospective"]
C_RC = {"S": "#F5B09D", "M": "#F07B62", "L": "#E64B35"}; C_LG = "#3C5488"
INK, MUTED, GRID = "#222222", "#787878", "#E6E6E6"; MM = 1 / 25.4
# axis ranges wide enough for the small-N points (a 2-trajectory fraction score ranks coarsely: ROC ~0.70, PR ~0.25)
METRICS = [("R2", "R²", (0, 0.6), np.arange(0, 0.61, 0.2)), ("ROC_AUC", "ROC-AUC", (0.4, 1.0), np.arange(0.4, 1.01, 0.2)),
           ("PR_AUC", "PR-AUC", (0, 0.8), np.arange(0, 0.81, 0.2))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", required=True); p.add_argument("--figure_dir", required=True)
    p.add_argument("--tables_dir", required=True, help="output/tables (writes mc_ablation.*; reads benchmark.csv for the LightGBM line)")
    p.add_argument("--models", nargs="+", required=True, help="MC anchor ids: reclaim-s-pretrain reclaim-m-pretrain reclaim-l-pretrain")
    p.add_argument("--cutoff_hnhc", default="30000")
    p.add_argument("--no_reference", action="store_true", help="draw no LightGBM reference line")
    p.add_argument("--plot_sizes", nargs="+", default=["L"], help="backbone sizes drawn in the figure (table always has all)")
    p.add_argument("--plot_ns", nargs="+", type=int, default=[1, 8, 16, 32, 64], help="trajectory counts drawn in the figure (table always has all)")
    return p.parse_args()


def read(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path); keys = [c for c in ("source", "model", "cost_type") if c in df.columns]
    return df.drop_duplicates(subset=keys, keep="last")


def cell(df, model, col):
    if df is None: return np.nan
    r = df[(df["model"] == model) & (df["source"] == "total")]
    if "cost_type" in df.columns: r = r[r["cost_type"] == "total"]
    return float(r[col].values[0]) if len(r) and pd.notna(r[col].values[0]) else np.nan


def write_tex(T, cost, path):
    """Supplementary table: accuracy and cost of the anchor by trajectory count, all sizes, both evaluations."""
    post = {}
    if cost is not None:
        c = cost[(cost["section"] == "evaluation_total") & cost["model"].str.match(r"^reclaim-[sml]-posttrain$")]
        post = {m[8].upper(): float(v) for m, v in zip(c["model"], c["gpu_s_per_patient"])}
    def val(sec, sz, N, col, fmt):
        r = T[(T.section == sec) & (T["size"] == sz) & (T.N == N)]
        return "--" if not len(r) or pd.isna(r[col].iloc[0]) else fmt % r[col].iloc[0]
    ptxt = (" (the post-trained one-token readout costs " + " / ".join(f"{post[z]:.4f}" for z in "SML" if z in post) +
            " GPU-s per enrollee for " + " / ".join(z for z in "SML" if z in post) + ")") if post else ""
    n = {sec: int(T[T.section == sec]["n"].max()) for sec in SEC}
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{\textcolor{blue}{Accuracy and cost of the pre-trained simulation estimator by number of Monte-Carlo simulations}. Pre-trained ReClaim-S, -M and -L "
         r"read out by Monte-Carlo simulation of $N$ future-year trajectories per enrollee in the held-out cohort (retrospective n = " + f"{n['retrospective']:,}" +
         r"; prospective n = " + f"{n['prospective']:,}" + r"): R$^2$ of the censored total, HNHC ROC-AUC and PR-AUC (score = fraction of the $N$ simulations with "
         r"next-year total $\geq$ \$30,000), and GPU-seconds per enrollee" + ptxt + r". $N = 1$ is greedy decoding of a single trajectory, not a Monte-Carlo estimate.}",
         r"\label{tab:mc_simulation_count}   % requires \usepackage{booktabs}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.12}",
         r"\begin{tabular}{@{}rrrrrrrrrrrrr@{}}", r"\hline",
         r" & \multicolumn{3}{c}{\textbf{R$^2$}} & \multicolumn{3}{c}{\textbf{ROC-AUC}} & \multicolumn{3}{c}{\textbf{PR-AUC}} & \multicolumn{3}{c}{\textbf{GPU-s / enrollee}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
         r"\textbf{$N$} & S & M & L & S & M & L & S & M & L & S & M & L \\", r"\hline"]
    for sec in SEC:
        L.append(r"\multicolumn{13}{@{}l}{\textit{" + sec.capitalize() + r" evaluation}} \\")
        for N in sorted(T["N"].unique()):
            row = [f"{N}" + (r"$^{\dagger}$" if N == 1 else "")]
            for col, fmt in [("R2", "%.3f"), ("ROC_AUC", "%.3f"), ("PR_AUC", "%.3f"), ("gpu_s_per_patient", "%.3f")]:
                row += [val(sec, sz, N, col, fmt) for sz in ["S", "M", "L"]]
            L.append(" & ".join(row) + r" \\")
        L.append(r"\hline")
    L += [r"\multicolumn{13}{@{}l}{\scriptsize $^{\dagger}$greedy decoding (temperature 0), shown for reference.} \\", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n"); print("wrote", path)


def main():
    a = parse_args(); out_dir = a.tables_dir; os.makedirs(out_dir, exist_ok=True); os.makedirs(a.figure_dir, exist_ok=True)
    cost = read(os.path.join(a.result_root, "inference_cost.csv"))
    rows = []
    for sec in SEC:
        base = os.path.join(a.result_root, sec)
        reg = read(os.path.join(base, "regression", "regression_metrics.csv"))
        cls = read(os.path.join(base, "classification", a.cutoff_hnhc, "classification_metrics.csv"))
        auc = read(os.path.join(base, "binary_auc", a.cutoff_hnhc, "auc_metrics.csv"))
        for model in a.models:
            g = re.match(r"^reclaim-([sml])-pretrain$", model); assert g, f"not an MC anchor id: {model}"
            ns = sorted({int(m.group(1)) for df in (reg, cls, auc) if df is not None
                         for x in df["model"].unique() if (m := re.match(re.escape(model) + r"-(\d+)$", x))})
            if not ns: print(f"[{sec}] {model}: no pretrain-<N> rows in the metric tables - skipped"); continue
            for N in ns:
                mid = f"{model}-{N}"; gsp = np.nan
                if cost is not None:
                    c = cost[(cost["model"] == mid) & (cost["section"] == sec)]
                    if len(c): gsp = float(c["gpu_s_per_patient"].iloc[0])
                r = dict(section=sec, model=mid, size=g.group(1).upper(), N=N, n=cell(reg, mid, "n_valid"),
                         R2=cell(reg, mid, "R2"), MAE=cell(reg, mid, "MAE"), RMSE=cell(reg, mid, "RMSE"),
                         F1_hnhc=cell(cls, mid, "f1-binary"), prec_hnhc=cell(cls, mid, "precision-binary"), rec_hnhc=cell(cls, mid, "recall-binary"),
                         ROC_AUC=cell(auc, mid, "roc_auc"), PR_AUC=cell(auc, mid, "pr_auc"), gpu_s_per_patient=gsp)
                rows.append(r)
                print(f"[{sec}] {mid:26s} R2={r['R2']:.4f}  MAE={r['MAE']:,.0f}  F1={r['F1_hnhc']:.3f}  ROC={r['ROC_AUC']:.4f}  PR={r['PR_AUC']:.4f}  gpu_s={gsp:.4f}")
    if not rows: raise SystemExit("no pretrain-<N> rows found; run e4 TASK=metrics first")
    T = pd.DataFrame(rows).sort_values(["section", "size", "N"]); path = os.path.join(out_dir, "mc_ablation.csv"); T.to_csv(path, index=False); print("wrote", path)
    write_tex(T, cost, os.path.join(out_dir, "mc_ablation.tex"))
    if T["ROC_AUC"].isna().all():
        print("NOTE: no AUC rows for any pretrain-<N>: run mc_aggregate.sh (writes prob_ge into cost_predictions_<N>.csv), then e4 TASK=metrics")

    ref = {}
    fs = os.path.join(a.tables_dir, "benchmark.csv")
    if not a.no_reference and os.path.exists(fs):
        F = pd.read_csv(fs, keep_default_na=False, na_values=[""])
        for _, r in F.iterrows():
            if r["role"] == "lightgbm":
                for k in ("R2", "ROC_AUC", "PR_AUC"):
                    if pd.notna(r[k]): ref[(r["section"], "lgbm", k)] = float(r[k])
    elif not a.no_reference:
        print(f"NOTE: {fs} not found - no LightGBM reference line (run final_tables.py first)")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 95 * MM), gridspec_kw=dict(left=0.07, right=0.985, bottom=0.21, top=0.93, wspace=0.32, hspace=0.42))
    Ns = sorted(n for n in T["N"].unique() if n in set(a.plot_ns))
    T_plot = T[T["N"].isin(Ns)]
    for r, sec in enumerate(SEC):
        for c, (col, name, ylim, ticks) in enumerate(METRICS):
            ax = axes[r, c]
            for size in [z.upper() for z in a.plot_sizes]:
                t = T_plot[(T_plot.section == sec) & (T_plot["size"] == size)].sort_values("N")
                if len(t) and t[col].notna().any():       # one connected line incl. N = 1 (greedy; said in the caption, not drawn apart)
                    ax.plot(t["N"], t[col], color=C_RC[size], lw=0.9, marker="o", ms=3.6, mfc="white", mec=C_RC[size], mew=0.9, zorder=3)
            lg = ref.get((sec, "lgbm", col))
            if lg is not None: ax.axhline(lg, color=C_LG, lw=1.0, ls=(0, (2.5, 1.5)), zorder=2)   # heavier than the main figure's hairline so the navy reads
            ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])                # linear axis: the true spacing of the drawn N
            ax.set_xlim(-1.5, max(Ns) + 2.5); ax.set_ylim(*ylim); ax.set_yticks(ticks)
            ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True); ax.set_ylabel(name, labelpad=2)
        fig.text(0.012, axes[r, 0].get_position().y1 + 0.01, "a" if r == 0 else "b", fontsize=7.5, fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(0.5, 0.115, "Monte-Carlo simulations per enrollee, N", ha="center", va="top", fontsize=6, color=INK)
    sizes = [z.upper() for z in a.plot_sizes]
    handles = [Line2D([], [], color=C_RC[z], lw=0.9, marker="o", ms=3.6, mfc="white", mec=C_RC[z], mew=0.9, label=f"ReClaim-{z} pre-training") for z in sizes]
    if ref:
        handles.append(Line2D([], [], color=C_LG, lw=1.0, ls=(0, (2.5, 1.5)), label="LightGBM"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=5.5, handlelength=2.2, columnspacing=2.0, bbox_to_anchor=(0.5, 0.01))
    for ext, kw in (("pdf", {}), ("png", {"dpi": 450})):
        fig.savefig(os.path.join(a.figure_dir, f"mc_ablation_figure.{ext}"), **kw)
    plt.close(fig); print("saved", os.path.join(a.figure_dir, "mc_ablation_figure.{pdf,png}"))


if __name__ == "__main__":
    main()
