#!/usr/bin/env python3
"""Main figure of the expenditure task (Figure 6), drawn from output/tables/ only.
Style: Helvetica/Arial 5-7 pt, 180 mm double column, vector PDF with editable Type-42 text, RGB;
LightGBM navy, ReClaim S/M/L on the red size ramp; training stage carried by position (pre above
post in each block) and by ink, never by colour: post-training bars take an ink outline,
post-training markers a solid fill (pre-training hollow).

  a/b  retrospective / prospective: one row of grouped horizontal bars per cohort, model labels once
       on the left, three metric axes grouped as Regression (R2) and HNHC identification (ROC-AUC |
       PR-AUC) with identical axis ranges in both rows; LightGBM, then ReClaim-S/M/L (pre-training,
       post-training)
  c    performance (enrollee-weighted over both settings; one band per metric) vs relative inference
       efficiency (1 / GPU-s per enrollee, pre-training L = 1, log x), one arrow per size; markers
       coloured by metric, hollow with a coloured edge for pre-training, solid with an ink edge for
       post-training

Inputs: <tables_dir>/benchmark.csv (columns role/size/stage identify LightGBM, the Monte-Carlo
anchor and the post-trained models) and <tables_dir>/inference_cost.csv, both written by
final_tables.py from the roster. No model is discovered by pattern; a missing cell is drawn as an
em dash, never substituted. Output: <figure_dir>/expenditure_main_figure.{pdf,png}, exactly 180 x 100 mm."""
import os, re, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.transforms as mtr
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

SEC = ["retrospective", "prospective"]; SIZES = ["S", "M", "L"]
C_LG = "#3C5488"                                                   # LightGBM, NPG 4
C_RC = {"S": "#F5B09D", "M": "#F07B62", "L": "#E64B35"}            # ReClaim size ramp (light -> dark)
INK, MUTED, GRID, TINT = "#222222", "#787878", "#E6E6E6", "#F3F3F3"
METRIC_COLORS = {"ROC-AUC": "#7B2D6B", "PR-AUC": "#AA3377", "R²": "#6A3D9A"}   # one purple family, mid-lightness throughout: plum / magenta-purple / violet
MM = 1 / 25.4
STYLE = os.environ.get("STYLE", "lines")     # lines (default): thin separators between model groups | boxes: one tint band per group


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tables_dir", required=True, help="output/tables (benchmark.csv, inference_cost.csv)")
    p.add_argument("--figure_dir", required=True)
    return p.parse_args()


def roster(summary):
    """model id per (size, stage) from the final table's own role/size/stage columns (no pattern matching)."""
    r = {}
    for _, row in summary[summary["role"].isin(["pretrain", "posttrain"])].iterrows():
        key = (row["size"], row["stage"])
        assert key not in r or r[key] == row["model"], f"two models for {key}: {r[key]} vs {row['model']}"
        r[key] = row["model"]
    return r


STAGES = ("pre-training", "post-training")


def rows_spec():
    return [("LightGBM", "", "lightgbm-{}-1year")] + [(f"ReClaim-{s}", st, None) for s in SIZES for st in STAGES]


def value(summary, ros, group, sub, metric, kind):
    """kind: 'reg' -> the roster's lightgbm-reg-*, 'cls' -> lightgbm-cls-*; reclaim rows via the roster."""
    if group == "LightGBM":
        lg = summary[(summary["role"] == "lightgbm") & summary["model"].str.startswith("lightgbm-reg-" if kind == "reg" else "lightgbm-cls-")]
        name = lg["model"].iloc[0] if len(lg) else None
    else:
        name = ros.get((group[-1], sub))
    if name is None or name not in set(summary["model"]): return None
    v = summary.loc[summary["model"] == name, metric].values
    return None if len(v) == 0 or pd.isna(v[0]) else float(v[0])


def bars(ax, summary, ros, metric, kind, xlabel, xlim, labels=True, ticks=None, gutter=(0.02, 0.075, 0.118), span_to=None):
    """labels=True draws the model-group labels in the gutter and one continuous tint band per group from
    gutter[0] to span_to (figure x; the right edge of the row's last axis); gutter[1] = group-name centre;
    stage labels (pre-/post-training) are centred between gutter[2] and the first axis."""
    spec = rows_spec(); ypos = []; cur = 0; prev = None
    for g, _, _ in spec:
        if prev is not None and g != prev: cur += 0.6
        ypos.append(cur); cur += 1; prev = g
    ypos = np.array(ypos); ypos = ypos.max() - ypos
    blocks = []; i = 0
    while i < len(spec):
        j = i
        while j < len(spec) and spec[j][0] == spec[i][0]: j += 1
        blocks.append((spec[i][0], ypos[i], ypos[j - 1])); i = j
    fig = ax.figure; pos = ax.get_position()
    trf = mtr.blended_transform_factory(fig.transFigure, ax.transData)     # x in figure fraction, y in data
    x1 = span_to if span_to is not None else pos.x1
    for k, (g, yt, yb) in enumerate(blocks):
        if not labels: continue
        if STYLE == "boxes":   # one unbroken band per model group, gutter through the last metric axis
            ax.add_patch(FancyBboxPatch((gutter[0], yb - 0.6), x1 - gutter[0], (yt - yb) + 1.2, boxstyle="square,pad=0",
                                        transform=trf, facecolor=TINT, edgecolor="none", zorder=0, clip_on=False))
        elif k > 0:            # thin separator above every group but the first
            ax.plot([gutter[0], x1], [yt + 0.8, yt + 0.8], transform=trf, color="#CCCCCC", lw=0.5, clip_on=False, zorder=0)
        ax.text(gutter[1], (yt + yb) / 2, g, transform=trf, ha="center", va="center", fontsize=6, color=INK)
    if labels:   # stage labels centred between the group name and the first axis
        xc = (gutter[2] + pos.x0) / 2
        for (g, sub, _), yy in zip(spec, ypos):
            if sub: ax.text(xc, yy, sub, transform=trf, ha="center", va="center", fontsize=6, color=INK)
    for (g, sub, _), yy in zip(spec, ypos):
        v = value(summary, ros, g, sub, metric, kind)
        if v is None:
            ax.text(xlim[0] + 0.005, yy, "—", va="center", fontsize=6, color="#BBBBBB"); continue
        col = C_LG if g == "LightGBM" else C_RC[g[-1]]
        post = sub == "post-training"                               # stage = ink outline, not colour
        ax.barh(yy, v, height=0.72, color=col, edgecolor=INK if post else "none", lw=0.9 if post else 0, zorder=2)
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, yy, f"{v:.3f}", va="center", fontsize=5.5, color=INK, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlabel(xlabel, labelpad=1.5); ax.set_xlim(*xlim)
    ax.set_ylim(ypos.min() - 0.75, ypos.max() + 0.75)
    if ticks is not None: ax.set_xticks(ticks)
    ax.grid(True, axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
    ref = value(summary, ros, "LightGBM", "", metric, kind)
    if ref is not None: ax.axvline(ref, color=C_LG, lw=0.6, ls=(0, (1.5, 1.5)))


def main():
    a = parse_args(); os.makedirs(a.figure_dir, exist_ok=True)
    F = pd.read_csv(os.path.join(a.tables_dir, "benchmark.csv"), keep_default_na=False, na_values=[""])
    S = {sec: F[F["section"] == sec].reset_index(drop=True) for sec in SEC}
    ros = roster(F)
    print("roster:", {f"{k[0]} {k[1]}": v for k, v in sorted(ros.items())})
    tot = pd.read_csv(os.path.join(a.tables_dir, "inference_cost.csv")).set_index("model")
    npat = {sec: int(S[sec]["n"].max()) for sec in SEC}
    W = {sec: npat[sec] / sum(npat.values()) for sec in SEC}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 6, "axes.titlesize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999", "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(180 * MM, 100 * MM))                    # exact double-column size; no tight bbox
    outer = fig.add_gridspec(2, 2, width_ratios=[7.0, 3.1], wspace=0.30, hspace=0.42,
                             left=0.215, right=0.965, bottom=0.095, top=0.90)
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        # R² | gap 3g | ROC-AUC | gap g | PR-AUC  (explicit spacer columns; the regression-classification gap is
        # three times the gap inside the HNHC pair)
        g = 0.09
        inner = outer[r, 0].subgridspec(1, 5, wspace=0, width_ratios=[1.1, 3 * g, 1, g, 1])
        axes[r, 0] = fig.add_subplot(inner[0, 0]); axes[r, 1] = fig.add_subplot(inner[0, 2]); axes[r, 2] = fig.add_subplot(inner[0, 4])
        for k in range(3): axes[r, k].set_facecolor("none")               # the group bands (drawn in axes 0) show through
    ax_e = fig.add_subplot(outer[:, 1])

    def letter(ax, ch, dx=-0.01):
        ax.text(dx, 1.0, ch, transform=ax.transAxes, fontsize=7.5, fontweight="bold", ha="right", va="bottom", color=INK)
    for r, sec in enumerate(SEC):
        s = S[sec]
        bars(axes[r, 0], s, ros, "R2", "reg", "R²", (0, 0.56), ticks=np.arange(0, 0.51, 0.1), span_to=axes[r, 2].get_position().x1)
        bars(axes[r, 1], s, ros, "ROC_AUC", "cls", "ROC-AUC", (0.7, 0.95), labels=False, ticks=[0.7, 0.8, 0.9])
        bars(axes[r, 2], s, ros, "PR_AUC", "cls", "PR-AUC", (0.3, 0.7), labels=False, ticks=np.arange(0.3, 0.71, 0.1))
        # panel letter at the far left (figure coords; no panel titles — the legend names the panels); metric-group headers above the axes
        y_top = axes[r, 0].get_position().y1 + 0.012
        fig.text(0.02, y_top, "a" if r == 0 else "b", fontsize=7.5, fontweight="bold", ha="left", va="bottom", color=INK)
        axes[r, 0].set_title("Regression", loc="center", pad=3, fontsize=6.5, color=MUTED)
        p1, p2 = axes[r, 1].get_position(), axes[r, 2].get_position()
        fig.text((p1.x0 + p2.x1) / 2, p1.y1 + 0.012, "HNHC identification", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    letter(ax_e, "c")

    # ---- panel e: performance vs relative inference efficiency
    def wavg(metric, sz, st):
        vals = [value(S[sec], ros, f"ReClaim-{sz}", st, metric, None) for sec in SEC]
        if any(v is None for v in vals):
            print(f"WARN panel c: {metric} missing for ReClaim-{sz} {st} in a section -> point skipped"); return None
        return sum(W[sec] * v for sec, v in zip(SEC, vals))
    def gsp(sz, st):
        return float(tot.loc[ros[(sz, st)], "gpu_s_per_patient"])     # same ids in both final tables
    unit = gsp("L", "pre-training")                                      # slowest configuration = 1
    eff = {st: [unit / gsp(sz, st) for sz in SIZES] for st in STAGES}
    ax = ax_e; MS = 8.5
    metrics = [("ROC-AUC", "ROC_AUC"), ("PR-AUC", "PR_AUC"), ("R²", "R2")]
    for mname, col_ in metrics:
        col = METRIC_COLORS[mname]
        for i, sz in enumerate(SIZES):
            x0, x1 = eff["pre-training"][i], eff["post-training"][i]; y0, y1 = wavg(col_, sz, "pre-training"), wavg(col_, sz, "post-training")
            if y0 is None or y1 is None: continue
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#C8C8C8", lw=0.6, shrinkA=5, shrinkB=5))
            ax.plot(x0, y0, marker="o", ms=MS, mfc="white", mec=col, mew=0.9, ls="none", zorder=3)           # pre: hollow, coloured edge
            ax.text(x0, y0, sz, ha="center", va="center", fontsize=5, color=col, fontweight="bold", zorder=4)
            ax.plot(x1, y1, marker="o", ms=MS, mfc=col, mec=INK, mew=0.6, ls="none", zorder=3)               # post: solid, ink edge
            ax.text(x1, y1, sz, ha="center", va="center", fontsize=5, color="white", fontweight="bold", zorder=4)
        # name the metric band above its post-train cluster, in the metric colour
        ys = [wavg(col_, sz, "post-training") for sz in SIZES]
        if all(y is not None for y in ys):
            ax.text(np.sqrt(eff["post-training"][0] * eff["post-training"][2]), max(ys) + 0.028,
                    mname, ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")
    preS, postS = wavg("ROC_AUC", "S", "pre-training"), wavg("ROC_AUC", "S", "post-training")
    preM, postM = wavg("ROC_AUC", "M", "pre-training"), wavg("ROC_AUC", "M", "post-training")
    if None in (preS, postS, preM, postM):                       # anchor ROC-AUC absent: put the ratio labels on the R2 band instead
        preS, postS = wavg("R2", "S", "pre-training"), wavg("R2", "S", "post-training")
        preM, postM = wavg("R2", "M", "pre-training"), wavg("R2", "M", "post-training")
    ybund = min(preS, preM) + (min(postS, postM) - min(preS, preM)) * 0.5
    xa = np.sqrt(max(eff["pre-training"]) * min(eff["post-training"]))
    spots = {"S": (ybund - 0.02, "top"), "M": (preM + (postM - preM) * 0.5, "center"), "L": (ybund + 0.026, "bottom")}
    for i, sz in enumerate(SIZES):
        y, va = spots[sz]
        ax.text(xa, y, f"{sz}: {eff['post-training'][i] / eff['pre-training'][i]:.0f}×", ha="center", va=va, fontsize=5.5, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9) if sz == "M" else None)
    ax.set_xscale("log"); ax.set_xlim(0.45, max(eff["post-training"]) * 3.2); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1×", "10×", "100×"]); ax.minorticks_off()
    ax.set_ylim(0.3, 1.0); ax.set_yticks(np.arange(0.3, 0.91, 0.1))
    ax.set_xlabel("Relative inference efficiency (log scale)", labelpad=1.5)
    ax.set_ylabel("Performance (weighted over retrospective and prospective)", labelpad=2)
    ax.text(1.25, 0.315, "pre-training L = 1", ha="left", va="bottom", fontsize=5.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", mfc="white", mec=INK, mew=0.8, ls="none", ms=5.5, label="Pre-training"),
               Line2D([], [], marker="o", mfc="#999999", mec=INK, mew=0.6, ls="none", ms=5.5, label="Post-training")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.56), frameon=False, fontsize=5.5, handletextpad=0.4, labelspacing=0.5,
              title="S / M / L: backbone size", title_fontsize=5.5)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 450})):
        fig.savefig(os.path.join(a.figure_dir, f"expenditure_main_figure.{ext}"), **kw)   # exact 180 x 100 mm, no tight bbox
    plt.close(fig); print("saved", os.path.join(a.figure_dir, "expenditure_main_figure.{pdf,png}"))


if __name__ == "__main__":
    main()
