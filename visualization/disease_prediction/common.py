"""Shared loading, styling and drawing helpers for the disease-onset figures.

Every input is a per-disease AUC table keyed by the 3-character ICD-10 code
(column `icd_code`), i.e. the output of `to_icd_code.py`. Integer
`disease_token` ids are deliberately not accepted: different checkpoints
order their vocabularies differently, so joining on ids silently pairs
unrelated diseases.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1] / "downstream_tasks" / "disease_prediction"
DEFAULT_OUT_DIR = TASK_ROOT / "output" / "figures"
ICD_MAP_FILE = TASK_ROOT / "data" / "icd_code_map_with_category.csv"

PANEL_FACE = "#F8F9F8"
GRID_COLOR = "#D8D8D8"
INK_COLOR = "#222222"
BOX_FACE = "#FFFFFF"
POINT_JITTER = 0.16

METHOD_PALETTE = {
    "LightGBM": "#3C5488",
    "Delphi": "#4DBBD5",
    "ReClaim": "#E64B35",
}

# Paul Tol-inspired palette, one color per ICD-10 chapter (I..XVII).
CHAPTER_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
    "#BBBBBB", "#332288", "#CC3311", "#009988", "#EE7733", "#0077BB",
    "#AA4499", "#44BB99", "#DDAA33", "#BB5566", "#117733",
]
FALLBACK_COLOR = "#DDDDDD"

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
          "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17}


# ── Loading ───────────────────────────────────────────────────────────────────
def read_auc(path: str | Path, unpooled: bool = False) -> pd.DataFrame:
    """Read a pooled (or unpooled) AUC table keyed by `icd_code`."""
    df = pd.read_csv(path)
    if "icd_code" not in df.columns:
        raise ValueError(
            f"{path} has no `icd_code` column. Convert it first with "
            "`python to_icd_code.py --tokenizer <model> --input <csv> --output <csv>`."
        )
    auc_col = "auc_delong" if unpooled and "auc_delong" in df.columns else "auc"
    df = df.rename(columns={auc_col: "auc", "n_control": "n_healthy", "n_controls": "n_healthy"})
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["icd_code", "auc"])
    if not unpooled and not df["icd_code"].is_unique:
        raise ValueError(f"{path} has duplicate icd_code rows")
    return df


def align(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide table of AUCs (one column per name) on the diseases shared by all frames."""
    wide = None
    for name, df in frames.items():
        sub = df.set_index("icd_code")["auc"].rename(name)
        wide = sub.to_frame() if wide is None else wide.join(sub, how="inner")
    return wide


def rare_codes(ref: pd.DataFrame, max_prevalence: float = 0.0005, min_cases: int = 10) -> set[str]:
    """Rare endpoints: case fraction below `max_prevalence` with more than `min_cases` cases."""
    ratio = ref["n_diseased"] / (ref["n_diseased"] + ref["n_healthy"])
    keep = (ratio < max_prevalence) & (ref["n_diseased"] > min_cases)
    return set(ref.loc[keep, "icd_code"])


def chapter_table() -> pd.DataFrame:
    """icd_code -> chapter label, chapter order and color."""
    icd = pd.read_csv(ICD_MAP_FILE)
    chapters = sorted(icd["ICD-10 Chapter (short)"].unique(), key=chapter_key)
    color = {c: CHAPTER_PALETTE[i % len(CHAPTER_PALETTE)] for i, c in enumerate(chapters)}
    out = icd[["ICD code", "ICD-10 Chapter (short)"]].rename(
        columns={"ICD code": "icd_code", "ICD-10 Chapter (short)": "chapter"}
    )
    out["color"] = out["chapter"].map(color)
    return out.drop_duplicates("icd_code")


def chapter_key(label: str) -> int:
    m = re.match(r"([IVX]+)\.", str(label))
    return _ROMAN.get(m.group(1), 9999) if m else 9999


def chapter_short(label: str) -> str:
    """'IX. Circulatory Diseases' -> 'IX'."""
    return str(label).split(".")[0]


def paired_summary(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Summary of the per-disease difference a - b."""
    diff = a - b
    n = len(diff)
    sem = diff.std(ddof=1) / np.sqrt(n)
    return {
        "n": n,
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "delta_mean": diff.mean(),
        "delta_ci95": 1.96 * sem,
        "frac_a_better": (diff > 0).mean(),
    }


# ── Styling and drawing ───────────────────────────────────────────────────────
def set_style(font_size: float = 11) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 1,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.facecolor": PANEL_FACE,
        "savefig.facecolor": PANEL_FACE,
        "axes.facecolor": PANEL_FACE,
        "axes.edgecolor": INK_COLOR,
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
        "text.color": INK_COLOR,
        "axes.labelcolor": INK_COLOR,
        "axes.titlecolor": INK_COLOR,
    })


def save(fig: plt.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_base.with_suffix(f'.{ext}')}")
    plt.close(fig)


def draw_box(ax: plt.Axes, groups: list[np.ndarray], colors: list[str],
             point_colors: list | None = None, width: float = 0.34, seed: int = 42) -> None:
    """Box plot with jittered points and a white diamond at each group mean.

    `point_colors[i]` may be a single color or an array of per-point colors.
    """
    rng = np.random.default_rng(seed)
    bp = ax.boxplot(
        groups,
        positions=np.arange(len(groups)),
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": INK_COLOR, "linewidth": 1.15},
        whiskerprops={"color": INK_COLOR, "linewidth": 1.05},
        capprops={"color": INK_COLOR, "linewidth": 1.05},
        medianprops={"color": INK_COLOR, "linewidth": 1.35},
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.44)
    for idx, vals in enumerate(groups):
        dense = len(vals) > 300
        pc = colors[idx] if point_colors is None else point_colors[idx]
        ax.scatter(
            idx + rng.uniform(-POINT_JITTER, POINT_JITTER, size=len(vals)),
            vals,
            s=8 if dense else 14,
            c=pc,
            edgecolor=BOX_FACE,
            linewidth=0.35,
            alpha=0.28 if dense else 0.46,
            zorder=3,
            rasterized=dense,
        )
    ax.scatter(np.arange(len(groups)), [np.mean(v) for v in groups], marker="D", s=30,
               facecolor=BOX_FACE, edgecolor=INK_COLOR, linewidth=0.9, zorder=4)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45, color=GRID_COLOR)
    ax.grid(axis="x", visible=False)


def draw_paired_scatter(ax: plt.Axes, x: pd.Series, y: pd.Series, x_label: str, y_label: str,
                        color: str, title: str = "Per-Disease Paired Comparison") -> dict[str, float]:
    """Per-disease scatter of y vs x with the identity line and a summary box."""
    s = paired_summary(y, x)
    ax.scatter(x, y, s=18, alpha=0.55, color=color, edgecolors=BOX_FACE, linewidth=0.2, rasterized=True)
    lo = max(0.0, float(min(x.min(), y.min())) - 0.02)
    hi = min(1.0, float(max(x.max(), y.max())) + 0.02)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#6E6E6E")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{x_label} AUC")
    ax.set_ylabel(f"{y_label} AUC")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, color=GRID_COLOR)
    ax.text(
        0.03, 0.97,
        f"N = {s['n']}\n"
        f"Mean {y_label}: {s['mean_a'] * 100:.2f}%\n"
        f"Mean {x_label}: {s['mean_b'] * 100:.2f}%\n"
        f"Mean delta: {s['delta_mean'] * 100:+.2f} pp ± {s['delta_ci95'] * 100:.2f} pp\n"
        f"Above diagonal: {s['frac_a_better'] * 100:.1f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": BOX_FACE, "edgecolor": "#BDBDBD", "alpha": 0.92},
    )
    return s
