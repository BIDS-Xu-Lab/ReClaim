#!/usr/bin/env python3
"""Shared library for the disease-embedding figures.

Loads a checkpoint's input-embedding matrix, maps disease tokens to ICD-10
labels, counts how often each disease occurs in the corpus, and draws the
scatter: position from the projection, colour from the chapter, marker size
from prevalence.

Imported by plot_umap.py and plot_tsne.py. Only the projection differs between
them, so the two figures are directly comparable.

Coordinates come from the model weights, never from patient records. The one
quantity read from the corpus is a per-disease occurrence count, which is
aggregate and is cached outside the repository.

The coordinate columns are named UMAP1 and UMAP2 throughout, including on the
t-SNE path, so that a single plotting stack serves both projections.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from tqdm import tqdm  # noqa: E402

matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': [PUBLICATION_FONT, 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
})


HERE = Path(__file__).resolve().parent

DEFAULT_LABELS_FILE = (HERE.parents[1] / "downstream_tasks" / "disease_prediction"
                       / "delphi_baseline"
                       / "delphi_labels_chapters_colours_icd.csv")

PUBLICATION_FONT = "Nimbus Sans"

ICD10_CHAPTER_ORDER = [
    'I: Infectious diseases', 'II: Neoplasms', 'III: Blood and immune disorders',
    'IV: Metabolic diseases', 'V: Mental disorders', 'VI: Nervous system diseases',
    'VII: Eye diseases', 'VIII: Ear diseases', 'IX: Circulatory diseases',
    'X: Respiratory diseases', 'XI: Digestive diseases', 'XII: Skin diseases',
    'XIII: Musculoskeletal diseases', 'XIV: Genitourinary diseases',
    'XV: Pregnancy and childbirth', 'XVI: Perinatal conditions',
    'XVII: Congenital abnormalities',
]

ICD10_CHAPTER_MAP = {
    'I. Infectious Diseases': 'I: Infectious diseases',
    'II. Neoplasms': 'II: Neoplasms',
    'III. Blood & Immune Disorders': 'III: Blood and immune disorders',
    'IV. Metabolic Diseases': 'IV: Metabolic diseases',
    'V. Mental Disorders': 'V: Mental disorders',
    'VI. Nervous System Diseases': 'VI: Nervous system diseases',
    'VII. Eye Diseases': 'VII: Eye diseases',
    'VIII. Ear Diseases': 'VIII: Ear diseases',
    'IX. Circulatory Diseases': 'IX: Circulatory diseases',
    'X. Respiratory Diseases': 'X: Respiratory diseases',
    'XI. Digestive Diseases': 'XI: Digestive diseases',
    'XII. Skin Diseases': 'XII: Skin diseases',
    'XIII. Musculoskeletal Diseases': 'XIII: Musculoskeletal diseases',
    'XIV. Genitourinary Diseases': 'XIV: Genitourinary diseases',
    'XV. Pregnancy & Childbirth': 'XV: Pregnancy and childbirth',
    'XVI. Perinatal Conditions': 'XVI: Perinatal conditions',
    'XVII. Congenital Abnormalities': 'XVII: Congenital abnormalities',
}

NATURE_PALETTE = {
    'I: Infectious diseases': '#4477AA',
    'II: Neoplasms': '#EE6677',
    'III: Blood and immune disorders': '#228833',
    'IV: Metabolic diseases': '#CCBB44',
    'V: Mental disorders': '#66CCEE',
    'VI: Nervous system diseases': '#AA3377',
    'VII: Eye diseases': '#BBBBBB',
    'VIII: Ear diseases': '#332288',
    'IX: Circulatory diseases': '#CC3311',
    'X: Respiratory diseases': '#009988',
    'XI: Digestive diseases': '#EE7733',
    'XII: Skin diseases': '#0077BB',
    'XIII: Musculoskeletal diseases': '#AA4499',
    'XIV: Genitourinary diseases': '#44BB99',
    'XV: Pregnancy and childbirth': '#DDAA33',
    'XVI: Perinatal conditions': '#BB5566',
    'XVII: Congenital abnormalities': '#117733',
}

NON_DISEASE_CHAPTERS = {'Technical', 'Sex', 'Smoking, Alcohol and BMI'}

DX_MAJOR_TOKEN_RE = re.compile(r"<DX-MAJOR_([^>\s]+)>")

SEQUENCE_COLUMN_CANDIDATES = ("eval_seq", "seq")

DEFAULT_FIGURE_WIDTH = 9.0

DEFAULT_FIGURE_HEIGHT = 8.0

DEFAULT_EXPORT_DPI = 2400

LEGEND_ENTRIES_PT = 8.65

LEGEND_ENTRIES_TITLE_PT = 9.65

# Height of the exported marker-size legend strip, in inches.
SIZE_LEGEND_HEIGHT_IN = 0.509

PREVALENCE_LEGEND_VALUES = [1_000, 100_000, 10_000_000, 100_000_000]

BASE_MARKER_SIZE_RANGE = (2.0, 14.0)

# Piecewise-linear mapping from log10(count) to a fraction of the size range,
# spaced so the highest prevalence tiers keep a visible gap.
SIZE_BP_LOG = np.array([3.0, 5.0, 7.0, 8.0])

SIZE_BP_FRAC = np.array([0.12, 0.25, 0.55, 1.0])



def extract_code_from_label(name: str) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    return name.split()[0]


# ── Model loading ────────────────────────────────────────────────────────────
def load_delphi_module(model_path: str | Path | None = None):
    """Import the Delphi ``model.py``, needed to rebuild a Delphi checkpoint.

    Searched beside the checkpoint first, then ``$DELPHI_REPO``, then the copy
    in this repository.
    """
    candidates = []
    if model_path is not None:
        p = Path(model_path).resolve()
        ckpt_dir = p if p.is_dir() else p.parent  # the directory holding ckpt.pt
        # The copy beside the checkpoint matches those weights, so it wins.
        candidates += [ckpt_dir / "model.py", ckpt_dir.parent / "model.py"]
    # An explicit override, for a Delphi checkout kept outside this tree.
    delphi_repo = os.environ.get("DELPHI_REPO")
    if delphi_repo:
        candidates.append(Path(delphi_repo) / "model.py")
    candidates.append(Path(__file__).resolve().parents[2]
                      / "downstream_tasks" / "disease_prediction"
                      / "delphi_baseline" / "model.py")

    module_path = next((c for c in candidates if c.exists()), None)
    if module_path is None:
        raise FileNotFoundError(
            "Could not find Delphi model.py; set DELPHI_REPO. Looked in: "
            + ", ".join(str(c) for c in candidates))
    spec = importlib.util.spec_from_file_location("delphi_model", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model_embeddings(model_path: str | Path, model_type: str,
                          device: str) -> tuple[str, object, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = Path(model_path)
    is_delphi = model_type == "delphi" or (model_type == "auto" and (path / "ckpt.pt").exists())

    if is_delphi:
        ckpt_path = path / "ckpt.pt" if path.is_dir() else path
        delphi_module = load_delphi_module(model_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        conf = delphi_module.DelphiConfig(**checkpoint["model_args"])
        model = delphi_module.Delphi(conf)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        wte = model.transformer.wte.weight.cpu().detach().float().numpy()
        return "delphi", None, wte

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.eval()
    try:
        wte = model.transformer.wte.weight
    except AttributeError:
        wte = model.get_input_embeddings().weight
    return "hf", tokenizer, wte.cpu().detach().float().numpy()


def extract_disease_mapping(tokenizer, labels_df: pd.DataFrame) -> dict[int, int]:
    vocab = tokenizer.get_vocab()
    dx_major_tokens = {}
    for token, tid in vocab.items():
        if token.startswith('<DX-MAJOR_') and token.endswith('>'):
            dx_major_tokens[token[10:-1]] = tid
    
    delphi_to_token = {}
    disease_labels = labels_df[~labels_df['ICD-10 Chapter'].isin(NON_DISEASE_CHAPTERS)]
    for _, row in disease_labels.iterrows():
        if row['code'] in dx_major_tokens:
            delphi_to_token[row['index']] = dx_major_tokens[row['code']]
    return delphi_to_token


# ── Token counting ───────────────────────────────────────────────────────────
def resolve_parquet_files(data_path: str | Path) -> list[Path]:
    path = Path(data_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.parquet") if p.is_file())
    raise FileNotFoundError(f"Data path does not exist: {data_path}")


def build_parquet_fingerprint(parquet_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for pf in parquet_files:
        stat = pf.stat()
        digest.update(f"{pf.resolve()}{stat.st_size}{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def count_tokens_in_data(data_path: str | Path, labels_df: pd.DataFrame,
                         cache_dir: str | Path) -> Counter:
    import pyarrow.parquet as pq

    parquet_files = resolve_parquet_files(data_path)
    fingerprint = build_parquet_fingerprint(parquet_files)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = cache_dir / f"disease_token_counts__{fingerprint}.csv"
    
    if counts_csv.exists():
        print(f"Loading cached counts from {counts_csv}")
        cached = pd.read_csv(counts_csv, dtype={"code": str})
        return Counter({r.code: int(r["count"]) for _, r in cached.iterrows()})
    
    print(f"Counting tokens in {len(parquet_files)} parquet files...")
    counts = Counter()
    for pf in tqdm(parquet_files, desc="Parquet files"):
        schema = pq.read_schema(pf)
        seq_col = next((c for c in SEQUENCE_COLUMN_CANDIDATES if c in schema.names), None)
        if not seq_col:
            continue
        for batch in pq.ParquetFile(pf).iter_batches(columns=[seq_col], batch_size=8192):
            for val in batch.column(0).to_pylist():
                if isinstance(val, str):
                    counts.update(DX_MAJOR_TOKEN_RE.findall(val))
    
    rows = [{"code": c, "count": n} for c, n in counts.items()]
    pd.DataFrame(rows).to_csv(counts_csv, index=False)
    print(f"Saved counts to {counts_csv}")
    return counts


# ── Marker size ──────────────────────────────────────────────────────────────
def _count_to_diameter(counts_arr, size_range: tuple[float, float]) -> np.ndarray:
    """Piecewise-linear mapping from log10(count) to marker diameter.

    Uses SIZE_BP_LOG / SIZE_BP_FRAC breakpoints so that higher prevalence
    tiers get progressively larger visual gaps (especially 10M → 100M).
    """
    log_c = np.log10(np.maximum(np.asarray(counts_arr, dtype=float), 1.0))
    frac = np.interp(log_c, SIZE_BP_LOG, SIZE_BP_FRAC)
    lo, hi = size_range
    return lo + (hi - lo) * frac


def scale_counts_to_sizes(counts, size_range: tuple[float, float] = BASE_MARKER_SIZE_RANGE
                          ) -> pd.Series:
    counts = pd.Series(counts).fillna(1).clip(lower=1).astype(float)
    if len(counts) == 0:
        return counts
    diameters = _count_to_diameter(counts.values, size_range)
    return pd.Series(diameters, index=counts.index)











# ── Legends ──────────────────────────────────────────────────────────────────
def create_chapter_legend(chapters: list[str],
                          fontsize: float = LEGEND_ENTRIES_PT) -> list[Line2D]:
    handles = []
    for chapter in ICD10_CHAPTER_ORDER:
        if chapter in chapters:
            handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                  markerfacecolor=NATURE_PALETTE.get(chapter, '#888888'),
                                  markeredgecolor='white', markersize=7, label=chapter))
    return handles


def _scatter_area_to_marker_diameter_pt(s_area) -> np.ndarray:
    """Match ``scatter(..., s=…)``: *s* is marker area in pt²; return circle diameter in pt."""
    s_area = np.maximum(np.asarray(s_area, dtype=float), 1e-12)
    return 2.0 * np.sqrt(s_area / np.pi)


def _size_legend_handles(size_range: tuple[float, float]) -> list[Line2D]:
    """Build Line2D handles for the standard prevalence legend values.

    Main UMAP uses ``s = d**2`` where *d* comes from ``_count_to_diameter``; legend
    ``markersize`` must be the resulting diameter in points, not *d*.
    """
    d = _count_to_diameter(np.array(PREVALENCE_LEGEND_VALUES, dtype=float), size_range)
    ms = _scatter_area_to_marker_diameter_pt(d ** 2)
    return [Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor='#555555', markeredgecolor='#555555',
                   markersize=float(m), markeredgewidth=0.25,
                   label=f'{v:,}')
            for v, m in zip(PREVALENCE_LEGEND_VALUES, ms)]


def _export_size_legend(size_range: tuple[float, float], umap_fig_width: float,
                        base_path: Path, dpi: int) -> None:
    """Horizontal 'Entries in dataset' legend strip.

    Width = 2 × UMAP figure width so the PDF matches two UMAP panels side-by-side
    in LaTeX with the legend row scaled to the same total width (no AUC gutter).
    """
    leg_w = 2.0 * float(umap_fig_width)
    leg_h = SIZE_LEGEND_HEIGHT_IN
    handles = _size_legend_handles(size_range)

    fig = plt.figure(figsize=(leg_w, leg_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fp = FontProperties(family=PUBLICATION_FONT, size=LEGEND_ENTRIES_PT)
    fp_title = FontProperties(family=PUBLICATION_FONT, size=LEGEND_ENTRIES_TITLE_PT)
    ax.legend(handles=handles, title="Entries in dataset",
              loc="center", ncol=len(handles) + 1, frameon=False,
              prop=fp, title_fontproperties=fp_title,
              handletextpad=0.4, columnspacing=1.2)

    for ext in (".png", ".pdf"):
        fig.savefig(str(base_path) + ext, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved size legend: {base_path}.pdf")


# ── Drawing ──────────────────────────────────────────────────────────────────
def plot_disease_embedding(labels_df: pd.DataFrame, output_path: str | Path,
                           figure_width: float = DEFAULT_FIGURE_WIDTH,
                           figure_height: float = DEFAULT_FIGURE_HEIGHT,
                           export_dpi: int = DEFAULT_EXPORT_DPI,
                           size_scale: float = 1.0) -> None:
    """Draw the projected disease embeddings, coloured by ICD-10 chapter.

    Writes `output_path` as png and pdf, and the marker-size legend beside it.
    """
    labels_df = labels_df.copy()
    labels_df["chapter_display"] = labels_df["ICD-10 Chapter (short)"].map(
        lambda x: ICD10_CHAPTER_MAP.get(x, x))
    chapters = [c for c in ICD10_CHAPTER_ORDER
                if c in labels_df["chapter_display"].values]

    scaled_range = tuple(s * size_scale for s in BASE_MARKER_SIZE_RANGE)
    has_counts = "count" in labels_df.columns and labels_df["count"].notna().any()
    marker_sizes = (
        scale_counts_to_sizes(labels_df["count"].fillna(1), scaled_range)
        if has_counts
        else pd.Series(np.mean(scaled_range), index=labels_df.index))
    # scatter takes an area, and the mapping above produces a diameter.
    scatter_sizes = marker_sizes.pow(2)

    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes([0.04, 0.05, 0.70, 0.90])

    for chapter in chapters:
        rows = labels_df[labels_df["chapter_display"] == chapter]
        if rows.empty:
            continue
        ax.scatter(rows["UMAP1"], rows["UMAP2"],
                   c=NATURE_PALETTE.get(chapter, "#888888"),
                   s=scatter_sizes.loc[rows.index].values,
                   alpha=0.8, edgecolors="white", linewidths=0.25,
                   rasterized=True)

    x_pad = (labels_df["UMAP1"].max() - labels_df["UMAP1"].min()) * 0.02
    y_pad = (labels_df["UMAP2"].max() - labels_df["UMAP2"].min()) * 0.02
    ax.set_xlim(labels_df["UMAP1"].min() - x_pad,
                labels_df["UMAP1"].max() + x_pad)
    ax.set_ylim(labels_df["UMAP2"].min() - y_pad,
                labels_df["UMAP2"].max() + y_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fp = FontProperties(family=PUBLICATION_FONT, size=LEGEND_ENTRIES_PT)
    fp_title = FontProperties(family=PUBLICATION_FONT,
                              size=LEGEND_ENTRIES_TITLE_PT)
    ax.legend(handles=create_chapter_legend(chapters), title="ICD-10 chapter",
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              prop=fp, title_fontproperties=fp_title,
              labelspacing=0.35, borderaxespad=0)

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=export_dpi, facecolor="white")
    print("Saved: %s" % (output_path,))
    pdf_path = output_path.with_suffix(".pdf")
    # Cap PDF rasterization DPI to limit RAM; vector text stays sharp.
    fig.savefig(pdf_path, dpi=min(int(export_dpi), 800), facecolor="white",
                format="pdf")
    print("Saved: %s" % (pdf_path,))
    plt.close(fig)

    _export_size_legend(scaled_range, figure_width,
                        output_path.with_name(f"{output_path.stem}_legend_size"),
                        export_dpi)
