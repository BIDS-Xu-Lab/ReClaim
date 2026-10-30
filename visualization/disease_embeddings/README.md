# Disease-Embedding Figures

Plotting scripts for the disease-embedding figures of the ReClaim paper, and
the clustering scores reported alongside them. Model checkpoints and the
tokenized corpus are not distributed; the scripts take them as inputs, and
everything written out is aggregate, one row per disease.

## Inputs: a model checkpoint

Each script embeds one model and is run once per model. Two types are handled:

- `--model_type hf` reads a Hugging Face directory and takes its input
  embeddings.
- `--model_type delphi` reads a `ckpt.pt`, which needs the Delphi baseline's
  `model.py` to rebuild. It is looked for beside the checkpoint, then at
  `$DELPHI_REPO`, then in
  [`../../downstream_tasks/disease_prediction/delphi_baseline/`](../../downstream_tasks/disease_prediction/delphi_baseline/).

The disease label table gives each token its name and ICD-10 chapter, and
defaults to the copy in that same directory, so it needs no configuring.
Required columns: `index`, `name`, `ICD-10 Chapter`, `ICD-10 Chapter (short)`.

`--data_file` is optional and points at the tokenized corpus. It is read only
to count how often each disease occurs, which sizes the markers; the counts are
cached under `--count_cache_dir` and keyed on a fingerprint of the corpus.
Without it every marker is drawn at one size and the layout is unchanged.

## Figures

Run from this directory; outputs go to `--out_dir` (default `figures`) as
`.png` and `.pdf`, named from `--stem`, with a `*_coords.npy` and a
`*_labels.csv` of the plotted coordinates.

| Paper panel | Script | Inputs |
|-------------|--------|--------|
| Fig. A6a, A6b | `plot_umap.py` | one checkpoint, one panel per model |
| Fig. A7a, A7b | `plot_tsne.py` | one checkpoint, one panel per model |
| NMI and ARI, reported in text | `cluster_metrics.py` | one checkpoint, and the ReClaim tokenizer to align the disease set |

What each script draws:

- **`plot_umap.py`** — scatter of the disease tokens' embedding rows reduced by
  UMAP, coloured by ICD-10 chapter, with marker area scaled from the disease's
  prevalence in the corpus and a chapter legend beside the axes; the marker-size
  legend is exported separately so two panels can share one row.
- **`plot_tsne.py`** — the same scatter from a t-SNE reduction. It uses openTSNE
  rather than scikit-learn, for a main-phase `exaggeration` separate from the
  early phase.
- **`cluster_metrics.py`** — k-means over the embeddings into as many clusters
  as there are chapters, scored against the chapter labels with normalized
  mutual information and the adjusted Rand index, averaged over several
  restarts. Beside its JSON it drops the scored embedding matrix and token ids
  as `*_embeddings.npy` and `*_delphi_ids.npy`; those are model weights, and
  the repository's `.gitignore` excludes `.npy` so they cannot be committed.

Nothing in the figures is annotated or highlighted. Which clusters to label
would be a judgement made after seeing the layout, so the claim rests on the
scatter and the scores, which treat every disease alike.

Shared model loading, token mapping, prevalence counting and drawing code is in
`common.py`; chapter labels and colours come from the Delphi label table named
above.

## Examples

Both figures and both scores, for Delphi and ReClaim, written to
`../../figures/disease_embeddings/` by default:

```bash
DELPHI_CKPT=/path/to/delphi RECLAIM_CKPT=/path/to/reclaim \
    bash run_disease_embedding_figures.sh
```

Marker sizes need per-disease counts from the corpus:

```bash
CORPUS_DIR=/path/to/tokenized/final_data DELPHI_CKPT=... RECLAIM_CKPT=... \
    bash run_disease_embedding_figures.sh
```

One UMAP panel on its own (Fig. A6b):

```bash
python plot_umap.py --model_path /path/to/reclaim --model_type hf --stem reclaim
```

One t-SNE panel, at the published settings (Fig. A7b):

```bash
python plot_tsne.py --model_path /path/to/reclaim --model_type hf --stem reclaim \
    --perplexity 16 --exaggeration 2 --learning_rate 10 --n_jobs 1
```

`--n_jobs 1` is deliberate: openTSNE's parallel neighbour search is not
deterministic, and at one thread the seed fixes the output. The run takes
seconds at this size.

Scores for one model, aligned to the disease set the ReClaim tokenizer covers
so the two models are scored over the same diseases:

```bash
python cluster_metrics.py --model_path /path/to/delphi --model_type delphi \
    --model_name Delphi --align_tokenizer_path /path/to/reclaim \
    --out_file figures/metrics_delphi.json
```

## Dependencies

`torch`, `numpy<2`, `pandas`, `pyarrow`, `matplotlib`, `tqdm`; `umap-learn` for
`plot_umap.py`, `openTSNE` for `plot_tsne.py`, `scikit-learn` for
`cluster_metrics.py`, and `transformers` to read a Hugging Face tokenizer.
