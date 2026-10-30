# Expenditure outputs

Formats the metric tables in `results/` into the paper's expenditure tables and figures; nothing is
re-scored.

```bash
bash output/scripts/make_all.sh        # from the repository root; PYTHON=<python> overrides env.sh
```

| Script | Writes |
|---|---|
| `scripts/roster.sh` | the reported model ids: Monte-Carlo anchor at `FINAL_N_GENERATION`, post-trained models, LightGBM window |
| `scripts/final_tables.py` | `tables/benchmark.*`, `tables/benchmark_by_payer.*`, `tables/inference_cost.*`, `tables/ROSTER.md` |
| `scripts/main_figure.py` | `figures/expenditure_main_figure.{pdf,png}`, from `tables/` only |
| `scripts/mc_ablation.py` | `tables/mc_ablation.*`, `figures/mc_ablation_figure.{pdf,png}` |
| `e4_eval/quantify_cost_discretization.py` (stage e4) | `figures/discretization_figure.{pdf,png}`; `make_all.sh` copies its table to `tables/discretization.*` |

Tables are written as CSV plus LaTeX. `tables/ROSTER.md` lists the ids behind them and any missing cell. `make_all.sh`
checks the metric tables in `results/` first and stops, leaving `output/` unchanged, if any is missing. The
discretization figure needs the raw claims and is not redrawn here.

Model ids: `trainmean`, `persistence`; `lightgbm-{reg,cls}-{1year,allhist}`;
`reclaim-{s,m,l}-pretrain-<N>` (Monte-Carlo anchor with N trajectories; N = 1 is greedy decoding);
`reclaim-{s,m,l}-posttrain`. Payer strata: `CR` commercial and Medicare supplemental, `MD` Medicaid.
