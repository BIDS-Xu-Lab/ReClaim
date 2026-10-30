# ReClaim healthcare-expenditure forecasting

Code for the expenditure-forecasting task of the ReClaim paper: next-year total healthcare
expenditure and identification of high-need, high-cost (HNHC) enrollees on MarketScan claims.

| Stage | What it does | Runs on |
|---|---|---|
| `e1_preprocessing/` | evaluation cohorts (retrospective, prospective), supervision pool, post-training pairs | CPU |
| `e2_baseline/` | LightGBM baseline, non-learned anchors, Monte-Carlo anchor of the pre-trained model | CPU, GPU |
| `e3_posttrain/` | expenditure post-training and the one-token readout | GPU |
| `e4_eval/` | scoring of every model; cost-token discretization audit | CPU |
| `output/` | every expenditure table and figure of the paper | the metric tables |
| `e0_common/` | token parsing, cost bins, sampling, inference-cost records | — |

## Design

- **Task.** For an enrollee and a prediction point, predict the total gross payments of the next
  calendar year, censored at $1,000,000 (R², MAE), and whether it reaches $30,000 (HNHC; ROC-AUC,
  PR-AUC). Retrospective evaluation: a random calendar-year boundary within the history.
  Prospective evaluation: history up to the training cutoff, outcome in the first year after it.
- **Post-trained ReClaim** (`reclaim-{s,m,l}-posttrain`). The prompt is the history followed by
  `<INSTRUCT-COST>`; the response is one cost token, the next-year total on the `<COST-me>` lattice
  of pre-training (59 bins). Loss: soft ordinal cross-entropy over the bins plus a Huber term on the
  expected log-dollars. Readout: one forward pass; the softmax over the bins gives the forecast
  (expectation over the bins' dollar values) and the probability of exceeding any cutoff.
  100,000 pairs, 2 epochs, learning rate 3e-5, global batch 32 (S, M) or 64 (L).
- **Monte-Carlo anchor** (`reclaim-{s,m,l}-pretrain`, scored as `…-pretrain-<N>`). The pre-trained
  model samples N one-year trajectories per enrollee; the forecast is the mean of their cost-token
  sums and the HNHC score the fraction at or above the cutoff. The paper reports N = 64; `output/`
  also tabulates every N generated.
- **LightGBM** (`lightgbm-{reg,cls}-{1year,allhist}`). One regressor and one HNHC classifier per
  feature window, on prior cost, age and bag-of-words counts of diagnosis, procedure and drug
  tokens; early stopping on the validation split.
- **Anchors.** `persistence` (last observed-year cost) and `trainmean`.
- All sampling (supervision pool, train/validation split, pairs, prediction points) is keyed on a
  salted hash of the enrollee id, so any copy of the corpus yields the same cohorts.

## Setup

```bash
conda env create -f environment.yml && conda activate reclaim-expenditure
cp env.sh.example env.sh      # checkpoints, data root, software; never committed
```

Checkpoints are set in `env.sh` as `RECLAIM_{S,M,L}_CKPT` and referred to as `reclaim-s|m|l`.
Pipeline constants (cohort rules, recipe, cutoffs) are in `config.sh`.

## Run

Submit from the repository root. Each stage has a launcher (`DRY=1` prints the `sbatch` commands).

```bash
e1_preprocessing/run.sh                                   # test cohort + supervision pool, then pairs
e2_baseline/run.sh                                        # LightGBM train -> predictions (+ anchors)
TASK=mc MODEL_SIZE=all e2_baseline/run.sh                 # Monte-Carlo anchor at NUM_GENERATION, then aggregation
MODEL_SIZE=all e3_posttrain/run.sh                        # post-training
ARM=readout MODEL_SIZE=all e3_posttrain/run.sh            # readout of the three checkpoints
e4_eval/run.sh                                            # score everything, then output/
TASK=discretization e4_eval/run.sh                        # raw-claims audit of the cost tokens (once; needs MS_RAW_DIR)
bash output/scripts/make_all.sh                           # tables and figures only, from results/
```

- Other trajectory counts: `NUM_GENERATION=<N> TASK=mc … e2_baseline/run.sh`; every N keeps its own
  files and is scored as its own model. `N = 1` decodes greedily. Host memory of the generation job
  grows with N (`MC_MEM`).
- A post-training recipe other than the reported one needs a name suffix:
  `LR=5e-5 NAME_SUFFIX=-lr5e5 MODEL_SIZE=s e3_posttrain/run.sh`.
- `EXTRA_MODELS="<id> <id>"` (or `auto`) adds prediction folders to the scoring; only the ids in
  `output/scripts/roster.sh` enter the paper's tables and figure.

## Data

No data, checkpoints or results are distributed; MarketScan needs its own license. A run writes
person-level files to `data/`, checkpoints to `models/` and metric tables to `results/`, all
gitignored. `output/` holds aggregates only.
