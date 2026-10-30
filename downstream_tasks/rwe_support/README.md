# ReClaim real-world-evidence studies

Code for the two real-world-evidence studies of the ReClaim paper.

| Module | Study | Data |
|---|---|---|
| `study1/` | Study 1: known-effect validation, replicating Schneeweiss et al. 2009 | MarketScan claims (MDCR) |
| `study2/marketscan/` | Study 2: target trial emulation of Tang et al. 2026 | MarketScan claims (held-out split) |
| `study2/ynhh/` | Study 2 on an EHR source | YNHH OMOP |
| `output/` | every RWE table and figure of the paper | the modules' results |

Each module runs four numbered SLURM stages: `<x>0` cohorts, `<x>1` hdPS covariates, `<x>2` embeddings (GPU),
`<x>3` estimation (`x` = `b`, `tm`, `ty`).

## Setup

```bash
conda env create -f environment.yml && conda activate reclaim-rwe
Rscript install_r_packages.R                    # CRAN_REPO=<mirror> on a no-egress cluster
cp <module>/env.sh.example <module>/env.sh      # data roots, conda env, checkpoints
```

Submit wrappers from the module root. ReClaim checkpoints are set in `env.sh` as `RECLAIM_{S,M,L}_CKPT` and
referred to as `reclaim-s|m|l`.

## Data

No data, checkpoints or results are distributed; MarketScan and YNHH data need their own licenses. Runs write
person-level files to `intermediate/` and aggregates to `results/`, both gitignored; `output/` holds aggregates only.
`bash output/scripts/make_all.sh` regenerates the RWE tables and figures from a run's `results/`; the cohort tables
`s1_cohort_*` and `s2_cohorts` also need the person-level `intermediate/` cohort files and are skipped without them.
