#!/bin/bash
#SBATCH --job-name=b3_3_summary
#SBATCH --partition=day
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_job_logs/prolog_%x_%j.out   # relative -> submit from the module root
# Stage b3_3: summary (summarize_results.R: forest plots + supplementary table) -> $RESULTS_ROOT/_summary/,
# plus the paper-comparison ladder printout for both controls. Covers every spec with an
# estimate.json, with the separation guard applied. Root: results/mdcr (RESULTS_ROOT overrides).
set -uo pipefail
MODULE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; case "$MODULE_ROOT" in */b[0-9_]*|*/tm[0-9]_*|*/ty[0-9]_*) MODULE_ROOT="${MODULE_ROOT%/*}";; esac
cd "$MODULE_ROOT" || { echo "ERROR: cd failed"; exit 1; }
. "$MODULE_ROOT/env.sh" || { echo "ERROR: env.sh missing -- copy env.sh.example to env.sh and fill in the paths"; exit 1; }
export PYTHONPATH="$MODULE_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # root-level imports (datasources, codesets, utils) from section dirs
[ -d config ] || { echo "ERROR: not the module root"; exit 1; }
_LOG="$MODULE_ROOT/slurm_job_logs/b3_3_summary_${SLURM_JOB_ID:-manual}"; mkdir -p "$(dirname "$_LOG")"
exec >"$_LOG.out" 2>"$_LOG.err"

module --force purge; module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"; set +u; conda activate "${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}"; set -u  # conda activate.d scripts reference unset vars
RS="${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}/bin/Rscript"
PY="${RECLAIM_ENV:?export RECLAIM_ENV (conda env path) in env.sh}/bin/python"

# ---- data roots (as in b0_2_build_cohorts) ----
R_ROOT="${RESULTS_ROOT:-results/mdcr}"
export INTER_ROOT="${INTER_ROOT:-intermediate/mdcr}"
echo "###### b3_3_summary summary: ($R_ROOT) ######"
echo "==== summarize_results: assemble forest + table (with separation guard) ===="
"$RS" b3_estimate/summarize_results.R "$R_ROOT"

echo
echo "==== _summary contents ===="
ls -l --time-style=+%Y-%m-%d_%H:%M "$R_ROOT/_summary/"
echo
echo "==== supplementary_table.csv: spec coverage + separation flags ===="
"$PY" - "$R_ROOT" <<'PYEOF'
import csv, collections, sys
rows=list(csv.DictReader(open(f"{sys.argv[1]}/_summary/supplementary_table.csv")))
print(f"  total rows: {len(rows)}")
spec=collections.Counter(r["spec"] for r in rows)
print("  by spec:", dict(sorted(spec.items())))
sep=[r for r in rows if r.get("separation","").upper()=="TRUE"]
print(f"  separation-flagged (non-estimable): {len(sep)}")
for r in sep:
    print(f"    {r['cell']:14s} {r['spec']:16s} {r['estimator']:14s} c={r['c_stat']} RR={r['rr']}")
PYEOF

echo
echo "############  PAPER-COMPARISON LADDERS (Schneeweiss 2009)  ############"
"$PY" - "$R_ROOT" <<'PYEOF'
import sys, os, json, glob, math, pandas as pd
root = sys.argv[1]
PAPER = {
    "positive_control_celecoxib_gi":
        ("POSITIVE control, paper Ex.1 (celecoxib -> GI bleed): crude 1.09 (0.91-1.30) -> "
         "demo 1.01 -> predefined 0.94 (rung not run here) -> hdPS 0.88 (0.73-1.06); SMR-weighted hdPS 0.77 (0.67-0.88)"),
    "negative_control_flu_hipfx":
        ("NEGATIVE control, paper Ex.3 (flu vaccine -> hip fx): crude 0.93 (0.80-1.08) -> "
         "demo+hdPS 1.02 (0.85-1.21)  [truth = 1.0]"),
}
ORDER = ["crude","demo","hdps",
         "delphi-only","reclaim-s-only","reclaim-m-only","reclaim-l-only"]
for cohort, target in PAPER.items():
    cells = sorted(os.path.basename(p) for p in glob.glob(f"{root}/{cohort}/*") if os.path.isdir(p))
    if not cells: continue
    print(f"\n== {target}")
    for cell in cells:
        coh = os.path.join(os.environ.get("INTER_ROOT", "intermediate/mdcr"), "cohorts", cohort, cell, "cohort.csv")
        if os.path.exists(coh):
            d = pd.read_csv(coh)
            g = d.groupby("arm")["outcome"].agg(n="size", ev="sum")
            exp = g.loc[1,"n"]/len(d) if 1 in g.index else float("nan")
            print(f"  -- {cell}: n={len(d):,}  events={int(d.outcome.sum()):,}  exposed frac={exp:.3f}")
        neg = cohort.startswith("negative_control")
        for spec in ORDER:
            f = f"{root}/{cohort}/{cell}/{spec}/decile_adjusted/estimate.json"
            if not os.path.exists(f): continue
            j = json.load(open(f))
            rr, lb, ub = j.get("rr"), j.get("rr_lb"), j.get("rr_ub")
            if not isinstance(rr, (int, float)): continue
            sep = " [SEPARATION]" if j.get("separation_flag") else ""
            null = ("  <- covers 1.0" if (lb or 0) <= 1.0 <= (ub or 0) else "  <- EXCLUDES 1.0") if neg else ""
            print(f"     {spec:16s} RR={rr:.3f} ({lb:.3f}-{ub:.3f})  [k={j.get('n_covariates')}, c={j.get('c_statistic')}]{sep}{null}")
PYEOF
echo "#######################################################################"
echo "DONE."
