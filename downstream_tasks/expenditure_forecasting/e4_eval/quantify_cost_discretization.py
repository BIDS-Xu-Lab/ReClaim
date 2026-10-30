#!/usr/bin/env python3
"""
Quantify the cost-token discretization error against raw MarketScan claims.

Background: the tokenization sums payments per (enrollee, month, payer,
visit type) and rounds each monthly sum to one significant digit
(<COST-me> = m x 10^e; s1_construct_tokens.encode_cost).  All expenditure
labels and predictions in the forecasting task therefore live in this
quantized space.  This script reconstructs the *exact* (unrounded) monthly
sums from the raw claims tables for the held-out test cohort, applies the
identical rounding operator, and quantifies the information lost at the
annual level - i.e., how well the tokenized world approximates true annual
gross payments.

Phases
------
reconstruct : stream raw claims (I: TOTPAY@ADMDATE, O: PAY@SVCDATE,
              D: PAY@SVCDATE), filter to the cohort, and write exact monthly
              sums per (enrollee, month, visit type, payer) to
              out_dir/monthly/<payer>_<table>_<year>.parquet
analyze     : apply the encode_cost rounding to each monthly group, aggregate
              exact vs tokenized annual totals per enrollee-year, and write the
              supplement items: one figure (exact vs tokenized hexbin), one table
              (enrollees, enrollee-years, R2, HNHC prevalence exact / tokenized,
              HNHC label changed by tokenization; overall and by payer) as CSV +
              LaTeX, and a README with the figure caption, definitions, files
              and provenance.  With --prospective_labels, also checks that the
              prospective (cutoff-year) token labels equal the tokenized
              reconstruction.
all         : both.

Faithfulness notes: the tokenizer applies only enrollee-level exclusions before cost
aggregation, and the evaluation cohort already passed those - so restricting to
cohort enrollee IDs reproduces the tokenized population.  Rows with a null
claim date are dropped (they cannot form a monthly event group).  Spark
F.round is HALF_UP; replicated here as floor(m + 0.5).

Negative monthly sums (refund / adjustment claims; ~0.02% of monthly groups,
~0.15% of enrollee-years) are mapped to $0 by the tokenizer, so in those years
the token total is not a rounding of the exact total; they are flagged
(has_negative_month) and excluded from the table and the figure.
"""

import os
import re
import argparse
import logging
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import setup_logging
from posttrain_bins import encode_cost

# (payment column, date column, visit type) per raw table letter
TABLE_SPECS = {
    'i': ('TOTPAY', 'ADMDATE', 'inpatient'),
    'o': ('PAY', 'SVCDATE', 'outpatient'),
    'd': ('PAY', 'SVCDATE', 'pharmacy'),
}
PAYER_PREFIX = {'CCAE': 'CR', 'MDCR': 'CR', 'MDCD': 'MD'}
HNHC_CUTOFF = 30000.0   # the task's only classification target (binary HNHC)


def parse_args():
    p = argparse.ArgumentParser(description="Quantify cost-token discretization error")
    p.add_argument("--phase", type=str, default="all",
                   choices=["reconstruct", "analyze", "all"])
    p.add_argument("--raw_dir", type=str, required=True,
                   help="RawData root (contains CCAE/ MDCR/ MDCD/)")
    p.add_argument("--cohort_dir", type=str, nargs='+', required=True,
                   help="One or more dirs holding preprocessed_input.parquet (or any "
                        "*.parquet) with an enrollee_id column; ids are unioned")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--figure_dir", type=str, default=None,
                   help="Also write the figure here (e.g. the paper's final-figures dir)")
    p.add_argument("--payers", type=str, nargs='+', default=['CCAE', 'MDCR', 'MDCD'])
    p.add_argument("--tables", type=str, nargs='+', default=['i', 'o', 'd'])
    p.add_argument("--years", type=int, nargs='+', default=list(range(2008, 2024)))
    p.add_argument("--batch_size", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers (one raw file each)")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Per-file batch cap (smoke tests only)")
    p.add_argument("--prospective_labels", type=str, default=None,
                   help="Optional prospective preprocessed_input.parquet for an "
                        "end-to-end token-label check on the cutoff year")
    p.add_argument("--cutoff_year", type=int, default=2023,
                   help="First post-cutoff year (prospective outcome year)")
    return p.parse_args()


# =============================================================================
# Quantization operator (numpy replica of s1_construct_tokens.encode_cost)
# =============================================================================

# =============================================================================
# Phase A: reconstruction
# =============================================================================

def find_raw_file(raw_dir: str, payer: str, table: str, year: int):
    if payer == 'MDCD':
        pattern = os.path.join(raw_dir, 'MDCD', f"Medicaid_{table}_{year}.snappy.parquet")
    else:
        pattern = os.path.join(raw_dir, payer, f"{payer}_{table.upper()}_{year}_v*.snappy.parquet")
    files = sorted(glob(pattern))
    return files[0] if files else None


def load_cohort_ids(cohort_dirs) -> dict:
    """Return {'CR': int64 array, 'MD': int64 array} of raw ENROLIDs, unioned
    over the given dirs (preprocessed_input.parquet if present, else *.parquet)."""
    files = []
    for d in cohort_dirs:
        pi = os.path.join(d, "preprocessed_input.parquet")
        files += [pi] if os.path.exists(pi) else sorted(glob(os.path.join(d, "*.parquet")))
    ids = pd.concat(
        [pd.read_parquet(f, columns=['enrollee_id']) for f in files]
    )['enrollee_id'].astype(str)
    out = {}
    for prefix in ('CR', 'MD'):
        sub = ids[ids.str.startswith(prefix)].str[len(prefix):]
        out[prefix] = np.unique(sub.astype(np.int64).values)
    return out


def reconstruct_one_file(raw_path: str, payer: str, table: str, year: int,
                         cohort_ids: np.ndarray, out_path: str,
                         batch_size: int, max_batches):
    """Stream one raw parquet; write exact monthly sums for cohort members."""
    pay_col, date_col, vt = TABLE_SPECS[table]
    pf = pq.ParquetFile(raw_path)
    parts = []
    n_rows = n_kept = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_size,
                                               columns=['ENROLID', date_col, pay_col])):
        if max_batches is not None and bi >= max_batches:
            break
        df = batch.to_pandas()
        n_rows += len(df)
        df = df.dropna(subset=['ENROLID', date_col])
        df['ENROLID'] = df['ENROLID'].astype(np.int64)
        df = df[np.isin(df['ENROLID'].values, cohort_ids)]
        if df.empty:
            continue
        n_kept += len(df)
        df['month'] = pd.to_datetime(df[date_col]).values.astype('datetime64[M]')
        g = (df.groupby(['ENROLID', 'month'])[pay_col]
               .agg(exact_sum='sum', n_claims='size').reset_index())
        parts.append(g)

    if parts:
        monthly = (pd.concat(parts)
                   .groupby(['ENROLID', 'month'])
                   .agg(exact_sum=('exact_sum', 'sum'), n_claims=('n_claims', 'sum'))
                   .reset_index())
    else:
        monthly = pd.DataFrame(columns=['ENROLID', 'month', 'exact_sum', 'n_claims'])
    monthly['payer'] = payer
    monthly['vt'] = vt
    monthly['year'] = year
    monthly.to_parquet(out_path, index=False)
    return raw_path, n_rows, n_kept, len(monthly)


def run_reconstruct(args):
    cohort = load_cohort_ids(args.cohort_dir)
    logging.info("  Cohort IDs: CR=%d, MD=%d", len(cohort['CR']), len(cohort['MD']))

    monthly_dir = os.path.join(args.out_dir, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)

    jobs = []
    for payer in args.payers:
        ids = cohort[PAYER_PREFIX[payer]]
        for table in args.tables:
            for year in args.years:
                raw_path = find_raw_file(args.raw_dir, payer, table, year)
                if raw_path is None:
                    logging.warning("  Missing raw file: %s %s %d", payer, table, year)
                    continue
                out_path = os.path.join(monthly_dir, f"{payer}_{table}_{year}.parquet")
                if os.path.exists(out_path):
                    logging.info("  Exists, skipping: %s", os.path.basename(out_path))
                    continue
                jobs.append((raw_path, payer, table, year, ids, out_path,
                             args.batch_size, args.max_batches))

    logging.info("  %d raw files to process (%d workers)", len(jobs), args.workers)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(reconstruct_one_file, *j): j for j in jobs}
        for fut in as_completed(futures):
            raw_path, n_rows, n_kept, n_groups = fut.result()
            logging.info("  Done %-45s rows=%d kept=%d monthly_groups=%d",
                         os.path.basename(raw_path), n_rows, n_kept, n_groups)


# =============================================================================
# Phase B: analysis
# =============================================================================

REL_ERR_FLOOR = 1.0   # $; the encoder maps 0 < v <= 1 to the $1 bin, so relative
                      # error below $1 is unbounded by construction (and sub-dollar
                      # exact sums are mostly +/- claim cancellation residue)

STRATA = [('overall', 'All'), ('payer_CR', 'CCAE + MDCR'), ('payer_MD', 'MDCD')]


def _stratum_metrics(df: pd.DataFrame, label: str) -> dict:
    """Six table rows + the two caption numbers, for one stratum of enrollee-years."""
    e, q = df['exact_total'].values, df['quant_total'].values
    ss_res = np.sum((q - e) ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    m = e >= REL_ERR_FLOOR
    rel = (q[m] - e[m]) / e[m]
    ye, yq = e >= HNHC_CUTOFF, q >= HNHC_CUTOFF
    near = (e >= 0.75 * HNHC_CUTOFF) & (e < 1.25 * HNHC_CUTOFF)
    return {
        'subset': label,
        'n_enrollees': int(df.groupby(['ENROLID', 'payer_group']).ngroups),
        'n_enrollee_years': int(len(df)),
        'R2_tokenized_vs_exact': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        'hnhc_prev_exact_pct': float(100 * ye.mean()),
        'hnhc_prev_tokenized_pct': float(100 * yq.mean()),
        'hnhc_flip_pct': float(100 * np.mean(ye != yq)),
        'hnhc_flip_near_pct': float(100 * np.mean(ye[near] != yq[near])) if near.any() else np.nan,
        'median_abs_rel_err_pct': float(100 * np.median(np.abs(rel))) if m.any() else np.nan,
    }


def run_analyze(args):
    monthly_dir = os.path.join(args.out_dir, "monthly")
    files = sorted(glob(os.path.join(monthly_dir, "*.parquet")))
    if not files:
        logging.error("  No monthly parquets in %s - run reconstruct first", monthly_dir)
        return
    monthly = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logging.info("  Loaded %d monthly groups from %d files", len(monthly), len(files))

    # Quantize each monthly group (the unit the tokenizer rounds at); v <= 0 -> $0 as in pretraining
    monthly['quant_sum'] = encode_cost(monthly['exact_sum'].values)
    n_neg = int((monthly['exact_sum'] < 0).sum())
    logging.info("  Negative monthly sums (refund/adjustment claims -> $0 in token space): %d (%.4f%%)",
                 n_neg, 100 * n_neg / len(monthly))

    # ---- Annual aggregation --------------------------------------------
    monthly['year'] = monthly['month'].astype('datetime64[ns]').dt.year
    monthly['payer_group'] = monthly['payer'].map({'CCAE': 'CR', 'MDCR': 'CR', 'MDCD': 'MD'})
    key = ['ENROLID', 'payer_group', 'year']
    annual = (monthly.groupby(key)[['exact_sum', 'quant_sum']].sum().reset_index()
              .rename(columns={'exact_sum': 'exact_total', 'quant_sum': 'quant_total'}))
    neg_years = monthly.loc[monthly['exact_sum'] < 0, key].drop_duplicates()
    annual = annual.merge(neg_years.assign(has_negative_month=True), on=key, how='left')
    annual['has_negative_month'] = annual['has_negative_month'].fillna(False).astype(bool)
    n_neg_years = int(annual['has_negative_month'].sum())
    logging.info("  Enrollee-years: %d (enrollees: %d); with a negative month, excluded: %d (%.3f%%)",
                 len(annual), annual.groupby(['ENROLID', 'payer_group']).ngroups,
                 n_neg_years, 100 * n_neg_years / len(annual))
    annual.to_parquet(os.path.join(args.out_dir, "annual_exact_vs_tokenized.parquet"), index=False)

    # ---- Table: overall and by payer, years without negative months -----
    clean = annual[~annual['has_negative_month']]
    records = [_stratum_metrics(clean, 'overall')]
    for pg, g in clean.groupby('payer_group'):
        records.append(_stratum_metrics(g, f'payer_{pg}'))
    metrics = pd.DataFrame(records)
    metrics.to_csv(os.path.join(args.out_dir, "discretization_metrics.csv"), index=False)
    for line in metrics.to_string(index=False).split("\n"):
        logging.info("    %s", line)

    _plot(clean, args.out_dir, args.figure_dir)
    prosp = run_prospective_check(args, annual) if args.prospective_labels else {}
    pct_neg_years = 100 * n_neg_years / len(annual)
    _write_table_and_readme(args, metrics, pct_neg_years, prosp)


def _plot(annual: pd.DataFrame, out_dir: str, figure_dir: str = None):
    """Supplementary figure: exact vs tokenized annual expenditure, one hexbin panel with a
    horizontal log-scale colour bar of enrollee-year counts, identity line and the HNHC
    threshold marked.  revision/reclaim_vis_style.md: 88 mm single column, Helvetica/Arial
    5-7 pt, vector PDF with Type-42 text, RGB; no model colour is used (identity and guides
    in neutral ink; viridis is the sanctioned >4-level ramp for a density)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
        'legend.fontsize': 5.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
        'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.default': 'regular',
        'figure.facecolor': 'white',
    })
    INK, MUTED = '#222222', '#787878'
    MM = 1 / 25.4
    e, q = annual['exact_total'].values, annual['quant_total'].values
    m = (e >= REL_ERR_FLOOR) & (q > 0)
    le, lq = np.log10(e[m]), np.log10(q[m])
    top = float(np.ceil(max(le.max(), lq.max()) * 2) / 2)   # next half-decade

    fig = plt.figure(figsize=(88 * MM, 100 * MM))            # exact single-column width; no tight bbox
    ax = fig.add_axes([0.16, 0.235, 0.80, 0.735])
    cax = fig.add_axes([0.16, 0.105, 0.80, 0.022])
    hb = ax.hexbin(le, lq, gridsize=56, bins='log', cmap='viridis', mincnt=1,
                   linewidths=0.15, edgecolors='face', rasterized=True)
    ax.plot([0, top], [0, top], color=INK, ls='--', lw=0.6, label='Identity (tokenized = exact)')
    h = np.log10(HNHC_CUTOFF)
    ax.axvline(h, color=MUTED, ls=':', lw=0.5); ax.axhline(h, color=MUTED, ls=':', lw=0.5)
    ax.text(h + 0.06, 0.12, 'HNHC threshold ($30,000)', rotation=90, fontsize=5.5,
            color=MUTED, va='bottom', ha='left')

    ticks = np.arange(0, top + 0.01, 1.0)
    fmt = FuncFormatter(lambda v, _: {0: '$1', 1: '$10', 2: '$100', 3: '$1K', 4: '$10K',
                                      5: '$100K', 6: '$1M', 7: '$10M'}.get(int(round(v)), ''))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(fmt)
    ax.set_xlim(-0.15, top); ax.set_ylim(-0.15, top); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Exact annual expenditure, raw claims (log scale)', labelpad=1.5)
    ax.set_ylabel('Tokenized annual expenditure (log scale)', labelpad=2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', frameon=False, handlelength=1.8, borderaxespad=0.3)
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal')
    cb.set_label('Enrollee-years per bin (log scale)', fontsize=6, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    dirs = [out_dir] + ([figure_dir] if figure_dir else [])
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 450})):
            fig.savefig(os.path.join(d, f"discretization_figure.{ext}"), **kw)
        logging.info("  Saved figure: %s", os.path.join(d, "discretization_figure.png"))
    plt.close(fig)


def run_prospective_check(args, annual: pd.DataFrame) -> dict:
    """Pipeline sanity check: the prospective outcome labels (token sums) must equal
    the tokenized raw-claims reconstruction for the cutoff year.  Enrollees with no
    raw claims in that year have exact = tokenized = $0."""
    logging.info("  Prospective end-to-end label check (year=%d)", args.cutoff_year)
    labels = pd.read_parquet(args.prospective_labels, columns=['enrollee_id', 'next_annual_cost'])
    labels['payer_group'] = labels['enrollee_id'].astype(str).str[:2]
    labels['ENROLID'] = labels['enrollee_id'].astype(str).str[2:].astype(np.int64)
    ann_y = annual[annual['year'] == args.cutoff_year]
    merged = labels.merge(ann_y, on=['ENROLID', 'payer_group'], how='left')
    n_raw = int(merged['exact_total'].notna().sum())
    merged['exact_total'] = merged['exact_total'].fillna(0.0)
    merged['quant_total'] = merged['quant_total'].fillna(0.0)
    t, r = merged['next_annual_cost'].values.astype(float), merged['quant_total'].values
    same = np.isclose(t, r, rtol=1e-9)
    stats = {'year': args.cutoff_year, 'n': int(len(merged)), 'n_with_raw_claims': n_raw,
             'pct_exact_match': float(100 * same.mean()), 'n_mismatch': int((~same).sum()),
             'max_abs_diff': float(np.max(np.abs(t - r)[~same])) if (~same).any() else 0.0}
    logging.info("    %d enrollees (%d with raw claims in %d): label == tokenized reconstruction for %.2f%%; "
                 "%d differ, max |diff| $%.0f", stats['n'], n_raw, args.cutoff_year,
                 stats['pct_exact_match'], stats['n_mismatch'], stats['max_abs_diff'])
    merged.to_parquet(os.path.join(args.out_dir, "prospective_label_check.parquet"), index=False)
    return stats


def _write_table_and_readme(args, metrics: pd.DataFrame, pct_neg_years: float, prosp: dict) -> None:
    from datetime import date
    m = metrics.set_index('subset')
    ov = m.loc['overall']
    job = os.environ.get('SLURM_JOB_ID', 'interactive')
    ROWS = [  # (label, formatter); shared by the LaTeX and the markdown table
        ('Enrollees', lambda r: f"{int(r['n_enrollees']):,}"),
        ('Enrollee-years', lambda r: f"{int(r['n_enrollee_years']):,}"),
        ('R2, tokenized vs exact', lambda r: f"{r['R2_tokenized_vs_exact']:.3f}"),
        ('HNHC prevalence, exact', lambda r: f"{r['hnhc_prev_exact_pct']:.2f}%"),
        ('HNHC prevalence, tokenized', lambda r: f"{r['hnhc_prev_tokenized_pct']:.2f}%"),
        ('HNHC label changed by tokenization', lambda r: f"{r['hnhc_flip_pct']:.2f}%"),
    ]
    caption_body = (
        "For every enrollee in the retrospective and prospective evaluation cohorts, exact annual "
        "expenditure was reconstructed from the raw MarketScan inpatient, outpatient and pharmacy claims "
        "and compared with the tokenized total, in which each monthly sum per claim type is rounded to one "
        "significant digit (the <COST-me> lattice) before summation, overall and by payer. R2: coefficient "
        "of determination of the tokenized annual total against the exact total. HNHC: high-need, "
        "high-cost, annual total >= $30,000; the label is changed by tokenization when it differs between "
        f"the exact and the tokenized total (among enrollee-years whose exact total lies within +/-25% of "
        f"the threshold, {ov['hnhc_flip_near_pct']:.1f}% change; the median absolute relative error of the tokenized "
        f"total is {ov['median_abs_rel_err_pct']:.1f}%). Enrollee-years containing a negative monthly sum (refund or "
        f"adjustment claims, which the tokenizer maps to $0; {pct_neg_years:.2f}%) are excluded.")

    # ---- LaTeX ----------------------------------------------------------
    def tex_escape(s):
        return (s.replace('%', '\\%').replace('$', '\\$').replace('>=', '$\\geq$').replace('+/-', '$\\pm$')
                 .replace('R2', 'R$^2$').replace('<COST-me>', '\\texttt{<COST-me>}').replace('30,000', '30{,}000'))
    tex = ["\\begin{table}[H]", "\\centering",
           "\\caption{\\textbf{Effect of cost-token discretization on expenditure labels.} " + tex_escape(caption_body) + "}",
           "\\label{tab:discretization}", "\\small", "\\renewcommand{\\arraystretch}{1.15}",
           "\\begin{tabular}{@{}lrrr@{}}", "\\hline",
           " & " + " & ".join(f"\\textbf{{{name}}}" for _, name in STRATA) + " \\\\", "\\hline"]
    for label, fmt in ROWS:
        tex.append(tex_escape(label) + " & " + " & ".join(tex_escape(fmt(m.loc[k])) for k, _ in STRATA) + " \\\\")
    tex += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(args.out_dir, "discretization_table.tex"), 'w') as f:
        f.write("\n".join(tex))

    # ---- README ---------------------------------------------------------
    rows = ["| | " + " | ".join(name for _, name in STRATA) + " |", "|---|" + "---:|" * len(STRATA)]
    for label, fmt in ROWS:
        rows.append("| " + label.replace('R2', 'R²') + " | " + " | ".join(fmt(m.loc[k]) for k, _ in STRATA) + " |")
    check = (f"* Prospective label check ({prosp['year']}): for all {prosp['n']:,} prospective enrollees "
             f"({prosp['n_with_raw_claims']:,} with raw claims in the outcome year, the rest labelled $0), the pipeline's "
             f"outcome label equals the tokenized raw-claims reconstruction for {prosp['pct_exact_match']:.2f}%; the "
             f"{prosp['n_mismatch']:,} remaining differ by at most ${prosp['max_abs_diff']:,.0f} (one low-value monthly "
             "token; claims without a service date cannot be placed in a month by the reconstruction).\n") if prosp else ""
    md = f"""# Cost-token discretization audit (evaluation cohorts)

Supplementary material for the healthcare-expenditure forecasting task: how far the tokenized
expenditure space, in which every label and prediction of the task lives, sits from the exact
dollar amounts in the raw claims.  Generated by `e4_eval/quantify_cost_discretization.py`
(SLURM job {job}, {date.today()}); the table is read from `discretization_metrics.csv`.

## Supplementary Figure | Tokenized versus exact annual expenditure

![discretization_figure](discretization_figure.png)

**Cost-token rounding preserves annual expenditure across six orders of magnitude.**
Two-dimensional histogram of {int(ov['n_enrollee_years']):,} enrollee-years from the {int(ov['n_enrollees']):,}
enrollees of the retrospective and prospective evaluation cohorts ({min(args.years)}–{max(args.years)}).
For each enrollee-year the *x* axis is the exact annual expenditure reconstructed from the raw
MarketScan inpatient, outpatient and pharmacy claims, and the *y* axis is the same year's total
after tokenization: payments are summed per enrollee, month, payer and claim type, each
monthly sum is rounded to one significant digit (the `<COST-me>` lattice, 59 value-ordered
bins), and the tokenized annual total is the sum of those rounded monthly values.  Both axes are logarithmic.  Each
hexagon is one bin of the plane; its colour gives the number of enrollee-years in the bin on a
logarithmic colour scale (dark, 1–10; yellow, >10,000).  The dashed red line is the identity,
on which tokenization changed nothing; the dotted grey lines mark the high-need, high-cost (HNHC)
threshold of $30,000 on both axes.  The cloud forms a band of near-constant width around the
identity, because one-significant-digit rounding moves a value by at most about one third
(±0.13 on the log axis); the slight periodic wobble is the lattice itself (values just below a
bin edge round up, just above round down).  Below about $10 the lattice is coarse ($1, $2, $3,
…), so tokenized totals snap to a few discrete levels; the sparse bins above the identity there
are years composed of many sub-dollar monthly sums, which the encoder lifts to the $1 bin.
Enrollee-years with exact totals below $1 and years containing a negative monthly sum
({pct_neg_years:.2f}%; refund or adjustment claims, which the tokenizer maps to $0) are not shown.
R² of the tokenized against the exact total {ov['R2_tokenized_vs_exact']:.3f}; median absolute relative error
{ov['median_abs_rel_err_pct']:.1f}%; HNHC label changed by tokenization in {ov['hnhc_flip_pct']:.2f}% of enrollee-years
(Supplementary Table below).

## Supplementary Table | Effect of cost-token discretization on expenditure labels, overall and by payer

{chr(10).join(rows)}

Definitions.  *Exact* = sum of raw claim payments in the calendar year; *tokenized* = the same
sum after each monthly value is rounded to one significant digit (the value the model sees).
R² = 1 − Σ(tokenized − exact)² / Σ(exact − mean)², tokenized regressed on exact.  HNHC = annual
total ≥ $30,000; *changed by tokenization* = the HNHC label differs between exact and tokenized
totals; among enrollee-years whose exact total lies within ±25% of the threshold
([$22,500, $37,500)) the share is {ov['hnhc_flip_near_pct']:.1f}%.  Median absolute relative error of the
tokenized total, (tokenized − exact) / exact over enrollee-years with exact ≥ ${REL_ERR_FLOOR:.0f} (the encoder
maps 0 < v ≤ $1 to the $1 bin): {ov['median_abs_rel_err_pct']:.1f}%.  Enrollee-years containing a negative
monthly sum are excluded from the table and the figure.

## Files

| file | content |
|---|---|
| `discretization_figure.png` / `.pdf` | the figure (88 mm single column; 450 dpi PNG; vector PDF, hexbins rasterised, fonts embedded as Type 42); also written to `--figure_dir` when given |
| `discretization_metrics.csv` | the table, one row per stratum (plus the two caption numbers) |
| `discretization_table.tex` | the table as a LaTeX supplement table with caption |
| `annual_exact_vs_tokenized.parquet` | one row per enrollee-year: exact and tokenized totals, negative-month flag |
| `prospective_label_check.parquet` | one row per prospective enrollee: pipeline label, tokenized and exact reconstruction |
| `monthly/` | exact monthly sums per enrollee, claim type, payer and year (reconstruction cache; reused by analyze-only reruns) |

## Provenance

* Cohort ids: {', '.join(f'`{d}`' for d in args.cohort_dir)}
* Raw claims: `{args.raw_dir}` (tables {', '.join(args.tables)}; payers {', '.join(args.payers)}; years {min(args.years)}–{max(args.years)})
* Rounding operator: `e0_common/posttrain_bins.encode_cost` (bit-identical replica of the pre-training tokenizer's rule, Spark ROUND_HALF_UP; v ≤ 0 → $0)
{check}* Command: `TASK=discretization e4_eval/run.sh` (reconstruct + analyze; ~12 min on one bigmem node, 24 workers)
"""
    with open(os.path.join(args.out_dir, "README.md"), 'w') as f:
        f.write(md)
    logging.info("  Saved discretization_table.tex and README.md")


# =============================================================================
# Main
# =============================================================================

def main():
    log_file = setup_logging(level=logging.INFO)
    args = parse_args()
    logging.info("DISCRETIZATION QUANTIFICATION STARTING (phase=%s)...", args.phase)
    logging.info("  Raw: %s", args.raw_dir)
    logging.info("  Cohort: %s", args.cohort_dir)
    logging.info("  Out: %s", args.out_dir)
    logging.info("  Payers=%s tables=%s years=%s", args.payers, args.tables, args.years)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase in ("reconstruct", "all"):
        run_reconstruct(args)
    if args.phase in ("analyze", "all"):
        run_analyze(args)

    logging.info("DISCRETIZATION QUANTIFICATION COMPLETED")
    logging.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
