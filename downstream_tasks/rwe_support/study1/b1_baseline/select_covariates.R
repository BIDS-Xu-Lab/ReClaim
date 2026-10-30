#!/usr/bin/env Rscript
# =============================================================================
# select_covariates.R -- thin wrapper around CRAN autoCovariateSelection.
#
# Implements Schneeweiss hdPS steps 2-4 (candidate -> recurrence -> prioritise) with the paper's
# defaults (n=200 per dimension, min 100 patients, k=500). Steps 1 and 5-7 are our own code
# (build_long_table builds the window/dimensions; estimate_ladder assembles the PS formula and
# estimates). The covariate engine is the package, not hand-coded.
#
# Inputs : --long_parquet (build_long_table output), --cohort_csv
# Output : --out_dir/hdps_covars.csv (patient x selected covariate 0/1) + multiplicative_bias.csv
# Usage  : Rscript select_covariates.R --long_parquet <dir> --cohort_csv <csv> --out_dir <dir>
#              --n 200 --min_num_patients 100 --k 500
#
# The package version is pinned so a CRAN update cannot silently shift covariate selection.
PINNED_VERSION <- "1.0.0"   # autoCovariateSelection version this pipeline is validated against
# =============================================================================

suppressWarnings(suppressMessages({
  library(optparse)
  library(dplyr)
  library(nanoparquet)            # reads the Spark-written long parquet without libarrow
  library(autoCovariateSelection) # the pinned covariate engine
}))
source(file.path(dirname(sub("--file=", "",
        commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])), "..", "utils.R"))
log_file <- setup_logging()

opt <- parse_args(OptionParser(option_list = list(
  make_option("--long_parquet", type = "character"),
  make_option("--cohort_csv",   type = "character"),
  make_option("--out_dir",      type = "character"),
  make_option("--n", type = "integer", default = 200L),
  make_option("--min_num_patients", type = "integer", default = 100L),
  make_option("--k", type = "integer", default = 500L),
  make_option("--recurrence", type = "character", default = "auto",
              help = paste("Step-3 engine: 'dense' = the pinned autoCovariateSelection function;",
                           "'sparse' = the equivalent that avoids its patient x covariate grid;",
                           "'auto' (default) = dense unless that grid would exceed R's",
                           ".Machine$integer.max. Use dense/sparse explicitly to verify they agree."))
)))

got <- as.character(utils::packageVersion("autoCovariateSelection"))
if (!identical(got, PINNED_VERSION)) {
  log_warning(sprintf("autoCovariateSelection version %s != pinned %s -- covariate selection may drift",
                      got, PINNED_VERSION))
}

# ---- inputs ----------------------------------------------------------------
log_info(sprintf("Reading long table: %s", opt$long_parquet))
df_long <- as.data.frame(read_spark_parquet(opt$long_parquet))
df_long$person_id <- as.character(df_long$person_id)

cohort <- read.csv(opt$cohort_csv)
cohort$person_id <- as.character(cohort$person_id)
ids <- unique(cohort$person_id)                          # carries covariate-less patients through
df_long <- df_long[df_long$person_id %in% ids, ]

# exposure/outcome vectors aligned to patientIdVector (ids) order
treat01   <- cohort$arm[match(ids, cohort$person_id)]
outcome01 <- cohort$outcome[match(ids, cohort$person_id)]

log_info(sprintf("Patients=%d  long_rows=%d  exposed=%d  events=%d",
                 length(ids), nrow(df_long), sum(treat01), sum(outcome01)))

# ---- Step 2: candidate covariates (top-n per dimension) --------------------
step1 <- get_candidate_covariates(
  df = df_long, domainVarname = "domain", eventCodeVarname = "event_code",
  patientIdVarname = "person_id", patientIdVector = ids,
  n = opt$n, min_num_patients = opt$min_num_patients)
log_info(sprintf("Step2 candidate covariates: %d", length(unique(step1$covars_data$event_code))))

# ---- Step 3: recurrence tiers (once / >=median / >=75th pct) ---------------
# Sparse re-implementation of autoCovariateSelection::get_recurrence_covariates.
#
# The library dcast()s the sparse (patient, code) table into a dense patient x code grid before
# computing anything. At full-MDCR scale that grid (millions of patients x 1,000 candidates)
# exceeds R's .Machine$integer.max and data.table aborts.
#
# The grid is unnecessary: the library derives the cutoffs as
#     median <- as.integer(median(vals[vals != 0])); uq <- as.integer(quantile(vals[vals != 0], .75))
# i.e. it filters the zeros out again, so the cutoffs depend only on the non-zero counts, which
# are exactly the sparse rows. A patient absent for a code has count 0, so with median >= 1 and
# uq >= 1 every indicator is 0 for them and their row never needs materialising.
#
# The four (median, uq) branches, as.integer() truncation, "rec_<code>_<tier>" naming and the
# alphabetical code order are reproduced exactly, so the output is identical to the library's
# (verified against the dense path where it fits; see --recurrence dense|sparse).
get_recurrence_covariates_sparse <- function(df, patientIdVector) {
  dt <- data.table::data.table(pid = as.character(df$person_id),
                               code = as.character(df$event_code))
  cnt <- dt[, .N, by = .(pid, code)]                       # sparse counts == dcast's fun.aggregate=length
  pos <- seq_along(patientIdVector); names(pos) <- patientIdVector
  cnt[, pi := pos[pid]]
  cnt <- cnt[!is.na(pi)]
  codes <- sort(unique(dt$code))                           # dcast orders columns by sorted code
  n <- length(patientIdVector)
  out <- list(person_id = patientIdVector)
  for (cv in codes) {
    sub <- cnt[code == cv]
    nz <- sub$N                                            # non-zero counts only -- the library's vals[vals!=0]
    med <- as.integer(stats::median(nz))
    uq  <- as.integer(stats::quantile(nz, 0.75))
    v <- integer(n); v[sub$pi] <- sub$N                    # absent patients stay 0
    out[[paste0("rec_", cv, "_once")]] <- as.integer(v >= 1L)
    if (med != 1L) out[[paste0("rec_", cv, "_sporadic")]]  <- as.integer(v >= med)
    if (uq  != med) out[[paste0("rec_", cv, "_frequent")]] <- as.integer(v >= uq)
  }
  list(recurrence_data = as.data.frame(out, stringsAsFactors = FALSE),
       patientIds = patientIdVector)
}

n_cand <- length(unique(step1$covars_data$event_code))
use_sparse <- switch(opt$recurrence,
  sparse = TRUE, dense = FALSE,
  # auto: the dense path is the pinned library, so prefer it and switch only when its grid would
  # overflow. 0.95 of the limit leaves headroom rather than discovering the abort after the work.
  as.numeric(length(ids)) * n_cand > 0.95 * .Machine$integer.max)
log_info(sprintf("Step3 recurrence path: %s (%d patients x %d candidates = %.2e cells)",
                 if (use_sparse) "SPARSE" else "dense (library)",
                 length(ids), n_cand, as.numeric(length(ids)) * n_cand))
step2 <- if (use_sparse) {
  suppressWarnings(suppressMessages(library(data.table)))
  get_recurrence_covariates_sparse(step1$covars_data, ids)
} else {
  get_recurrence_covariates(
    df = step1$covars_data, patientIdVarname = "person_id",
    eventCodeVarname = "event_code", patientIdVector = ids)
}
# get_recurrence_covariates returns list(recurrence_data, patientIds); the recurrence-covariate
# count is the columns of recurrence_data minus person_id.
log_info(sprintf("Step3 recurrence covariates: %d", ncol(step2$recurrence_data) - 1L))

# ---- Step 4: prioritise by |log(Bross multiplicative bias)|, keep top k ----
# Standard library implementation (autoCovariateSelection::get_prioritised_covariates).
# DISCLOSED DEVIATION from Schneeweiss 2009 step 4: the package applies the bias term with the
# unflipped RR_CD for RR_CD < 1 (the paper substitutes 1/RR_CD), and covariates with RR_CD = Inf
# rank as NaN and are dropped. Identical to the paper for RR_CD >= 1; the paper's own
# sensitivity analyses (additive term, no prioritization) bound the effect of the ranking rule
# at ~0.03 on the RR, so the standard implementation is kept deliberately.
step3 <- get_prioritised_covariates(
  df = step2$recurrence_data, patientIdVarname = "person_id",
  exposureVector = treat01, outcomeVector = outcome01,
  patientIdVector = ids, k = opt$k)
hdps_covars <- step3$autoselected_covariate_df   # patient x selected-covariate 0/1
log_info(sprintf("Step4 selected covariates: %d", ncol(hdps_covars) - 1))

# ---- outputs ---------------------------------------------------------------
dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(hdps_covars, file.path(opt$out_dir, "hdps_covars.csv"), row.names = FALSE)
# multiplicative_bias holds the abs-log Bross term per covariate
if (!is.null(step3$multiplicative_bias)) {
  write.csv(step3$multiplicative_bias, file.path(opt$out_dir, "multiplicative_bias.csv"), row.names = FALSE)
}
log_info(sprintf("Wrote hdps_covars.csv (%d x %d). Log: %s",
                 nrow(hdps_covars), ncol(hdps_covars), log_file))
