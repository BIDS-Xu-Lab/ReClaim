#!/usr/bin/env Rscript
# =============================================================================
# select_covariates.R -- Schneeweiss hdPS covariate selection per emulation (ty1_1).
#
# Thin wrapper around CRAN autoCovariateSelection (steps 2-4: candidate -> recurrence ->
# Bross prioritisation, n=200/k=500 defaults), run on the two-arm emulation cohort against
# the union hdPS long table. The Bross ranking needs one exposure and one outcome vector:
#   exposure = treatment class (GLP-1RA = 1)
#   outcome  = COMPOSITE any-primary-psychiatric-event (the hdPS algorithm is single-outcome;
#              ranking against the composite selects confounder proxies relevant to the
#              psychiatric outcome family as a whole)
#
# Inputs: --long_parquet (build_long_table), --cohort_csv (emulation cohort).
# Output: <out_dir>/hdps_covars.csv (+ multiplicative_bias.csv).
# =============================================================================

# ensure_packages (utils.R) installs missing CRAN packages before library() can fail;
# conda-provided ones (survival) are only checked.
source(file.path(dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])), "..", "utils.R"))
ensure_packages(c("optparse", "dplyr", "nanoparquet", "autoCovariateSelection", "data.table", "glue"))
suppressWarnings(suppressMessages({
  library(optparse)
  library(dplyr)
  library(nanoparquet)
  library(autoCovariateSelection)
}))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()

opt <- parse_args(OptionParser(option_list = list(
  make_option("--long_parquet", type = "character"),
  make_option("--cohort_csv",   type = "character"),
  make_option("--treat_class",  type = "character", default = "glp1ra"),
  make_option("--comp_class",   type = "character", default = "sglt2i"),
  make_option("--out_dir",      type = "character"),
  make_option("--n", type = "integer", default = 200L),
  make_option("--min_num_patients", type = "integer", default = 50L),
  make_option("--k", type = "integer", default = 500L)
)))

# ---- inputs ----------------------------------------------------------------
log_info(sprintf("Comparison: %s vs %s", opt$treat_class, opt$comp_class))
log_info(sprintf("Reading long table: %s", opt$long_parquet))
df_long <- as.data.frame(read_spark_parquet(opt$long_parquet))
df_long$person_id <- as.character(df_long$person_id)

cohort <- read.csv(opt$cohort_csv)
cohort$person_id <- as.character(cohort$person_id)
cohort <- cohort[cohort$drug_class %in% c(opt$treat_class, opt$comp_class), ]
ids <- unique(cohort$person_id)
df_long <- df_long[df_long$person_id %in% ids, ]

treat01 <- as.integer(cohort$drug_class[match(ids, cohort$person_id)] == opt$treat_class)
ev_cols <- grep("^out_.*_event$", names(cohort), value = TRUE)
outcome01 <- as.integer(rowSums(cohort[match(ids, cohort$person_id), ev_cols, drop = FALSE]) > 0)

log_info(sprintf("Patients=%d  long_rows=%d  treated=%d  composite_events=%d",
                 length(ids), nrow(df_long), sum(treat01), sum(outcome01)))

# Degenerate-cohort guard: with an empty arm, zero composite events, or n < min_num_patients
# (no candidate code can reach the carrier threshold) the Bross selection is undefined. Write a
# header-only hdps_covars.csv and exit cleanly; estimate_ladder's hdps spec then writes its
# non-analyzable stub, so the chain flows end-to-end instead of failing on a missing file.
if (sum(treat01) == 0 || sum(treat01) == length(ids) || sum(outcome01) == 0 ||
    length(ids) < opt$min_num_patients) {
  log_warning(sprintf(
    "NON-ANALYZABLE for hdPS selection (n=%d, treated=%d, composite_events=%d, min_num_patients=%d) -- writing header-only hdps_covars.csv and exiting cleanly",
    length(ids), sum(treat01), sum(outcome01), opt$min_num_patients))
  dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(data.frame(person_id = character(0)),
            file.path(opt$out_dir, "hdps_covars.csv"), row.names = FALSE)
  quit(save = "no", status = 0)
}

# ---- Step 2: candidate covariates (top-n per dimension) --------------------
step1 <- get_candidate_covariates(
  df = df_long, domainVarname = "domain", eventCodeVarname = "event_code",
  patientIdVarname = "person_id", patientIdVector = ids,
  n = opt$n, min_num_patients = opt$min_num_patients)
log_info(sprintf("Step2 candidate covariates: %d", length(unique(step1$covars_data$event_code))))

# ---- Step 3: recurrence tiers (sparse re-implementation; reproduces the library's output
#      exactly, used when the dense patient x candidate matrix would overflow) ----
get_recurrence_covariates_sparse <- function(df, patientIdVector) {
  dt <- data.table::data.table(pid = as.character(df$person_id),
                               code = as.character(df$event_code))
  cnt <- dt[, .N, by = .(pid, code)]
  pos <- seq_along(patientIdVector); names(pos) <- patientIdVector
  cnt[, pi := pos[pid]]
  cnt <- cnt[!is.na(pi)]
  codes <- sort(unique(dt$code))
  n <- length(patientIdVector)
  out <- list(person_id = patientIdVector)
  for (cv in codes) {
    sub <- cnt[code == cv]
    nz <- sub$N
    med <- as.integer(stats::median(nz))
    uq  <- as.integer(stats::quantile(nz, 0.75))
    v <- integer(n); v[sub$pi] <- sub$N
    out[[paste0("rec_", cv, "_once")]] <- as.integer(v >= 1L)
    if (med != 1L) out[[paste0("rec_", cv, "_sporadic")]]  <- as.integer(v >= med)
    if (uq  != med) out[[paste0("rec_", cv, "_frequent")]] <- as.integer(v >= uq)
  }
  list(recurrence_data = as.data.frame(out, stringsAsFactors = FALSE),
       patientIds = patientIdVector)
}

n_cand <- length(unique(step1$covars_data$event_code))
use_sparse <- as.numeric(length(ids)) * n_cand > 0.95 * .Machine$integer.max
log_info(sprintf("Step3 recurrence path: %s (%d patients x %d candidates)",
                 if (use_sparse) "SPARSE" else "dense (library)", length(ids), n_cand))
step2 <- if (use_sparse) {
  suppressWarnings(suppressMessages(library(data.table)))
  get_recurrence_covariates_sparse(step1$covars_data, ids)
} else {
  get_recurrence_covariates(
    df = step1$covars_data, patientIdVarname = "person_id",
    eventCodeVarname = "event_code", patientIdVector = ids)
}
log_info(sprintf("Step3 recurrence covariates: %d", ncol(step2$recurrence_data) - 1L))

# ---- Step 4: prioritise by |log(Bross multiplicative bias)|, keep top k ----
step3 <- get_prioritised_covariates(
  df = step2$recurrence_data, patientIdVarname = "person_id",
  exposureVector = treat01, outcomeVector = outcome01,
  patientIdVector = ids, k = opt$k)
hdps_covars <- step3$autoselected_covariate_df
log_info(sprintf("Step4 selected covariates: %d", ncol(hdps_covars) - 1))

# ---- outputs ---------------------------------------------------------------
dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(hdps_covars, file.path(opt$out_dir, "hdps_covars.csv"), row.names = FALSE)
if (!is.null(step3$multiplicative_bias)) {
  write.csv(step3$multiplicative_bias, file.path(opt$out_dir, "multiplicative_bias.csv"), row.names = FALSE)
}
log_info(sprintf("Wrote hdps_covars.csv (%d x %d). Log: %s",
                 nrow(hdps_covars), ncol(hdps_covars), log_file))
