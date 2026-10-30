#!/usr/bin/env Rscript
# =============================================================================
# estimate_ladder.R -- the reference paper's primary estimator (decile-adjusted), one spec per call.
#
# Holds cohort + outcome fixed; varies ONLY the PS covariate spec:
#   crude | demo | hdps | delphi-only | reclaim-<key>-only
#   (embedding specs are SUBSTITUTION arms: age+sex + rep_*; stacking hdps+<model> would blur
#   attribution between the two covariate blocks, so it is rejected below.)
# Estimator: PS deciles as indicator terms in the paper's own outcome model -- logistic OR for the
# positive control (their Examples 1-2), Cox HR for the negative control (Example 3) -- with a
# model-based Wald 95% CI, as the paper reported.
#
# Inputs : --cohort_csv, --hdps_covars (hdps spec), --embed_parquet (embedding specs), --design
# Output : --out_dir/estimate.json  (out_dir = results/mdcr/{cohort}/{cell}/{spec}/decile_adjusted)
# Sparse cells (< 10 exposed events) are flagged, not refit.
# =============================================================================

suppressWarnings(suppressMessages({
  library(optparse); library(dplyr); library(glmnet)
  library(survival)
  library(jsonlite); library(speedglm)
}))

# Fast binomial fit for the wide hdPS models: speedglm fits the SAME unpenalised model as glm()
# via BLAS (glm does not finish a ~500-covariate PS on large cells). Falls back to glm on error.
fast_binom <- function(formula, data) {
  tryCatch(speedglm::speedglm(formula, data = data, family = binomial()),
           error = function(e) glm(formula, family = binomial(), data = data))
}
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()
INDENT1 <- "  "; INDENT2 <- "    "; INDENT3 <- "      "
CV_FOLDS <- 5L   # one global CV setting (fit_dr_models uses the same)
# Deterministic uniform subsample of size <= m (same helper fit_dr_models uses for its lambda search).
stripe_idx <- function(n, m) if (n <= m) seq_len(n) else which(((seq_len(n) - 1L) %% ceiling(n / m)) == 0L)
LAMBDA_SUBSAMPLE <- 100000L

opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohort_csv", type = "character"),
  make_option("--hdps_covars", type = "character", default = NULL, help = "select_covariates hdps_covars.csv"),
  make_option("--embed_parquet", type = "character", default = NULL, help = "cohort_with_<model>_embeddings.parquet"),
  make_option("--spec", type = "character", default = "hdps"),
  make_option("--out_dir", type = "character"),
  make_option("--design", type = "character", default = "cohort_logistic",
              help = "cohort_logistic|landmark_cox -- the paper's outcome model: logistic on the new-user cohort design (positive control) or Cox on the landmark design (negative control)")
)))

# =============================================================================
# PS helpers
# =============================================================================
# PS deciles with de-duplicated breaks: heavily tied PS values (tiny cells, near-constant PS)
# collapse quantile breaks and make cut() abort; unique() keeps however many distinct strata
# the PS actually supports (>=1).
ps_deciles <- function(ps) {
  br <- unique(quantile(ps, 0:10 / 10, na.rm = TRUE))
  if (length(br) < 2) return(rep(1L, length(ps)))
  cut(ps, breaks = br, include.lowest = TRUE, labels = FALSE)
}

# =============================================================================
# load cohort + assemble the covariate set for the requested spec
# =============================================================================
log_info(sprintf("spec=%s estimator=decile_adjusted", opt$spec))
# Reject stacking specs explicitly: the coarse-family switch below would otherwise map
# 'hdps+delphi' onto plain hdps covariates and run it mislabelled.
if (grepl("^hdps\\+", opt$spec))
  stop(sprintf("spec '%s': hdps+<model> stacking arms are not part of this benchmark -- use the substitution spec (<model>-only) or plain hdps", opt$spec))
data <- read.csv(opt$cohort_csv)
data$treatment <- as.integer(data$arm)
data$person_id <- as.character(data$person_id)
if (!"cal_year" %in% names(data) && "index_date" %in% names(data))
  data$cal_year <- as.integer(format(as.Date(data$index_date), "%Y"))
demo_vars <- intersect(c("age_at_index", "sex", "cal_year", "region", "plan"), names(data))
# `payer` enters the PS only when it varies (a constant factor cannot be contrasted).
if ("payer" %in% names(data)) {
  data$payer <- as.factor(data$payer)
  if (nlevels(data$payer) >= 2) demo_vars <- c(demo_vars, "payer")
}
# Minimal demographics for the embedding SUBSTITUTION arms: age + sex only (see xvars switch).
emb_demo_vars <- intersect(c("age_at_index", "sex"), names(data))

merge_by_pid <- function(data, path, cols_prefix) {
  # Parquet preferred for the wide embedding tables: column-pruned read, no silent truncation.
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
    schema <- nanoparquet::read_parquet_schema(path)
    keep <- c("person_id", grep(cols_prefix, schema$name, value = TRUE))
    add <- as.data.frame(nanoparquet::read_parquet(path, col_select = keep))
  } else {
    add <- read.csv(path)
    keep <- c("person_id", grep(cols_prefix, names(add), value = TRUE))
    add <- add[, keep]
  }
  add$person_id <- as.character(add$person_id)
  # Covariate files that do not cover the cohort would merge as NAs -> 0 ("no signal" rather than
  # "missing"). Fail loudly instead.
  matched <- sum(data$person_id %in% add$person_id)
  if (matched < nrow(data)) {
    log_warning(sprintf("%s: only %d/%d cohort patients present in %s -- the remainder will be zero-filled",
                        cols_prefix, matched, nrow(data), basename(path)))
    if (matched < 0.99 * nrow(data))
      stop(sprintf("covariate file %s covers only %.2f%% of the cohort (%d/%d). Refusing to proceed: ",
                   basename(path), 100 * matched / nrow(data), matched, nrow(data)),
           "a zero-filled majority would produce a meaningless estimate.")
  }
  merge(data, add, by = "person_id", all.x = TRUE)
}

hdps_vars <- character(0); rep_vars <- character(0)
uses_hdps <- grepl("^hdps", opt$spec)
uses_embed <- grepl("delphi|reclaim", opt$spec)

if (uses_hdps && !is.null(opt$hdps_covars)) {
  # autoCovariateSelection names each selected covariate rec_<domain>_<code>_<once|sporadic|frequent>;
  # the only non-covariate column is person_id.
  data <- merge_by_pid(data, opt$hdps_covars, "^rec_")
  hdps_vars <- setdiff(grep("^rec_", names(data), value = TRUE), demo_vars)
  data[hdps_vars][is.na(data[hdps_vars])] <- 0
}
if (uses_embed && !is.null(opt$embed_parquet)) {
  data <- merge_by_pid(data, opt$embed_parquet, "^rep_")
  rep_vars <- grep("^rep_", names(data), value = TRUE)
  data[rep_vars][is.na(data[rep_vars])] <- 0
}
# An embedding spec without rep_* covariates would silently collapse to demo-only and mislabel
# the result; fail loudly.
if (uses_embed && length(rep_vars) == 0)
  stop(sprintf("spec '%s' requires embedding covariates but none loaded -- pass --embed_parquet (run b2_embeddings/b2_1_embeddings.sh first).", opt$spec))

xvars <- switch(
  sub("[-+].*$", "", opt$spec),  # coarse family
  # crude = the reference's unadjusted rung: no covariates, constant PS, one decile,
  # outcome ~ treatment -- what makes our ladder readable against theirs end to end.
  "crude"         = character(0),
  "demo"          = demo_vars,
  # hdps = demographics + the k empirical covariates (the paper's model 5b, and exactly its
  # Example 3 design).
  "hdps"          = c(demo_vars, hdps_vars),
  # Substitution arms carry only age+sex alongside rep_* -- the same demographics both models
  # read from their sequence header -- so everything beyond age/sex must come from the
  # embedding, which is the substitution question. Region/plan/payer/cal_year are deliberately
  # NOT added: ReClaim's tokens already encode them (double-counting), Delphi's cannot (and
  # hand-feeding them would mask that limitation instead of measuring it).
  "reclaim"       = c(emb_demo_vars, rep_vars),
  "delphi"        = c(emb_demo_vars, rep_vars),
  stop(sprintf("unknown spec family '%s'", opt$spec)))
xvars <- unique(xvars[xvars %in% names(data)])
log_info(sprintf("n=%d  exposed=%d  events=%d  #covariates=%d",
                 nrow(data), sum(data$treatment), sum(data$outcome), length(xvars)))
if (sum(data$outcome[data$treatment == 1]) < 10)
  log_warning("UNSTABLE cell: < 10 exposed events")

# -----------------------------------------------------------------------------
# Events-per-variable guard on the PROPENSITY model. The PS is a logistic fit of treatment on
# xvars, so its stability is governed by the SMALLER arm, not the outcome count: with k
# covariates and only m minority-arm members, k >= m makes separation essentially certain and
# the fitted scores collapse to 0/1 (a "significant" RR that is numerical breakdown). Graded:
#   epv_ps <  1  -> non-estimable; flagged so summarize_results keeps it out of the forest plots.
#   epv_ps < 10  -> below the conventional rule of thumb: warn, record, still report.
n_minority <- min(sum(data$treatment == 1), sum(data$treatment == 0))
epv_ps <- if (length(xvars) > 0) n_minority / length(xvars) else NA_real_
epv_outcome <- if (length(xvars) > 0) sum(data$outcome[data$treatment == 1]) / length(xvars) else NA_real_
overparameterized <- isTRUE(is.finite(epv_ps) && epv_ps < 1)
if (overparameterized) {
  log_warning(sprintf(paste("OVER-PARAMETERISED PS: %d covariates vs %d in the smaller arm",
                            "(EPV=%.2f) -- marking NON-ESTIMABLE"),
                      length(xvars), n_minority, epv_ps))
} else if (isTRUE(is.finite(epv_ps) && epv_ps < 10)) {
  log_warning(sprintf("LOW EPV on the PS model: %d covariates vs %d in the smaller arm (EPV=%.2f, rule of thumb >=10)",
                      length(xvars), n_minority, epv_ps))
}

# =============================================================================
# propensity score
# =============================================================================
# Zero-covariate case (spec = crude): the PS is the marginal treatment prevalence, identical
# for everyone; ps_deciles collapses to one level and the decile-adjusted model reduces to
# outcome ~ treatment. c-statistic is undefined and left NA.
f_ps <- if (length(xvars)) as.formula(paste0("treatment ~ ", paste0(xvars, collapse = " + "))) else NULL

# PS ENGINE RULE:
#   * REFERENCE-PROTOCOL specs (crude/demo/hdps) -> UNPENALISED multivariate logistic, exactly as
#     Schneeweiss 2009 specifies. Replicating the method means replicating its estimator.
#   * EMBEDDING specs (delphi / reclaim-*) -> RIDGE. These arms are our addition, so no
#     replication constraint applies, and a 1-2k-dim dense representation is the wrong object
#     for an unpenalised fit (EPV can approach or cross 1). Ridge shrinks the whole
#     representation rather than selecting a sparse subset of correlated dimensions, which is
#     what an embedding is.
ps_glm <- if (is.null(f_ps) || uses_embed) NULL else fast_binom(f_ps, data)
if (uses_embed) {
  X_ps <- model.matrix(f_ps, data)[, -1, drop = FALSE]
  sub <- stripe_idx(nrow(X_ps), LAMBDA_SUBSAMPLE)   # deterministic striped subsample for the lambda search (see utils.R)
  # Lambda is selected ONCE on a deterministic stripe subsample (<=100k rows), then the model is
  # fit at that lambda on the full cohort (a full cv.glmnet path over millions of rows at 2k dims
  # is not tractable). Folds are person_id rank modulo k: seed-invariant and interleaved rather
  # than blocked (ENROLIDs grow over calendar time, so blocks would confound folds with era).
  .fold <- ((match(data$person_id[sub], sort(data$person_id[sub])) - 1L) %% CV_FOLDS) + 1L
  cvfit <- glmnet::cv.glmnet(X_ps[sub, , drop = FALSE], data$treatment[sub],
                             family = "binomial", alpha = 0, foldid = .fold)
  lam <- cvfit$lambda.min
  fit_ps <- glmnet::glmnet(X_ps, data$treatment, family = "binomial", alpha = 0,
                           lambda = lam * c(4, 2, 1))   # short warm-start path, keep last
  data$propensityScore <- as.numeric(predict(fit_ps, newx = X_ps, s = lam, type = "response"))
  log_info(sprintf("%sRIDGE PS (embedding spec): %d covariates, lambda=%.6g (selected on %s of %s rows)",
                   INDENT2, ncol(X_ps), lam,
                   format(length(sub), big.mark = ","), format(nrow(X_ps), big.mark = ",")))
}
# CRUDE fallback: only when there are genuinely NO covariates. Condition on f_ps, NOT ps_glm --
# ps_glm is also NULL for the ridge embedding specs, and testing it would silently overwrite
# every ridge propensity score with the constant crude one.
if (is.null(f_ps)) {
  data$propensityScore <- rep(mean(data$treatment), nrow(data))
  log_info(sprintf("%sspec=crude: no covariates -- constant PS = %.4f (unadjusted contrast)",
                   INDENT2, mean(data$treatment)))
}
# speedglm's predict needs newdata explicitly (it does not retain fitted.values); glm ignores it.
if (!is.null(ps_glm)) data$propensityScore <- tryCatch({
  p <- as.numeric(predict(ps_glm, newdata = data, type = "response"))
  if (!all(is.finite(p))) stop("non-finite PS predictions")
  p
}, error = function(err) {
  # Complete separation (small cohort, wide hdPS design): refit as ridge logistic -- keeps
  # every covariate in the PS with shrunk coefficients instead of dropping the cell.
  log_warning(sprintf("unpenalised PS fit unusable (%s) -- ridge (glmnet alpha=0) fallback",
                      conditionMessage(err)))
  X <- model.matrix(f_ps, data)[, -1, drop = FALSE]
  .fold <- ((match(data$person_id, sort(data$person_id)) - 1L) %% 5L) + 1L
  cvfit <- glmnet::cv.glmnet(X, data$treatment, family = "binomial", alpha = 0, foldid = .fold)
  as.numeric(predict(cvfit, newx = X, s = "lambda.min", type = "response"))
})
# A constant score cannot discriminate; report NA rather than a meaningless 0.5.
c_stat <- if (length(unique(data$propensityScore)) < 2) NA_real_ else
  tryCatch(as.numeric(pROC::auc(data$treatment, data$propensityScore)), error = function(e) NA)
e <- pmin(pmax(data$propensityScore, 1e-6), 1 - 1e-6)

# =============================================================================
# The reference paper's outcome model
# =============================================================================
# Schneeweiss fits ONE outcome model per example cohort -- logistic for Examples 1-2 (our
# positive control), Cox for Example 3 (our negative control); the PS enters as deciles
# ("adjusted ... for deciles of the estimated PS"). SE: model-based, as the paper reported.
fit_paper_model <- function(df, outcome_col, kind, adjust = NULL) {
  rhs <- paste(c("treatment", adjust), collapse = " + ")
  if (identical(kind, "cohort_logistic")) {
    f <- as.formula(paste0(outcome_col, " ~ ", rhs))
    m <- tryCatch(suppressWarnings(glm(f, family = binomial(), data = df)),
         error = function(e) NULL)
    if (is.null(m)) return(c(logRR = NA, seLogRR = NA))
    cf <- summary(m)$coefficients
    return(c(logRR = cf["treatment", 1], seLogRR = cf["treatment", 2]))
  }
  # Negative control (landmark Cox): follow-up censored at disenrollment via person_time (build_cohort),
  # matching the paper's Cohort-3 design ("censored after the occurrence of the study end
  # point, death, or disenrollment").
  df$.tt <- if ("person_time" %in% names(df)) pmax(as.numeric(df$person_time), 1) else df$followup_days
  f <- as.formula(paste0("Surv(.tt, ", outcome_col, ") ~ ", rhs))
  m <- tryCatch(suppressWarnings(coxph(f, data = df)), error = function(e) NULL)
  if (is.null(m)) return(c(logRR = NA, seLogRR = NA))
  cf <- summary(m)$coefficients
  c(logRR = cf["treatment", "coef"], seLogRR = cf["treatment", "se(coef)"])
}

estimate_one <- function(outcome_col) {
  # A constant PS (spec=crude) collapses to a single decile; factor() on one level is an
  # aliased column glm would drop anyway, so it is omitted and the crude rung is literally
  # outcome ~ treatment. Deciles containing only one arm are NOT dropped -- the paper adjusted
  # for deciles without exclusions, so a replication must too; sparse cells lean on the flags
  # (<10 exposed events, EPV<1, c-stat>=0.99) instead.
  data$decile <- ps_deciles(data$propensityScore)
  adj <- if (length(unique(data$decile)) > 1) "factor(decile)" else NULL
  fit_paper_model(data, outcome_col, opt$design, adjust = adj)
}

# =============================================================================
# estimate + write results
# =============================================================================
if (isTRUE(is.finite(c_stat) && c_stat >= 0.99))
  log_warning(sprintf("NON-ESTIMABLE: PS c-statistic=%.4f (>=0.99) with %d covariates on n=%d -- the arms are perfectly separated; this RR is numerical breakdown, not an estimate.",
                      c_stat, length(xvars), nrow(data)))
# --out_dir is the leaf directory (.../{spec}/decile_adjusted/ -- summarize_results/fit_dr_models walk that layout).
est <- "decile_adjusted"
out_dir <- opt$out_dir
primary <- estimate_one("outcome")
log_info(sprintf("%s%-15s logRR=%.3f se=%.3f  RR=%.3f", INDENT2, est,
                 primary["logRR"], primary["seLogRR"], exp(primary["logRR"])))
rr_lb <- unname(exp(primary["logRR"] - 1.96 * primary["seLogRR"]))
rr_ub <- unname(exp(primary["logRR"] + 1.96 * primary["seLogRR"]))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
res <- list(
  spec = opt$spec, estimator = est,
  n = nrow(data), n_exposed = sum(data$treatment), events = sum(data$outcome),
  n_exposed_events = sum(data$outcome[data$treatment == 1]),
  unstable_flag = sum(data$outcome[data$treatment == 1]) < 10,
  n_covariates = length(xvars), c_statistic = c_stat,
  ps_engine = if (uses_embed) "ridge" else "logistic", cv_folds = CV_FOLDS,
  # separation_flag: c-stat ~1 means the PS separates the arms perfectly -- the RR is
  # numerical breakdown. overparameterized_flag is its pre-fit companion (epv_ps < 1).
  # summarize_results treats either flag as non-estimable (kept in the supplementary table, out of the plots).
  separation_flag = isTRUE(is.finite(c_stat) && c_stat >= 0.99),
  epv_ps = epv_ps, epv_outcome = epv_outcome,
  overparameterized_flag = overparameterized,
  rr = unname(exp(primary["logRR"])), rr_lb = rr_lb, rr_ub = rr_ub,
  logRR = unname(primary["logRR"]), seLogRR = unname(primary["seLogRR"]))
write_json(res, file.path(out_dir, "estimate.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
log_info(sprintf("Wrote estimate.json -> %s. Log: %s", out_dir, log_file))
