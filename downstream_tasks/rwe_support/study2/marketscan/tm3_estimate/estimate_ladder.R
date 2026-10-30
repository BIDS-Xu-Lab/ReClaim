#!/usr/bin/env Rscript
# =============================================================================
# estimate_ladder.R -- PS-spec x estimator grid for the GLP-1 TTE replication.
#
# The reference protocol (Tang et al. 2026) supplies the estimator panel; the PS covariate
# SPEC is the experimental axis (crude | demo | paper | hdps | <key>-only). ONE PS fit per
# spec; three estimators are functions of it:
#   siptw_cox (PRIMARY) : stabilized IPTW (p1/p99 truncated) -> weighted Cox, robust SE
#                         (the reference's primary estimator; estimand ATE)
#   psm_cox             : 1:1 nearest-neighbour, caliper 0.2 SD, seed 1 -> Cox in matched
#                         pairs, pair-clustered SE (the reference's matching analysis; ATT)
#   strat_cox           : Cox stratified on PS deciles (conditional estimand)
# Estimands differ legitimately: the cross-estimator claim is about direction and
# null-coverage, not point identity. Each estimator gets its own balance diagnostics and
# NCO -> fitNull -> EASE -> calibrated CI/p pass; Schoenfeld PH checks on every Cox fit.
# PS engines: plain logistic for the reference specs (as the reference specifies); RIDGE for
# the embedding specs, where an unpenalised fit cannot be estimated (EPV<2 at 1-2k rep_*
# dims) and L2 suits a dense correlated representation (L1 would arbitrarily keep one member
# of each correlated group). CV uses deterministic folds, k = CV_FOLDS.
#
# Output: <out_dir>/<estimator>/estimate.json (+ balance/nco/km CSVs per estimator).
# Usage : Rscript tm3_estimate/estimate_ladder.R --cohort_csv <csv> --comparison glp1_vs_sglt2
#           --treat_class glp1ra --comp_class sglt2i --spec paper --out_dir results/<comparison>/paper
# =============================================================================

suppressWarnings(suppressMessages({
  library(optparse); library(dplyr); library(glmnet)
  library(sandwich); library(lmtest); library(survival)
  library(jsonlite); library(speedglm); library(MatchIt)
  library(EmpiricalCalibration)
}))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()
INDENT2 <- "    "

opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohort_csv", type = "character"),
  make_option("--hdps_covars", type = "character", default = NULL),
  make_option("--embed_parquet", type = "character", default = NULL,
              help = "embed_reclaim cohort_with_<key>_embeddings.parquet (embedding specs only)"),
  make_option("--comparison", type = "character", default = "glp1_vs_sglt2"),
  make_option("--treat_class", type = "character", default = "glp1ra"),
  make_option("--comp_class", type = "character", default = "sglt2i"),
  make_option("--spec", type = "character", default = "paper",
              help = paste("crude|demo|paper|hdps  or an embedding spec:",
                           "<model>-only (substitution: age+sex+rep_*)")),
  make_option("--estimators", type = "character",
              default = "siptw_cox,psm_cox,strat_cox",
              help = "comma-separated subset of siptw_cox|psm_cox|strat_cox"),
  # 0.001 = the reference's rule (Table S4 footnote: follow-up risk < 0.1% excluded)
  make_option("--nco_min_prevalence", type = "double", default = 0.001),
  make_option("--caliper_sweep", type = "character", default = "",
              help = "comma-separated calipers, e.g. '0.05,0.1,0.2,0.5,1'; empty = off"),
  # NCOs entering the calibration null: default = the reference's reported panel (Table S4,
  # 24 outcomes) with the >=0.1% risk rule on top; "all" = the full master list.
  make_option("--nco_set", type = "character", default = "lookup_table/glp_nco_reference24.txt",
              help = "path to a one-name-per-line NCO subset file, or 'all'"),
  make_option("--out_dir", type = "character",
              help = "SPEC directory; each estimator writes <out_dir>/<estimator>/")
)))

fast_binom <- function(formula, data) {
  tryCatch(speedglm::speedglm(formula, data = data, family = binomial()),
           error = function(e) glm(formula, family = binomial(), data = data))
}

ps_deciles <- function(ps) {
  br <- unique(quantile(ps, 0:10 / 10, na.rm = TRUE))
  if (length(br) < 2) return(rep(1L, length(ps)))
  cut(ps, breaks = br, include.lowest = TRUE, labels = FALSE)
}

# DETERMINISTIC CV folds: fold = rank of person_id modulo k, so lambda selection is
# seed-invariant. Interleaving (not blocking) matters: ENROLIDs grow over calendar time, so
# contiguous blocks would confound folds with enrollment era; ids are administrative, so
# the stripes are quasi-random with respect to covariates and outcomes.
det_foldid <- function(ids, k) ((match(ids, sort(ids)) - 1L) %% k) + 1L

# One global CV setting: the CV curve is flat near its minimum at this n, so more folds add
# cost without inferential gain.
CV_FOLDS <- 5L

# =============================================================================
# load cohort, subset to the comparison's two arms
# =============================================================================
log_info(sprintf("comparison=%s spec=%s (%s vs %s)  estimators=%s",
                 opt$comparison, opt$spec, opt$treat_class, opt$comp_class, opt$estimators))
data <- read.csv(opt$cohort_csv)
data$person_id <- as.character(data$person_id)
data <- data[data$drug_class %in% c(opt$treat_class, opt$comp_class), ]
data$treatment <- as.integer(data$drug_class == opt$treat_class)
# NON-ANALYZABLE guard: an empty arm makes every estimator undefined -> flagged stub.
if (sum(data$treatment == 1) == 0 || sum(data$treatment == 0) == 0) {
  dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(
    list(comparison = opt$comparison, spec = opt$spec, non_analyzable = TRUE,
         n = nrow(data), n_treat = sum(data$treatment), n_comp = sum(data$treatment == 0)),
    file.path(opt$out_dir, "estimate.json"), auto_unbox = TRUE, pretty = TRUE)
  log_warning(sprintf("NON-ANALYZABLE: empty arm (n=%d, treated=%d) -- stub written, exiting cleanly",
                      nrow(data), sum(data$treatment)))
  quit(save = "no", status = 0)
}

data$sex <- factor(data$sex)
if ("region" %in% names(data)) {
  data$region <- as.character(data$region); data$region[is.na(data$region) | data$region == ""] <- "UNK"
  data$region <- factor(data$region)
}
data$cal_year <- factor(data$cal_year)
if ("payer" %in% names(data)) data$payer <- factor(data$payer)

# `demo` = the minimal rung, defined identically in every data source: age, sex, calendar
# year. region (claims-only) and payer (the database label) are excluded; the reference's
# insurance covariate is `ins_category`, which enters the `paper` spec where Table S5 lists it.
demo_vars <- intersect(c("age_at_index", "sex", "cal_year"), names(data))

outcome_names <- sub("_event$", "", grep("^out_.*_event$", names(data), value = TRUE))
nco_names_all <- sub("_event$", "", grep("^nco_.*_event$", names(data), value = TRUE))
# Restrict to the reference's NCO panel unless --nco_set all. A listed NCO that does not
# resolve to a cohort column is a hard ERROR: a silently dropped negative control would
# shrink the calibration null without a trace.
if (!identical(opt$nco_set, "all")) {
  if (!file.exists(opt$nco_set))
    stop(sprintf("--nco_set file not found: %s (use 'all' for the full master list)", opt$nco_set))
  want <- trimws(readLines(opt$nco_set, warn = FALSE)); want <- want[nzchar(want)]
  .norm <- function(v) gsub("[^a-z0-9]", "", tolower(v))
  hit <- match(.norm(want), .norm(sub("^nco_", "", nco_names_all)))
  if (anyNA(hit))
    stop(sprintf("%d NCO(s) in %s have no cohort column: %s", sum(is.na(hit)), opt$nco_set,
                 paste(want[is.na(hit)], collapse = ", ")))
  nco_names_all <- nco_names_all[hit]
  log_info(sprintf("NCO panel restricted to %s: %d of %d master NCOs",
                   basename(opt$nco_set), length(nco_names_all),
                   length(grep("^nco_.*_event$", names(data)))))
}

bin_vars <- grep("^(cov_|med_)", names(data), value = TRUE)
prev <- vapply(data[bin_vars], function(x) mean(x, na.rm = TRUE), numeric(1))
ut_vars <- grep("^ut_", names(data), value = TRUE)

# =============================================================================
# paper: the reference's own covariate list (Supplement Table S5), strictly: a covariate
# that cannot be built the same way in claims is DROPPED, never substituted. Omitted:
# race/ethnicity, labs/vitals (HbA1c/BMI/SBP/DBP/eGFR), ED visits (no native ED flag in
# MarketScan), the Charlson index (coding and weights unstated; every constituent already
# enters individually), and baseline study-class use (the two compared classes are
# structurally zero under the washout; third-class use is not carried). Insurance enters as
# `ins_category`, guarded on >=2 levels. The result is a strict subset of their list.
# =============================================================================
S5_CLUSTERS <- c(
  # -- 23 exact PennCIL cluster-name matches
  "acute_myocardial_infarction", "headache", "menstrual_disorders", "arrhythmias", "asthma",
  "biliary_tract_disease", "bronchitis", "cardiomyopathy", "cerebrovascular_disease",
  "chronic_kidney_disease", "covid19", "cognitive_impairment", "fractures", "glaucoma",
  "hypertension", "obesity", "osteoarthritis", "osteoporosis", "pneumonia", "skin_cancers",
  "thyroid_disorders", "malnutrition", "benign_neoplasms",
  # -- 6 renamed cluster matches
  "peripheral_and_visceral_vascular_disease",              # Peripheral vascular disease
  "chronic_obstructive_pulmonary_disease_and_bronchiectasis",  # COPD
  "coronary_atherosclerosis_and_other_heart_disease",      # Coronary artery disease
  "disorders_of_lipid_metabolism",                         # Lipid disorders
  "pancreatic_disorders_excluding_diabetes",               # Pancreatic disorders
  "seizures_and_epilepsy",                                 # Seizures/epilepsy
  # -- 5 diabetes-severity covariates defined on the E11 tree in build_cohort
  "dm_nephropathy", "dm_retinopathy", "dm_neuropathy",
  "dm_other_specified_complications", "dm_unspecified_complications")
# Substance use disorders = union of the substance-specific clusters, tobacco EXCLUDED (it is
# a separate cluster, and an ~18%-prevalent smoking flag would redefine this ~3% covariate).
S5_SUD_PARTS <- c("alcohol_related_disorders", "opioid_related_disorders",
                  "cannabis_related_disorders", "sedative_related_disorders",
                  "stimulant_related_disorders", "inhalant_related_disorders",
                  "other_specified_substance_related_disorders")
S5_MEDS <- c("arb", "ccb", "diuretics", "lipid", "corticosteroids", "nsaids", "anticoagulants",
             "antiplatelet_agents", "proton_pump_inhibitors", "opioids", "other_aom",
             "hormone_replacement_therapy", "insulin", "metformin", "sulfonylureas",
             "thiazolidinediones", "a_glucosidase_inhibitors")

build_s5_vars <- function(data) {
  have <- names(data)
  cl <- paste0("cov_", S5_CLUSTERS); miss_cl <- setdiff(cl, have)
  md <- paste0("med_", S5_MEDS);     miss_md <- setdiff(md, have)
  if (length(miss_cl) || length(miss_md))
    stop(sprintf("paper_s5: %d covariate column(s) absent from the cohort: %s",
                 length(miss_cl) + length(miss_md),
                 paste(c(miss_cl, miss_md), collapse = ", ")))
  sud <- paste0("cov_", S5_SUD_PARTS); sud <- sud[sud %in% have]
  if (length(sud) == 0) stop("paper_s5: no substance-use cluster columns found")
  data$cov_substance_use_disorders <- as.integer(rowSums(data[, sud, drop = FALSE] > 0) > 0)
  # Utilization as Table S5 parameterises it: inpatient >=1; outpatient 0 / 1-20 / >20
  data$ut_ip_any <- as.integer(data$ut_n_ip_admits >= 1)
  data$ut_op_cat <- cut(data$ut_n_op_days, breaks = c(-Inf, 0, 20, Inf),
                        labels = c("0", "1-20", ">20"))
  # Insurance (Table S5): a constant factor would abort model.matrix -> dropped, logged
  ins <- character(0)
  if ("ins_category" %in% have) {
    if (length(unique(na.omit(data$ins_category))) >= 2) {
      data$ins_category <- factor(data$ins_category)
      ins <- "ins_category"
      log_info(sprintf("    paper_s5: insurance covariate ACTIVE (%s)",
                       paste(levels(data$ins_category), collapse = "/")))
    } else {
      log_warning("paper_s5: ins_category has <2 levels in this cohort -- covariate DROPPED")
    }
  } else {
    log_warning("paper_s5: no ins_category column in the cohort file -- insurance covariate ABSENT")
  }
  list(data = data,
       vars = c(intersect(c("age_at_index", "sex", "cal_year"), have), ins,
                cl, "cov_substance_use_disorders", md, "ut_ip_any", "ut_op_cat"))
}

s5_vars <- character(0)
if (opt$spec == "paper") {
  .s5 <- build_s5_vars(data); data <- .s5$data; s5_vars <- .s5$vars
}

# ---- embedding specs: <model>-only (substitution) ----
uses_embed <- grepl("-only$", opt$spec)
rep_vars <- character(0)
if (uses_embed) {
  if (is.null(opt$embed_parquet)) stop(sprintf("spec '%s' requires --embed_parquet", opt$spec))
  schema <- nanoparquet::read_parquet_schema(opt$embed_parquet)
  keep <- c("person_id", grep("^rep_", schema$name, value = TRUE))
  add <- as.data.frame(nanoparquet::read_parquet(opt$embed_parquet, col_select = keep))
  add$person_id <- as.character(add$person_id)
  matched <- sum(data$person_id %in% add$person_id)
  if (matched < nrow(data))
    stop(sprintf("embedding file covers only %d/%d of the comparison cohort -- refusing to zero-fill",
                 matched, nrow(data)))
  data <- merge(data, add, by = "person_id", all.x = TRUE)
  rep_vars <- grep("^rep_", names(data), value = TRUE)
  log_info(sprintf("embedding covariates: %d rep_* dims from %s",
                   length(rep_vars), basename(opt$embed_parquet)))
}
emb_demo_vars <- intersect(c("age_at_index", "sex"), names(data))

hdps_vars <- character(0)
if (opt$spec == "hdps") {
  if (is.null(opt$hdps_covars)) stop("spec=hdps requires --hdps_covars")
  add <- read.csv(opt$hdps_covars)
  # header-only file = select_covariates' degenerate-cohort guard fired -> standard stub
  if (nrow(add) == 0 || !length(grep("^rec_", names(add)))) {
    dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
    jsonlite::write_json(
      list(comparison = opt$comparison, spec = opt$spec, non_analyzable = TRUE,
           n = nrow(data), n_treat = sum(data$treatment), n_comp = sum(data$treatment == 0),
           reason = "hdPS selection non-analyzable (header-only covariate file from select_covariates guard)"),
      file.path(opt$out_dir, "estimate.json"), auto_unbox = TRUE, pretty = TRUE)
    log_warning("NON-ANALYZABLE: hdPS covariate file is header-only (select_covariates guard) -- stub written, exiting cleanly")
    quit(save = "no", status = 0)
  }
  add$person_id <- as.character(add$person_id)
  keep <- c("person_id", grep("^rec_", names(add), value = TRUE))
  add <- add[, keep]
  matched <- sum(data$person_id %in% add$person_id)
  if (matched < 0.99 * nrow(data))
    stop(sprintf("hdps covariate file covers only %d/%d of the comparison cohort", matched, nrow(data)))
  data <- merge(data, add, by = "person_id", all.x = TRUE)
  hdps_vars <- grep("^rec_", names(data), value = TRUE)
  data[hdps_vars][is.na(data[hdps_vars])] <- 0
  hv <- vapply(data[hdps_vars], function(x) var(as.numeric(x)) > 0, logical(1))
  hdps_vars <- hdps_vars[hv]
}

xvars <- if (grepl("-only$", opt$spec)) {
  c(emb_demo_vars, rep_vars)                          # SUBSTITUTION
} else switch(opt$spec,
  "crude" = character(0),
  "demo"  = demo_vars,
  "paper" = s5_vars,
  "hdps"  = c(demo_vars, hdps_vars),
  stop(sprintf("unknown spec '%s'", opt$spec)))
xvars <- unique(xvars[xvars %in% names(data)])
log_info(sprintf("n=%d  treated=%d  #covariates=%d", nrow(data), sum(data$treatment), length(xvars)))

n_minority <- min(sum(data$treatment == 1), sum(data$treatment == 0))
epv_ps <- if (length(xvars) > 0) n_minority / length(xvars) else NA_real_
if (isTRUE(is.finite(epv_ps) && epv_ps < 10) && !uses_embed)
  log_warning(sprintf("LOW EPV on the PS model: %d covariates vs %d in the smaller arm (EPV=%.2f)",
                      length(xvars), n_minority, epv_ps))

# =============================================================================
# ONE propensity score per spec (logistic / ridge), shared by every estimator
# =============================================================================
f_ps <- if (length(xvars)) as.formula(paste0("treatment ~ ", paste0(xvars, collapse = " + "))) else NULL
ps_glm <- NULL
if (!is.null(f_ps) && uses_embed) {
  X_ps <- model.matrix(f_ps, data)[, -1, drop = FALSE]
  cvfit <- glmnet::cv.glmnet(X_ps, data$treatment, family = "binomial", alpha = 0,
                             foldid = det_foldid(data$person_id, CV_FOLDS))
  data$ps <- as.numeric(predict(cvfit, newx = X_ps, s = "lambda.min", type = "response"))
  log_info(sprintf("%sRIDGE PS (embedding spec): %d covariates, all shrunk (lambda.min=%.6g)",
                   INDENT2, ncol(X_ps), cvfit$lambda.min))
} else if (!is.null(f_ps)) {
  ps_glm <- fast_binom(f_ps, data)
}
if (is.null(f_ps)) {
  data$ps <- rep(mean(data$treatment), nrow(data))
  log_info(sprintf("%sspec=crude: constant PS = %.4f (unadjusted contrast)", INDENT2, mean(data$treatment)))
} else if (!is.null(ps_glm)) {
  data$ps <- tryCatch({
    p <- as.numeric(predict(ps_glm, newdata = data, type = "response"))
    if (!all(is.finite(p))) stop("non-finite PS predictions")
    p
  }, error = function(err) {
    log_warning(sprintf("unpenalised PS fit unusable (%s) -- ridge (glmnet alpha=0) fallback",
                        conditionMessage(err)))
    X <- model.matrix(f_ps, data)[, -1, drop = FALSE]
    cvfit <- glmnet::cv.glmnet(X, data$treatment, family = "binomial", alpha = 0,
                               foldid = det_foldid(data$person_id, CV_FOLDS))
    as.numeric(predict(cvfit, newx = X, s = "lambda.min", type = "response"))
  })
}
c_stat <- if (length(unique(data$ps)) < 2) NA_real_ else
  tryCatch(as.numeric(pROC::auc(data$treatment, data$ps, quiet = TRUE)), error = function(e) NA)
e <- pmin(pmax(data$ps, 1e-6), 1 - 1e-6)
log_info(sprintf("PS c-statistic = %s", ifelse(is.na(c_stat), "NA", sprintf("%.4f", c_stat))))

# stabilized IPTW (ATE), truncated at p1/p99 (Xu et al., the reference's sIPTW citation)
pt <- mean(data$treatment)
w_raw <- ifelse(data$treatment == 1, pt / e, (1 - pt) / (1 - e))
w <- pmin(pmax(w_raw, quantile(w_raw, 0.01)), quantile(w_raw, 0.99))
ess <- function(wv) sum(wv)^2 / sum(wv^2)
log_info(sprintf("%ssIPTW: raw max=%.2f -> truncated max=%.2f  ESS treated=%.0f/%d comparator=%.0f/%d",
                 INDENT2, max(w_raw), max(w),
                 ess(w[data$treatment == 1]), sum(data$treatment == 1),
                 ess(w[data$treatment == 0]), sum(data$treatment == 0)))

# clinical equipoise (preference score in [0.3, 0.7]) -- a property of the PS, shared
odds_e <- e / (1 - e); odds_p <- pt / (1 - pt)
pref <- (odds_e / odds_p) / (odds_e / odds_p + 1)
equipoise <- mean(pref >= 0.3 & pref <= 0.7)

data$decile <- ps_deciles(data$ps)

# =============================================================================
# shared machinery
# =============================================================================
wmean <- function(x, wv) sum(wv * x) / sum(wv)

smd_table_for <- function(df, wv = NULL) {
  # SMDs between arms of `df` (wv = NULL: unweighted). Denominator = the full 2-arm cohort's
  # unweighted SD, so before/after and across-estimator SMDs share a scale.
  if (is.null(f_ps)) return(NULL)
  Xa <- model.matrix(f_ps, data)[, -1, drop = FALSE]
  t1a <- data$treatment == 1
  sd_pool <- sqrt((apply(Xa[t1a, , drop = FALSE], 2, var) +
                   apply(Xa[!t1a, , drop = FALSE], 2, var)) / 2)
  Xd <- model.matrix(f_ps, df)[, -1, drop = FALSE]
  t1 <- df$treatment == 1
  if (is.null(wv)) wv <- rep(1, nrow(df))
  m1 <- apply(Xd[t1, , drop = FALSE], 2, wmean, wv = wv[t1])
  m0 <- apply(Xd[!t1, , drop = FALSE], 2, wmean, wv = wv[!t1])
  tb <- data.frame(covariate = colnames(Xd), smd = as.numeric((m1 - m0) / sd_pool))
  tb[is.finite(tb$smd), ]
}

cox_one <- function(df, wv, prefix, cluster_col = NULL, strata_col = NULL) {
  # one Cox fit -> logHR/se/Schoenfeld p. ZERO-EVENT-ARM GUARD: with no events in one arm the
  # partial likelihood is monotone and the HR diverges -> non-estimable NA.
  tt <- df[[paste0(prefix, "_time")]]; ev <- df[[paste0(prefix, "_event")]]
  if (sum(ev[df$treatment == 1]) == 0 || sum(ev[df$treatment == 0]) == 0)
    return(list(logHR = NA, se = NA, schoenfeld_p = NA))
  dd <- data.frame(tt = tt, ev = ev, treatment = df$treatment)
  rhs <- "treatment"
  if (!is.null(strata_col)) { dd$st <- df[[strata_col]]; rhs <- paste0(rhs, " + strata(st)") }
  if (!is.null(cluster_col)) { dd$cl <- df[[cluster_col]]; rhs <- paste0(rhs, " + cluster(cl)") }
  f <- as.formula(paste0("Surv(tt, ev) ~ ", rhs))
  fit <- tryCatch({
    if (!is.null(wv)) coxph(f, data = dd, weights = wv, robust = TRUE)
    else if (!is.null(cluster_col)) coxph(f, data = dd, robust = TRUE)
    else coxph(f, data = dd)
  }, error = function(e2) NULL)
  if (is.null(fit)) return(list(logHR = NA, se = NA, schoenfeld_p = NA))
  cf <- summary(fit)$coefficients
  se <- if ("robust se" %in% colnames(cf)) cf["treatment", "robust se"] else cf["treatment", "se(coef)"]
  zph_p <- tryCatch(cox.zph(fit)$table["treatment", "p"], error = function(e2) NA_real_)
  list(logHR = unname(cf["treatment", "coef"]), se = unname(se), schoenfeld_p = unname(zph_p))
}

ir_by_arm <- function(df, wv, prefix) {
  tt <- df[[paste0(prefix, "_time")]]; ev <- df[[paste0(prefix, "_event")]]
  out <- list()
  for (a in c(1, 0)) {
    m <- df$treatment == a
    py <- sum(wv[m] * tt[m]) / 365.25
    out[[ifelse(a == 1, "treat", "comp")]] <-
      list(events = unname(sum(ev[m])), w_events = unname(sum(wv[m] * ev[m])),
           ir_1000py = unname(1000 * sum(wv[m] * ev[m]) / py))
  }
  out
}

nco_estimates <- function(df, wv, cluster_col = NULL, strata_col = NULL) {
  prev_nco <- vapply(nco_names_all, function(p) mean(df[[paste0(p, "_event")]]), numeric(1))
  keep <- nco_names_all[prev_nco >= opt$nco_min_prevalence]
  rows <- lapply(keep, function(p) {
    r <- cox_one(df, wv, p, cluster_col, strata_col)
    data.frame(nco = p, logHR = r$logHR, seLogHR = r$se, prevalence = mean(df[[paste0(p, "_event")]]))
  })
  tb <- do.call(rbind, rows)
  tb[is.finite(tb$logHR) & is.finite(tb$seLogHR) & tb$seLogHR > 0, ]
}

fit_null_safe <- function(nco_tb) {
  if (is.null(nco_tb) || nrow(nco_tb) < 5) return(NULL)
  tryCatch(fitNull(nco_tb$logHR, nco_tb$seLogHR), error = function(e2) NULL)
}

calibrate_outcomes <- function(rows, null_fit) {
  rows$hr_cal <- NA; rows$lb_cal <- NA; rows$ub_cal <- NA; rows$p_cal <- NA
  if (is.null(null_fit)) return(rows)
  err_model <- tryCatch(convertNullToErrorModel(null_fit), error = function(e2) NULL)
  for (i in seq_len(nrow(rows))) {
    if (!is.finite(rows$logHR[i]) || !is.finite(rows$se[i])) next
    if (!is.null(err_model)) {
      ci <- tryCatch(calibrateConfidenceInterval(rows$logHR[i], rows$se[i], err_model),
                     error = function(e2) NULL)
      if (!is.null(ci)) {
        rows$hr_cal[i] <- exp(ci$logRr); rows$lb_cal[i] <- exp(ci$logLb95Rr); rows$ub_cal[i] <- exp(ci$logUb95Rr)
      }
    }
    p <- tryCatch(calibrateP(null_fit, rows$logHR[i], rows$se[i]), error = function(e2) NA)
    rows$p_cal[i] <- suppressWarnings(as.numeric(p))
  }
  rows
}

outcome_block <- function(df, wv, null_fit, cluster_col = NULL, strata_col = NULL, km_path = NULL) {
  rows <- lapply(outcome_names, function(p) {
    r <- cox_one(df, wv, p, cluster_col, strata_col)
    ir <- ir_by_arm(df, if (is.null(wv)) rep(1, nrow(df)) else wv, p)
    data.frame(outcome = sub("^out_", "", p), logHR = r$logHR, se = r$se,
               schoenfeld_p = r$schoenfeld_p,
               events_treat = ir$treat$events, events_comp = ir$comp$events,
               ir_treat_1000py = ir$treat$ir_1000py, ir_comp_1000py = ir$comp$ir_1000py)
  })
  tb <- do.call(rbind, rows)
  tb$hr <- exp(tb$logHR); tb$lb <- exp(tb$logHR - 1.96 * tb$se); tb$ub <- exp(tb$logHR + 1.96 * tb$se)
  tb$p <- 2 * pnorm(-abs(tb$logHR / tb$se))
  tb <- calibrate_outcomes(tb, null_fit)
  if (!is.null(km_path)) {
    km_rows <- list()
    for (p in outcome_names) {
      dd <- data.frame(tt = df[[paste0(p, "_time")]], ev = df[[paste0(p, "_event")]],
                       treatment = df$treatment)
      sf <- tryCatch(survfit(Surv(tt, ev) ~ treatment, data = dd,
                             weights = if (is.null(wv)) NULL else wv),
                     error = function(e2) NULL)
      if (is.null(sf)) next
      st <- summary(sf, censored = TRUE)
      if (is.null(st$strata) || length(st$time) == 0) next   # single/empty stratum (degenerate arm)
      km <- data.frame(outcome = sub("^out_", "", p), arm = as.character(st$strata),
                       time = st$time, surv = st$surv)
      km <- do.call(rbind, lapply(split(km, km$arm), function(g) {
        if (nrow(g) > 250) g[seq(1, nrow(g), length.out = 250), ] else g
      }))
      km_rows[[p]] <- km
    }
    if (length(km_rows)) write.csv(do.call(rbind, km_rows), km_path, row.names = FALSE)
  }
  tb
}

# =============================================================================
# obesity subgroup (primary estimator only): PS refit within subgroup, sIPTW Cox
# =============================================================================
obesity_block_fn <- function() {
  if (!("cov_obesity" %in% names(data)) || opt$spec == "crude") return(NULL)
  ob <- data[data$cov_obesity == 1, ]
  if (nrow(ob) < 200 || min(table(ob$treatment)) < 50) {
    log_info("obesity subgroup too small -- skipped"); return(NULL)
  }
  log_info(sprintf("== obesity subgroup (n=%d, treated=%d) ==", nrow(ob), sum(ob$treatment)))
  xv_ob <- xvars[vapply(xvars, function(v) {
    x <- ob[[v]]
    if (is.factor(x)) nlevels(droplevels(x)) >= 2 else var(as.numeric(x), na.rm = TRUE) > 0
  }, logical(1))]
  ob[] <- lapply(ob, function(x) if (is.factor(x)) droplevels(x) else x)
  f_ob <- as.formula(paste0("treatment ~ ", paste0(xv_ob, collapse = " + ")))
  ps_ob <- tryCatch({
    if (uses_embed) {
      Xo <- model.matrix(f_ob, ob)[, -1, drop = FALSE]
      cvo <- glmnet::cv.glmnet(Xo, ob$treatment, family = "binomial", alpha = 0,
                               foldid = det_foldid(ob$person_id, CV_FOLDS))
      as.numeric(predict(cvo, newx = Xo, s = "lambda.min", type = "response"))
    } else {
      as.numeric(predict(fast_binom(f_ob, ob), newdata = ob, type = "response"))
    }
  }, error = function(e2) NULL)
  if (is.null(ps_ob) || !all(is.finite(ps_ob))) {
    log_warning("obesity-subgroup PS fit failed -- skipped"); return(NULL)
  }
  e_ob <- pmin(pmax(ps_ob, 1e-6), 1 - 1e-6)
  pt_ob <- mean(ob$treatment)
  w_ob0 <- ifelse(ob$treatment == 1, pt_ob / e_ob, (1 - pt_ob) / (1 - e_ob))
  w_ob <- pmin(pmax(w_ob0, quantile(w_ob0, 0.01)), quantile(w_ob0, 0.99))
  list(n = nrow(ob), n_treat = sum(ob$treatment),
       outcomes = outcome_block(ob, w_ob, NULL))
}

# =============================================================================
# estimator loop -- everything below is a cheap function of the shared PS
# =============================================================================
ests <- trimws(strsplit(opt$estimators, ",")[[1]])
ests <- ests[nzchar(ests)]
if (opt$spec == "crude") {
  # constant PS: siptw/strat both reduce to the unadjusted Cox; matching is undefined.
  ests <- intersect(ests, "siptw_cox")
  log_info("spec=crude: estimator panel reduced to siptw_cox (== unadjusted Cox)")
}

md <- NULL   # matched cohort, built once if psm_cox requested
if ("psm_cox" %in% ests) {
  set.seed(1)
  mi <- tryCatch(matchit(treatment ~ ps, data = data, method = "nearest", distance = data$ps,
                         ratio = 1, caliper = 0.2, std.caliper = TRUE),
                 error = function(e2) NULL)
  if (!is.null(mi)) {
    md <- match.data(mi)
    log_info(sprintf("%sPSM: matched %d treated / %d comparator (of %d / %d)", INDENT2,
                     sum(md$treatment == 1), sum(md$treatment == 0),
                     sum(data$treatment == 1), sum(data$treatment == 0)))
  } else {
    log_warning("matchit failed -- psm_cox skipped")
    ests <- setdiff(ests, "psm_cox")
  }
}

# =============================================================================
# CALIPER SWEEP on the ALREADY-FITTED PS: if the embedding-PS advantage survives at loose
# calipers (near-full retention) it is not an artifact of a fixed caliper shrinking the
# matched cohort (estimand shift). The PS is never re-fit per caliper.
# =============================================================================
if (nzchar(opt$caliper_sweep) && "psm_cox" %in% ests) {
  cals <- suppressWarnings(as.numeric(trimws(strsplit(opt$caliper_sweep, ",")[[1]])))
  cals <- cals[is.finite(cals) & cals > 0]
  n_tr0 <- sum(data$treatment == 1)
  sweep_rows <- lapply(cals, function(cal) {
    set.seed(1)   # same seed as the primary PSM fit -- only the caliper varies
    mi_c <- tryCatch(matchit(treatment ~ ps, data = data, method = "nearest",
                             distance = data$ps, ratio = 1,
                             caliper = cal, std.caliper = TRUE),
                     error = function(e2) NULL)
    if (is.null(mi_c)) return(data.frame(caliper = cal, n_matched_treat = NA_integer_,
      n_matched_comp = NA_integer_, retention = NA_real_, max_smd = NA_real_,
      n_smd_gt0.1 = NA_integer_, n_nco = NA_integer_, ease = NA_real_))
    md_c <- match.data(mi_c)
    b <- tryCatch(smd_table_for(md_c, NULL), error = function(e2) NULL)
    nt <- tryCatch(nco_estimates(md_c, NULL), error = function(e2) NULL)
    nf <- fit_null_safe(nt)
    es <- if (!is.null(nf))
      tryCatch(as.numeric(computeExpectedAbsoluteSystematicError(nf)[1]),
               error = function(e2) NA_real_) else NA_real_
    data.frame(caliper = cal,
               n_matched_treat = sum(md_c$treatment == 1),
               n_matched_comp  = sum(md_c$treatment == 0),
               retention       = if (n_tr0 > 0) sum(md_c$treatment == 1) / n_tr0 else NA_real_,
               max_smd         = if (is.null(b)) NA_real_ else max(abs(b$smd), na.rm = TRUE),
               n_smd_gt0.1     = if (is.null(b)) NA_integer_ else sum(abs(b$smd) > 0.1, na.rm = TRUE),
               n_nco           = if (is.null(nt)) NA_integer_ else nrow(nt),
               ease            = es)
  })
  sw <- do.call(rbind, sweep_rows)
  sw$comparison <- opt$comparison; sw$spec <- opt$spec
  sw_dir <- file.path(opt$out_dir, "psm_cox"); dir.create(sw_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(sw, file.path(sw_dir, "caliper_sweep.csv"), row.names = FALSE)
  log_info(sprintf("%scaliper sweep: %s -> retention %s", INDENT2,
                   paste(sprintf("%.2f", sw$caliper), collapse = "/"),
                   paste(sprintf("%.0f%%", 100 * sw$retention), collapse = "/")))
}

for (est in ests) {
  log_info(sprintf("== estimator: %s ==", est))
  out_dir <- file.path(opt$out_dir, est)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (est == "siptw_cox") {
    df <- data; wv <- w; cl <- NULL; st <- NULL
    bal <- smd_table_for(data, w)
  } else if (est == "psm_cox") {
    df <- md; wv <- NULL; cl <- "subclass"; st <- NULL
    bal <- smd_table_for(md, NULL)
  } else if (est == "strat_cox") {
    df <- data; wv <- NULL; cl <- NULL; st <- "decile"
    bal <- NULL   # balance is within-stratum by construction; no single-number SMD applies
  } else stop(sprintf("unknown estimator '%s'", est))

  nco_tb <- nco_estimates(df, wv, cl, st)
  null_fit <- fit_null_safe(nco_tb)
  ease <- if (!is.null(null_fit))
    tryCatch(computeExpectedAbsoluteSystematicError(null_fit), error = function(e2) NA_real_) else NA_real_
  log_info(sprintf("%sNCOs used: %d  EASE = %s", INDENT2, nrow(nco_tb),
                   ifelse(is.na(ease), "NA", sprintf("%.4f", ease))))

  tb <- outcome_block(df, wv, null_fit, cl, st,
                      km_path = if (est == "siptw_cox") file.path(out_dir, "km_curves.csv") else NULL)
  for (i in seq_len(nrow(tb)))
    log_info(sprintf("%s%-45s HR=%.3f (%.3f-%.3f)  cal HR=%s", INDENT2, tb$outcome[i],
                     tb$hr[i], tb$lb[i], tb$ub[i],
                     ifelse(is.na(tb$hr_cal[i]), "NA",
                            sprintf("%.3f (%.3f-%.3f)", tb$hr_cal[i], tb$lb_cal[i], tb$ub_cal[i]))))

  ob <- if (est == "siptw_cox") obesity_block_fn() else NULL

  res <- list(
    comparison = opt$comparison, spec = opt$spec, estimator = est,
    ps_engine = if (uses_embed) "ridge" else "logistic", cv_folds = CV_FOLDS,
    treat_class = opt$treat_class, comp_class = opt$comp_class,
    n = nrow(data), n_treat = sum(data$treatment), n_comp = sum(data$treatment == 0),
    n_analyzed = nrow(df),
    n_matched_treat = if (est == "psm_cox") sum(df$treatment == 1) else NULL,
    n_matched_comp = if (est == "psm_cox") sum(df$treatment == 0) else NULL,
    n_covariates = length(xvars), c_statistic = c_stat, epv_ps = epv_ps,
    ess_treat = if (est == "siptw_cox") ess(w[data$treatment == 1]) else NULL,
    ess_comp = if (est == "siptw_cox") ess(w[data$treatment == 0]) else NULL,
    max_weight = if (est == "siptw_cox") max(w) else NULL,
    max_weight_raw = if (est == "siptw_cox") max(w_raw) else NULL,
    equipoise_proportion = equipoise,
    balance = if (!is.null(bal)) list(
      n_covariates = nrow(bal),
      max_smd = max(abs(bal$smd), na.rm = TRUE),
      n_gt_0.1 = sum(abs(bal$smd) > 0.1, na.rm = TRUE)) else NULL,
    nco = list(n_used = nrow(nco_tb), ease = ease,
               null_mean = if (!is.null(null_fit)) unname(null_fit[1]) else NA,
               null_sd = if (!is.null(null_fit)) unname(null_fit[2]) else NA),
    outcomes = tb,
    obesity = ob)
  write_json(res, file.path(out_dir, "estimate.json"),
             auto_unbox = TRUE, pretty = TRUE, na = "null", dataframe = "rows")
  if (!is.null(bal)) write.csv(bal, file.path(out_dir, "balance_smd.csv"), row.names = FALSE)
  write.csv(nco_tb, file.path(out_dir, "nco_estimates.csv"), row.names = FALSE)
  log_info(sprintf("Wrote %s", file.path(out_dir, "estimate.json")))
}
log_info(sprintf("Done (%d estimator(s)). Log: %s", length(ests), log_file))
