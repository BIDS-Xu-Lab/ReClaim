#!/usr/bin/env Rscript
# =============================================================================
# estimate_dr.R -- the doubly robust estimates of Study 1, computed from the nuisances that
# fit_dr_models.R has cross-fitted and checkpointed. Nothing is refitted: this reads the cohort and
# the per-nuisance .rds files, so it runs in minutes on a few GB.
#
# REPORTED: aipw_att, the effect in the treated (ATT). Doubly robust score with the untreated-risk
# model g0 only (fitted on the comparator arm); comparators weighted by
# e / (1 - e), the weights truncated at their 1st and 99th percentiles (truncation = wpctl_1_99, the
# rule Study 2 applies to its weights). iptw_att_hajek is the same weighting without an outcome
# model.
#
# Also reported, for every (PS row x outcome-model column) with checkpoints in --cache_dir:
#   aipw        whole-cohort (ATE) AIPW, the estimator first reported
#   aipw_hajek  the same with weights normalised within each arm; tmle, targeted maximum
#               likelihood (weighted-intercept fluctuation of each arm's outcome model)
#   aipw_atu    the effect in the UNTREATED (treated weighted by (1 - e) / e; needs g1 for every
#               comparator). The whole-cohort effect is a mixture of the treated and untreated
#               effects, so this row shows which half a whole-cohort estimate owes its value to
#   iptw_hajek  whole-cohort weighting without an outcome model; gcomp, the outcome model alone
#   ps_decile_adj  the ladder's own estimator on this PS row's cross-fitted score: the outcome on
#               treatment + decile indicators with the ladder's model (--design cohort_logistic:
#               logistic; landmark_cox: Cox on person_time)
# All are marginal risk ratios with influence-function standard errors (ps_decile_adj: model SE).
# Propensity rules: clip_0.01 (as checkpointed) and pctl_1_99 (propensities truncated at their
# 1st / 99th percentiles); the ATT rows also carry wpctl_1_99. --out_root/dr_ps_decile_table.csv
# gives, per PS row and decile, arm sizes, events, the within-decile risk ratio and weight shares.
#
# Inputs : --cohort_csv, --cache_dir, --population [--risk_start_days], [--limit]
# Output : --out_root/dr_estimators_grid.csv
# =============================================================================

suppressWarnings(suppressMessages({ library(optparse) }))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()
IND <- "    "

opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohort_csv", type = "character"),
  make_option("--cache_dir", type = "character"),
  make_option("--out_root", type = "character"),
  make_option("--population", type = "character", default = "all", help = "all|at_risk"),
  make_option("--risk_start_days", type = "integer", default = NULL),
  make_option("--limit", type = "integer", default = NULL),
  make_option("--design", type = "character", default = "cohort_logistic",
              help = "cohort_logistic|landmark_cox -- the ladder's outcome model for this cohort (used by the ps_decile rows)")
)))
stopifnot(opt$population %in% c("all", "at_risk"),
          opt$design %in% c("cohort_logistic", "landmark_cox"))

coh <- read.csv(opt$cohort_csv)
if (!is.null(opt$limit)) coh <- coh[seq_len(min(opt$limit, nrow(coh))), ]
if (opt$population == "at_risk") {
  if (is.null(opt$risk_start_days)) stop("--population at_risk needs --risk_start_days")
  coh <- coh[coh$person_time > opt$risk_start_days | coh$outcome == 1, ]
}
n <- nrow(coh); W <- as.integer(coh$arm); Y <- as.integer(coh$outcome)
log_info(sprintf("population=%s n=%s treated=%s events=%s", opt$population,
                 format(n, big.mark = ","), format(sum(W), big.mark = ","), format(sum(Y), big.mark = ",")))

# checkpoint names: ps_<row><CTAG>.rds, om_<col>_arm{0,1}<CTAG>.rds
CTAG <- paste0(if (opt$population == "all") "" else paste0("__", opt$population),
               if (is.null(opt$limit)) "" else paste0("__limit", opt$limit))
# keep the checkpoints whose tag is exactly CTAG: strip it, then refuse any other "__" tag
stems <- sub("\\.rds$", "", list.files(opt$cache_dir, pattern = "\\.rds$"))
stems <- stems[endsWith(stems, CTAG)]
base <- substr(stems, 1, nchar(stems) - nchar(CTAG))
base <- base[!grepl("__", base, fixed = TRUE)]
ps_rows <- sub("^ps_", "", base[grepl("^ps_", base)])
om_cols <- unique(sub("_arm[01]$", "", sub("^om_", "", base[grepl("^om_.*_arm[01]$", base)])))
rd <- function(stem) {
  cc <- readRDS(file.path(opt$cache_dir, paste0(stem, CTAG, ".rds")))
  if (!identical(as.integer(cc$n), as.integer(n))) stop(sprintf("%s was fitted on n=%s, cohort has n=%s", stem, cc$n, n))
  cc
}
e_hat <- lapply(setNames(ps_rows, ps_rows), function(ps) rd(paste0("ps_", ps))$e)
g_hat <- lapply(setNames(om_cols, om_cols), function(om) {
  list(g1 = rd(sprintf("om_%s_arm1", om))$res$g, g0 = rd(sprintf("om_%s_arm0", om))$res$g)
})
log_info(sprintf("PS rows: %s | outcome models: %s", paste(ps_rows, collapse = ", "), paste(om_cols, collapse = ", ")))

TRUNC <- list(
  "clip_0.01" = function(e) e,                                   # as checkpointed
  "pctl_1_99" = function(e) { q <- quantile(e, c(0.01, 0.99)); pmin(pmax(e, q[1]), q[2]) })

rr_from <- function(phi1, phi0) {   # marginal RR + influence-function SE of its log
  r1 <- mean(phi1); r0 <- mean(phi0)
  se <- sqrt(var(phi1 / r1 - phi0 / r0) / n)
  c(rr = r1 / r0, lb = exp(log(r1 / r0) - 1.96 * se), ub = exp(log(r1 / r0) + 1.96 * se), r1 = r1, r0 = r0)
}
est <- function(e, g1, g0) {
  w1 <- W / e; w0 <- (1 - W) / (1 - e)
  out <- list()
  out$aipw <- rr_from(g1 + w1 * (Y - g1), g0 + w0 * (Y - g0))
  out$aipw_hajek <- rr_from(g1 + w1 / mean(w1) * (Y - g1), g0 + w0 / mean(w0) * (Y - g0))
  # TMLE: shift each arm's outcome model on the logit scale by the intercept of a weighted
  # logistic regression (weights 1/e or 1/(1-e)) with the model's own logit as offset. The
  # weighted residuals then sum to zero, so the estimate is the mean of the updated predictions.
  fluct <- function(g, w) {
    k <- w > 0
    eps <- coef(suppressWarnings(glm(Y[k] ~ 1, offset = qlogis(g[k]), weights = w[k], family = quasibinomial())))[1]
    plogis(qlogis(g) + eps)
  }
  g1s <- fluct(g1, w1); g0s <- fluct(g0, w0)
  out$tmle <- rr_from(g1s + w1 * (Y - g1s), g0s + w0 * (Y - g0s))
  out
}
ess <- function(e) { w <- W / e + (1 - W) / (1 - e); sum(w)^2 / sum(w^2) / n }
# effect in the treated: r1 is the treated arm's observed risk; r0 is what the treated would have
# had untreated. The common 1 / P(W = 1) factor cancels in the ratio, so it is left out.
att <- function(e, g0 = NULL, wtrunc = FALSE) {
  w0 <- (1 - W) * e / (1 - e)
  # every treated weight is 1, so only the comparator weights vary: truncate those at their own
  # 1st and 99th percentiles
  if (wtrunc) { c0 <- W == 0; q <- quantile(w0[c0], c(0.01, 0.99)); w0[c0] <- pmin(pmax(w0[c0], q[1]), q[2]) }
  w0 <- w0 * sum(W) / sum(w0)                                      # normalised to the treated count
  phi1 <- W * Y
  phi0 <- if (is.null(g0)) w0 * Y else W * g0 + w0 * (Y - g0)
  r1 <- mean(phi1); r0 <- mean(phi0)
  se <- sqrt(var(phi1 / r1 - phi0 / r0) / n)
  c(rr = r1 / r0, lb = exp(log(r1 / r0) - 1.96 * se), ub = exp(log(r1 / r0) + 1.96 * se),
    r1 = r1 / mean(W), r0 = r0 / mean(W))
}
atu <- function(e, g1) {   # effect in the untreated: r0 observed, r1 = what the comparators would have had treated
  w1 <- W * (1 - e) / e; w1 <- w1 * sum(1 - W) / sum(w1)
  phi0 <- (1 - W) * Y
  phi1 <- (1 - W) * g1 + w1 * (Y - g1)
  r1 <- mean(phi1); r0 <- mean(phi0)
  se <- sqrt(var(phi1 / r1 - phi0 / r0) / n)
  c(rr = r1 / r0, lb = exp(log(r1 / r0) - 1.96 * se), ub = exp(log(r1 / r0) + 1.96 * se),
    r1 = r1 / mean(1 - W), r0 = r0 / mean(1 - W))
}
# the ladder's regression on this run's rows: treatment + propensity-decile indicators; returns the
# treatment coefficient as a ratio
ladder_fit <- function(d) {
  df <- data.frame(Y = Y, W = W, d = factor(d))
  rhs <- "W + d"
  if (opt$design == "landmark_cox") {
    df$tt <- pmax(as.numeric(coh$person_time), 1)
    cf <- summary(survival::coxph(as.formula(paste("survival::Surv(tt, Y) ~", rhs)), data = df))$coefficients["W", c("coef", "se(coef)")]
  } else {
    cf <- summary(glm(as.formula(paste("Y ~", rhs)), family = binomial(), data = df))$coefficients["W", 1:2]
  }
  c(rr = exp(cf[[1]]), lb = exp(cf[[1]] - 1.96 * cf[[2]]), ub = exp(cf[[1]] + 1.96 * cf[[2]]), r1 = NA, r0 = NA)
}
decile_of <- function(e) cut(e, unique(quantile(e, 0:10 / 10)), include.lowest = TRUE, labels = FALSE)

# what each row estimates: the whole cohort (ate), the treated (att), the untreated (atu), or a
# decile-conditional effect
ESTIMAND <- c(gcomp = "ate", iptw_hajek = "ate", aipw = "ate", aipw_hajek = "ate", tmle = "ate",
              iptw_att_hajek = "att", aipw_att = "att", aipw_atu = "atu", ps_decile_adj = "conditional")
rows <- list()
add <- function(ps, om, estimator, trunc, v, e = NULL)
  rows[[length(rows) + 1L]] <<- data.frame(
    ps_spec = ps, om = om, estimator = estimator, estimand = ESTIMAND[[estimator]], truncation = trunc,
    rr = v[["rr"]], rr_lb = v[["lb"]], rr_ub = v[["ub"]], r1 = v[["r1"]], r0 = v[["r0"]],
    ess_frac = if (is.null(e)) NA else ess(e), population = opt$population, n = n)
for (om in om_cols) {
  gg <- g_hat[[om]]; r1 <- mean(gg$g1); r0 <- mean(gg$g0)
  add("gcomp", om, "gcomp", NA, c(rr = r1 / r0, lb = NA, ub = NA, r1 = r1, r0 = r0))
}
dec_rows <- list()
for (ps in ps_rows) {
  # the ladder's estimator on this row's cross-fitted score, and the per-decile picture behind it
  d <- decile_of(e_hat[[ps]])
  add(ps, NA, "ps_decile_adj", NA, ladder_fit(d))
  w <- W / e_hat[[ps]] + (1 - W) / (1 - e_hat[[ps]])
  dec_rows[[ps]] <- do.call(rbind, lapply(sort(unique(d)), function(k) {
    i <- d == k; n1 <- sum(W[i]); n0 <- sum(1 - W[i]); y1 <- sum(Y[i] * W[i]); y0 <- sum(Y[i] * (1 - W[i]))
    data.frame(ps_spec = ps, decile = k, e_mean = mean(e_hat[[ps]][i]), n_treated = n1, n_comparator = n0,
               events_treated = y1, events_comparator = y0, rr = (y1 / n1) / (y0 / n0),
               ate_weight_share = sum(w[i]) / sum(w), treated_weight_share = sum((W / e_hat[[ps]])[i]) / sum(W / e_hat[[ps]]))
  }))
}
for (ps in ps_rows) for (tn in names(TRUNC)) {
  e <- TRUNC[[tn]](e_hat[[ps]])
  w1 <- W / e; w0 <- (1 - W) / (1 - e)
  add(ps, NA, "iptw_hajek", tn, rr_from(w1 / mean(w1) * Y, w0 / mean(w0) * Y), e)
  add(ps, NA, "iptw_att_hajek", tn, att(e), e)
  for (om in om_cols) {
    v <- est(e, g_hat[[om]]$g1, g_hat[[om]]$g0)
    for (nm in names(v)) add(ps, om, nm, tn, v[[nm]], e)
    add(ps, om, "aipw_att", tn, att(e, g_hat[[om]]$g0), e)
    add(ps, om, "aipw_atu", tn, atu(e, g_hat[[om]]$g1), e)
  }
}
for (ps in ps_rows) {   # the reported rule: weights truncated at their 1st / 99th percentiles
  e <- e_hat[[ps]]
  add(ps, NA, "iptw_att_hajek", "wpctl_1_99", att(e, wtrunc = TRUE), e)
  for (om in om_cols) add(ps, om, "aipw_att", "wpctl_1_99", att(e, g_hat[[om]]$g0, wtrunc = TRUE), e)
}
grid <- do.call(rbind, rows)
dir.create(opt$out_root, recursive = TRUE, showWarnings = FALSE)
write.csv(grid, file.path(opt$out_root, "dr_estimators_grid.csv"), row.names = FALSE)
write.csv(do.call(rbind, dec_rows), file.path(opt$out_root, "dr_ps_decile_table.csv"), row.names = FALSE)
for (i in seq_len(nrow(grid))) with(grid[i, ], log_info(sprintf(
  "%s%-22s %-20s %-15s %-10s RR=%.3f (%.3f-%.3f) ess=%.2f", IND, ps_spec, ifelse(is.na(om), "-", om),
  estimator, ifelse(is.na(truncation), "-", truncation), rr, rr_lb, rr_ub, ess_frac)))
log_info(sprintf("Wrote %d rows -> %s", nrow(grid), file.path(opt$out_root, "dr_estimators_grid.csv")))
