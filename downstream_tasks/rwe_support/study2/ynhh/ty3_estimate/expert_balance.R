#!/usr/bin/env Rscript
# =============================================================================
# expert_balance.R -- balance of one PS specification on the reference's covariate set
# (Tang 2026 Supplement Table S5, i.e. the `paper` spec's covariates), before adjustment, after
# sIPTW and after 1:1 PSM.
#
# The spec's PS, weights and matched set are those of estimate_ladder.R itself: this script
# evaluates estimate_ladder.R's own code up to the shared PS and weights (same cohort, covariates,
# PS engine, deterministic CV folds, p1/p99 truncation), then repeats its PSM call (seed 1,
# caliper 0.2 SD). No outcome model is fitted. SMD convention as estimate_ladder.R: difference in
# (weighted) means over the full two-arm cohort's unweighted pooled SD.
#
# Takes estimate_ladder.R's arguments (same --out_dir).
# Output: <out_dir>/{siptw_cox,psm_cox}/balance_expert_smd.csv  (covariate, smd_before, smd_after)
# =============================================================================

here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
# (dot-prefixed names: estimate_ladder.R's globals, e.g. its clipped PS `e`, must not be overwritten)
.el_exprs <- parse(file.path(here, "estimate_ladder.R"), keep.source = FALSE)
.el_stop <- which(vapply(.el_exprs, function(x) startsWith(paste(deparse(x), collapse = " "), "ests <- trimws("), logical(1)))
if (length(.el_stop) != 1) stop("estimate_ladder.R: cannot locate the start of the estimator loop")
for (.el_i in seq_len(.el_stop - 1)) eval(.el_exprs[[.el_i]], envir = globalenv())

# ---- the expert covariate set on this cohort ----
if (opt$spec != "paper") { .s5 <- build_s5_vars(data); data <- .s5$data; s5_vars <- .s5$vars }
f_ex <- as.formula(paste0("treatment ~ ", paste0(s5_vars, collapse = " + ")))

smd_on <- function(f, df, wv = NULL) {
  Xa <- model.matrix(f, data)[, -1, drop = FALSE]
  t1a <- data$treatment == 1
  sd_pool <- sqrt((apply(Xa[t1a, , drop = FALSE], 2, var) + apply(Xa[!t1a, , drop = FALSE], 2, var)) / 2)
  Xd <- model.matrix(f, df)[, -1, drop = FALSE]
  t1 <- df$treatment == 1
  if (is.null(wv)) wv <- rep(1, nrow(df))
  m1 <- apply(Xd[t1, , drop = FALSE], 2, wmean, wv = wv[t1])
  m0 <- apply(Xd[!t1, , drop = FALSE], 2, wmean, wv = wv[!t1])
  data.frame(covariate = colnames(Xd), smd = as.numeric((m1 - m0) / sd_pool))
}

before <- smd_on(f_ex, data, NULL)
write_bal <- function(est, after) {
  # a term constant in the cohort has no SMD (zero pooled SD); dropped as in estimate_ladder.R
  out <- data.frame(covariate = before$covariate, smd_before = before$smd, smd_after = after$smd)
  out <- out[is.finite(out$smd_before) & is.finite(out$smd_after), ]
  dir.create(file.path(opt$out_dir, est), recursive = TRUE, showWarnings = FALSE)
  write.csv(out, file.path(opt$out_dir, est, "balance_expert_smd.csv"), row.names = FALSE)
  log_info(sprintf("%s%s: expert set (%d terms) max |SMD| before %.3f -> after %.3f; share < 0.1 after %.1f%%",
                   INDENT2, est, nrow(out), max(abs(out$smd_before)), max(abs(out$smd_after)),
                   100 * mean(abs(out$smd_after) < 0.1)))
}

# sIPTW: the stabilized, p1/p99-truncated weights `w` of estimate_ladder.R
write_bal("siptw_cox", smd_on(f_ex, data, w))

# 1:1 PSM, exactly as estimate_ladder.R (undefined for crude: constant PS)
if (opt$spec != "crude") {
  set.seed(1)
  mi <- matchit(treatment ~ ps, data = data, method = "nearest", distance = data$ps,
                ratio = 1, caliper = 0.2, std.caliper = TRUE)
  md <- match.data(mi)
  log_info(sprintf("%sPSM: matched %d treated / %d comparator", INDENT2, sum(md$treatment == 1), sum(md$treatment == 0)))
  write_bal("psm_cox", smd_on(f_ex, md, NULL))
}
log_info(sprintf("Done: %s / %s. Log: %s", opt$comparison, opt$spec, log_file))
