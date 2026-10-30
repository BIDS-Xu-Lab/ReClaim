#!/usr/bin/env Rscript
# =============================================================================
# test_estimate_dr.R -- checks the estimator functions of estimate_dr.R on a simulation
# whose true effects are known and differ by population (protective where the treated are,
# harmful elsewhere), so the whole-cohort, treated and untreated effects are three different
# numbers. Each doubly robust estimator must recover its own truth when EITHER nuisance is
# correct, and must not when both are wrong. Run from the module root:
#   Rscript b3_estimate/test_estimate_dr.R
# =============================================================================
ex <- parse("b3_estimate/estimate_dr.R")
for (e in ex) if (is.call(e) && identical(e[[1]], as.name("<-")) && is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))) eval(e)
set.seed(1); n <- 400000
X <- rnorm(n); e_true <- plogis(-2.2 + 1.0 * X)            # ~13% treated, treated have high X
W <- rbinom(n, 1, e_true)
p0 <- plogis(-4 + 0.8 * X)                                  # untreated risk rises with X
p1 <- p0 * ifelse(X > 0, 0.6, 1.3)                          # protective where treated live, harmful elsewhere
Y1 <- rbinom(n, 1, p1); Y0 <- rbinom(n, 1, p0); Y <- ifelse(W == 1, Y1, Y0)
truth <- c(ATE = mean(p1) / mean(p0), ATT = mean(p1[W == 1]) / mean(p0[W == 1]), ATU = mean(p1[W == 0]) / mean(p0[W == 0]))
cat("truth:", sprintf("%s=%.3f", names(truth), truth), "| crude:", sprintf("%.3f", mean(Y[W==1]) / mean(Y[W==0])), "\n")
f <- function(v) sprintf("%.3f (%.3f-%.3f)", v[["rr"]], v[["lb"]], v[["ub"]])
bad_e <- rep(mean(W), n); bad_g0 <- rep(mean(Y[W == 0]), n); bad_g1 <- rep(mean(Y[W == 1]), n)
cat("ATT  true e, true g0   :", f(att(e_true, p0)), "\n")
cat("ATT  true e, WRONG g0  :", f(att(e_true, bad_g0)), "\n")
cat("ATT  WRONG e, true g0  :", f(att(bad_e, p0)), "\n")
cat("ATT  WRONG e, WRONG g0 :", f(att(bad_e, bad_g0)), "  <- should be off (crude)\n")
cat("ATT  weighting only    :", f(att(e_true)), "\n")
cat("ATU  true e, true g1   :", f(atu(e_true, p1)), "\n")
cat("ATU  true e, WRONG g1  :", f(atu(e_true, bad_g1)), "\n")
v <- est(e_true, p1, p0); cat("ATE  aipw/hajek/tmle   :", f(v$aipw), "|", f(v$aipw_hajek), "|", f(v$tmle), "\n")
v <- est(e_true, bad_g1, bad_g0); cat("ATE  true e, WRONG g   :", f(v$aipw), "|", f(v$aipw_hajek), "|", f(v$tmle), "\n")
v <- est(bad_e, p1, p0); cat("ATE  WRONG e, true g   :", f(v$aipw), "|", f(v$aipw_hajek), "|", f(v$tmle), "\n")
