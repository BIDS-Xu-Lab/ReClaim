#!/usr/bin/env Rscript
# =============================================================================
# summarize_results.R -- collect the (comparison x spec x estimator) grid into tables + figures (ty3_2).
#
# Input: results/<cell>/<comparison>/<spec>/<estimator>/estimate.json (estimate_ladder).
# Outputs under results/_summary_<cell>/:
#   summary_outcomes.csv      per-outcome HR/CI/p + calibrated, all grid cells
#   summary_diagnostics.csv   per-cell: n, c-stat, EASE, equipoise, balance, ESS/matched-N
#   summary_obesity.csv       obesity-subgroup rows (primary estimator)
#   forest_primary.png        siptw_cox: outcomes x specs, before/after NCO calibration
#   forest_estimators.png     estimator-consistency: outcomes x specs, colored by estimator
#   forest_nco_<comp>.png     NCO estimates per spec (siptw_cox)
#   km_<comp>.png             weighted KM curves (paper spec, siptw_cox)
#   RESULTS.md                readable digest
# =============================================================================

# ensure_packages (utils.R) installs missing CRAN packages before library() can fail.
source(file.path(dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])), "..", "utils.R"))
ensure_packages(c("optparse", "jsonlite", "dplyr", "ggplot2", "glue"))
suppressWarnings(suppressMessages({
  library(optparse); library(jsonlite); library(dplyr); library(ggplot2)
}))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()

opt <- parse_args(OptionParser(option_list = list(
  make_option("--results_root", type = "character", default = "results/ynhh_src"),
  make_option("--out_dir", type = "character", default = "results/_summary_ynhh_src")
)))
dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)

spec_order <- c("crude", "demo", "paper", "hdps",
                "reclaim-s-only", "reclaim-m-only", "reclaim-l-only")
est_order <- c("siptw_cox", "psm_cox", "strat_cox")
# factor(levels=spec_order) would silently NA-drop a spec not in the list; spec_fct() appends
# unknown specs at the tail, so spec_order only pins the display order of the known ladder.
spec_fct <- function(x) factor(x, levels = c(spec_order, setdiff(unique(as.character(x)), spec_order)))
# specs actually present on disk (union with spec_order so ordering is stable)
specs_on_disk <- function(comp_dir) union(spec_order,
  basename(list.dirs(comp_dir, recursive = FALSE)))
nz <- function(x) if (is.null(x) || length(x) == 0) NA else x

jsons <- Sys.glob(file.path(opt$results_root, "*", "*", "*", "estimate.json"))
log_info(sprintf("Found %d estimate.json files under %s", length(jsons), opt$results_root))
# estimate_ladder's NON-ANALYZABLE stubs live one level UP (<comparison>/<spec>/estimate.json, no
# estimator dir). Count them so an all-stub run reports itself instead of erroring on 0 files.
stubs <- Sys.glob(file.path(opt$results_root, "*", "*", "estimate.json"))
if (length(stubs) > 0)
  log_warning(sprintf("%d NON-ANALYZABLE spec-level stubs present: %s", length(stubs),
                      paste(sub(paste0("^", opt$results_root, "/?"), "", dirname(stubs)), collapse = ", ")))
if (length(jsons) == 0) {
  if (length(stubs) > 0) {
    dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
    writeLines(c("ALL CELLS NON-ANALYZABLE (empty arm / degenerate cohort).",
                 sprintf("Spec-level stubs (%d):", length(stubs)),
                 sub(paste0("^", opt$results_root, "/?"), "", dirname(stubs))),
               file.path(opt$out_dir, "ALL_NON_ANALYZABLE.txt"))
    log_warning(sprintf("No analyzable estimates -- wrote %s and exiting cleanly",
                        file.path(opt$out_dir, "ALL_NON_ANALYZABLE.txt")))
    quit(save = "no", status = 0)
  }
  stop("no estimates found (expected <comparison>/<spec>/<estimator>/estimate.json)")
}

outc_rows <- list(); diag_rows <- list(); ob_rows <- list()
for (jp in jsons) {
  j <- fromJSON(jp)
  key <- sprintf("%s/%s/%s", j$comparison, j$spec, j$estimator)
  ot <- as.data.frame(j$outcomes)
  ot$comparison <- j$comparison; ot$spec <- j$spec; ot$estimator <- j$estimator
  outc_rows[[key]] <- ot
  diag_rows[[key]] <- data.frame(
    comparison = j$comparison, spec = j$spec, estimator = j$estimator,
    n = j$n, n_treat = j$n_treat, n_comp = j$n_comp, n_analyzed = nz(j$n_analyzed),
    n_matched_treat = nz(j$n_matched_treat),
    n_covariates = j$n_covariates,
    c_statistic = nz(j$c_statistic), epv_ps = nz(j$epv_ps),
    lasso_nonzero = nz(j$lasso_nonzero),
    ess_treat = nz(j$ess_treat), ess_comp = nz(j$ess_comp),
    max_weight = nz(j$max_weight),
    equipoise = j$equipoise_proportion,
    n_nco = j$nco$n_used, ease = nz(j$nco$ease),
    max_smd = nz(j$balance$max_smd),
    n_smd_gt0.1 = nz(j$balance$n_gt_0.1))
  if (is.data.frame(j$obesity$outcomes) && nrow(j$obesity$outcomes) > 0) {
    obt <- as.data.frame(j$obesity$outcomes)
    obt$comparison <- j$comparison; obt$spec <- j$spec; obt$estimator <- j$estimator
    obt$n_subgroup <- j$obesity$n
    ob_rows[[key]] <- obt
  }
}
outc <- bind_rows(outc_rows) %>%
  mutate(spec = spec_fct(spec),
         estimator = factor(estimator, levels = est_order))
diag <- bind_rows(diag_rows) %>%
  arrange(comparison, spec_fct(spec), factor(estimator, levels = est_order))
write.csv(outc, file.path(opt$out_dir, "summary_outcomes.csv"), row.names = FALSE)
write.csv(diag, file.path(opt$out_dir, "summary_diagnostics.csv"), row.names = FALSE)
if (length(ob_rows)) write.csv(bind_rows(ob_rows), file.path(opt$out_dir, "summary_obesity.csv"), row.names = FALSE)

# Plot failures must not kill the summary (a tiny cohort can leave nothing to facet); the CSVs
# and digest are the deliverable, plots are best-effort.
safe_plot <- function(label, expr) tryCatch(expr, error = function(e)
  log_warning(sprintf("plot '%s' skipped: %s", label, conditionMessage(e))))

# ---- forest: primary estimator, uncalibrated vs calibrated -----------------
prim <- outc %>% filter(estimator == "siptw_cox")
fp <- prim %>%
  select(comparison, spec, outcome, hr, lb, ub, hr_cal, lb_cal, ub_cal) %>%
  tidyr::pivot_longer(cols = c(hr, hr_cal), names_to = "kind", values_to = "HR") %>%
  mutate(LB = ifelse(kind == "hr", lb, lb_cal), UB = ifelse(kind == "hr", ub, ub_cal),
         kind = ifelse(kind == "hr", "uncalibrated", "NCO-calibrated")) %>%
  filter(is.finite(HR))
safe_plot("forest_primary", {
g <- ggplot(fp, aes(x = HR, y = outcome, color = kind)) +
  geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
  geom_point(position = position_dodge(width = 0.6), size = 1.6) +
  geom_errorbarh(aes(xmin = LB, xmax = UB), position = position_dodge(width = 0.6), height = 0.25) +
  facet_grid(comparison ~ spec) + scale_x_log10() +
  labs(x = "Hazard ratio (95% CI, log scale)", y = NULL,
       title = "GLP-1RA vs comparator: psychiatric outcomes (sIPTW Cox, primary estimator)",
       subtitle = "columns = PS covariate spec; colors = before/after NCO calibration") +
  theme_bw(base_size = 9) + theme(legend.position = "bottom")
ggsave(file.path(opt$out_dir, "forest_primary.png"), g,
       width = max(12, 2 + 2.2 * length(unique(fp$spec))), height = 6, dpi = 200, limitsize = FALSE)
})

# ---- forest: estimator consistency (uncalibrated HRs, colored by estimator) ----
fe <- outc %>% filter(is.finite(hr), spec != "crude")
safe_plot("forest_estimators", {
ge <- ggplot(fe, aes(x = hr, y = outcome, color = estimator)) +
  geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
  geom_point(position = position_dodge(width = 0.7), size = 1.4) +
  geom_errorbarh(aes(xmin = lb, xmax = ub), position = position_dodge(width = 0.7), height = 0.3) +
  facet_grid(comparison ~ spec) + scale_x_log10() +
  labs(x = "Hazard ratio (95% CI, log scale)", y = NULL,
       title = "Estimator consistency: weighting (sIPTW) vs matching (PSM) vs stratification (deciles)",
       subtitle = "same PS fit per spec; estimands differ (ATE / ATT / conditional) -- read direction and null-coverage") +
  theme_bw(base_size = 9) + theme(legend.position = "bottom")
ggsave(file.path(opt$out_dir, "forest_estimators.png"), ge,
       width = max(12, 2 + 2.2 * length(unique(fe$spec))), height = 6.5, dpi = 200, limitsize = FALSE)
})

# ---- NCO forests (siptw_cox) ------------------------------------------------
for (comp in unique(outc$comparison)) {
  ncs <- list()
  for (sp in specs_on_disk(file.path(opt$results_root, comp))) {
    f <- file.path(opt$results_root, comp, sp, "siptw_cox", "nco_estimates.csv")
    # nrow guard: a tiny cohort can leave an empty nco_estimates.csv
    if (file.exists(f)) { d <- read.csv(f); if (nrow(d)) { d$spec <- sp; ncs[[sp]] <- d } }
  }
  if (!length(ncs)) next
  nd <- bind_rows(ncs) %>% mutate(spec = spec_fct(spec),
                                  hr = exp(logHR), lb = exp(logHR - 1.96 * seLogHR),
                                  ub = exp(logHR + 1.96 * seLogHR))
  safe_plot(sprintf("forest_nco_%s", comp), {
  gn <- ggplot(nd, aes(x = hr, y = reorder(nco, hr))) +
    geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
    geom_errorbarh(aes(xmin = lb, xmax = ub), height = 0, color = "grey70", linewidth = 0.3) +
    geom_point(size = 0.7, color = "#2166AC") +
    facet_wrap(~spec, nrow = 1) + scale_x_log10() +
    labs(x = "NCO hazard ratio (95% CI)", y = NULL,
         title = sprintf("Negative control outcomes (sIPTW) -- %s", comp)) +
    theme_bw(base_size = 7) + theme(axis.text.y = element_text(size = 4))
  ggsave(file.path(opt$out_dir, sprintf("forest_nco_%s.png", comp)), gn,
         width = max(12, 2 + 2.2 * length(ncs)), height = 9, dpi = 200, limitsize = FALSE)
  })
}

# ---- KM curves (paper spec, siptw_cox) --------------------------------------
for (comp in unique(outc$comparison)) {
  f <- file.path(opt$results_root, comp, "paper", "siptw_cox", "km_curves.csv")
  if (!file.exists(f)) next
  km <- read.csv(f)
  km$arm <- ifelse(grepl("=1$|treatment=1", km$arm), "GLP-1RA", "comparator")
  safe_plot(sprintf("km_%s", comp), {
  gk <- ggplot(km, aes(x = time / 365.25, y = 1 - surv, color = arm)) +
    geom_step(linewidth = 0.5) + facet_wrap(~outcome, scales = "free_y", ncol = 3) +
    labs(x = "Years since initiation", y = "Weighted cumulative incidence",
         title = sprintf("Weighted KM cumulative incidence -- %s (paper spec, sIPTW)", comp)) +
    theme_bw(base_size = 9) + theme(legend.position = "bottom")
  ggsave(file.path(opt$out_dir, sprintf("km_%s.png", comp)), gk, width = 10, height = 6, dpi = 200)
  })
}

# ---- readable digest --------------------------------------------------------
md <- c("# TTE replication -- results digest", "",
        sprintf("Generated by summarize_results.R. %d (comparison x spec x estimator) cells.", nrow(diag)), "",
        "## Diagnostics", "",
        paste(capture.output(print(as.data.frame(diag), row.names = FALSE)), collapse = "\n"), "",
        "## Primary outcomes -- siptw_cox (HR then NCO-calibrated HR)", "")
for (comp in unique(outc$comparison)) {
  md <- c(md, sprintf("### %s", comp), "")
  sub <- prim %>% filter(comparison == comp) %>% arrange(outcome, spec)
  md <- c(md, "| outcome | spec | HR (95% CI) | p | cal HR (95% CI) | cal p |",
          "|---|---|---|---|---|---|")
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    md <- c(md, sprintf("| %s | %s | %.2f (%.2f-%.2f) | %.3f | %s | %s |",
                        r$outcome, r$spec, r$hr, r$lb, r$ub, r$p,
                        ifelse(is.na(r$hr_cal), "NA", sprintf("%.2f (%.2f-%.2f)", r$hr_cal, r$lb_cal, r$ub_cal)),
                        ifelse(is.na(r$p_cal), "NA", sprintf("%.3f", r$p_cal))))
  }
  md <- c(md, "")
}
md <- c(md, "## Estimator consistency -- anxiety & depression, uncalibrated HR by estimator", "")
cons <- outc %>%
  filter(outcome %in% c("anxiety_disorders", "depression"), spec != "crude") %>%
  arrange(comparison, outcome, spec, estimator)
md <- c(md, "| comparison | outcome | spec | siptw | psm | strat |", "|---|---|---|---|---|---|")
for (grp in split(cons, list(cons$comparison, cons$outcome, cons$spec), drop = TRUE)) {
  fmt <- function(e) {
    r <- grp[grp$estimator == e, ]
    if (nrow(r) == 0 || !is.finite(r$hr[1])) "NA" else sprintf("%.2f (%.2f-%.2f)", r$hr[1], r$lb[1], r$ub[1])
  }
  md <- c(md, sprintf("| %s | %s | %s | %s | %s | %s |",
                      grp$comparison[1], grp$outcome[1], grp$spec[1],
                      fmt("siptw_cox"), fmt("psm_cox"), fmt("strat_cox")))
}
writeLines(md, file.path(opt$out_dir, "RESULTS.md"))
log_info(sprintf("Wrote summary tables + figures -> %s. Log: %s", opt$out_dir, log_file))
