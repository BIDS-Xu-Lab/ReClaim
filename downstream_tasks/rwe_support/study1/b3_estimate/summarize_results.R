#!/usr/bin/env Rscript
# =============================================================================
# summarize_results.R -- assemble the summary deliverables (invoked by b3_3_summary.sh).
#   1. Main figure: two-panel forest (positive_control / negative_control), RR (95% Wald CI)
#      per spec x estimator, with the null line at 1. For negative_control the main figure
#      headlines the primary season (config season.primary / arg 2).
#   2. negative_control per-season figure (one season in the design; the machinery is per cell).
#   3. Supplementary table: every cell, covariate counts (n,k), c-statistics and the
#      non-estimable flags per spec.
#
# Walks results/mdcr/{cohort}/{cell}/{spec}/{estimator}/estimate.json, where {cell} is the payer
# stratum for positive_control and '{stratum}_s{season}' for negative_control.
#
# Usage: Rscript summarize_results.R [results_root] [v2_primary_season]
#   results_root       default results/mdcr
#   v2_primary_season  e.g. 2011 -- overrides config season.primary for negative_control's main panel.
# =============================================================================
suppressWarnings(suppressMessages({ library(jsonlite); library(dplyr); library(ggplot2) }))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R")); log_file <- setup_logging()
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

cargs <- commandArgs(TRUE)
RESULTS <- if (length(cargs) >= 1) cargs[1] else "results/mdcr"
PRIMARY_SEASON <- if (length(cargs) >= 2 && nzchar(cargs[2])) cargs[2] else NA_character_
# Fall back to the config value (config/negative_control_flu_hipfx.yaml: season.primary) when no season
# is passed on the CLI. No `yaml` pkg dependency: season.primary is the only 4-digit `primary:` value
# in the file (outcome.primary is a word like 'base').
if (is.na(PRIMARY_SEASON)) {
  cfg_path <- file.path(here, "..", "config", "negative_control_flu_hipfx.yaml")
  if (file.exists(cfg_path)) {
    hit <- grep("^\\s*primary:\\s*[0-9]{4}\\b", readLines(cfg_path, warn = FALSE), value = TRUE)
    if (length(hit) >= 1) {
      PRIMARY_SEASON <- sub("^\\s*primary:\\s*([0-9]{4}).*$", "\\1", hit[1])
      log_info(sprintf("negative_control primary season from config season.primary: %s", PRIMARY_SEASON))
    }
  }
}
OUT <- file.path(RESULTS, "_summary"); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

files <- list.files(RESULTS, pattern = "estimate.json$", recursive = TRUE, full.names = TRUE)
log_info(sprintf("Found %d estimate.json files under %s", length(files), RESULTS))

rows <- lapply(files, function(f) {
  j <- tryCatch(fromJSON(f), error = function(e) NULL); if (is.null(j)) return(NULL)
  parts <- strsplit(sub(paste0(RESULTS, "/"), "", f), "/")[[1]]  # cohort/cell/spec/estimator/estimate.json
  cell <- parts[2]
  # cell = payer stratum (positive_control) OR '{stratum}_s{season}' (negative_control per-season). Parse the season back out.
  mm <- regmatches(cell, regexec("^(.*)_s([0-9]{4})$", cell))[[1]]
  if (length(mm) == 3) { payer <- mm[2]; season <- mm[3] } else { payer <- cell; season <- NA_character_ }
  data.frame(cohort = parts[1], cell = cell, payer = payer, season = season,
             spec = parts[3], estimator = parts[4],
             rr = j$rr %||% NA, lb = j$rr_lb %||% NA, ub = j$rr_ub %||% NA,
             c_stat = j$c_statistic %||% NA,
             k = j$n_covariates %||% NA, events = j$n_exposed_events %||% NA,
             unstable = j$unstable_flag %||% NA,
             epv_ps = j$epv_ps %||% NA,
             stringsAsFactors = FALSE)
})
df <- do.call(rbind, Filter(Negate(is.null), rows))
if (is.null(df) || !nrow(df)) { log_warning("No results to summarize."); quit(save = "no") }

# NON-ESTIMABLE guard, three independent routes (flagged rows stay in the supplementary table
# and are dropped from the forest plots):
#  1. PS c-statistic at (or numerically at) 1: the PS separates treated from untreated perfectly,
#     every score is 0 or 1, strata carry no information and the "estimate" is numerical
#     breakdown. It can also manufacture a spuriously significant CI on a known-null contrast.
#     Derived from the stored c_statistic so it applies to results already on disk.
SEP_CUT <- 0.99
df$sep_cstat <- !is.na(df$c_stat) & df$c_stat >= SEP_CUT
#  2. estimate_ladder's epv_ps (minority-arm size / #covariates) < 1: a PS with more covariates
#     than minority-arm members is unsupported before it is fitted, even where a ridge fallback
#     keeps the c-statistic just under the cut. NA (absent) defaults to FALSE.
df$overparam <- !is.na(df$epv_ps) & df$epv_ps < 1
#  3. Too few exposed events (< 10, estimate_ladder's unstable_flag) for any propensity method to
#     mean anything.
df$too_few <- !is.na(df$unstable) & df$unstable
df$separation <- df$sep_cstat | df$overparam | df$too_few
if (any(df$separation)) {
  log_warning(sprintf("%d/%d estimates NON-ESTIMABLE (%d by PS c-statistic >= %.2f, %d by EPV < 1, %d by < 10 exposed events) -> dropped from the forest plots; they remain in supplementary_table.csv with separation=TRUE.",
                      sum(df$separation), nrow(df), sum(df$sep_cstat), SEP_CUT, sum(df$overparam),
                      sum(df$too_few)))
  for (i in which(df$separation))
    log_warning(sprintf("   non-estimable: %s / %s / %s / %s  (c=%.3f, k=%s, EPV=%s, RR=%.4g) [%s]",
                        df$cohort[i], df$cell[i], df$spec[i], df$estimator[i],
                        df$c_stat[i], df$k[i],
                        ifelse(is.na(df$epv_ps[i]), "NA", sprintf("%.2f", df$epv_ps[i])), df$rr[i],
                        paste(c("c-stat", "EPV", "<10 events")[c(df$sep_cstat[i], df$overparam[i], df$too_few[i])], collapse = "+")))
}

df$is_primary <- is.na(df$season) | (!is.na(PRIMARY_SEASON) & df$season == PRIMARY_SEASON)
seasoned <- df[!is.na(df$season), , drop = FALSE]
if (nrow(seasoned)) {
  if (is.na(PRIMARY_SEASON)) {
    log_warning("No v2_primary_season given -> negative_control main panel shows every season (headlines none). Pass the outcome-blind primary season as arg 2.")
  } else if (!PRIMARY_SEASON %in% seasoned$season) {
    log_warning(sprintf("v2_primary_season=%s not found among built seasons {%s}.",
                        PRIMARY_SEASON, paste(sort(unique(seasoned$season)), collapse = ", ")))
  } else {
    log_info(sprintf("negative_control primary season = %s (main panel); %d seasons shown in the consistency figure.",
                     PRIMARY_SEASON, length(unique(seasoned$season))))
  }
}

write.csv(df, file.path(OUT, "supplementary_table.csv"), row.names = FALSE)

panel_of <- function(cohort) ifelse(cohort == "positive_control_celecoxib_gi",
                                    "positive_control: celecoxib->GI (expect RR<1)", "negative_control: flu->hip fx (expect RR~1)")

# --- (1) MAIN forest: positive_control = its estimate; negative_control = the primary season only.
# With no primary set, every negative_control season is shown in the main panel.
# Plots use only estimable rows (flagged cases stay in supplementary_table.csv).
plotdf <- df[!df$separation, , drop = FALSE]
main <- if (is.na(PRIMARY_SEASON)) plotdf else plotdf[plotdf$is_primary, , drop = FALSE]
if (nrow(main)) {
  main$panel <- panel_of(main$cohort)
  main$row <- paste(main$spec, main$estimator, sep = " | ")
  xlab_main <- if (is.na(PRIMARY_SEASON)) "RR (95% CI)  [negative_control = all seasons, no primary set]" else
    sprintf("RR (95%% CI)  [negative_control = primary season %s]", PRIMARY_SEASON)
  p <- ggplot(main, aes(x = rr, y = row, xmin = lb, xmax = ub, color = payer)) +
    geom_pointrange(position = position_dodge(width = 0.5), size = 0.25, na.rm = TRUE) +
    geom_vline(xintercept = 1, lty = 2) +
    facet_wrap(~panel, scales = "free_x") +
    scale_x_log10() + xlab(xlab_main) + ylab("") +
    theme_bw() + theme(axis.text.y = element_text(size = 6))
  ggsave(file.path(OUT, "forest_main.png"), p, width = 12, height = 9)
}

# --- (2) negative_control per-season forest, faceted by payer ---------
if (nrow(seasoned)) {
  neg <- seasoned[!seasoned$separation, , drop = FALSE]   # drop non-estimable
  neg$row <- paste(neg$spec, neg$estimator, sep = " | ")
  pv <- ggplot(neg, aes(x = rr, y = row, xmin = lb, xmax = ub, color = season)) +
    geom_pointrange(position = position_dodge(width = 0.7), size = 0.22, na.rm = TRUE) +
    geom_vline(xintercept = 1, lty = 2) +
    facet_wrap(~payer) +
    scale_x_log10() + xlab("RR (95% CI) -- negative_control flu->hip fx, per season (expect ~1 in every season)") + ylab("") +
    theme_bw() + theme(axis.text.y = element_text(size = 6))
  ggsave(file.path(OUT, "forest_v2_by_season.png"), pv, width = 12, height = 9)
  log_info("Wrote forest_v2_by_season.png (per-season consistency, no pooling)")
}

log_info(sprintf("Wrote supplementary_table.csv + forest_main.png to %s. Log: %s", OUT, log_file))

