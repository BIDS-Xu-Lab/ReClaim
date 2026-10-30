#!/usr/bin/env Rscript
# summarize_expert_balance.R -- collect every <comparison>/<spec>/<estimator>/balance_expert_smd.csv
# under a results root into one table: per comparison x spec x estimator, the number of expert-set
# terms, max |SMD| and the share of terms with |SMD| < 0.1, before and after adjustment.
# Usage : Rscript tm3_estimate/summarize_expert_balance.R <results_root> <out_csv>
args <- commandArgs(trailingOnly = TRUE)
root <- args[1]; out_csv <- args[2]
files <- list.files(root, pattern = "^balance_expert_smd\\.csv$", recursive = TRUE, full.names = TRUE)
if (!length(files)) stop(sprintf("no balance_expert_smd.csv under %s", root))
rows <- lapply(files, function(f) {
  p <- strsplit(sub(paste0("^", normalizePath(root), "/"), "", normalizePath(f)), "/")[[1]]
  b <- read.csv(f)
  data.frame(comparison = p[length(p) - 3], spec = p[length(p) - 2], estimator = p[length(p) - 1],
             n_terms = nrow(b),
             max_abs_smd_before = max(abs(b$smd_before)), share_lt_0.1_before = mean(abs(b$smd_before) < 0.1),
             max_abs_smd_after = max(abs(b$smd_after)), share_lt_0.1_after = mean(abs(b$smd_after) < 0.1),
             n_gt_0.1_after = sum(abs(b$smd_after) > 0.1))
})
tb <- do.call(rbind, rows)
tb <- tb[order(tb$comparison, tb$estimator, match(tb$spec, c("crude", "demo", "paper", "hdps",
          "reclaim-s-only", "reclaim-m-only", "reclaim-l-only"))), ]
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
write.csv(tb, out_csv, row.names = FALSE)
cat(sprintf("wrote %s (%d rows)\n", out_csv, nrow(tb)))
