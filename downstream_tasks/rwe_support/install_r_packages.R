#!/usr/bin/env Rscript
# CRAN packages for every module, non-interactive. Run from a `conda activate`d shell so the
# env's C++ toolchain is on PATH (nanoparquet and MatchIt compile from source under conda R).
#
#   Rscript install_r_packages.R
#
# survival / MASS / Matrix are NOT listed: they come from conda r-recommended (environment.yml);
# current CRAN Matrix needs R >= 4.4, so CRAN cannot backfill them on R 4.3.
# The mirror is env-driven: default = public CRAN; on a no-egress cluster export
# CRAN_REPO=<cluster-local Posit/CRAN mirror> (see env.sh).
options(repos = c(CRAN = Sys.getenv("CRAN_REPO", "https://cloud.r-project.org")))
pkgs <- c("optparse", "jsonlite", "glue", "dplyr", "tidyr", "data.table",
          "glmnet", "sandwich", "lmtest", "speedglm", "MatchIt", "EmpiricalCalibration",
          "nanoparquet", "ggplot2", "pROC", "autoCovariateSelection")
install.packages(setdiff(pkgs, rownames(installed.packages())))

# verify: a missing package must fail THIS script, not a pipeline stage
bad <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(bad)) stop("FAILED to install: ", paste(bad, collapse = ", "))
cat("All", length(pkgs), "R packages installed and loadable.\n")
