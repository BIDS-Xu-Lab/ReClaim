#!/usr/bin/env Rscript
# utils.R -- shared R helpers: logging, Spark-parquet reading, package self-heal.

#' Create logs/<script>_<timestamp>.log and define log_info / log_warning / log_error /
#' log_debug / log_cat in the global environment (console + file). Returns the log path.
setup_logging <- function(script_path = NULL) {
  if (!dir.exists("logs")) {
    dir.create("logs", showWarnings = FALSE)
  }
  if (is.null(script_path)) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- args[grep("--file=", args)]
    if (length(file_arg) > 0) {
      script_name <- basename(gsub("--file=", "", file_arg))
      script_name <- sub("\\.R$", "", script_name)
    } else {
      script_name <- "r_script"
    }
  } else {
    script_name <- basename(sub("\\.R$", "", script_path))
  }

  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  log_file <- file.path("logs", paste0(script_name, "_", timestamp, ".log"))
  log_con <- file(log_file, open = "wt")

  log_message <- function(msg, type = "INFO") {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    formatted_msg <- sprintf("[%s] %s: %s", timestamp, type, msg)
    cat(formatted_msg, "\n")
    cat(formatted_msg, "\n", file = log_con, append = TRUE)
  }

  assign("log_info", function(msg) log_message(msg, "INFO"), envir = .GlobalEnv)
  assign("log_warning", function(msg) log_message(msg, "WARNING"), envir = .GlobalEnv)
  assign("log_error", function(msg) log_message(msg, "ERROR"), envir = .GlobalEnv)
  assign("log_debug", function(msg) log_message(msg, "DEBUG"), envir = .GlobalEnv)
  assign("log_cat", function(...) {
    msg <- paste0(...)
    log_message(msg)
  }, envir = .GlobalEnv)

  reg.finalizer(.GlobalEnv, function(e) {
    if (exists("log_con") && isOpen(log_con)) {
      close(log_con)
    }
  }, onexit = TRUE)

  log_info(paste0("Log started at: ", Sys.time()))
  log_info(paste0("Script: ", script_name))
  log_info(paste0("Log file: ", log_file))
  log_info(paste0("System: ", Sys.info()["sysname"], " ", Sys.info()["release"]))
  log_info(paste0("R version: ", R.version.string))
  log_info(paste0(rep("-", 80), collapse = ""))
  return(log_file)
}

#' Read a Spark-written parquet dataset (a DIRECTORY of part-*.parquet files) into a data.frame.
#' nanoparquet is used rather than arrow: this R env has no system libarrow, and nanoparquet
#' is a small pure-C reader that handles snappy. It reads one file at a time, so the parts
#' are row-bound. Accepts a single .parquet file too.
read_spark_parquet <- function(path) {
  if (dir.exists(path)) {
    parts <- list.files(path, pattern = "\\.parquet$", full.names = TRUE)
    parts <- parts[!startsWith(basename(parts), ".")]   # skip hidden .crc sidecars
    if (length(parts) == 0L) stop(sprintf("no .parquet part files under %s", path))
    parts <- sort(parts)
    dfs <- lapply(parts, nanoparquet::read_parquet)
    do.call(rbind, dfs)
  } else {
    nanoparquet::read_parquet(path)
  }
}

#' Format a duration in seconds as "1h 30m 45s".
format_time_delta <- function(seconds) {
  hours <- floor(seconds / 3600)
  remainder <- seconds %% 3600
  minutes <- floor(remainder / 60)
  seconds <- remainder %% 60

  if (hours > 0) {
    return(glue::glue("{hours}h {minutes}m {floor(seconds)}s"))
  } else if (minutes > 0) {
    return(glue::glue("{minutes}m {floor(seconds)}s"))
  } else {
    return(glue::glue("{round(seconds, 2)}s"))
  }
}
# NULL-coalescing shim: %||% entered base R in 4.4.0, but this env pins R 4.3.1
# (environment.yml; current CRAN Matrix needs R >= 4.4).
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# Dependency guard: install a missing CRAN package up-front instead of dying on library()
# mid-pipeline. On a no-egress cluster export CRAN_REPO=<local Posit/CRAN mirror> (env.sh);
# compiled packages (MatchIt, nanoparquet) need a `conda activate`d shell so the env's
# toolchain is on PATH. survival/MASS/Matrix are conda-provided and never auto-installed
# (current CRAN Matrix needs R >= 4.4): if one is missing the env is broken.
.CONDA_ONLY_PKGS <- c("survival", "MASS", "Matrix")
ensure_packages <- function(pkgs) {
  miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(miss)) return(invisible())
  conda_miss <- intersect(miss, .CONDA_ONLY_PKGS)
  if (length(conda_miss))
    stop("conda-provided package(s) missing: ", paste(conda_miss, collapse = ", "),
         " -- broken env; rebuild from the repo-level environment.yml (do NOT CRAN-install these)")
  message("ensure_packages: installing missing CRAN package(s): ", paste(miss, collapse = ", "))
  install.packages(miss, repos = Sys.getenv("CRAN_REPO", "https://cloud.r-project.org"))
  bad <- miss[!vapply(miss, requireNamespace, logical(1), quietly = TRUE)]
  if (length(bad))
    stop("could not install: ", paste(bad, collapse = ", "),
         if (!nzchar(Sys.getenv("CRAN_REPO")))
           " (on a no-egress cluster export CRAN_REPO=<local Posit/CRAN mirror> first; see env.sh)"
         else "")
  invisible()
}
