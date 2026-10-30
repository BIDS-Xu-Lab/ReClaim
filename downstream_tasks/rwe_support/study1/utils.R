#!/usr/bin/env Rscript
# utils.R -- shared logging / parquet helpers for the R stages.

#' Set up logging to logs/<script>_<timestamp>.log and the console.
#'
#' Defines log_info / log_warning / log_error / log_debug / log_cat in the global environment.
#'
#' @param script_path Optional path to the script file. If NULL, taken from the command line.
#' @return Character string with path to the log file
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

#' Read a Spark-written parquet dataset into a data.frame.
#'
#' Spark writes a DIRECTORY of `part-*.snappy.parquet` files (plus `_SUCCESS`/`.crc`), not a
#' single file. nanoparquet (a pure-C reader that needs no system libarrow) reads one file at a
#' time, so the parts are row-bound. Accepts a single .parquet file too.
#'
#' @param path Directory of part files (or a single .parquet file)
#' @return data.frame of all rows
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

#' Format a duration in seconds as a readable string (e.g. "1h 30m 45s").
#'
#' @param seconds Numeric seconds to format
#' @return Character string with formatted time
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