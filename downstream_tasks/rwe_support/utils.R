#!/usr/bin/env Rscript


#' Setup logging function
#' 
#' Creates a log file with timestamp and returns the path to the log file.
#' Enables simultaneous output to both console and log file.
#' 
#' @param script_path Optional path to the script file. If NULL, tries to determine from call stack.
#' @return Character string with path to the log file
setup_logging <- function(script_path = NULL) {
  # Create logs directory if it doesn't exist
  if (!dir.exists("logs")) {
    dir.create("logs", showWarnings = FALSE)
  }
  
  # Generate log filename with script name and timestamp
  if (is.null(script_path)) {
    # Extract script name from command arguments
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- args[grep("--file=", args)]
    if (length(file_arg) > 0) {
      script_name <- basename(gsub("--file=", "", file_arg))
      script_name <- sub("\\.R$", "", script_name)
    } else {
      # Fallback if --file not found
      script_name <- "r_script"
    }
  } else {
    script_name <- basename(sub("\\.R$", "", script_path))
  }
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  log_file <- file.path("logs", paste0(script_name, "_", timestamp, ".log"))
  
  # Create a logging function that writes to both console and log file
  log_con <- file(log_file, open = "wt")
  
  # Create a new log function to replace cat and message
  log_message <- function(msg, type = "INFO") {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    formatted_msg <- sprintf("[%s] %s: %s", timestamp, type, msg)
    
    # Write to console
    cat(formatted_msg, "\n")
    
    # Write to log file
    cat(formatted_msg, "\n", file = log_con, append = TRUE)
  }
  
  # Assign to global environment to make available in calling script
  assign("log_info", function(msg) log_message(msg, "INFO"), envir = .GlobalEnv)
  assign("log_warning", function(msg) log_message(msg, "WARNING"), envir = .GlobalEnv)
  assign("log_error", function(msg) log_message(msg, "ERROR"), envir = .GlobalEnv)
  assign("log_debug", function(msg) log_message(msg, "DEBUG"), envir = .GlobalEnv)
  
  # Also redefine cat for this session to use our logging
  assign("log_cat", function(...) {
    msg <- paste0(...)
    log_message(msg)
  }, envir = .GlobalEnv)
  
  # Register cleanup to close the connections when the R session ends
  reg.finalizer(.GlobalEnv, function(e) {
    if (exists("log_con") && isOpen(log_con)) {
      close(log_con)
    }
  }, onexit = TRUE)
  
  # Print log header
  log_info(paste0("Log started at: ", Sys.time()))
  log_info(paste0("Script: ", script_name))
  log_info(paste0("Log file: ", log_file))
  log_info(paste0("System: ", Sys.info()["sysname"], " ", Sys.info()["release"]))
  log_info(paste0("R version: ", R.version.string))
  log_info(paste0(rep("-", 80), collapse = ""))
  
  # Return log file name
  return(log_file)
}

#' Function to format time delta
#' 
#' Formats a time duration in seconds to a readable string (e.g., "1h 30m 45s").
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

#' Time execution decorator equivalent (higher-order function)
#' 
#' Returns a function that wraps the input function with timing code.
#' 
#' @param func Function to time
#' @return Function that executes the input function and logs execution time
time_execution <- function(func) {
  function(...) {
    start_time <- Sys.time()
    result <- func(...)
    end_time <- Sys.time()
    execution_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    formatted_time <- format_time_delta(execution_time)
    func_name <- deparse(substitute(func))
    message(glue::glue("Function '{func_name}' executed in {formatted_time}"))
    return(result)
  }
} 