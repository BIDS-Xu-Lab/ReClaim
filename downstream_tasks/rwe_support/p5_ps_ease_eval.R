#!/usr/bin/env Rscript

# # Install R and all required packages via conda-forge first in the conda environment
# conda install -c conda-forge -y \
#     r-base \
#     r-data.table \
#     r-dplyr \
#     r-stringr \
#     r-lubridate \
#     r-tidyr \
#     r-rlang \
#     r-janitor \
#     r-matchit \
#     r-scales \
#     r-ggplot2 \
#     r-optparse \
#     r-jsonlite \
#     r-glmnet \
#     r-sandwich \
#     r-lmtest

# # EmpiricalCalibration may not be on conda-forge, install via R
# Rscript -e "install.packages('EmpiricalCalibration', repos='https://cloud.r-project.org')"

# ============================================================================
# PACKAGE LOADING (no installation at runtime)
# ============================================================================
# Prerequisites: install packages in your conda environment before running the script.

#!/usr/bin/env Rscript

# EASE Score Calculator Function
# Adapted for HPC environment

# ============================================================================
# PACKAGE LOADING (no installation at runtime)
# ============================================================================
# Prerequisites: Run setup_r_env.sh once on login node to install packages
# in your conda environment before submitting jobs.

# Source logging utilities
source("utils.R")

# Setup logging
log_file <- setup_logging()

# List of required packages
packages <- c(
  "data.table", "dplyr", "stringr", "lubridate", 
  "tidyr", "rlang", "janitor", "MatchIt", "scales",
  "ggplot2", "EmpiricalCalibration", "optparse", "jsonlite",
  "glmnet",    # Required for MatchIt with distance = "lasso"
  "sandwich",  # Robust variance estimators (HC0) for Poisson regression
  "lmtest"     # coeftest() for applying robust SEs
)

# Simple load function - fails fast if packages missing
load_packages <- function(packages) {
  missing <- c()
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      missing <- c(missing, pkg)
    } else {
      library(pkg, character.only = TRUE, quietly = TRUE)
      log_info(sprintf("Loaded package: %s", pkg))
    }
  }
  
  if (length(missing) > 0) {
    stop(paste(
      "Missing packages:", paste(missing, collapse = ", "),
      "\nPlease install them in your conda environment before running.",
      "\nRun: conda install -c conda-forge", 
      paste0("r-", tolower(missing), collapse = " ")
    ))
  }
}

# Load all packages (will fail immediately if any missing)
load_packages(packages)

# Rest of your script continues here...

#' Compute Preference Score from Propensity Score
#' @param data Data frame with propensityScore column
#' @param unfilteredData Optional unfiltered data for proportion calculation
#' @return Data frame with preferenceScore column added
computePreferenceScore <- function(data, unfilteredData = NULL) {
  if (is.null(unfilteredData)) {
    proportion <- sum(data$treatment) / nrow(data)
  } else {
    proportion <- sum(unfilteredData$treatment) / nrow(unfilteredData)
  }
  
  propensityScore <- data$propensityScore
  propensityScore[propensityScore > 0.9999999] <- 0.9999999
  x <- exp(log(propensityScore / (1 - propensityScore)) - log(proportion / (1 - proportion)))
  data$preferenceScore <- x / (x + 1)
  return(data)
}

#' Select top-k Negative Control Outcomes (NCO) by prevalence
#'
#' Finds columns matching a pattern (default: "^NCO_binary_") in the input
#' data, computes each column's prevalence as mean(x == 1, na.rm = TRUE), logs
#' the sorted prevalence list (descending), and returns the top-k column names.
#'
#' @param data Data frame containing NCO columns
#' @param pattern Regular expression to select NCO variables (default: "^NCO_binary_")
#' @param k Number of top NCO variables to return (default: 30)
#' @param low_prevalence_threshold Drop NCOs with prevalence < threshold (default: 0.01)
#' @return Character vector of selected NCO column names (length ≤ k)
select_top_nco_variables <- function(data, pattern = "^NCO_binary_", k = 30, low_prevalence_threshold = 0.005) {
  nco_cols <- grep(pattern, colnames(data), value = TRUE)
  if (length(nco_cols) == 0) {
    log_warning("No NCO variables found matching pattern")
    return(character(0))
  }
  prevalences <- sapply(nco_cols, function(col) {
    x <- data[[col]]
    mean(x == 1, na.rm = TRUE)
  })
  prevalences[is.na(prevalences)] <- 0
  # Filter low-prevalence NCOs first
  keep_idx <- which(prevalences >= low_prevalence_threshold)
  if (length(keep_idx) == 0) {
    log_warning(paste("All NCOs filtered out with prevalence <",
                      paste0(round(low_prevalence_threshold * 100, 2), "%")))
    return(character(0))
  }
  if (length(keep_idx) < length(prevalences)) {
    log_info(paste(
      "Filtered out", length(prevalences) - length(keep_idx),
      "low-prevalence NCOs (<",
      paste0(round(low_prevalence_threshold * 100, 2), "%"), ")"
    ))
  }
  nco_cols <- nco_cols[keep_idx]
  prevalences <- prevalences[keep_idx]
  ord <- order(prevalences, decreasing = TRUE)
  sorted_nco <- nco_cols[ord]
  sorted_prev <- prevalences[ord]
  log_info("Sorted NCO prevalences (desc):")
  for (i in seq_along(sorted_nco)) {
    log_info(paste0(i, ". ", sorted_nco[i], ": ", round(sorted_prev[i] * 100, 3), "%"))
  }
  top_k <- head(sorted_nco, k)
  log_info(paste("Selected top", length(top_k), "NCO variables for EASE."))
  return(top_k)
}

#' Select top-k Primary Outcomes by prevalence
#'
#' Finds columns matching a pattern (default: "^visits_binary_") in the input
#' data, computes each column's prevalence as mean(x == 1, na.rm = TRUE), logs
#' the sorted prevalence list (descending), and returns the top-k column names.
#'
#' @param data Data frame containing primary outcome columns
#' @param pattern Regular expression to select outcome variables (default: "^visits_binary_")
#' @param k Number of top outcome variables to return (default: 10)
#' @param low_prevalence_threshold Drop outcomes with prevalence < threshold (default: 0.01)
#' @return Character vector of selected outcome column names (length ≤ k)
select_top_primary_outcomes <- function(data, pattern = "^visits_binary_", k = 30, low_prevalence_threshold = 0.005) {
  outcome_cols <- grep(pattern, colnames(data), value = TRUE)
  if (length(outcome_cols) == 0) {
    log_warning("No primary outcome variables found matching pattern")
    return(character(0))
  }
  prevalences <- sapply(outcome_cols, function(col) {
    x <- data[[col]]
    mean(x == 1, na.rm = TRUE)
  })
  prevalences[is.na(prevalences)] <- 0
  # Filter low-prevalence outcomes first
  keep_idx <- which(prevalences >= low_prevalence_threshold)
  if (length(keep_idx) == 0) {
    log_warning(paste("All primary outcomes filtered out with prevalence <",
                      paste0(round(low_prevalence_threshold * 100, 2), "%")))
    return(character(0))
  }
  if (length(keep_idx) < length(prevalences)) {
    log_info(paste(
      "Filtered out", length(prevalences) - length(keep_idx),
      "low-prevalence primary outcomes (<",
      paste0(round(low_prevalence_threshold * 100, 2), "%"), ")"
    ))
  }
  outcome_cols <- outcome_cols[keep_idx]
  prevalences <- prevalences[keep_idx]
  ord <- order(prevalences, decreasing = TRUE)
  sorted_outcomes <- outcome_cols[ord]
  sorted_prev <- prevalences[ord]
  log_info("Sorted primary outcome prevalences (desc):")
  for (i in seq_along(sorted_outcomes)) {
    log_info(paste0(i, ". ", sorted_outcomes[i], ": ", round(sorted_prev[i] * 100, 3), "%"))
  }
  top_k <- head(sorted_outcomes, k)
  log_info(paste("Selected top", length(top_k), "primary outcome variables."))
  return(top_k)
}

#' Calculate EASE Score with Propensity Score Matching
#' 
#' This function performs the computationally expensive parts that are shared
#' across all primary outcomes: propensity score matching, NCO analysis, and EASE calculation.
#' 
#' @param data Input data frame
#' @param treatment_var Name of treatment variable (should be 0/1)
#' @param xvars Vector of covariate names for propensity score matching
#' @param nco_outcomes Vector of negative control outcome variable names
#' @param caliper Caliper for matching (default: 0.2)
#' @param ratio Matching ratio (default: 1)
#' @param distance Distance method for matching (default: "lasso")
#' @param output_dir Directory to save plots and results (optional)
#' @param save_plots Whether to save diagnostic plots (default: FALSE)
#' 
#' @return List containing:
#'   - ease_score: The calculated EASE score
#'   - matched_data: The matched dataset
#'   - nco_results: Results for negative control outcomes
#'   - valid_nco: Filtered valid NCO results
#'   - matching_summary: Summary of matching procedure
#'   - equipoise_proportion: Proportion in equipoise
#'   - systematic_error_model: Model for empirical calibration
#'   
calculate_ease_score <- function(data, 
                                treatment_var, 
                                xvars, 
                                nco_outcomes,
                                caliper = 0.2,
                                ratio = 1,
                                distance = "lasso",
                                output_dir = NULL,
                                save_plots = FALSE) {
  
  # Validate inputs
  if (!treatment_var %in% colnames(data)) {
    stop(paste("Treatment variable", treatment_var, "not found in data"))
  }
  if (!all(xvars %in% colnames(data))) {
    missing_vars <- xvars[!xvars %in% colnames(data)]
    stop(paste("Covariate(s) not found in data:", paste(missing_vars, collapse = ", ")))
  }
  if (!all(nco_outcomes %in% colnames(data))) {
    missing_nco <- nco_outcomes[!nco_outcomes %in% colnames(data)]
    stop(paste("NCO variable(s) not found in data:", paste(missing_nco, collapse = ", ")))
  }
  
  # Rename treatment variable to 'treatment' for consistency
  data_work <- data
  data_work$treatment <- data_work[[treatment_var]]
  
  # Remove low-incidence variables (< 1% prevalence) - only for binary variables
  prevalence_threshold <- 0.01
  # Convert to data.frame for consistent column selection
  if (is.data.table(data_work)) {
    xvar_data <- data_work[, ..xvars]
  } else {
    xvar_data <- data_work[xvars]
  }

  
  # Only check prevalence for binary/numeric variables that could be low-prevalence indicators
  low_prevalence <- sapply(xvar_data, function(x) {
    if (is.numeric(x) && all(x %in% c(0, 1, NA), na.rm = TRUE)) {
      # Binary variable - check prevalence
      return(sum(x, na.rm = TRUE) / length(x) < prevalence_threshold)
    } else {
      # Non-binary variable - keep it
      return(FALSE)
    }
  })
  
  xvars_filtered <- xvars[!low_prevalence]
  
  if (length(xvars_filtered) != length(xvars)) {
    log_info(paste("Removed", length(xvars) - length(xvars_filtered), "low-prevalence variables"))
  }
  
  # Create propensity score formula
  formula_prop <- as.formula(paste0("treatment ~ ", paste0(xvars_filtered, collapse = " + ")))
  
  # Perform propensity score matching
  set.seed(1)

  match_out <- matchit(formula_prop, 
                      data = data_work,
                      caliper = caliper,
                      method = "nearest",
                      distance = distance,
                      ratio = ratio,
                      verbose = FALSE,
                      estimand = "ATT",
                      distance.options = list(s = "lambda.min"))
  
  # Add propensity and preference scores
  data_work$propensityScore <- match_out$distance
  data_work <- computePreferenceScore(data_work)
  
  # Diagnostic: propensity score range
  log_info(sprintf("PS range: %.4f to %.4f",
    min(data_work$propensityScore, na.rm = TRUE),
    max(data_work$propensityScore, na.rm = TRUE)))
  
  # Calculate equipoise proportion
  equipoiseBounds <- c(0.3, 0.7)
  equipoise_proportion <- mean(data_work$preferenceScore >= equipoiseBounds[1] & 
                              data_work$preferenceScore <= equipoiseBounds[2], na.rm = TRUE)
  
  # Get matched data
  matched_data <- match.data(match_out)
  
  # Analyze negative control outcomes
  nco_results <- data.frame()
  

  for (nco in nco_outcomes) {
    if (sum(matched_data[[nco]], na.rm = TRUE) > 0) {  # Only analyze if there are events
      nco_formula <- as.formula(paste0(nco, " ~ treatment"))
      
      tryCatch({
        nco_model <- glm(nco_formula, family = poisson(link = "log"), data = matched_data)
        # Use sandwich robust SEs (HC0) — model-based SEs from Poisson on
        # binary outcomes are typically inflated, which causes fitNull to
        # over-attribute variance to sampling noise
        robust_nco <- lmtest::coeftest(nco_model, vcov = sandwich::vcovHC(nco_model, type = "HC0"))
        
        nco_result <- data.frame(
          nco_name = nco,
          logRR = robust_nco[2, 1],
          seLogRR = robust_nco[2, 2]
        )
        nco_results <- rbind(nco_results, nco_result)
        }, error = function(e) {
          log_warning(paste("Could not fit model for NCO", nco, ":", e$message))
        })
    }
  }
  
  # Remove only non-finite regression results (Inf/NaN from failed fits).
  # No arbitrary thresholds on logRR or seLogRR: the NCO set is fixed by
  # select_top_nco_variables() (prevalence-based, model-independent), and
  # fitNull is likelihood-based — it naturally down-weights noisy NCOs
  # (large SE → flat likelihood contribution) so explicit filtering is
  # redundant and would make the NCO set model-dependent.
  valid_nco <- nco_results[is.finite(nco_results$logRR) & is.finite(nco_results$seLogRR) &
                           nco_results$seLogRR > 0, ]
  
  # Diagnostic logging for NCO analysis
  log_info(sprintf("N total NCO results: %d", nrow(nco_results)))
  log_info(sprintf("N valid NCOs (finite & SE > 0): %d", nrow(valid_nco)))
  if (nrow(valid_nco) > 0) {
    log_info(sprintf("logRR range: %.4f to %.4f", min(valid_nco$logRR), max(valid_nco$logRR)))
    log_info(sprintf("seLogRR quantiles (10/50/90%%): %.4f, %.4f, %.4f",
      quantile(valid_nco$seLogRR, 0.1),
      quantile(valid_nco$seLogRR, 0.5),
      quantile(valid_nco$seLogRR, 0.9)))
  }
  if (nrow(nco_results) > nrow(valid_nco)) {
    log_warning(sprintf("Dropped %d NCOs with non-finite logRR/seLogRR or SE <= 0",
      nrow(nco_results) - nrow(valid_nco)))
  }
  
  # Calculate EASE score using EmpiricalCalibration package
  if (nrow(valid_nco) > 0) {
    # Capture return value from tryCatch to avoid scoping bug in error handler
    ease_score <- tryCatch({
      fitnull <- fitNull(valid_nco$logRR, valid_nco$seLogRR)
      score <- computeExpectedAbsoluteSystematicError(fitnull)
      # Log fitNull parameters; wrap separately so a logging failure can't
      # prevent the already-computed score from being returned.
      tryCatch({
        log_info(sprintf("fitNull: mu = %.6f, sigma = %.6f", fitnull$mean, fitnull$sd))
      }, error = function(e2) {
        log_info(sprintf("fitNull completed (could not extract mu/sigma for logging: %s)", e2$message))
      })
      score
    }, error = function(e) {
      log_warning(paste("fitNull failed, falling back to mean(|logRR|):", e$message))
      mean(abs(valid_nco$logRR))
    })
  } else {
    ease_score <- NA
    warning("No valid negative control outcomes for EASE calculation")
  }
  
  # Log EASE with full precision to distinguish true zero from rounding artifact
  log_info(sprintf("EASE score = %.10f", ifelse(is.na(ease_score), NaN, ease_score)))
  
  # Prepare systematic error model for empirical calibration
  # Use convertNullToErrorModel (standard pathway for NCO-only calibration)
  systematic_error_model <- NULL
  if (nrow(valid_nco) > 0) {
    systematic_error_model <- tryCatch({
      fitnull <- fitNull(valid_nco$logRR, valid_nco$seLogRR)
      convertNullToErrorModel(fitnull)
    }, error = function(e) {
      log_warning(paste("Could not fit systematic error model:", e$message))
      NULL
    })
  }
  
  # Save plots if requested
  if (save_plots && !is.null(output_dir)) {
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    # Equipoise plot with enhanced functionality
    tryCatch({
      # Ensure propensity scores are capped at 0.9999999 for preference score calculation
      data_work_plot <- data_work
      data_work_plot$propensityScore[data_work_plot$propensityScore > 0.9999999] <- 0.9999999
      data_work_plot <- computePreferenceScore(data_work_plot)
      
      # Define equipoise bounds
      equipoiseBounds <- c(0.3, 0.7)
      
      # Calculate equipoise proportion
      equipoise <- mean(data_work_plot$preferenceScore >= equipoiseBounds[1] & 
                       data_work_plot$preferenceScore <= equipoiseBounds[2], na.rm = TRUE)
      
      # Create equipoise plot with area shading
      equipoise_plot <- plotPs_mis(data_work_plot, 
                                  scale = "preference",
                                  showCountsLabel = TRUE, 
                                  showAucLabel = TRUE, 
                                  showEquiposeLabel = TRUE,
                                  equipoiseBounds = equipoiseBounds)
      
      # Save plot
      ggsave(filename = file.path(output_dir, "equipoise.png"), 
             plot = equipoise_plot, width = 6, height = 6)
      
      # Log equipoise value
      log_info(paste("Equipoise proportion:", round(equipoise * 100, 2), "%"))
      
    }, error = function(e) {
      cat("Warning: Could not create equipoise plot:", e$message, "\n")
    })
    
    # NCO forest plot
    if (nrow(nco_results) > 0) {
      tryCatch({
        nco_plot_data <- nco_results
        nco_plot_data$ll <- nco_plot_data$logRR - 1.96 * nco_plot_data$seLogRR
        nco_plot_data$ul <- nco_plot_data$logRR + 1.96 * nco_plot_data$seLogRR
        
        nco_forest <- ggplot(nco_plot_data, aes(x = logRR, y = nco_name, xmin = ll, xmax = ul)) +
          geom_pointrange(size = 0.2) +
          geom_vline(xintercept = 0, lty = 2) +
          ylab("") + xlab("logRR (95% CI)") +
          theme_bw() +
          theme(legend.position = "none")
        
        ggsave(filename = file.path(output_dir, "nco_forest.png"), 
               plot = nco_forest, width = 8, height = max(4, nrow(nco_results) * 0.3))
      }, error = function(e) {
        cat("Warning: Could not create NCO forest plot:", e$message, "\n")
      })
    }
    
    # EASE calibration plot
    if (nrow(valid_nco) > 0) {
      tryCatch({
        fitnull <- fitNull(valid_nco$logRR, valid_nco$seLogRR)
        ease_plot <- plotCalibrationEffect(logRrNegatives = valid_nco$logRR,
                                          seLogRrNegatives = valid_nco$seLogRR,
                                          showExpectedAbsoluteSystematicError = TRUE,
                                          null = fitnull)
        ## explicitly increase the size of the annotation text to 6
        ease_annotation_size <- 5
        ease_plot$layers <- lapply(ease_plot$layers, function(layer) {
          if (inherits(layer$geom, "GeomText") || inherits(layer$geom, "GeomLabel")) {
            layer$aes_params$size <- ease_annotation_size
          }
          layer
        })
        ggsave(filename = file.path(output_dir, "ease.png"), 
               plot = ease_plot, width = 6, height = 4)
      }, error = function(e) {
        cat("Warning: Could not create EASE plot:", e$message, "\n")
      })
    }
  }
  
  # Return shared analysis results
  return(list(
    ease_score = ease_score,
    matched_data = matched_data,
    nco_results = nco_results,
    valid_nco = valid_nco,
    matching_summary = summary(match_out),
    equipoise_proportion = equipoise_proportion,
    systematic_error_model = systematic_error_model
  ))
}

#' Analyze Individual Primary Outcome Using Shared Results
#' 
#' This function analyzes a single primary outcome using the pre-computed
#' shared analysis results (matching, EASE score, etc.)
#' 
#' @param shared_results Results from calculate_ease_score()
#' @param outcome_var Name of primary outcome variable
#' @param output_dir Directory to save results (optional)
#' 
#' @return List containing:
#'   - results: Data frame with detailed results for this outcome
#'   - calibrated_results: Empirically calibrated results (if available)
#'   
analyze_primary_outcome <- function(shared_results, outcome_var, output_dir = NULL) {
  
  matched_data <- shared_results$matched_data
  
  # Validate outcome variable exists
  if (!outcome_var %in% colnames(matched_data)) {
    stop(paste("Outcome variable", outcome_var, "not found in matched data"))
  }
  
  # Calculate outcome rates
  r_0 <- sum(matched_data[[outcome_var]][matched_data$treatment == 0])
  r_1 <- sum(matched_data[[outcome_var]][matched_data$treatment == 1])
  n_0 <- sum(matched_data$treatment == 0)
  n_1 <- sum(matched_data$treatment == 1)
  
  # Fit outcome model with robust (sandwich) standard errors
  outcome_formula <- as.formula(paste0(outcome_var, " ~ treatment"))
  outcome_model <- glm(outcome_formula, family = poisson(link = "log"), data = matched_data)
  
  # Use sandwich robust SEs (HC0) — consistent with NCO analysis
  robust_outcome <- lmtest::coeftest(outcome_model, vcov = sandwich::vcovHC(outcome_model, type = "HC0"))
  logRR <- robust_outcome[2, 1]
  seLogRR <- robust_outcome[2, 2]
  
  # Empirical calibration (if systematic error model is available)
  calibrated_results <- NULL
  if (!is.null(shared_results$systematic_error_model)) {
    tryCatch({
      calibrated_results <- calibrateConfidenceInterval(
        logRr = logRR,
        seLogRr = seLogRR,
        shared_results$systematic_error_model, 
        ciWidth = 0.95
      )
    }, error = function(e) {
      log_warning(paste("Could not perform empirical calibration for", outcome_var, ":", e$message))
    })
  }
  
  # Compile results with clear separation of original vs calibrated
  results <- list(
    # Basic outcome information
    outcome = outcome_var,
    n_treated = n_1,
    n_control = n_0,
    events_treated = r_1,
    events_control = r_0,
    
    # Shared analysis results (same for all outcomes)
    ease_score = shared_results$ease_score,
    n_valid_nco = nrow(shared_results$valid_nco),
    equipoise_proportion = shared_results$equipoise_proportion,
    
    # Original (uncalibrated) results
    original_results = list(
      logRR = logRR,
      seLogRR = seLogRR,
      RR = exp(logRR),
      RR_LB = exp(logRR - 1.96 * seLogRR),
      RR_UB = exp(logRR + 1.96 * seLogRR)
    )
  )
  
  # Add calibrated results if available (for comparison with original)
  if (!is.null(calibrated_results)) {
    results$calibrated_results = list(
      logRR = calibrated_results$logRr,
      seLogRR = calibrated_results$seLogRr,
      RR = exp(calibrated_results$logRr),
      RR_LB = exp(calibrated_results$logRr - 1.96 * calibrated_results$seLogRr),
      RR_UB = exp(calibrated_results$logRr + 1.96 * calibrated_results$seLogRr)
    )
  } else {
    results$calibrated_results = NULL
  }
  
  # Save results if output directory specified
  if (!is.null(output_dir)) {
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    # Save results as JSON
    tryCatch({
      outcome_results <- list(
        results = results,
        calibrated_results = calibrated_results
      )
      json_path <- file.path(output_dir, "outcome_results.json")
      jsonlite::write_json(outcome_results, path = json_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
      log_info(paste("Outcome results saved to:", json_path))
    }, error = function(e) {
      log_warning(paste("Could not save outcome results:", e$message))
    })
  }
  
  return(results)
}

# Helper function from original code for plotting (simplified version)
plotPs_mis <- function(data, scale = "preference", showCountsLabel = FALSE, 
                      showAucLabel = FALSE, showEquiposeLabel = FALSE,
                      targetLabel = "Target", comparatorLabel = "Comparator",
                      equipoiseBounds = c(0.3, 0.7)) {
  
  if (scale == "preference") {
    data$score <- data$preferenceScore
    label <- "Preference score"
  } else {
    data$score <- data$propensityScore
    label <- "Propensity score"
  }
  
  # Calculate equipoise proportion
  equipoise <- mean(data$score >= equipoiseBounds[1] & data$score <= equipoiseBounds[2], na.rm = TRUE)
  
  d1 <- density(data$score[data$treatment == 1], from = 0, to = 1, n = 200)
  d0 <- density(data$score[data$treatment == 0], from = 0, to = 1, n = 200)
  d <- data.frame(x = c(d1$x, d0$x), 
                  y = c(d1$y, d0$y), 
                  treatment = c(rep(targetLabel, length(d1$x)),
                               rep(comparatorLabel, length(d0$x))))
  d$treatment <- factor(d$treatment, levels = c(targetLabel, comparatorLabel))
  
  plot <- ggplot(d, aes(x = x, y = y)) +
    geom_density(stat = "identity", aes(color = treatment, group = treatment, fill = treatment)) +
    scale_fill_manual(values = c(rgb(0.8, 0, 0, alpha = 0.5), rgb(0, 0, 0.8, alpha = 0.5))) +
    scale_color_manual(values = c(rgb(0.8, 0, 0, alpha = 0.5), rgb(0, 0, 0.8, alpha = 0.5))) +
    scale_x_continuous(label, limits = c(0, 1)) +
    scale_y_continuous("Density") +
    theme(legend.title = element_blank(), legend.position = "top")
  
  # Add equipoise area shading
  if (showEquiposeLabel) {
    # Add vertical lines for equipoise bounds
    plot <- plot + 
      geom_vline(xintercept = equipoiseBounds[1], linetype = "dashed", color = "gray50", alpha = 0.8) +
      geom_vline(xintercept = equipoiseBounds[2], linetype = "dashed", color = "gray50", alpha = 0.8) +
      # Add shaded area for equipoise region
      annotate("rect", xmin = equipoiseBounds[1], xmax = equipoiseBounds[2], 
               ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "green") +
      # Add equipoise label
      annotate("text", x = 0.02, y = Inf, vjust = 1.2, hjust = 0,
               label = paste0("Equipoise: ", round(equipoise * 100, 1), "%"),
               size = 3.5, color = "black")
  }
  
  return(plot)
}

 

# Function to run EASE analysis on a dataset with multiple outcomes
run_ease_analysis <- function(input_file, analysis_name, base_output_dir, primary_outcomes, nco_outcomes, treatment = "glp1", control = "dpp4") {
  log_info(paste("=== Starting", analysis_name, "analysis ==="))
  log_info(paste("Input file:", input_file))
  
  # Check if file exists
  if (!file.exists(input_file)) {
    log_error(paste("Input file not found:", input_file))
    return(NULL)
  }
  
  data = read.csv(input_file)
  log_info(paste("Loaded data with", nrow(data), "rows and", ncol(data), "columns"))

  # log_info("Subset data on only the CCAE and Medicare data...")
  # data <- data %>%
  #   filter(grepl("^[CR]", pid))

  # log_info(paste("CCAE and Medicare patients:", nrow(data)))


  data <- data %>%
    rename(
      medication_type = drug_name,
      medication_type_class = drug_class
    )

  treatment_var <- "treatment"  # 0/1 treatment variable

  data = data %>% filter(medication_type_class %in% c(treatment, control))
  data$treatment = data$medication_type_class == treatment

  xvars = c(
          "age_at_entry",
          "gender", 
          # "demo_plantyp",
          # "demo_eeclass",
          # "demo_eestatu",
          #"demo_indstry",
          colnames(data%>%dplyr::select(starts_with('medical_'))),
          colnames(data%>%dplyr::select(starts_with('med_'))), 
          colnames(data%>%dplyr::select(starts_with('rep_'))) # representation columns
  )

  # # Filter out rows where any xvar is NA or "MISSING"
  # log_info("Filtering data to remove rows with NA or 'MISSING' values in covariates...")
  # original_count <- nrow(data)
  
  # data <- data %>%
  #   filter(if_all(all_of(xvars), ~ !is.na(.) & . != "MISSING"))
  
  # filtered_count <- nrow(data)
  # excluded_count <- original_count - filtered_count
  
  # log_info(paste("Original data size:", original_count))
  # log_info(paste("Filtered data size:", filtered_count))
  # log_info(paste("Excluded rows:", excluded_count, "(" , round(excluded_count/original_count*100, 2), "%)"))

  
  
  
  # Log analysis details
  log_info(paste("Treatment variable:", treatment_var))
  log_info(paste("Number of covariates (X vars):", length(xvars)))
  log_info(paste("Number of primary outcomes:", length(primary_outcomes)))
  log_info(paste("Number of NCO outcomes:", length(nco_outcomes)))
  log_info(paste("Number of representation features:", sum(grepl("^rep_", xvars))))

  log_info("Starting shared analysis (matching and EASE calculation)...")

  # Perform shared analysis once
  timed_calculate_ease_score <- time_execution(calculate_ease_score)
  
  # All results go directly under base_output_dir (no shared_analysis subfolder)
  shared_results <- timed_calculate_ease_score(
    data = data,
    treatment_var = treatment_var,
    xvars = xvars,
    nco_outcomes = nco_outcomes,
    output_dir = base_output_dir,
    save_plots = TRUE
  )

  # Log shared results
  log_info("Shared Analysis Results:")
  log_info(paste("EASE Score:", round(shared_results$ease_score, 4)))
  log_info(paste("Number of matched pairs:", nrow(shared_results$matched_data)))
  log_info(paste("Number of valid NCOs used:", nrow(shared_results$valid_nco)))
  log_info(paste("Equipoise proportion:", round(shared_results$equipoise_proportion * 100, 2), "%"))

  # Save shared results
  tryCatch({
    serializable_shared <- shared_results
    # Remove matched_data to avoid large file sizes
    serializable_shared$matched_data <- NULL
    # Remove systematic_error_model as it's not JSON serializable
    serializable_shared$systematic_error_model <- NULL
    # Remove matching_summary to keep JSON clean
    serializable_shared$matching_summary <- NULL
    json_path <- file.path(base_output_dir, "shared_results.json")
    jsonlite::write_json(serializable_shared, path = json_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
    log_info(paste("Shared results saved to:", json_path))
  }, error = function(e) {
    log_warning(paste("Could not save shared results:", e$message))
  })

  # Now analyze each primary outcome using shared results
  log_info("Analyzing individual primary outcomes...")
  
  outcome_results <- list()
  
  for (i in seq_along(primary_outcomes)) {
    outcome_var <- primary_outcomes[i]
    log_info(paste("=== Processing outcome", i, "of", length(primary_outcomes), ":", outcome_var, "==="))
    
    tryCatch({
      # Analyze this outcome using shared results (no individual output directory)
      outcome_result <- analyze_primary_outcome(
        shared_results = shared_results,
        outcome_var = outcome_var,
        output_dir = NULL  # Don't save individual files
      )
      
      outcome_results[[outcome_var]] <- outcome_result
      
      # Log results for this outcome
      log_info(paste("Results for", outcome_var, ":"))
      log_info(paste("  Original RR:", round(outcome_result$original_results$RR, 3)))
      log_info(paste("  Original 95% CI:", round(outcome_result$original_results$RR_LB, 3), "-", round(outcome_result$original_results$RR_UB, 3)))
      log_info(paste("  Log RR:", round(outcome_result$original_results$logRR, 3)))
      log_info(paste("  Standard Error:", round(outcome_result$original_results$seLogRR, 3)))

      # Log calibrated results if available
      if (!is.null(outcome_result$calibrated_results)) {
        log_info("  Empirically Calibrated Results:")
        log_info(paste("    Calibrated RR:", round(outcome_result$calibrated_results$RR, 3)))
        log_info(paste("    Calibrated 95% CI:", round(outcome_result$calibrated_results$RR_LB, 3), "-", round(outcome_result$calibrated_results$RR_UB, 3)))
      }
      
      log_info(paste("Successfully completed analysis for", outcome_var))
      
    }, error = function(e) {
      log_error(paste("Failed to analyze", outcome_var, ":", e$message))
      outcome_results[[outcome_var]] <- NULL
    })
  }
  
  # Save all primary outcome results together in one JSON file
  tryCatch({
    primary_outcomes_json_path <- file.path(base_output_dir, "primary_outcome_results.json")
    jsonlite::write_json(outcome_results, path = primary_outcomes_json_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
    log_info(paste("Primary outcome results saved to:", primary_outcomes_json_path))
  }, error = function(e) {
    log_warning(paste("Could not save primary outcome results:", e$message))
  })

  log_info(paste(analysis_name, "analysis completed successfully!"))
  log_info(paste("Results saved to:", base_output_dir))
  
  return(list(
    shared_results = shared_results,
    outcome_results = outcome_results
  ))
}

# Main analysis (single-run with CLI argument)
# Parse CLI arguments using optparse
option_list <- list(
  optparse::make_option("--model", type = "character", default = "no",
                        help = "Model name; reads latte_t2e/cohort_with_<model>_embeddings.csv"),
  optparse::make_option("--top_k_nco", type = "integer", default = 100,
                        help = "Number of top NCO variables to select"),
  optparse::make_option("--top_k_primary_outcomes", type = "integer", default = 100,
                        help = "Number of top primary outcome variables to analyze"),
  optparse::make_option("--treatment", type = "character", default = "glp1",
                        help = "Treatment arm in medication_type_class; choices: glp1,dpp4,sglt2"),
  optparse::make_option("--control", type = "character", default = "dpp4",
                        help = "Control arm in medication_type_class; choices: glp1,dpp4,sglt2"),
  optparse::make_option("--low_prevalence_threshold", type = "double", default = 0.005,
                        help = "Drop NCOs/outcomes with prevalence below this threshold [default: 0.005]")
)
parser <- optparse::OptionParser(option_list = option_list)
opts <- optparse::parse_args(parser)

# Set working directory to current directory
working_dir <- getwd()

# Validate treatment/control choices
allowed_arms <- c("glp1", "dpp4", "sglt2")
opts$treatment <- tolower(opts$treatment)
opts$control <- tolower(opts$control)
if (!(opts$treatment %in% allowed_arms)) {
  stop(sprintf("Invalid --treatment '%s'. Choose from: %s", opts$treatment, paste(allowed_arms, collapse = ", ")))
}
if (!(opts$control %in% allowed_arms)) {
  stop(sprintf("Invalid --control '%s'. Choose from: %s", opts$control, paste(allowed_arms, collapse = ", ")))
}
if (opts$treatment == opts$control) {
  stop("--treatment and --control must be different")
}

# Extract version number from model ID (e.g., "v6-s" -> 6, "v7-no" -> 7)
# "delphi" is a standalone v6-era model without the version prefix
if (opts$model == "delphi") {
  version <- 6L
} else {
  version <- as.integer(gsub("^v(\\d+)-.*$", "\\1", opts$model))
}
log_info(paste("Model:", opts$model, "-> Version:", version))

input_file <- file.path(working_dir, "intermediate", paste0("v", version), "final_cohort", sprintf("cohort_with_%s_embeddings.csv", opts$model))
config_name <- paste0(opts$treatment, "_vs_", opts$control)
base_output_dir <- file.path(working_dir, "results", paste0("v", version), config_name)

# Load data to select primary outcomes
log_info("Loading data to select primary outcomes...")
data <- read.csv(input_file)
log_info(paste("Loaded data with", nrow(data), "rows and", ncol(data), "columns"))

# Select top primary outcomes
primary_outcomes <- select_top_primary_outcomes(data, k = opts$top_k_primary_outcomes, low_prevalence_threshold = opts$low_prevalence_threshold)
log_info(paste("Selected", length(primary_outcomes), "primary outcomes for analysis"))

# Select top NCO outcomes (independent of treatment/control configuration)
nco_outcomes <- select_top_nco_variables(data, pattern = "^NCO_binary_", k = opts$top_k_nco, low_prevalence_threshold = opts$low_prevalence_threshold)
log_info(paste("Selected", length(nco_outcomes), "NCO outcomes for analysis"))

# Run optimized analysis for all outcomes at once
analysis_name <- paste("With", opts$model, "Embeddings")
final_output_dir <- file.path(base_output_dir, paste0(opts$model, "_embeddings"))

tryCatch({
  all_results <- run_ease_analysis(
    input_file = input_file,
    analysis_name = analysis_name,
    base_output_dir = final_output_dir,
    primary_outcomes = primary_outcomes,
    nco_outcomes = nco_outcomes,
    treatment = opts$treatment,
    control = opts$control
  )
  
  log_info("=== SUMMARY ===")
  log_info(paste("EASE Score (same for all outcomes):", round(all_results$shared_results$ease_score, 4)))
  log_info("Primary outcome results:")
  
  for (outcome_var in names(all_results$outcome_results)) {
    if (!is.null(all_results$outcome_results[[outcome_var]])) {
      result <- all_results$outcome_results[[outcome_var]]
      original <- result$original_results
      
      # Format the summary line
      summary_line <- paste("  ", outcome_var, "- Original RR:", round(original$RR, 3), 
                           "(", round(original$RR_LB, 3), "-", round(original$RR_UB, 3), ")")
      
      if (!is.null(result$calibrated_results)) {
        calibrated <- result$calibrated_results
        summary_line <- paste(summary_line, "| Calibrated RR:", round(calibrated$RR, 3), 
                             "(", round(calibrated$RR_LB, 3), "-", round(calibrated$RR_UB, 3), ")")
      }
      
      log_info(summary_line)
    }
  }
  
  log_info("All analyses completed successfully!")
  
}, error = function(e) {
  log_error(paste("Failed to run analysis:", e$message))
})
