#!/usr/bin/env Rscript
# =============================================================================
# fit_dr_models.R -- fits and checkpoints the cross-fitted propensity and outcome models of the
# doubly robust step, one cohort cell per call. estimate_dr.R turns the checkpoints into estimates.
#
#   PS rows (engines = the ladder's own):  hdps | delphi-only | reclaim-s-only            (default)
#   Outcome-model columns:                 om-delphi | om-reclaim-s | om-reclaim-m | om-reclaim-l
#   --ps_specs / --om_specs select other rows and columns: any model with an embedding parquet.
#
# Cross-fitting: K deterministic folds (person_id-rank modulo K); every prediction comes from a model
# that did not see that person. Propensities are clipped to [e_clip, 1 - e_clip]. PS engines mirror
# the ladder: unpenalised logistic for hdps, ridge for the embedding rows.
# Outcome models: one per (column, arm), each fitted on that arm alone and predicted for everyone
# (g1 from the treated arm, g0 from the comparator arm); ridge on age, sex and the embedding. The
# reported effect in the treated uses g0 only. Lambda is selected ONCE per fitted model on a
# deterministic stripe subsample of that arm, then refit at that fixed lambda on each fold-complement.
# Population (--population): `all` cohort rows, or `at_risk` = still enrolled when the outcome
# window opens (person_time > --risk_start_days) or a case. On a landmark cohort whose outcome
# window opens after the index date, `at_risk` is the binary analogue of the ladder's Cox, which
# never holds an earlier drop-out in a risk set.
#
# Inputs : --cohort_csv, --hdps_covars, --embed_dir (cohort_with_<model>_embeddings.parquet)
# Output : --cache_dir/ps_<row><tag>.rds and om_<column>_arm{0,1}<tag>.rds, where <tag> marks the
#          population and any --limit. A rerun loads a valid checkpoint instead of refitting.
# =============================================================================

suppressWarnings(suppressMessages({
  library(optparse); library(glmnet); library(speedglm); library(parallel)
}))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(here, "..", "utils.R"))
log_file <- setup_logging()
IND <- "    "

opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohort_csv", type = "character",
              default = "intermediate/mdcr/cohorts/positive_control_celecoxib_gi/MDCR/cohort.csv"),
  make_option("--hdps_covars", type = "character",
              default = "intermediate/mdcr/hdps_covars/positive_control_celecoxib_gi_MDCR/hdps_covars.csv"),
  make_option("--embed_dir", type = "character",
              default = "intermediate/mdcr/embeddings/positive_control_celecoxib_gi_MDCR"),
  make_option("--cache_dir", type = "character", default = NULL,
              help = "directory for the per-nuisance .rds checkpoints (required)"),
  make_option("--k_folds", type = "integer", default = 5L),
  make_option("--lambda_subsample", type = "integer", default = 100000L),
  make_option("--e_clip", type = "double", default = 0.01),
  make_option("--population", type = "character", default = "all",
              help = "all|at_risk -- at_risk keeps rows with person_time > --risk_start_days, or a case"),
  make_option("--risk_start_days", type = "integer", default = NULL,
              help = "days from index_date to the opening of the outcome window (required for at_risk)"),
  make_option("--om_specs", type = "character", default = NULL,
              help = "comma-separated outcome-model columns to run (default: every column with embeddings)"),
  make_option("--ps_specs", type = "character", default = NULL,
              help = "comma-separated PS rows to run (default: hdps,delphi-only,reclaim-s-only)"),
  make_option("--limit", type = "integer", default = NULL, help = "smoke: first N rows")
)))

stopifnot(opt$population %in% c("all", "at_risk"))
csv_opt <- function(x, default) if (is.null(x)) default else trimws(strsplit(x, ",")[[1]])
OM_WANT <- csv_opt(opt$om_specs, c("om-delphi", "om-reclaim-s", "om-reclaim-m", "om-reclaim-l"))
PS_WANT <- csv_opt(opt$ps_specs, c("hdps", "delphi-only", "reclaim-s-only"))
ps_model <- function(ps) sub("-only$", "", ps)     # reclaim-m-pca120-only -> reclaim-m-pca120
om_model <- function(om) sub("^om-", "", om)       # om-reclaim-l          -> reclaim-l
is_model <- function(m) grepl("^(delphi|reclaim-[sml](-pca[0-9]+)?)$", m)
if (length(bad <- OM_WANT[!(grepl("^om-", OM_WANT) & is_model(om_model(OM_WANT)))]))
  stop("unknown --om_specs: ", paste(bad, collapse = ", "))
if (length(bad <- PS_WANT[!(PS_WANT == "hdps" | (grepl("-only$", PS_WANT) & is_model(ps_model(PS_WANT))))]))
  stop("unknown --ps_specs: ", paste(bad, collapse = ", "))

det_foldid <- function(ids, k) ((match(ids, sort(ids)) - 1L) %% k) + 1L
stripe_idx <- function(n, m) {                 # deterministic uniform subsample of size <= m
  if (n <= m) seq_len(n) else which(((seq_len(n) - 1L) %% ceiling(n / m)) == 0L)
}
auc_fast <- function(y, p) {                   # rank-based AUC (no pROC dependency)
  r <- rank(p); n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(sum(y == 0))
  if (n1 == 0 || n0 == 0) return(NA_real_)
  # as.numeric throughout: n1*n0 overflows R integers at cohort scale
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# =============================================================================
# load cohort + covariate blocks
# =============================================================================
log_info("Loading cohort + covariate blocks...")
coh <- read.csv(opt$cohort_csv)
if (!is.null(opt$limit)) coh <- coh[seq_len(min(opt$limit, nrow(coh))), ]
crude_rr <- function(d) mean(d$outcome[d$arm == 1]) / mean(d$outcome[d$arm == 0])
log_info(sprintf("population=%s  PS rows: %s  outcome models: %s", opt$population,
                 paste(PS_WANT, collapse = ", "), paste(OM_WANT, collapse = ", ")))
if (opt$population == "at_risk") {
  if (is.null(opt$risk_start_days) || !"person_time" %in% names(coh))
    stop("--population at_risk needs --risk_start_days and a person_time column")
  at_risk <- coh$person_time > opt$risk_start_days | coh$outcome == 1
  log_info(sprintf("at_risk: kept %s of %s rows (dropped %.1f%% of treated, %.1f%% of comparators); crude RR %.3f -> %.3f",
                   format(sum(at_risk), big.mark = ","), format(nrow(coh), big.mark = ","),
                   100 * mean(!at_risk[coh$arm == 1]), 100 * mean(!at_risk[coh$arm == 0]),
                   crude_rr(coh), crude_rr(coh[at_risk, ])))
  coh <- coh[at_risk, ]
}
coh$person_id <- as.character(coh$person_id)
n <- nrow(coh)
W <- as.integer(coh$arm); Y <- as.integer(coh$outcome)
log_info(sprintf("n=%s treated=%s events=%s (treated events=%s)",
                 format(n, big.mark = ","), format(sum(W), big.mark = ","),
                 format(sum(Y), big.mark = ","), format(sum(Y[W == 1]), big.mark = ",")))

coh$cal_year <- as.integer(format(as.Date(coh$index_date), "%Y"))
# constant factors are dropped, not errored: the negative control indexes within a single
# flu season, so cal_year has one level there and model.matrix would halt.
demo_terms <- c("age_at_index",
                if (length(unique(coh$sex)) > 1L) "factor(sex)",
                if (length(unique(coh$cal_year)) > 1L) "factor(cal_year)")
M_demo <- model.matrix(reformulate(demo_terms), data = coh)[, -1, drop = FALSE]

read_block <- function(path, prefix, ids) {
  # column-pruned parquet -> matrix aligned to cohort order; refuses partial coverage
  schema <- nanoparquet::read_parquet_schema(path)
  keep <- c("person_id", grep(prefix, schema$name, value = TRUE))
  d <- as.data.frame(nanoparquet::read_parquet(path, col_select = keep))
  d$person_id <- as.character(d$person_id)
  ix <- match(ids, d$person_id)
  if (anyNA(ix)) stop(sprintf("%s covers only %d/%d cohort ids", basename(path),
                              sum(!is.na(ix)), length(ids)))
  m <- as.matrix(d[ix, setdiff(keep, "person_id"), drop = FALSE])
  storage.mode(m) <- "double"; m
}

# Each covariate block is read once, and only when a requested PS row or outcome-model column
# needs it.
M_rec <- NULL
if ("hdps" %in% PS_WANT) {
  hd <- read.csv(opt$hdps_covars)
  hd$person_id <- as.character(hd$person_id)
  ixh <- match(coh$person_id, hd$person_id)
  rec_cols <- grep("^rec_", names(hd), value = TRUE)
  M_rec <- as.matrix(hd[ixh, rec_cols, drop = FALSE]); storage.mode(M_rec) <- "double"
  M_rec[is.na(M_rec)] <- 0
  rm(hd)
}
M_agesex <- model.matrix(~ age_at_index + factor(sex), data = coh)[, -1, drop = FALSE]
# An embedding design is age + sex + the representation. The same matrix serves as that model's
# PS row and as its outcome-model column, so it is built once and shared, not copied.
EMB <- list()
for (m in unique(c(ps_model(setdiff(PS_WANT, "hdps")), om_model(OM_WANT)))) {
  path <- file.path(opt$embed_dir, sprintf("cohort_with_%s_embeddings.parquet", m))
  if (!file.exists(path)) { log_info(sprintf("NOTE: no embeddings for %s (%s)", m, basename(path))); next }
  EMB[[m]] <- cbind(M_agesex, read_block(path, "^rep_", coh$person_id))
}
log_info(sprintf("blocks: demo %d, rec %s; embedding designs: %s", ncol(M_demo),
                 if (is.null(M_rec)) "not read" else ncol(M_rec),
                 paste(sprintf("%s %d", names(EMB), vapply(EMB, ncol, integer(1))), collapse = ", ")))

fold <- det_foldid(coh$person_id, opt$k_folds)
K <- opt$k_folds

# =============================================================================
# design matrices per nuisance setting
# =============================================================================
# PS rows: crude and demo are not run (a constant PS makes AIPW collapse toward the g-computation
# corner, which the grid reports separately); hdps = demographics + k empirical covariates,
# mirroring the ladder's hdps spec.
PS_X <- list()
for (ps in PS_WANT) {
  x <- if (ps == "hdps") cbind(M_demo, M_rec) else EMB[[ps_model(ps)]]
  if (is.null(x)) stop("no design matrix for PS row ", ps)
  PS_X[[ps]] <- x
}
# OM columns: no om-hdps -- the hdPS covariate set is SELECTED for propensity modelling (Bross
# bias ranking), so reusing it as an outcome model would duplicate the hdps PS row's information.
# The outcome side is the foundation-model representations only. A column joins the grid when its
# embeddings exist.
OM_X <- list()
for (om in OM_WANT) { x <- EMB[[om_model(om)]]; if (!is.null(x)) OM_X[[om]] <- x }
rm(EMB, x)

# NUISANCE ENGINE RULE -- the engine follows what the covariates are, on both sides:
#   * dense foundation-model EMBEDDINGS (delphi / reclaim-*)  -> RIDGE (alpha = 0): L1 would keep
#     one member of each correlated group arbitrarily, ridge shrinks the whole representation.
#   * sparse binary claim-code blocks (hdps rec_*)            -> LASSO / unpenalised.
# Cross-fitting licenses any learner here, so the engine is a modelling choice, not an identification one.
is_embed_block <- function(nm) grepl("delphi|reclaim", nm)
om_alpha <- function(nm) if (is_embed_block(nm)) 0 else 1

# =============================================================================
# cross-fitted PS predictions (engines = the ladder's)
# =============================================================================
# Nuisance checkpointing: every finished nuisance is saved to CACHE_DIR the moment it completes,
# and a rerun loads instead of refits. Cache validity is guarded on (n, k_folds, design-matrix
# signature): a changed covariate block invalidates its cached nuisance automatically.
# File names carry the population and any --limit (CTAG), and the outcome-model fitting mode, so a
# restricted or smoke run can share a cache directory with the full run without overwriting it.
CTAG <- paste0(if (opt$population == "all") "" else paste0("__", opt$population),
               if (is.null(opt$limit)) "" else paste0("__limit", opt$limit))
xsig <- function(X) {
  if (is.null(X)) return("null")
  s <- stripe_idx(nrow(X), min(nrow(X), 20000L))
  paste(ncol(X), format(sum(colSums(X[s, , drop = FALSE])), digits = 15),
        paste(head(colnames(X), 3), collapse = "|"), paste(tail(colnames(X), 3), collapse = "|"))
}
CACHE_DIR <- opt$cache_dir
if (is.null(CACHE_DIR)) stop("--cache_dir is required: the checkpoints are this script's only output")
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)

log_info("== PS nuisances (cross-fitted, ladder engines) ==")
# Embedding PS rows are fit with ridge (lambda selected once on a stripe subsample); the hdps
# row keeps the unpenalised logistic that Schneeweiss specifies. Rows are independent, so
# PS_CORES > 1 fits them in parallel (forked workers share the design matrices copy-on-write).
ps_fit_one <- function(ps) {
  t0 <- Sys.time()
  X <- PS_X[[ps]]
  sig <- xsig(X)
  cf <- if (is.null(CACHE_DIR)) NULL else file.path(CACHE_DIR, paste0("ps_", ps, CTAG, ".rds"))
  if (!is.null(cf) && file.exists(cf)) {
    cc <- readRDS(cf)
    if (identical(cc$n, n) && identical(cc$k, K) && identical(cc$sig, sig)) {
      log_info(sprintf("%s%-22s CACHED", IND, ps))
      return(list(ps = ps, e = cc$e))
    }
  }
  e <- numeric(n)
  ps_embed <- is_embed_block(ps)
  ps_lam <- NA_real_
  if (ps_embed) {
    sub <- stripe_idx(n, opt$lambda_subsample)
    cvl <- cv.glmnet(X[sub, , drop = FALSE], W[sub], family = "binomial", alpha = 0,
                     foldid = det_foldid(coh$person_id[sub], K))
    ps_lam <- cvl$lambda.min
  }
  for (k in seq_len(K)) {
    tr <- fold != k; te <- fold == k
    if (is.null(X)) { e[te] <- mean(W[tr]); next }
    if (ps_embed) {
      fitp <- glmnet(X[tr, , drop = FALSE], W[tr], family = "binomial", alpha = 0,
                     lambda = ps_lam * c(4, 2, 1))   # short warm-start path, keep last
      e[te] <- as.numeric(predict(fitp, newx = X[te, , drop = FALSE],
                                  s = ps_lam, type = "response"))
      next
    }
    df_tr <- as.data.frame(X[tr, , drop = FALSE]); df_tr$.w <- W[tr]
    fit <- tryCatch(speedglm(.w ~ ., data = df_tr, family = binomial()),
                    error = function(err) glm(.w ~ ., data = df_tr, family = binomial()))
    e[te] <- as.numeric(predict(fit, newdata = as.data.frame(X[te, , drop = FALSE]),
                                type = "response"))
  }
  bad <- !is.finite(e); if (any(bad)) e[bad] <- mean(W)
  n_clip <- sum(e < opt$e_clip | e > 1 - opt$e_clip)
  e <- pmin(pmax(e, opt$e_clip), 1 - opt$e_clip)
  if (!is.null(cf)) {   # atomic write: tmp + rename
    tmp <- paste0(cf, ".", Sys.getpid(), ".tmp"); saveRDS(list(n = n, k = K, sig = sig, e = e), tmp); file.rename(tmp, cf)
  }
  log_info(sprintf("%s%-22s [%s] cross-fitted c=%.4f  clipped=%d  (%.1f min)", IND, ps,
                   if (ps_embed) "ridge" else if (is.null(X)) "const" else "logistic",
                   ifelse(is.null(X), NA, auc_fast(W, e)), n_clip,
                   as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  list(ps = ps, e = e)
}
PS_CORES <- min(length(PS_X), max(1L, as.integer(Sys.getenv("PS_CORES", "1"))))
ps_res <- if (PS_CORES > 1L) {
  mclapply(names(PS_X), ps_fit_one, mc.cores = PS_CORES, mc.preschedule = FALSE)
} else lapply(names(PS_X), ps_fit_one)
e_hat <- list()
for (r in ps_res) {
  if (!is.list(r) || is.null(r$e)) stop("a PS task failed: ", paste(deparse(r), collapse = " "))
  e_hat[[r$ps]] <- r$e
}
rm(ps_res)

# =============================================================================
# cross-fitted outcome models (per om x arm), at a stripe-selected fixed lambda
# =============================================================================
log_info("== outcome-model nuisances (per arm, cross-fitted) ==")
# PS design matrices are no longer needed -- free them before forking.
PS_SPECS <- names(PS_X); rm(PS_X); invisible(gc())
# The (om x arm) tasks run in parallel via mclapply (forked workers share the design matrices
# copy-on-write; results are collected by task id). OM_CORES overrides the default of 5 workers.
OM_TASKS <- expand.grid(om = names(OM_X), a = c(1, 0),
                        stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
# heaviest-first (comparator arm, widest block) so the makespan is not gated on a straggler
OM_TASKS <- OM_TASKS[order(OM_TASKS$a, -vapply(OM_X[OM_TASKS$om], ncol, integer(1))), ]
# Fits on > 1M rows use thresh 1e-4 / maxit 5e3 (defaults 1e-7 / 1e5): coordinate descent at
# the small cv-selected lambda converges very slowly on a large rare-event training set, and the
# residual imprecision is far below the event-count noise these predictions feed into.
om_fit_one <- function(i) {
  om <- OM_TASKS$om[i]; a <- OM_TASKS$a[i]
  cf <- if (is.null(CACHE_DIR)) NULL else file.path(CACHE_DIR, sprintf("om_%s_arm%d%s.rds", om, a, CTAG))
  X <- OM_X[[om]]
  sig <- xsig(X)
  if (!is.null(cf) && file.exists(cf)) {
    cc <- readRDS(cf)
    if (identical(cc$n, n) && identical(cc$k, K) && identical(cc$sig, sig)) return(cc$res)
  }
  t0 <- Sys.time()
  arm_rows <- which(W == a)
  # lambda: ONE deterministic-fold cv.glmnet on a deterministic stripe of this arm
  sub <- arm_rows[stripe_idx(length(arm_rows), opt$lambda_subsample)]
  .a <- om_alpha(om)
  cvf <- cv.glmnet(X[sub, , drop = FALSE], Y[sub], family = "binomial", alpha = .a,
                   foldid = det_foldid(coh$person_id[sub], K))
  lam <- cvf$lambda.min
  g <- numeric(n)
  for (k in seq_len(K)) {
    tr <- arm_rows[fold[arm_rows] != k]           # this arm's fold-complement
    te <- which(fold == k)                        # predict for EVERYONE in fold k
    big <- length(tr) > 1e6
    fit <- glmnet(X[tr, , drop = FALSE], Y[tr], family = "binomial", alpha = .a,
                  lambda = lam * c(8, 4, 2, 1),   # short warm-start path, keep last
                  thresh = if (big) 1e-4 else 1e-7,
                  maxit  = if (big) 5e3  else 1e5)
    g[te] <- as.numeric(predict(fit, newx = X[te, , drop = FALSE],
                                s = lam, type = "response"))
    log_info(sprintf("%s%-13s arm=%d  fold %d/%d fit (%.1f min in)", IND, om, a, k, K,
                     as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
  g <- pmin(pmax(g, 1e-6), 1 - 1e-6)
  res <- list(om = om, a = a, g = g, lambda = lam,
              engine = if (.a == 0) "ridge" else "lasso",
              auc = auc_fast(Y[arm_rows], g[arm_rows]),   # within-arm, cross-fitted
              mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  if (!is.null(cf)) {   # atomic write; parallel workers write distinct files
    tmp <- paste0(cf, ".", Sys.getpid(), ".tmp"); saveRDS(list(n = n, k = K, sig = sig, res = res), tmp); file.rename(tmp, cf)
  }
  res
}
OM_CORES <- min(nrow(OM_TASKS), max(1L, as.integer(Sys.getenv("OM_CORES", "5"))))
log_info(sprintf("%d (om x arm) tasks on %d workers", nrow(OM_TASKS), OM_CORES))
om_res <- if (OM_CORES > 1L) {
  mclapply(seq_len(nrow(OM_TASKS)), om_fit_one, mc.cores = OM_CORES, mc.preschedule = FALSE)
} else lapply(seq_len(nrow(OM_TASKS)), om_fit_one)
g_hat <- list()   # g_hat[[om]] = list(g1=, g0=, lambda1=, lambda0=, auc1=, auc0=)
for (r in om_res) {
  if (!is.list(r) || is.null(r$g))                 # mclapply returns the worker error object
    stop("an outcome-model task failed: ", paste(deparse(r), collapse = " "))
  g_hat[[r$om]][[paste0("g", r$a)]] <- r$g
  g_hat[[r$om]][[paste0("lambda", r$a)]] <- r$lambda
  g_hat[[r$om]][[paste0("auc", r$a)]] <- r$auc
  log_info(sprintf("%s%-13s arm=%d  lambda=%.5f  in-arm AUC=%.4f  (%.1f min)", IND,
                   r$om, r$a, r$lambda, r$auc, r$mins))
}

log_info(sprintf("Fitted %d PS rows and %d outcome-model columns (both arms); checkpoints in %s. Log: %s",
                 length(e_hat), length(g_hat), CACHE_DIR, log_file))
