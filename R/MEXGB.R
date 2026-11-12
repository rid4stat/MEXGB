# Gaussian EM/BLUP helpers sig(), bay() and logV() are adapted from MEGB 0.2
# by Oyebayo Ridwan Olaniran and Saidat Fehintola Olaniran (GPL-2). See inst/NOTICE.
#' Mixed Effect Extreme Gradient Boosting (MEXGB) Algorithm
#'
#' Fits a flexible fixed effect and subject-specific random effects to Gaussian,
#' binary or clustered right-censored survival data. The fixed component combines
#' optional penalised linear terms with extreme gradient boosting, following Chen
#' and Guestrin (2016) \doi{10.1145/2939672.2939785}.
#'
#' The implementation is motivated by mixed-effects tree ensembles, including
#' Hajjem et al. (2014) \doi{10.1080/00949655.2012.741599} and Capitaine et al.
#' (2021) \doi{10.1177/0962280220946080}. It extends the EM/BLUP approach with
#' within-subject cross-fitting, optional linear terms, predictor screening,
#' calibration and approximate non-Gaussian variance-component updates.
#'
#' The conditional linear predictor is
#' \deqn{\eta_{ij}=f(x_{ij})+z_{ij}^{\top}\beta_z+z_{ij}^{\top}b_i,\quad b_i\sim N(0,D).}
#' Gaussian responses use an identity link and independent residual errors given
#' the random effects; binary responses use a logit link; survival responses use
#' a proportional-hazards model with a Breslow baseline hazard. The survival
#' implementation supports clustered right-censored event times, not start-stop
#' records or a longitudinal time-varying-covariate survival likelihood.
#'
#' Gaussian variance updates use BLUP/EM calculations. Binary and survival
#' updates use PQL, optionally followed by a Laplace-style approximation. The Cox
#' update uses a Poisson working representation and is an approximation rather
#' than a claim of equivalence to a full joint frailty likelihood implementation.
#' Failed Laplace optimization retains the PQL update and records diagnostics.
#'
#' Cross-fitting distributes observations within subjects. It supports tuning for
#' additional observations from represented subjects, not independent evaluation
#' on unseen subjects or future-time forecasting. Reusing folds for adaptive
#' selection is not external validation. Use an independent test set or nested
#' validation matched to the prediction task for performance reporting.
#' Gaussian screening and covariance shrinkage use a subject-clustered
#' standard-error threshold as a selection heuristic, not a hypothesis test.
#' The returned fit may be an earlier selected iteration rather than the final
#' attempted iteration; the baseline hazard is computed from the returned fit.
#'
#' @param X Finite numeric predictor matrix, with one row per observation and unique column names if supplied. Missing values must be handled before fitting.
#' @param Y [vector]: A vector containing the output trajectories (numeric for \code{family="gaussian"}; 0/1 for \code{family="binomial"}). For \code{family="cox"} a two-column matrix (observed time, event indicator 0/1) or a right-censored \code{survival::Surv} object. Counting-process and interval-censored responses are rejected.
#' @param id [vector]: Is the vector of the identifiers for the different trajectories.
#' @param Z [matrix]: A \code{N} x \code{q} matrix containing the \code{q} predictor of the random effects.
#' @param iter [numeric]: Maximal number of iterations of the algorithm. The default is set to \code{iter=100}
#' @param ntree Positive integer maximum number of boosting rounds in each fold model. Default 5000. With CV, the selected count may be zero; the full-data count is scaled by V/(V-1) and can exceed this fold-model limit. With fixed selection, exactly ntree rounds are used.
#' @param time [vector]: Is the vector of the measurement times associated with the trajectories in \code{Y},\code{Z} and \code{X}.
#' @param shrinkage [numeric]: a shrinkage parameter applied to each tree in the expansion (the \code{xgboost} learning rate \code{eta}). Default 0.1 (MEXGB <= 0.4: 0.05).
#' @param interaction.depth [numeric]: The maximum depth of each tree (the \code{xgboost} parameter \code{max_depth}). The default value is set to 1.
#' @param n.minobsinnode Leaf-size control mapped to the minimum Hessian weight
#'   in a leaf. Default 5. This equals an observation count for Gaussian loss;
#'   for other families it is a heuristic, not a guaranteed minimum leaf count.
#' @param lambda [numeric]: L2 penalty on the leaf weights (\code{xgboost} parameter \code{lambda}). Default 1.
#' @param alpha [numeric]: L1 penalty on the leaf weights (\code{xgboost} parameter \code{alpha}). Default 0.
#' @param gamma [numeric]: Minimum loss reduction (complexity charge per leaf) required to make a split (\code{xgboost} parameter \code{gamma}). Default 0.
#' @param subsample [numeric]: Row subsampling rate of each tree. Default 1 (deterministic, as \code{bag.fraction = 1} in MEGB).
#' @param colsample_bytree [numeric]: Column subsampling rate of each tree; useful when \code{p >> n}. Default 1.
#' @param min_child_weight [numeric]: Optional override of the minimum Hessian weight in a leaf. If \code{NULL} (default) it is \code{n.minobsinnode} for \code{family="gaussian"} and \code{n.minobsinnode/4} for binary and Cox outcomes (a heuristic; for binary outcomes the Bernoulli Hessian is at most 1/4).
#' @param family [character]: \code{"gaussian"} (continuous outcome, default), \code{"binomial"} (binary outcome) or \code{"cox"} (right-censored time-to-event outcome; mixed-effects (frailty) Cox model).
#' @param delta [numeric]: The algorithm stops when the relative difference in log likelihood between two iterations is smaller than \code{delta}. The default value is set to 0.001
#' @param stall_iter Positive integer: the EM stops when the selection criterion has not
#'   improved for \code{stall_iter} consecutive iterations and returns the best iteration
#'   (a safeguard for stochastic boosting). Inf disables it. Default 5. Selection uses the
#'   Gaussian held-out loss when available, otherwise the chosen likelihood criterion.
#' @param eps [numeric]: Lower bound for the PQL working weights (binary and cox outcomes). Default \code{1e-6}.
#' @param crossfit Number of within-subject folds: 0 (disabled) or an integer from
#'   2 to the number of observations. Default 5. Empty folds, insufficient training
#'   rows, binary training folds with one class, and survival training folds without
#'   events are rejected. Change seed, reduce folds, or provide more data if necessary.
#'   With crossfit = 0 the linear part is not fitted (linear = TRUE gives a warning).
#' @param init [character]: Starting values of the EM. \code{"auto"} (default): \code{"zero"} for
#'   \code{family = "gaussian"}, \code{"lmm"} otherwise. \code{"lmm"}: the random effects,
#'   their covariance and the residual variance of the mixed model with a constant fixed effect
#'   (PQL for binary outcomes; ignored for \code{family = "cox"}); \code{"zero"}: \eqn{b_i=0},
#'   \eqn{D=I}, \eqn{\sigma^2=1} (MEXGB 0.2).
#' @param dispersion [character]: residual structure of the PQL working model of binary and cox
#'   outcomes: \code{"none"} \eqn{V_i=Z_iDZ_i'+W_i^{-1}} (MEXGB 0.2); \code{"multiplicative"}
#'   \eqn{V_i=Z_iDZ_i'+\phi W_i^{-1}} (as \code{MASS::glmmPQL}); \code{"additive"}
#'   \eqn{V_i=Z_iDZ_i'+\tau^2I+W_i^{-1}} (\eqn{\tau^2}: variance of the out-of-fold error of the linear
#'   predictor). \code{"auto"} (default): \code{"multiplicative"} for \code{family = "cox"} with the
#'   cross-fitted E-step and calibrate = FALSE, \code{"none"} otherwise. All
#'   families stop and return the best iteration if the likelihood becomes non-finite.
#' @param inner [numeric]: maximum number of E-step iterations per boosting refit (default 50). The
#'   BLUP / variance-component updates (and, for binary and cox outcomes, the PQL linearisation)
#'   are cheap, so they are iterated to convergence (relative change < \code{inner_tol}) for the
#'   current fixed effect before the next boosting refit. \code{inner = 1} gives one step per
#'   refit (MEXGB <= 0.3), which stops far from the fixed point because variance-component EM
#'   converges slowly.
#' @param inner_tol [numeric]: convergence tolerance of the inner iterations (default 1e-4).
#' @param calibrate [logical]: Enable calibration for binary and Cox cross-fitted models. For Cox: rescale the boosted fixed effect by
#'   its out-of-fold calibration slope (Cox model of the outcome on the out-of-fold linear predictor,
#'   offset \eqn{Zb}), in the E-step and in prediction (default \code{TRUE}).
#'   Calibration may improve stability when the fixed-effect predictor is overdispersed;
#'   slopes are restricted to [0.05, 1.5].
#' @param calibrate_binomial Logical; additionally enable binary calibration when calibrate is TRUE. Default TRUE.
#' @param tree_method [character]: \code{xgboost} tree construction algorithm. \code{"auto"}
#'   (default) uses the exact greedy algorithm (as \code{gbm}) up to 10000 observations and the
#'   histogram algorithm above.
#' @param linear [logical or \code{"auto"}]: \code{"auto"} (default): for
#'   \code{family = "gaussian"} the fixed effect with and without the linear part is fitted at the
#'   first \code{screen_iter} M-steps (on all predictors) and the one with the smaller
#'   out-of-fold loss is kept, when \eqn{p>\sqrt{N}} (with fewer predictors the linear part is
#'   always kept); for the other
#'   families \code{"auto"} means \code{TRUE}. If \code{TRUE} the fixed effect is
#'   \eqn{f(x)=a+x^\top\beta+T(x)}: a penalised linear part (\code{glmnet}, elastic-net mixing
#'   \code{lin_alpha}, lambda and the relaxation chosen on the cross-fitting folds by the
#'   out-of-fold deviance) plus boosted trees \eqn{T} grown on top of it. When the signal is
#'   linear the out-of-fold selection keeps few or no trees; when it is not, the trees capture the
#'   remainder. \code{FALSE}: trees only (MEXGB <= 0.4).
#' @param lin_alpha [numeric]: elastic-net mixing parameter of the linear part (1 = lasso).
#' @param ntree_select [character]: \code{"cv"} (default): the number of trees is chosen at every
#'   EM iteration as the minimiser of the out-of-fold loss on the cross-fitting folds (0 trees
#'   allowed; \code{ntree} is then the maximum); \code{"fixed"}: always \code{ntree} trees (MEXGB <= 0.4).
#' @param criterion [character]: binary / cox outcomes. \code{"laplace"} (default): the EM is
#'   monitored with the Laplace approximation of the marginal log-likelihood, which is comparable
#'   between iterations; \code{"working"}: the PQL working likelihood (MEXGB <= 0.4), which changes
#'   with the linearisation point and made the best-iteration rule return early iterations.
#' @param estep Binary/Cox variance update. "laplace" (default) follows PQL with
#'   optimization of the approximate criterion; "pql" retains the PQL update.
#'   The Laplace-style Cox criterion uses a Poisson working representation.
#'   If the optimizer stops at its iteration limit its result is used and the code is
#'   recorded in laplace_diagnostics; if it fails (error or non-finite criterion) the
#'   PQL update is retained with a warning. These approximations do not establish inferential guarantees.
#' @param lin_relax [logical]: also consider relaxed-lasso versions of the linear part (blend of the lasso and of the unpenalised refit on its active set), chosen out of fold. Default \code{TRUE}. At prediction the linear part is clamped to the range of its fitted values on the training data, so that the unpenalised refit cannot extrapolate for rare predictor values.
#' @param screen [character]: \code{"auto"} (default): at the first EM iteration the trees are refitted on the K top-ranked predictors (split gain) of every fold model, K in \code{c(5, 10, 20, 40)}, and the predictor set (all or top K) with the smallest out-of-fold loss is kept for the following iterations. The ranking of each fold is computed from that fold's own model, so feature ranking excludes the validation rows; selection still requires independent performance evaluation. \code{"none"}: all predictors.
#' @param screen_iter [numeric]: number of EM iterations at which the screening decision is (re)made (default 2: the ranking is reconsidered after the initial random-effect updates).
#' @param zfixed [logical]: estimate population coefficients \eqn{\beta_z} for the columns of \code{Z} (default \code{TRUE}), so that the random effects are deviations with mean zero: the model is \eqn{f(x)+z^\top\beta_z+z^\top b_i}. Without them a population trend in a column of \code{Z} that \eqn{f} does not capture is absorbed by the mean of the random effects and inflates \eqn{D}. For \code{family = "cox"} the intercept of \eqn{\beta_z} is not identified and is fixed at 0.
#' @param heldout_blup [logical]: Gaussian with cross-fitting (default \code{TRUE}): in every out-of-fold
#'   loss the random effects of a held-out row are predicted from the other folds' rows of the same
#'   individual only (as for a new measurement), and the returned iteration is the one with the
#'   smallest held-out squared error when heldout_dscale is also TRUE. This is a tuning criterion, not an unbiased performance estimate.
#' @param heldout_dscale [logical]: Gaussian with \code{heldout_blup = TRUE} (default \code{TRUE}): after every
#'   E-step D is scaled by the factor c in \{0, 1/16, 1/8, 1/4, 1/2, 1\} that minimises the held-out
#'   out-of-fold squared error, used only when it is better than c = 1 by a paired z statistic above 2 (a selection heuristic, not a hypothesis test; c = 1 keeps the EM estimate).
#' @param heldout_offset [logical]: with \code{heldout_blup = TRUE} (default \code{TRUE}), use the held-out
#'   random-effect prediction as the offset of the held-out rows in the out-of-fold losses of the
#'   M-step (number of trees, linear part, screening). Not used at the first iteration with
#'   \code{init = "zero"}, where the offset is zero for every row.
#' @param nthread [numeric]: Number of threads used by \code{xgboost}. Default 1.
#' @param seed Non-negative integer seed for the folds and for supported XGBoost versions. Default 1. With XGBoost 1.x, stochastic tree fitting also uses the caller's R random-number stream.
#' @param verbose [boolean]: If TRUE, MEXGB will print out number of iterations to achieve convergence. Default is TRUE.
#'
#' @import xgboost
#' @import stats
#' @return A fitted MEXGB model which is a list of the following elements: \itemize{
#' \item \code{forest:} List containing the selected booster (possibly NULL), linear coefficients, intercept and predictor indices.
#' \item \code{random_effects :} Predictions of random effects for different trajectories.
#' \item \code{id_btilde:} Identifiers of individuals associated with the predictions \code{random_effects}.
#' \item \code{var_random_effects: } Estimation of the variance covariance matrix of random effects.
#' \item \code{sigma: } Estimation of the residual variance parameter (gaussian) the PQL dispersion (\eqn{\phi} for \code{dispersion = "multiplicative"}, when selected; \eqn{\tau^2} for \code{"additive"}; 1 for \code{"none"}, the default for binomial).
#' \item \code{time: } The vector of the measurement times associated with the trajectories in \code{Y},\code{Z} and \code{X}.
#' \item \code{LL:} Minus twice the approximate log-likelihood criterion at each attempted iteration, up to an additive constant; its definition depends on family and criterion.
#' \item \code{id: } Vector of the identifiers for the different trajectories.
#' \item \code{OOB: } Training error of the fitted boosting at each iteration (mean squared error for \code{gaussian}, mean deviance for \code{binomial}, \code{-2} partial log-likelihood per observation for \code{cox}).
#' \item \code{importance: } Normalised mean absolute combined SHAP and linear contributions when a linear component exists; otherwise normalised split gain. All entries are zero if there is no contribution.
#' \item \code{baseline: } (cox only) Breslow cumulative baseline hazard (\code{time}, \code{cumhaz}).
#' \item \code{family}, \code{converged}, \code{n_iter}, \code{best_iter}: Outcome family and iteration diagnostics.
#' \item \code{feature_names}, \code{z_names}: Training design column names.
#' \item \code{ntree}, \code{nrounds}: Requested fold-model limit and selected full-model tree count.
#' \item \code{beta_z:} Population coefficients for Z.
#' \item \code{laplace_diagnostics:} Per-iteration convergence flag, optimizer code and message (NULL when no Laplace update was attempted).
#' \item \code{heldout_loss, d_scale:} Gaussian held-out loss and selected covariance scale by iteration.
#' }
#'
#' @export
#'
#' @examples
#' set.seed(1)
#' d <- simLong(n = 8, p = 6, time_points = 4)
#' X <- as.matrix(d[paste0("Var", 1:6)])
#' Z <- as.matrix(d[c("RandomIntercept", "RandomSlope")])
#' fit <- MEXGB(X, d$Y, d$id, Z, time = d$time, ntree = 10,
#'              iter = 2, inner = 2, crossfit = 2, linear = FALSE,
#'              screen = "none", verbose = FALSE)
#' head(predict(fit, X, Z, d$id, d$time))
#' summary(fit)
#' # Small binary and clustered survival fits; PQL keeps examples inexpensive.
#' db <- simLong(8, 6, time_points = 4, family = "binomial")
#' Xb <- as.matrix(db[paste0("Var", 1:6)])
#' fb <- MEXGB(Xb, db$Y, db$id, Z, time = db$time, family = "binomial",
#'             ntree = 10, iter = 2, inner = 2, crossfit = 0,
#'             linear = FALSE, screen = "none", estep = "pql", verbose = FALSE)
#' head(predict(fb, Xb, Z, db$id, db$time))
#' ds <- simLong(8, 6, time_points = 4, family = "cox", censoring = 0)
#' Xs <- as.matrix(ds[paste0("Var", 1:6)])
#' fs <- MEXGB(Xs, cbind(ds$Y, ds$status), ds$id, Z, time = ds$time,
#'             family = "cox", ntree = 10, iter = 2, inner = 2, crossfit = 0,
#'             linear = FALSE, screen = "none", estep = "pql", verbose = FALSE)
#' predict(fs, Xs[1:2, ], Z[1:2, ], ds$id[1:2], ds$time[1:2],
#'         type = "survival", times = c(0, 1))
#'
MEXGB = function (X, Y, id, Z, iter = 100, ntree = 5000, time, shrinkage = 0.1,
                  interaction.depth = 1, n.minobsinnode = 5, lambda = 1, alpha = 0,
                  gamma = 0, subsample = 1, colsample_bytree = 1, min_child_weight = NULL,
                  family = c("gaussian", "binomial", "cox"), delta = 0.001, stall_iter = 5,
                  eps = 1e-6, crossfit = 5, init = c("auto", "lmm", "zero"), dispersion = c("auto", "additive", "multiplicative", "none"),
                  inner = 50, inner_tol = 1e-4, calibrate = TRUE, calibrate_binomial = TRUE,
                  tree_method = "auto", linear = "auto", lin_alpha = 1, lin_relax = TRUE,
                  screen = c("auto", "none"), screen_iter = 2, zfixed = TRUE, heldout_blup = TRUE, heldout_dscale = TRUE, heldout_offset = TRUE,
                  ntree_select = c("cv", "fixed"), criterion = c("laplace", "working"),
                  estep = c("laplace", "pql"),
                  nthread = 1, seed = 1, verbose = TRUE)
{
  family <- match.arg(family)
  X <- mexgb_matrix(X, "X")
  Z <- mexgb_matrix(Z, "Z")
  mexgb_rows(id, time, nrow(X))
  if (is.factor(id)) id <- as.character(id)
  for (nm in c("iter", "ntree", "interaction.depth", "inner", "screen_iter", "nthread"))
    mexgb_number(get(nm), nm, 1, integer = TRUE)
  mexgb_number(seed, "seed", 0, .Machine$integer.max, integer = TRUE)
  mexgb_number(stall_iter, "stall_iter", 1, integer = TRUE, allow_inf = TRUE)
  for (nm in c("delta", "eps", "inner_tol", "shrinkage", "n.minobsinnode"))
    mexgb_number(get(nm), nm, 0, open_lower = TRUE)
  for (nm in c("lambda", "alpha", "gamma")) mexgb_number(get(nm), nm, 0)
  for (nm in c("subsample", "colsample_bytree"))
    mexgb_number(get(nm), nm, 0, 1, open_lower = TRUE)
  mexgb_number(lin_alpha, "lin_alpha", 0, 1)
  if (!is.null(min_child_weight)) mexgb_number(min_child_weight, "min_child_weight", 0)
  for (nm in c("calibrate", "calibrate_binomial", "lin_relax", "zfixed", "heldout_blup",
               "heldout_dscale", "heldout_offset", "verbose")) mexgb_flag(get(nm), nm)
  if (!identical(linear, "auto")) mexgb_flag(linear, "linear")
  if (is.null(crossfit)) crossfit <- 0
  mexgb_number(crossfit, "crossfit", 0, integer = TRUE)
  if (crossfit == 1 || crossfit > nrow(X))
    stop("crossfit must be 0 or an integer from 2 to the number of observations.")
  if (crossfit == 0) {
    if (identical(linear, "auto")) linear <- FALSE
    if (isTRUE(linear)) {
      warning("linear = TRUE requires crossfit >= 2; fitting the boosted trees without the linear part.",
              call. = FALSE)
      linear <- FALSE
    }
  }
  tree_method <- match.arg(tree_method, c("auto", "exact", "hist", "approx"))
  init <- match.arg(init)
  ## init = "auto": "zero" for continuous outcomes (the first M-step is fitted before the random
  ## effects can absorb between-individual variation that x explains), "lmm" otherwise
  if (init == "auto") init <- if (family == "gaussian") "zero" else "lmm"
  ## linear = "auto": Gaussian -> linear part + trees versus trees only decided by the held-out
  ## out-of-fold loss at the first M-step; binomial / cox -> linear part always included
  if (identical(linear, "auto") && family != "gaussian") linear <- TRUE
  ## few predictors (p <= sqrt(N)): the linear part has negligible variance and is kept; the
  ## out-of-fold choice between the two structures is only worth its selection noise when p is large
  if (identical(linear, "auto") && ncol(as.matrix(X)) <= sqrt(length(Y))) linear <- TRUE
  lin_auto <- identical(linear, "auto"); n_lin <- 0
  dispersion <- match.arg(dispersion)
  ntree_select <- match.arg(ntree_select)
  criterion <- match.arg(criterion)
  screen <- match.arg(screen)
  screen_sets_cur <- NULL; n_screen <- 0; lin_dec <- NULL; scr_dec <- NULL
  estep <- match.arg(estep)
  X <- as.matrix(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("Var", seq_len(ncol(X)))
  if (family == "cox") {
    if (inherits(Y, "Surv") && !identical(attr(Y, "type"), "right"))
      stop("Only right-censored Surv responses are supported; start-stop and interval data are not supported.")
    Ymat <- as.matrix(Y)
    if (!is.numeric(Ymat) || any(!is.finite(Ymat)))
      stop("Survival responses must be finite numeric values.")
    if (ncol(Ymat) != 2) stop("For family = 'cox', Y must be a two-column matrix (time, status) or a Surv object.")
    stime <- as.numeric(Ymat[, 1]); status <- as.numeric(Ymat[, ncol(Ymat)])
    if (!all(status %in% c(0, 1))) stop("The status column must be coded 0 (censored) / 1 (event).")
    if (any(stime <= 0)) stop("Survival times must be positive.")
    if (!any(status == 1)) stop("At least one observed event is required.")
    Y <- stime
  } else {
    if (!is.numeric(Y) || any(!is.finite(Y)) ||
        (!is.null(dim(Y)) && (length(dim(Y)) != 2 || ncol(Y) != 1)))
      stop("Y must be a finite numeric vector or one-column matrix.")
    Y <- as.numeric(Y)
  }
  Z <- as.matrix(Z)
  if (nrow(X) != length(Y) || nrow(Z) != length(Y) || length(id) != length(Y))
    stop("X, Y, Z and id must have the same number of observations.")
  if (family == "binomial" && !all(Y %in% c(0, 1)))
    stop("For family = 'binomial', Y must be coded 0/1.")
  if (family == "binomial" && length(unique(Y)) != 2)
    stop("Both binary outcome classes must be present.")
  if (is.null(min_child_weight))
    min_child_weight <- if (family == "gaussian") n.minobsinnode else n.minobsinnode / 4

  params <- list(objective = "reg:squarederror", eta = shrinkage,
                 max_depth = interaction.depth, min_child_weight = min_child_weight,
                 lambda = lambda, alpha = alpha, gamma = gamma, subsample = subsample,
                 colsample_bytree = colsample_bytree, nthread = nthread,
                 tree_method = if (identical(tree_method, "auto"))
                   (if (nrow(X) <= 10000) "exact" else "hist") else tree_method)
  ## xgboost >= 2 takes its seed as a parameter; xgboost 1.x uses R's RNG (set.seed)
  if (utils::packageVersion("xgboost") >= "2.0.0") params$seed <- seed

  q <- dim(Z)[2]
  nind <- length(unique(id))
  btilde <- matrix(0, nind, q)
  sigmahat <- 1
  Btilde <- diag(rep(1, q))
  epsilonhat <- rep(0, length(Y))
  id_btilde <- unique(id)
  Tiime <- sort(unique(time))
  sigma2 <- 1
  Vrai <- NULL
  inc <- 1
  OOB <- NULL
  converged <- FALSE
  baseline <- NULL
  f0 <- 0
  best <- list(ll = Inf); stall <- 0; stopped_stall <- FALSE
  keep_best <- function(i, val = Vrai[i]) {   # remember the state with the smallest criterion
    if (val < best$ll) {
      best <<- list(ll = val, iter = i, forest = forest, btilde = btilde, Btilde = Btilde,
                    sigmahat = sigmahat, fhat = fhat, fhat_full = fhat_full, cal_slope = cal_slope,
                    cal_center = cal_center, nrounds = nrounds_i, beta_z = beta_z, fz = fz,
                    screen_sets = screen_sets_cur, linear = linear)
      stall <<- 0
    } else stall <<- stall + 1
  }
  indiv_list <- lapply(id_btilde, function(u) which(id == u))
  ## relative change of the variance components between two inner (E/M) iterations
  ## Cox: a common shift of the random intercepts is absorbed by the (Breslow) baseline hazard,
  ## so the likelihood is flat in the mean of b0 and it drifts, inflating D_11 until the EM
  ## diverges. Identifiability constraint of frailty models: random intercepts centred to mean 0.
  int_col <- which(apply(Z, 2, function(z) all(abs(z - 1) < 1e-12)))
  ## Population (fixed) coefficients of the random-effect design, beta_z: the random effects are
  ## deviations around them (E b_i = 0), as in a linear mixed model with Z also in the fixed
  ## part. Without beta_z, a population trend in a column of Z that f(x) does not capture (e.g.
  ## a common slope on time) is absorbed by the mean of the BLUPs, which inflates D (its mean
  ## enters D = mean(b b') + C) and under-shrinks the random effects (MEXGB <= 0.4). After every
  ## BLUP update the mean of the random effects is moved into beta_z. Cox: the intercept of
  ## beta_z is not identified (absorbed by the baseline hazard) and is simply dropped, which is
  ## the identifiability constraint of frailty models.
  beta_z <- rep(0, ncol(Z)); fz <- rep(0, length(Y))
  centre_b <- function(b) {
    m <- colMeans(b)
    if (family == "cox" && length(int_col) == 1) m_fix <- replace(m, int_col, 0) else m_fix <- m
    if (!isTRUE(zfixed)) { if (family == "cox" && length(int_col) == 1) m <- replace(m * 0, int_col, m[int_col]) else m <- m * 0; m_fix <- m_fix * 0 }
    beta_z <<- beta_z + m_fix
    fz <<- as.numeric(Z %*% beta_z)
    sweep(b, 2, m, "-")
  }
  vc_change <- function(B0, s0, B1, s1)
    max(abs(B1 - B0)) / max(max(abs(B0)), 1e-8) + abs(as.numeric(s1) - as.numeric(s0)) / max(abs(as.numeric(s0)), 1e-8)
  Zb <- function(b) {
    out <- rep(0, length(Y))
    for (k in 1:nind) {
      indiv <- indiv_list[[k]]
      out[indiv] <- Z[indiv, , drop = FALSE] %*% b[k, ]
    }
    out
  }

  ## Cross-fitted (out-of-fold) E-step. The boosted fit of f is evaluated on the training
  ## rows themselves; with hundreds of trees it interpolates the partial residuals, so the
  ## in-sample residuals Y - f_hat no longer contain the subject-level deviations and the
  ## E-step shrinks the random effects (and D) towards zero -- MEXGB then collapses to plain
  ## XGBoost. With crossfit = V >= 2 the observations of every individual are spread over V
  ## folds; f is refitted V times and the BLUPs, residual variance, random-effect covariance
  ## and likelihood are computed from out-of-fold predictions of f (individuals are still
  ## seen in every fold). The booster used for prediction is the full-data fit.
  N <- length(Y)
  use_cf <- !is.null(crossfit) && crossfit >= 2
  ## Residual structure of the binary / cox PQL working model.
  ##   "none"          : V_i = Z D Z' + W^{-1}            (dispersion 1; MEXGB 0.2)
  ##   "multiplicative": V_i = Z D Z' + phi W^{-1}        (as MASS::glmmPQL)
  ##   "additive"      : V_i = Z D Z' + tau2 I + W^{-1}   (tau2 = variance of the out-of-fold
  ##                                                       error of the linear predictor)
  ## "auto": cox with cross-fitting -> "multiplicative" (with dispersion 1 the extra variation of
  ## the out-of-fold Cox predictions inflates D and the EM diverges at high p; "additive" does not
  ## prevent it); binary -> "none" (best test AUC in the 0.3.1 benchmark; the drift of D over
  ## the iterations is a property of PQL with few binary measurements per subject -- glmer
  ## shows the same upward bias -- and the best-likelihood iteration is returned).
  disp_mode <- if (family == "gaussian") "none" else if (dispersion == "auto")
    (if (use_cf && family == "cox" && !isTRUE(calibrate)) "multiplicative" else "none") else dispersion
  cal_slope <- 1     # calibration slope of the boosted Cox fixed effect (see 'calibrate')
  cal_center <- 0    # the slope rescales (fixed effect - cal_center)
  nrounds_i <- ntree
  status_all <- if (family == "cox") status else rep(1, length(Y))
  cf_arg <- NULL
  tau2 <- 1          # starting value on the linear-predictor scale (0 is a fixed point of EM)
  diverged <- FALSE
  if (use_cf) {
    cf_fold <- cf_make_folds(id, crossfit, seed)
    for (v in seq_len(crossfit)) {
      tr <- cf_fold != v; te <- !tr
      if (!any(te) || sum(tr) < 2L)
        stop("Cross-fitting produced an empty fold or fewer than two training rows; reduce crossfit or supply more observations.")
      if (family == "binomial" && length(unique(Y[tr])) < 2L)
        stop("Each cross-fitting training set must contain both binary classes; change seed or crossfit.")
      if (family == "cox" && !any(status[tr] == 1))
        stop("Each survival training fold must contain an event; change seed or reduce crossfit.")
    }
    cf_arg <- cf_fold
  }
  laplace_diagnostics <- vector("list", iter)
  oof <- function(fit_rows) {
    out <- numeric(N)
    for (v in seq_len(crossfit)) {
      te <- which(cf_fold == v)
      if (length(te) == 0) next
      out[te] <- fit_rows(setdiff(seq_len(N), te), te)
    }
    out
  }

  ## Starting values (init = "lmm"): the random effects, D and sigma^2 of the linear mixed
  ## model with a constant fixed effect (Gaussian: Y = mu + Z b + e; binomial: PQL around the
  ## logit of the event rate), so that the first boosting fit is already made on Y - Z b
  ## instead of on Y (where it would absorb the subject-level variation).
  if (init == "lmm" && family != "cox") {
    if (family == "gaussian") { f_init <- rep(mean(Y), N); yw <- Y; w_init <- rep(1, N) }
    else {
      f0i <- stats::qlogis(min(max(mean(Y), 1e-4), 1 - 1e-4)); f_init <- rep(f0i, N)
    }
    for (it0 in 1:25) {
      if (family == "binomial") {
        eta0 <- pmin(pmax(f_init + fz + Zb(btilde), -30), 30); mu0 <- stats::plogis(eta0)
        w_init <- pmax(mu0 * (1 - mu0), eps); yw <- eta0 + (Y - mu0) / w_init
        s2 <- 1
      } else s2 <- as.numeric(sigmahat)
      for (k in 1:nind) {
        indiv <- indiv_list[[k]]; Zk <- Z[indiv, , drop = FALSE]
        V <- Zk %*% Btilde %*% t(Zk) + diag(s2 / w_init[indiv], length(indiv), length(indiv))
        btilde[k, ] <- Btilde %*% t(Zk) %*% solve(V) %*% (yw[indiv] - f_init[indiv] - fz[indiv])
      }
      btilde <- centre_b(btilde)
      epsilonhat <- yw - f_init - fz - Zb(btilde)
      if (family == "gaussian") {
        sigm <- sigmahat
        sigmahat <- sig(sigma = sigmahat, id = id, Z = Z, epsilon = epsilonhat, Btilde = Btilde)
        Btilde <- bay(bhat = btilde, Bhat = Btilde, Z = Z, id = id, sigmahat = sigm)
      } else Btilde <- bay_pql(bhat = btilde, Bhat = Btilde, Z = Z, id = id, w = w_init)
    }
  }

  ## BLUP of every row's random effect from the OTHER cross-fitting folds of its individual
  zb_heldout <- function(r, B, s2) {
    out <- numeric(N)
    for (k in 1:nind) {
      indiv <- indiv_list[[k]]
      for (v in unique(cf_fold[indiv])) {
        te <- indiv[cf_fold[indiv] == v]; trr <- indiv[cf_fold[indiv] != v]
        if (length(trr) == 0) next
        Zt <- Z[trr, , drop = FALSE]
        Vt <- Zt %*% B %*% t(Zt) + diag(s2, length(trr), length(trr))
        bk <- B %*% t(Zt) %*% solve(Vt, r[trr])
        out[te] <- as.numeric(Z[te, , drop = FALSE] %*% bk)
      }
    }
    out
  }
  heldout_loss <- rep(NA_real_, iter); d_scale <- rep(NA_real_, iter)
  f_prev0 <- if (init == "lmm" && family == "gaussian") f_init else rep(mean(Y), N)

  if (family == "gaussian") {

    for (i in 1:iter) {
      ystar <- rep(NA, length(Y))
      for (k in 1:nind) {
        indiv <- indiv_list[[k]]
        ystar[indiv] <- Y[indiv] - fz[indiv] - Z[indiv, , drop = FALSE] %*%
          btilde[k, ]
      }

      zb <- Zb(btilde) + fz
      ## held-out out-of-fold offset: for the rows of fold v, the BLUP computed from the other
      ## rows of the same individual only (as for a new measurement at prediction time)
      ## (with init = "zero" the first offset is 0 for all rows, which already excludes the held-out rows)
      zb_oof <- if (use_cf && heldout_blup && heldout_offset && !(i == 1 && init == "zero")) zb_heldout(Y - fz - (if (i == 1) f_prev0 else fhat), Btilde,
                                                as.numeric(sigmahat)) + fz else NULL
      ms <- mexgb_mstep(X, Y, status_all, offset = zb, offset_oof = zb_oof, family = "gaussian", params = params,
                        ntree = ntree, cf_fold = cf_arg, f0 = mean(ystar), linear = if (lin_auto && n_lin < screen_iter) "auto" else linear,
                        lin_alpha = lin_alpha, ntree_select = ntree_select, lin_relax = lin_relax,
                        screen = screen, screen_sets = if (screen == "auto") NULL else screen_sets_cur)
      forest <- ms$model; nrounds_i <- ms$nrounds
        if (lin_auto && n_lin < screen_iter) { lin_dec <- rbind(lin_dec, ms$cv_loss_both); linear <- isTRUE(ms$linear_chosen); n_lin <- n_lin + 1 }
        if (!is.null(ms$cv_loss_screen)) scr_dec <- rbind(scr_dec, ms$cv_loss_screen)
        if (!is.null(ms$screen_sets)) {   # screening (re)decided at the first 'screen_iter' M-steps
          screen_sets_cur <- if (is.list(ms$screen_sets)) ms$screen_sets else NULL
          n_screen <- n_screen + 1
          if (n_screen >= screen_iter) screen <- "none"
        }
      fhat_full <- ms$fixed_full
      fhat <- ms$fixed_cf
      OOB[i] <- mean((ystar - fhat)^2)
      ## E-step: the BLUP / variance-component EM updates are cheap (no boosting), so they are
      ## iterated to convergence for the current f ('inner' steps); a single step per boosting
      ## refit (MEGB/MEXGB <= 0.3) leaves D and sigma^2 far from their fixed point when the EM
      ## stops, because variance-component EM converges slowly.
      for (j in seq_len(max(1, inner))) {
        B0 <- Btilde; s0 <- sigmahat
        for (k in 1:nind) {
          indiv <- indiv_list[[k]]
          V <- Z[indiv, , drop = FALSE] %*% Btilde %*%
            t(Z[indiv, , drop = FALSE]) + diag(as.numeric(sigmahat),
                                               length(indiv), length(indiv))
          btilde[k, ] <- Btilde %*% t(Z[indiv, , drop = FALSE]) %*%
            solve(V) %*% (Y[indiv] - fhat[indiv] - fz[indiv])
        }
        btilde <- centre_b(btilde)
        epsilonhat <- Y - fhat - fz - Zb(btilde)
        sigm <- sigmahat
        sigmahat <- sig(sigma = sigmahat, id = id, Z = Z,
                        epsilon = epsilonhat, Btilde = Btilde)
        Btilde <- bay(bhat = btilde, Bhat = Btilde, Z = Z,
                      id = id, sigmahat = sigm)
        if (!all(is.finite(Btilde)) || vc_change(B0, s0, Btilde, sigmahat) < inner_tol) break
      }
      ## Honest predictive check of the variance components: each row's random effect is
      ## predicted from the OTHER cross-fitting folds of its individual and D is scaled by the
      ## factor c that minimises this out-of-fold squared error. c = 1 (the EM estimate) when
      ## the random effects predict new measurements as the model says; c < 1 when they do not
      ## (e.g. the between-individual variation is explained by x, and the random effects only
      ## absorb what f has not yet learned). The shrunken D is used for the BLUPs of the next
      ## M-step, so that f is fitted on the part of the signal the random effects do not predict.
      hon <- NA_real_
      if (use_cf && heldout_blup && heldout_dscale && all(is.finite(Btilde))) {
        r_cf <- Y - fhat - fz
        cs <- c(0, 1/16, 1/8, 1/4, 1/2, 1)
        Ec <- vapply(cs, function(cc) if (cc == 0) r_cf^2 else
          (r_cf - zb_heldout(r_cf, cc * Btilde, as.numeric(sigmahat)))^2, numeric(N))
        Lc <- colSums(Ec)
        ## D is shrunk only when the best scale is significantly better than the EM estimate
        ## (paired comparison of the held-out squared errors, z > 2)
        k_best <- which.min(Lc); dd <- Ec[, length(cs)] - Ec[, k_best]
        zc <- if (k_best < length(cs)) mean(dd) / (stats::sd(dd) / sqrt(N)) else 0
        if (!is.finite(zc) || zc <= 2) k_best <- length(cs)
        c_best <- cs[k_best]; hon <- Lc[k_best] / N
        d_scale[i] <- c_best
        if (c_best < 1) {
          Btilde <- max(c_best, 1e-4) * Btilde
          for (k in 1:nind) {
            indiv <- indiv_list[[k]]
            Vk <- Z[indiv, , drop = FALSE] %*% Btilde %*% t(Z[indiv, , drop = FALSE]) +
              diag(as.numeric(sigmahat), length(indiv), length(indiv))
            btilde[k, ] <- Btilde %*% t(Z[indiv, , drop = FALSE]) %*%
              solve(Vk) %*% (Y[indiv] - fhat[indiv] - fz[indiv])
          }
          btilde <- centre_b(btilde)
        }
      }
      heldout_loss[i] <- hon
      Vrai <- c(Vrai, logV(Y, fhat + fz, Z, time, id, Btilde,
                           0, sigmahat))
      if (!is.finite(Vrai[i]) || any(!is.finite(Btilde))) { diverged <- TRUE; break }
      keep_best(i, if (is.finite(hon)) hon else Vrai[i])
      if (i > 1)
        inc <- abs((Vrai[i - 1] - Vrai[i])/Vrai[i - 1])
      if (inc < delta) {
        converged <- TRUE
        break
      }
      if (stall >= stall_iter) { stopped_stall <- TRUE; break }
    }

  } else {

    ## Non-Gaussian outcomes: same EM/BLUP loop.
    ## M-step: f is fitted by extreme gradient boosting of the exact log-likelihood
    ##   (binomial: logistic loss; cox: Cox partial likelihood) with the current
    ##   random-effect contribution Z b as an offset -- the link-scale analogue of the
    ##   partial residual Y - Z b used for continuous outcomes.
    ## E-step: penalised quasi-likelihood (PQL) BLUP, i.e. the Gaussian E-step applied to
    ##   the working variate ytilde = eta + (y - mu)/w with residual covariance
    ##   W^{-1} = diag(1/w). Binomial: mu = expit(eta), w = mu(1 - mu). Cox (Poisson
    ##   representation of the partial likelihood): mu = Lambda0(t) exp(eta) with the Breslow
    ##   cumulative baseline hazard Lambda0, y = event indicator, w = mu.
    if (family == "binomial") {
      f0 <- stats::qlogis(min(max(mean(Y), 1e-4), 1 - 1e-4))
      params$objective <- "binary:logistic"
    } else {
      params$objective <- "survival:cox"
      lab <- ifelse(status == 1, Y, -Y)
    }
    fhat <- rep(f0, length(Y))
    for (i in 1:iter) {
      zb <- Zb(btilde) + fz

      if (family == "binomial") {
        ms <- mexgb_mstep(X, Y, status_all, offset = zb, family = "binomial", params = params,
                          ntree = ntree, cf_fold = cf_arg, f0 = f0, linear = linear,
                          lin_alpha = lin_alpha, ntree_select = ntree_select, lin_relax = lin_relax,
                        screen = screen, screen_sets = if (screen == "auto") NULL else screen_sets_cur)
        forest <- ms$model; nrounds_i <- ms$nrounds
        if (identical(linear, "auto")) { lin_dec <- ms$cv_loss_both; linear <- isTRUE(ms$linear_chosen) }
        if (!is.null(ms$screen_sets)) {   # screening (re)decided at the first 'screen_iter' M-steps
          screen_sets_cur <- if (is.list(ms$screen_sets)) ms$screen_sets else NULL
          n_screen <- n_screen + 1
          if (n_screen >= screen_iter) screen <- "none"
        }
        fhat_full <- ms$fixed_full; fhat <- ms$fixed_cf
        cal_center <- ms$model$intercept
        ## out-of-fold calibration slope of the boosted logit (see the cox branch)
        if (use_cf && isTRUE(calibrate) && calibrate_binomial) {
          cf_fit <- tryCatch(suppressWarnings(stats::glm(Y ~ I(fhat - cal_center) + offset(cal_center + zb),
                                                         family = stats::binomial())), error = function(e) NULL)
          g <- if (is.null(cf_fit)) NA else unname(stats::coef(cf_fit)[2])
          cal_slope <- if (is.finite(g)) min(max(g, 0.05), 1.5) else 1
          fhat <- cal_center + cal_slope * (fhat - cal_center)
          fhat_full <- cal_center + cal_slope * (fhat_full - cal_center)
        }
      } else {
        ## (the Cox log-hazard is identified up to a constant: every fold model is aligned to
        ##  the full-data fit on its own training rows, inside mexgb_mstep)
        ms <- mexgb_mstep(X, Y, status_all, offset = zb, family = "cox", params = params,
                          ntree = ntree, cf_fold = cf_arg, f0 = 0, linear = linear,
                          lin_alpha = lin_alpha, ntree_select = ntree_select, lin_relax = lin_relax,
                        screen = screen, screen_sets = if (screen == "auto") NULL else screen_sets_cur)
        forest <- ms$model; nrounds_i <- ms$nrounds
        if (identical(linear, "auto")) { lin_dec <- ms$cv_loss_both; linear <- isTRUE(ms$linear_chosen) }
        if (!is.null(ms$screen_sets)) {   # screening (re)decided at the first 'screen_iter' M-steps
          screen_sets_cur <- if (is.list(ms$screen_sets)) ms$screen_sets else NULL
          n_screen <- n_screen + 1
          if (n_screen >= screen_iter) screen <- "none"
        }
        fhat_full <- ms$fixed_full; fhat <- ms$fixed_cf
        ## Recalibration. A boosted Cox model over-fits the partial likelihood: its linear
        ## predictor ranks well (C-index) but its scale is inflated, which makes the PQL working
        ## residuals and hence D explode, and over-weights f against Z b in prediction. The
        ## out-of-fold calibration slope (Cox model of the outcome on the out-of-fold predictor
        ## with offset Z b) rescales the fixed effect in the E-step and in prediction.
        if (use_cf && isTRUE(calibrate)) {
          cf_fit <- tryCatch(survival::coxph(survival::Surv(Y, status) ~ fhat + offset(zb),
                                             ties = "breslow"), error = function(e) NULL)
          g <- if (is.null(cf_fit)) NA else unname(stats::coef(cf_fit)[1])
          cal_slope <- if (is.finite(g)) min(max(g, 0.05), 1.5) else 1
          fhat <- cal_slope * fhat; fhat_full <- cal_slope * fhat_full
        }
      }
      ## E-step: PQL linearisation + variance components iterated to convergence for the
      ## current f ('inner' steps; standard PQL with f as offset). One step per boosting refit
      ## (MEXGB <= 0.3) stopped far from the fixed point: D was under-estimated and the
      ## random effects over-shrunk.
      for (j in seq_len(max(1, inner))) {
        B0 <- Btilde; s0 <- sigmahat
        eta <- pmin(pmax(fhat + fz + Zb(btilde), -30), 30)
        if (family == "binomial") {
          mu <- stats::plogis(eta)
          w <- pmax(mu * (1 - mu), eps)
          ytilde <- eta + (Y - mu) / w          # PQL working (adjusted dependent) variate
        } else {
          mu <- breslow(Y, status, eta)$cumhaz_at * exp(eta)
          w <- pmax(mu, eps)
          ytilde <- eta + (status - mu) / w     # Poisson-PQL working variate
        }
        if (disp_mode == "additive") {
          upd <- pql_em_additive(ytilde - fhat - fz, Z, Btilde, tau2, w, indiv_list)
          btilde <- centre_b(upd$b); Btilde <- upd$D; tau2 <- upd$tau2; sigmahat <- tau2
          ll_i <- upd$ll
        } else {
          for (k in 1:nind) {
            indiv <- indiv_list[[k]]
            V <- Z[indiv, , drop = FALSE] %*% Btilde %*%
              t(Z[indiv, , drop = FALSE]) + diag(as.numeric(sigmahat) / w[indiv], length(indiv), length(indiv))
            btilde[k, ] <- Btilde %*% t(Z[indiv, , drop = FALSE]) %*%
              solve(V) %*% (ytilde[indiv] - fhat[indiv] - fz[indiv])
          }
          btilde <- centre_b(btilde)
          if (disp_mode == "multiplicative") {
            ## rows scaled by W^{1/2} give the homoscedastic Gaussian model -> sig() and bay()
            sw <- sqrt(w)
            res_w <- sw * (ytilde - fhat - fz - Zb(btilde))
            phi_old <- as.numeric(sigmahat)
            sigmahat <- as.numeric(sig(sigma = phi_old, id = id, Z = Z * sw, epsilon = res_w,
                                       Btilde = Btilde))
            Btilde <- bay(bhat = btilde, Bhat = Btilde, Z = Z * sw, id = id, sigmahat = phi_old)
          } else {
            Bold <- Btilde
            Btilde <- bay_pql(bhat = btilde, Bhat = Bold, Z = Z, id = id, w = w)
          }
          ll_i <- logV_pql(ytilde, fhat + fz, Z, id, Btilde, w / as.numeric(sigmahat))
        }
        if (!is.finite(ll_i) || !all(is.finite(Btilde)) || vc_change(B0, s0, Btilde, sigmahat) < inner_tol) break
      }
      ## Laplace E-step: the PQL fixed point under-estimates D (Breslow and Lin, 1996). D is
      ## re-estimated by maximising the Laplace approximation of the marginal likelihood
      ## (random effects at their posterior modes given D, Newton/PQL iterations), as in
      ## lme4::glmer (nAGQ = 1) and coxme, with the PQL solution as starting value.
      if (estep == "laplace" && disp_mode == "none" && all(is.finite(Btilde))) {
        la <- laplace_estep(family, Y, status_all, fhat + fz, btilde, Btilde, Z, indiv_list, eps,
                            centre = (family == "cox" && length(int_col) == 1), int_col = int_col)
        laplace_diagnostics[[i]] <- la[c("converged", "code", "message")]
        if (!is.null(la$D)) { btilde <- centre_b(la$b); Btilde <- la$D }
      }
      ## Comparable criterion across iterations: -2 x Laplace approximation of the marginal
      ## log-likelihood at the cross-fitted fixed effect (the PQL working likelihood changes
      ## with the linearisation point and cannot be compared between outer iterations)
      Vrai <- c(Vrai, if (criterion == "laplace")
        laplace_crit(family, Y, status_all, fhat + fz, Zb(btilde), btilde, Btilde, Z, indiv_list, eps) else ll_i)
      zb_new <- Zb(btilde) + fz
      if (family == "binomial") {
        mu_new <- pmin(pmax(stats::plogis(fhat + zb_new), 1e-12), 1 - 1e-12)
        OOB[i] <- -2 * mean(Y * log(mu_new) + (1 - Y) * log(1 - mu_new))
      } else {
        OOB[i] <- -2 * cox_partial_loglik(Y, status, fhat + zb_new) / length(Y)
      }
      if (!is.finite(Vrai[i]) || any(!is.finite(Btilde))) { diverged <- TRUE; break }
      keep_best(i)
      if (i > 1)
        inc <- abs((Vrai[i - 1] - Vrai[i])/Vrai[i - 1])
      if (inc < delta) {
        converged <- TRUE
        break
      }
      if (stall >= stall_iter) { stopped_stall <- TRUE; break }
    }
  }
  if (diverged && !is.finite(best$ll))
    stop("the EM diverged in its first iteration (non-finite likelihood)")

  ## Not converged within 'delta': return the state with the best (smallest) -2 log-likelihood
  if ((!converged || (use_cf && heldout_blup && family == "gaussian")) && is.finite(best$ll) && best$iter != i) {
    forest <- best$forest; btilde <- best$btilde; Btilde <- best$Btilde
    sigmahat <- best$sigmahat; fhat <- best$fhat; fhat_full <- best$fhat_full
    cal_slope <- best$cal_slope; cal_center <- best$cal_center; nrounds_i <- best$nrounds
    beta_z <- best$beta_z; fz <- best$fz
    screen_sets_cur <- best$screen_sets; linear <- best$linear
  }
  if (family == "cox") {
    bl <- breslow(Y, status, fhat_full + fz + Zb(btilde))
    baseline <- list(time = bl$time, cumhaz = bl$cumhaz)
  }

  failed_laplace <- which(vapply(laplace_diagnostics, function(x)
    !is.null(x) && is.na(x$code), logical(1)))
  if (length(failed_laplace)) warning("Laplace update failed at iteration(s) ",
    paste(failed_laplace, collapse = ", "), "; retained the PQL update. See laplace_diagnostics.", call. = FALSE)
  if (verbose) {
    if (converged) print(paste0("stopped after ", i, " iterations."))
    else if (diverged) print(paste0("non-finite likelihood at iteration ", i,
                                    ": stopped, returning iteration ", best$iter, "."))
    else if (stopped_stall) print(paste0("no improvement of the log-likelihood in ", stall_iter,
                                            " iterations: stopped after ", i,
                                            " iterations, returning iteration ", best$iter, "."))
    else print(paste0("maximum number of iterations (", iter, ") reached without convergence; returning iteration ",
                      best$iter, "."))
  }
  results <- list(forest = forest, random_effects = btilde,
                  var_random_effects = Btilde, sigma = sigmahat,
                  id_btilde = unique(id), LL = Vrai,
                  id = id, time = time, OOB = OOB, heldout_loss = heldout_loss, d_scale = d_scale,
                  importance = if (!is.null(forest$beta)) mexgb_importance(forest, X) else
                    if (!is.null(forest$booster)) xgbimportance(forest$booster, colnames(X)) else
                      stats::setNames(rep(0, ncol(X)), colnames(X)),
                  z_names = colnames(Z), laplace_diagnostics = laplace_diagnostics[seq_len(i)],
                  family = family, converged = converged, n_iter = i, diverged = diverged,
                  best_iter = if (converged && !(use_cf && heldout_blup && family == "gaussian")) i else best$iter,
                  feature_names = colnames(X), ntree = ntree, f0 = f0, cal_slope = cal_slope, cal_center = cal_center,
                  nrounds = nrounds_i, linear = linear, beta_z = beta_z,
                  lin_decision = lin_dec, screen_decision = scr_dec, screened = if (is.list(screen_sets_cur)) colnames(X)[screen_sets_cur$full] else NULL,
                  crossfit = if (use_cf) crossfit else 0,
                  baseline = baseline)
  class(results) <- "MEXGB"
  return(results)
}


#' Predict with a fitted MEXGB model.
#'
#' Predictions combine the boosted fixed effect with the predicted random effects of the
#' individuals seen during training. For individuals that were not in the training sample the
#' random effects are set to zero while retaining the population coefficients of Z.
#' For binary and survival outcomes this is conditional prediction at zero random
#' effects, not integration over the random-effects distribution.
#'
#' @param object : a \code{MEXGB} output of the MEXGB function.
#' @param X [matrix]: matrix of the fixed effects for the new observations to be predicted.
#' @param Z [matrix]: matrix of the random effects for the new observations to be predicted.
#' @param id [vector]: vector of the identifiers of the new observations to be predicted.
#' @param time [vector]: vector of the time measurements of the new observations to be predicted.
#' @param type [character]: \code{"response"} (default; identity for gaussian, probability for binomial, relative hazard \eqn{exp(\eta)} for cox), \code{"link"} (linear predictor; log relative hazard for cox) or \code{"survival"} (cox only: matrix of predicted survival probabilities \eqn{S(t|x)=\exp\{-\Lambda_0(t)e^{\eta}\}} at \code{times}, with the Breslow baseline estimated from the returned fit).
#' @param times [vector]: time points for \code{type = "survival"} (default: the distinct training event times).
#' @param ... : low levels arguments (ignored; kept for compatibility with \code{predict.MEGB}).
#'
#' @import stats
#' @import xgboost
#'
#' @return Numeric prediction vector, or a matrix with observations in rows and requested times in columns for survival predictions.
#'
#' @export
#'
#' @examples
#' # See MEXGB() for fitted-model prediction examples.
#'
predict.MEXGB <- function(object, X, Z, id, time, type = c("response", "link", "survival"),
                          times = NULL, ...){
  type <- match.arg(type)
  if (type == "survival" && object$family != "cox")
    stop("type = 'survival' is only available for family = 'cox'.")
  X <- mexgb_matrix(X, "X")
  Z <- mexgb_matrix(Z, "Z")
  mexgb_rows(id, time, nrow(X))
  if (is.factor(id)) id <- as.character(id)
  if (nrow(Z) != nrow(X) || ncol(Z) != ncol(object$random_effects))
    stop("Z must have one row per prediction and the training number of columns.")
  if (ncol(X) != length(object$feature_names))
    stop("X must have the same number of columns as the training predictors.")
  if (!is.null(colnames(X)) && !setequal(colnames(X), object$feature_names))
    stop("X column names must match the training predictors.")
  if (!is.null(object$z_names) && !is.null(colnames(Z))) {
    if (!setequal(colnames(Z), object$z_names)) stop("Z column names must match the training design.")
    Z <- Z[, object$z_names, drop = FALSE]
  }
  if (!is.null(times) && (!is.numeric(times) || any(!is.finite(times)) || any(times < 0)))
    stop("times must contain finite, non-negative prediction times.")
  if (is.null(colnames(X))) colnames(X) <- object$feature_names
  X <- X[, object$feature_names, drop = FALSE]
  Z <- as.matrix(Z)
  f <- fixedMEXGB(object, X)
  if (!is.null(object$beta_z)) f <- f + as.numeric(Z %*% object$beta_z)   # population part of Z
  Ypred <- f
  for (i in unique(id)){
    w <- which(id == i)
    k <- match(i, object$id_btilde)
    if (!is.na(k))
      Ypred[w] <- f[w] + Z[w, , drop = FALSE] %*% object$random_effects[k, ]
  }
  Ypred <- as.numeric(Ypred)
  if (type == "link") return(Ypred)
  if (object$family == "binomial") return(stats::plogis(Ypred))
  if (object$family == "cox") {
    if (type == "response") return(exp(Ypred))
    if (is.null(times)) times <- object$baseline$time
    H0 <- c(0, object$baseline$cumhaz)[findInterval(times, object$baseline$time) + 1]
    S <- exp(-outer(exp(Ypred), H0))
    colnames(S) <- format(times)
    return(S)
  }
  if (type == "survival") stop("type = 'survival' is only available for family = 'cox'.")
  Ypred
}


#' Relative influence of the predictors in a fitted MEXGB model.
#'
#' @param object : a \code{MEXGB} output of the MEXGB function.
#' @param top [numeric]: number of predictors to return (default all predictors with non-zero influence).
#' @param ... : not used.
#'
#' @return A data frame with columns \code{var} and \code{rel.inf} (percentage of the total
#' contribution), sorted by decreasing influence. With a linear component, importance
#' combines mean absolute SHAP and centred linear contributions; without one it
#' uses split gain. An empty data frame is returned when all contributions are zero.
#'
#' @export
summary.MEXGB <- function(object, top = NULL, ...){
  imp <- sort(object$importance, decreasing = TRUE)
  imp <- imp[imp > 0]
  if (!is.null(top)) {
    mexgb_number(top, "top", 1, integer = TRUE)
    imp <- utils::head(imp, top)
  }
  out <- data.frame(var = names(imp), rel.inf = as.numeric(imp), row.names = names(imp))
  out
}


#' Print a fitted MEXGB model
#' @param x A fitted MEXGB model.
#' @param ... Unused.
#' @return The model, invisibly.
#' @export
print.MEXGB <- function(x, ...){
  cat("Mixed Effect Extreme Gradient Boosting (MEXGB)\n")
  cat("  family:", x$family, "\n")
  cat("  individuals:", length(x$id_btilde), " observations:", length(x$id), "\n")
  cat("  random effects:", ncol(x$random_effects), " selected boosting rounds:", x$nrounds, "\n")
  cat("  EM iterations:", x$n_iter, if (x$converged) "(converged)" else "(not converged)", "\n")
  cat("  estimated random-effect covariance:\n")
  print(round(x$var_random_effects, 4))
  if (x$family == "gaussian") cat("  estimated residual variance:", round(as.numeric(x$sigma), 4), "\n")
  invisible(x)
}


#' Folds of the cross-fitted E-step (internal).
#'
#' The observations of every individual are spread as evenly as possible over \code{V} folds,
#' so every individual keeps observations in every training part. Drawn under a private seed;
#' the caller's random-number stream is left untouched.
#' @keywords internal
cf_make_folds <- function(id, V, seed = 1){
  old_seed <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = globalenv()) else
    rm(".Random.seed", envir = globalenv()))
  set.seed(seed)
  fold <- integer(length(id))
  for (u in unique(id)) {
    r <- which(id == u)
    f <- rep_len(sample.int(V), length(r))
    fold[r] <- f[sample.int(length(r))]
  }
  fold
}

#' Fit one xgboost regression ensemble (internal).
#'
#' @import xgboost
#' @keywords internal
xgbfit <- function(X, label, weight = NULL, params, ntree){
  if (is.null(weight)) {
    dtrain <- xgboost::xgb.DMatrix(data = X, label = label)
    params$base_score <- mean(label)
  } else {
    dtrain <- xgboost::xgb.DMatrix(data = X, label = label, weight = weight)
    params$base_score <- sum(weight * label) / sum(weight)
  }
  xgboost::xgb.train(params = params, data = dtrain, nrounds = ntree, verbose = 0)
}

#' Fit one logistic xgboost ensemble with an offset (internal).
#'
#' The offset (random-effect contribution plus the constant \code{f0}) enters as the
#' \code{xgboost} \code{base_margin}; \code{base_score = 0.5} makes the global bias zero on the
#' logit scale, so \code{predict(..., outputmargin = TRUE)} on a matrix returns the tree sum only.
#'
#' @import xgboost
#' @keywords internal
xgbfit_logit <- function(X, Y, offset, params, ntree){
  dtrain <- xgboost::xgb.DMatrix(data = X, label = Y, base_margin = offset)
  params$base_score <- 0.5
  xgboost::xgb.train(params = params, data = dtrain, nrounds = ntree, verbose = 0)
}

#' Population (fixed-effect) prediction of a MEXGB model on the link scale (internal).
#'
#' @keywords internal
fixedMEXGB <- function(object, X){
  f <- mexgb_fixed_predict(object$forest, X)
  if (object$family %in% c("binomial", "cox")) {
    cc <- object$cal_center %||% 0
    cc + (object$cal_slope %||% 1) * (f - cc)
  } else f
}

#' Fit one Cox-partial-likelihood xgboost ensemble with an offset (internal).
#'
#' \code{label} is the xgboost survival label (time for events, -time for censored
#' observations); the offset (random-effect contribution) enters as \code{base_margin}.
#'
#' @import xgboost
#' @keywords internal
xgbfit_cox <- function(X, label, offset, params, ntree){
  dtrain <- xgboost::xgb.DMatrix(data = X, label = label, base_margin = offset)
  xgboost::xgb.train(params = params, data = dtrain, nrounds = ntree, verbose = 0)
}

#' Breslow estimator of the cumulative baseline hazard (internal).
#'
#' @param time,status observed times and event indicators.
#' @param eta linear predictors (log relative hazards).
#' @param at times at which the cumulative baseline hazard is returned (default \code{time}).
#' @return list with the distinct event times \code{time}, the cumulative baseline hazard
#'   \code{cumhaz} at those times and \code{cumhaz_at} evaluated at \code{at}.
#' @keywords internal
breslow <- function(time, status, eta, at = time){
  r <- exp(eta)
  ut <- sort(unique(time[status == 1]))
  if (length(ut) == 0) return(list(time = numeric(0), cumhaz = numeric(0), cumhaz_at = rep(0, length(at))))
  d <- tabulate(match(time[status == 1], ut), nbins = length(ut))
  o <- order(time)
  ts <- time[o]
  cs <- rev(cumsum(rev(r[o])))                     # risk-set sums for sorted times
  R <- cs[findInterval(ut, ts, left.open = TRUE) + 1]
  H <- cumsum(d / R)
  list(time = ut, cumhaz = H, cumhaz_at = c(0, H)[findInterval(at, ut) + 1])
}

#' Cox partial log-likelihood with Breslow ties (internal).
#'
#' @keywords internal
cox_partial_loglik <- function(time, status, eta){
  r <- exp(eta)
  o <- order(time)
  ts <- time[o]
  cs <- rev(cumsum(rev(r[o])))
  ev <- which(status == 1)
  R <- cs[findInterval(time[ev], ts, left.open = TRUE) + 1]
  sum(eta[ev] - log(R))
}

#' Relative influence from an xgboost model (internal).
#'
#' @keywords internal
xgbimportance <- function(forest, feature_names){
  imp <- stats::setNames(rep(0, length(feature_names)), feature_names)
  tab <- tryCatch(as.data.frame(xgboost::xgb.importance(model = forest)),
                  error = function(e) NULL)
  if (!is.null(tab) && nrow(tab) > 0) {
    m <- match(tab$Feature, feature_names)
    ok <- !is.na(m)
    imp[m[ok]] <- 100 * tab$Gain[ok] / sum(tab$Gain[ok])
  }
  imp
}

#' Title
#'
#'
#' @import stats
#'
#' @keywords internal
sig <- function(sigma,id,Z, epsilon, Btilde){ #### update of the residual variance
  nind <- length(unique(id))
  Nombre <- length(id)
  sigm <- 0
  for (j in 1:nind){
    w <- which(id==unique(id)[j])
    V <- Z[w,, drop=FALSE]%*%Btilde%*%t(Z[w,, drop=FALSE])+diag(as.numeric(sigma),length(w),length(w))
    sigm <- sigm + t(epsilon[w])%*%epsilon[w] + sigma*(length(w)-sigma*(sum(diag(solve(V)))))
  }
  sigm <- sigm/Nombre
  return(sigm)
}

#' Title
#'
#' @import stats
#'
#' @keywords internal
bay <- function(bhat,Bhat,Z,id, sigmahat){ #### update of the random-effects covariance B
  nind <- length(unique(id))
  q <- dim(Z)[2]
  Nombre <- length(id)
  D <- 0
  for (j in 1:nind){
    w <- which(id==unique(id)[j])
    V <- Z[w,, drop=FALSE]%*%Bhat%*%t(Z[w,, drop=FALSE])+diag(as.numeric(sigmahat),length(w),length(w))
    D <- D+ (bhat[j,]%*%t(bhat[j,]))+ (Bhat- Bhat%*%t(Z[w,, drop=FALSE])%*%solve(V)%*%Z[w,, drop=FALSE]%*%Bhat)
  }
  D <- D/nind
  return(D)
}

#' Title
#'
#' @import stats
#'
#' @keywords internal
logV <- function(Y,f,Z,time,id,B,gamma,sigma){ # -2 log-likelihood (up to a constant) at M stage of EM
  Vraisem <- 0
  for (i in 1:length(unique(id))){
    w <- which(id==unique(id)[i])
    V <- Z[w,,drop=FALSE]%*%B%*%t(Z[w,,drop=FALSE])+diag(as.numeric(sigma),length(w),length(w))
    Vraisem <- Vraisem + log(det(V))+ t(Y[w]-f[w])%*%solve(V)%*%(Y[w]-f[w])
  }
  return(as.numeric(Vraisem))
}

#' One EM step of the additive PQL working model (internal).
#'
#' Working model for the cross-fitted E-step of binary / cox outcomes:
#' \eqn{r_i=\tilde y_i-\hat f_i=Z_ib_i+e_i+\epsilon_i}, \eqn{b_i\sim N(0,D)}, \eqn{e_i\sim N(0,\tau^2I)}
#' (out-of-fold error of the linear predictor), \eqn{\epsilon_i\sim N(0,W_i^{-1})} (PQL working error), so
#' \eqn{V_i=Z_iDZ_i'+\tau^2I+W_i^{-1}}. E-step at the current \eqn{(D,\tau^2)}: BLUPs and conditional
#' moments of \eqn{b_i} and \eqn{e_i}; M-step: \eqn{D=n^{-1}\sum_i E(b_ib_i')},
#' \eqn{\tau^2=N^{-1}\sum_i E(e_i'e_i)}. Returns the BLUPs, the updated parameters and the
#' -2 working log-likelihood (up to a constant) at the updated parameters.
#' @keywords internal
pql_em_additive <- function(r, Z, D, tau2, w, indiv_list){
  nind <- length(indiv_list); N <- length(r)
  b <- matrix(0, nind, ncol(Z)); Dn <- 0; tn <- 0
  for (k in seq_len(nind)) {
    ii <- indiv_list[[k]]; Zk <- Z[ii, , drop = FALSE]; n <- length(ii)
    V <- Zk %*% D %*% t(Zk) + diag(tau2 + 1 / w[ii], n, n)
    Vi <- solve(V); Vr <- Vi %*% r[ii]
    b[k, ] <- D %*% t(Zk) %*% Vr
    Dn <- Dn + b[k, ] %*% t(b[k, ]) + D - D %*% t(Zk) %*% Vi %*% Zk %*% D
    tn <- tn + tau2^2 * sum(Vr^2) + n * tau2 - tau2^2 * sum(diag(Vi))
  }
  Dn <- Dn / nind; tn <- max(as.numeric(tn) / N, 0)
  ll <- 0
  for (k in seq_len(nind)) {
    ii <- indiv_list[[k]]; Zk <- Z[ii, , drop = FALSE]; n <- length(ii)
    V <- Zk %*% Dn %*% t(Zk) + diag(tn + 1 / w[ii], n, n)
    ll <- ll + as.numeric(determinant(V, logarithm = TRUE)$modulus) + sum(r[ii] * solve(V, r[ii]))
  }
  list(b = b, D = Dn, tau2 = tn, ll = ll)
}

#' Update of the random-effects covariance under the PQL working model (internal).
#'
#' Same as \code{bay} with the residual covariance \eqn{\sigma^2 I} replaced by \eqn{W^{-1}}.
#'
#' @keywords internal
bay_pql <- function(bhat, Bhat, Z, id, w){
  nind <- length(unique(id))
  D <- 0
  for (j in 1:nind){
    ww <- which(id==unique(id)[j])
    V <- Z[ww,, drop=FALSE]%*%Bhat%*%t(Z[ww,, drop=FALSE])+diag(1/w[ww],length(ww),length(ww))
    D <- D+ (bhat[j,]%*%t(bhat[j,]))+ (Bhat- Bhat%*%t(Z[ww,, drop=FALSE])%*%solve(V)%*%Z[ww,, drop=FALSE]%*%Bhat)
  }
  D <- D/nind
  return(D)
}

#' PQL working -2 log-likelihood (internal).
#'
#' Same as \code{logV} with the residual covariance \eqn{\sigma^2 I} replaced by \eqn{W^{-1}}.
#'
#' @keywords internal
logV_pql <- function(Y, f, Z, id, B, w){
  Vraisem <- 0
  for (i in 1:length(unique(id))){
    ww <- which(id==unique(id)[i])
    V <- Z[ww,,drop=FALSE]%*%B%*%t(Z[ww,,drop=FALSE])+diag(1/w[ww],length(ww),length(ww))
    Vraisem <- Vraisem + as.numeric(determinant(V, logarithm = TRUE)$modulus) +
      t(Y[ww]-f[ww])%*%solve(V)%*%(Y[ww]-f[ww])
  }
  return(as.numeric(Vraisem))
}


#' Simulate Low/High Dimensional and Linear/Nonlinear Longitudinal dataset (continuous, binary or survival).
#'
#' Simulates the design of the MEXGB papers. Each of the \code{n} individuals is observed at
#' \code{time_points} equally spaced occasions \eqn{t_{ij}} rescaled to \eqn{[0,1]}. The \code{p}
#' fixed-effect predictors of every observation are drawn from \eqn{N_p(0,\Sigma_W)} with AR(1)
#' correlation \eqn{(\Sigma_W)_{gg'}=\rho_W^{|g-g'|}}; only the first six are active.
#' \itemize{
#' \item linear: \eqn{f(x)=2x_1-2x_2+1.5x_3-1.5x_4+x_5-x_6}.
#' \item nonlinear: \eqn{f(x)=2\sin(\pi x_1x_2)+4(x_3-0.5)^2+2x_4+\exp(x_5)+x_6x_1}.
#' }
#' A correlated random intercept and slope \eqn{b_i\sim N_2(0,D)} with
#' \eqn{D_{11}=\tau_0^2}, \eqn{D_{22}=\tau_1^2}, \eqn{D_{12}=\rho_Z\tau_0\tau_1} enters through
#' \eqn{z_{ij}=(1,t_{ij})}.
#' \itemize{
#' \item \code{family="gaussian"}: \eqn{y_{ij}=f(x_{ij})+b_{0i}+b_{1i}t_{ij}+\epsilon_{ij}} with
#' within-subject AR(1) errors \eqn{Cov(\epsilon_{ij},\epsilon_{ij'})=\sigma^2\rho_W^{|j-j'|}}.
#' \item \code{family="binomial"}: \eqn{\eta_{ij}=c_0+\kappa f^*(x_{ij})+b_{0i}+b_{1i}t_{ij}},
#' \eqn{y_{ij}\sim Bernoulli(expit(\eta_{ij}))}, where \eqn{f^*} is the standardised fixed effect.
#' The signal strength \eqn{\kappa} and intercept \eqn{c_0} are either supplied (\code{signal},
#' \code{intercept}) or calibrated by \code{\link{calibrateSim}} so that the pilot-sample expected fixed-effect
#' AUC approximates \code{target_auc} and the event rate equals \code{prevalence}. Without a target the
#' original design is used (\eqn{\kappa=1}, \eqn{c_0} solved so that the mean event probability equals
#' \code{prevalence}).
#' \item \code{family="cox"}: frailty Cox model with Weibull baseline,
#' \eqn{\lambda_{ij}(t)=\lambda\nu t^{\nu-1}\exp(\eta_{ij})}, \eqn{\eta_{ij}=\kappa f^*(x_{ij})+b_{0i}+b_{1i}t_{ij}};
#' \eqn{\kappa} is supplied or calibrated so that the population (fixed-effect) concordance index
#' approximates \code{target_c}. Exponential censoring is calibrated on the realised sample,
#' so its rate is sample-dependent and is not an independently fixed censoring DGP.
#' Rows represent clustered event times, not repeated records of a single terminal event.
#' }
#'
#' @param n Integer number of individuals, at least 2.
#' @param p Positive integer number of predictors (\code{p >= 6} for the nonlinear model).
#' @param rel_p [numeric]: Number of relevant predictors in the linear model (coefficients recycled from \code{c(2,-2,1.5,-1.5,1,-1)}). The default value is \code{rel_p=6}. The nonlinear model always uses the first six predictors.
#' @param time_points Integer number of realizations per individual, at least 2. The default value is \code{time_points=10}.
#' @param rho_W AR(1) correlation strictly between -1 and 1, among predictors and within-subject errors. The default value is \code{rho_W=0.6}.
#' @param rho_Z Correlation between -1 and 1 (inclusive) between the random intercept and slope. The default value is \code{rho_Z=0.6}.
#' @param random_sd_intercept [numeric]: Standard deviation of the random intercept. Default \eqn{\sqrt{0.5}}.
#' @param random_sd_slope [numeric]: Standard deviation of the random slope. Default \eqn{\sqrt{3}}.
#' @param noise_sd [numeric]: Residual standard deviation \eqn{\sigma} (continuous outcome). The default value is \code{noise_sd=0.5}.
#' @param linear [boolean]: If TRUE, the linear fixed effect is simulated, otherwise the nonlinear one.
#' @param family [character]: \code{"gaussian"} (default), \code{"binomial"} or \code{"cox"}.
#' @param target_auc [numeric]: binary only; target AUC in [0.5, 1) of the fixed effect (e.g. 0.85). \code{NULL} = no calibration.
#' @param prevalence Binary target event rate strictly between 0 and 1 (default 0.5).
#' @param target_c [numeric]: cox only; target concordance in [0.5, 1) of the fixed effect (e.g. 0.75). \code{NULL} = \eqn{\kappa=1}.
#' @param signal,intercept [numeric]: optional fixed values of \eqn{\kappa} and \eqn{c_0} (override calibration).
#' @param censoring Cox target censored fraction in [0, 1) (default 0.3).
#' @param weibull_shape,weibull_scale [numeric]: cox only; Weibull baseline \eqn{\nu} (default 1.5) and \eqn{\lambda} (default 1).
#' @param pilot_n [numeric]: number of individuals in the pilot sample used by the calibration (default 300).
#' @param pilot_seed [numeric]: seed of the pilot sample (the calibration is therefore identical across replicates and does not disturb the random-number stream).
#' @param calibration [list]: optional output of \code{\link{calibrateSim}}; when supplied it is used as is (no re-calibration), which keeps the data-generating process identical across simulation replicates and avoids repeating the pilot computation.
#'
#' @import MASS
#' @import stats
#'
#' @return A data frame with n*time_points rows and p+5 columns (p+6 for Cox) containing the following elements: \itemize{
#' \item \code{id:} vector of the individual IDs.
#' \item \code{time:} vector of the time realizations (rescaled to [0,1]).
#' \item \code{Y:} vector of the outcomes variable (observed survival time for \code{cox}).
#' \item \code{RandomIntercept:} vector of the Random Intercept design (ones).
#' \item \code{RandomSlope:} vector of the Random Slope design (time).
#' \item \code{Vars :} Remainder columns corresponding to the fixed effect variables.
#' \item \code{status :} (cox only, last column) event indicator.
#' }
#' The true fixed effect, the true linear predictor (binary, cox), the true random effects and the
#' calibration are returned as attributes \code{f}, \code{eta}, \code{b} and \code{calibration}.
#'
#' @export
#'
#' @examples
#' set.seed(1)
#' data = simLong(n = 20, p = 6, rel_p = 6, time_points = 10, rho_W = 0.6, rho_Z = 0.6,
#'                random_sd_intercept = sqrt(0.5),
#'                random_sd_slope = sqrt(3),
#'                noise_sd = 0.5, linear = FALSE) # Generate the data
#' head(data)   # first six rows of the data.
#' # binary outcome with population AUC 0.85 and prevalence 0.3
#' datab <- simLong(n = 20, p = 6, time_points = 10, linear = FALSE, family = "binomial",
#'                  target_auc = 0.85, prevalence = 0.3)
#' mean(datab$Y); attr(datab, "calibration")
#' # clustered survival outcome with concordance 0.75 and 30% censoring
#' datas <- simLong(n = 20, p = 6, time_points = 10, linear = TRUE, family = "cox",
#'                  random_sd_intercept = sqrt(0.5), random_sd_slope = sqrt(0.5),
#'                  target_c = 0.75, censoring = 0.3, pilot_n = 50)
#' mean(datas$status)
#'
simLong <- function(
    n, p, rel_p = 6, time_points = 10, rho_W = 0.6, rho_Z = 0.6,
    random_sd_intercept = sqrt(0.5), random_sd_slope = sqrt(3), noise_sd = 0.5,
    linear = TRUE, family = c("gaussian", "binomial", "cox"),
    target_auc = NULL, prevalence = 0.5, target_c = NULL, signal = NULL, intercept = NULL,
    censoring = 0.3, weibull_shape = 1.5, weibull_scale = 1,
    pilot_n = 300, pilot_seed = 1, calibration = NULL
) {
  family <- match.arg(family)
  mexgb_sim_inputs(linear, rel_p, time_points, rho_W, rho_Z,
                   random_sd_intercept, random_sd_slope, pilot_n, pilot_seed)
  mexgb_number(n, "n", 2, integer = TRUE)
  mexgb_number(p, "p", 1, integer = TRUE)
  mexgb_number(noise_sd, "noise_sd", 0)
  mexgb_number(prevalence, "prevalence", 0, 1, open_lower = TRUE, open_upper = TRUE)
  mexgb_number(censoring, "censoring", 0, 1, open_upper = TRUE)
  mexgb_number(weibull_shape, "weibull_shape", 0, open_lower = TRUE)
  mexgb_number(weibull_scale, "weibull_scale", 0, open_lower = TRUE)
  if (!is.null(target_auc)) mexgb_number(target_auc, "target_auc", 0.5, 1, open_upper = TRUE)
  if (!is.null(target_c)) mexgb_number(target_c, "target_c", 0.5, 1, open_upper = TRUE)
  if (!is.null(signal)) mexgb_number(signal, "signal", 0)
  if (!is.null(intercept)) mexgb_number(intercept, "intercept")
  if (!is.null(calibration)) {
    if (!is.list(calibration) || !all(c("signal", "f_mean", "f_sd") %in% names(calibration)))
      stop("calibration must contain signal, f_mean and f_sd.")
    mexgb_number(calibration$signal, "calibration$signal", 0)
    mexgb_number(calibration$f_mean, "calibration$f_mean")
    mexgb_number(calibration$f_sd, "calibration$f_sd", 0, open_lower = TRUE)
    if (!is.null(calibration$intercept)) mexgb_number(calibration$intercept, "calibration$intercept")
  }
  if (!linear && p < 6) stop("The nonlinear model requires p >= 6.")
  if (linear && rel_p > p) stop("rel_p must not exceed p.")
  N <- n * time_points
  id <- rep(1:n, each = time_points)
  tt <- rep(seq(0, 1, length.out = time_points), n)

  # Fixed effect predictors: x_ij ~ N_p(0, Sigma_W), (Sigma_W)_{gg'} = rho_W^|g-g'|
  X <- sim_X(N, p, rho_W)

  f <- sim_f(X, linear, rel_p)

  # Random intercept and slope
  b <- sim_b(n, random_sd_intercept, random_sd_slope, rho_Z)
  Z <- cbind(1, tt)
  random_effects <- rowSums(Z * b[id, , drop = FALSE])

  cal <- NULL
  status <- NULL
  if (family == "gaussian") {
    # Within-subject AR(1) errors: Cov(e_ij, e_ij') = sigma^2 rho_W^|j-j'|
    R <- noise_sd^2 * rho_W^abs(outer(1:time_points, 1:time_points, "-"))
    noise <- as.vector(t(MASS::mvrnorm(n, mu = rep(0, time_points), Sigma = R)))
    Y <- f + random_effects + noise
    eta <- NULL
  } else {
    if (!is.null(calibration)) {
      cal <- calibration                     # output of calibrateSim(): fixed DGP across replicates
    } else if (!is.null(signal) || (family == "binomial" && is.null(target_auc)) ||
        (family == "cox" && is.null(target_c))) {
      ## no calibration target: standardise f within the data set
      cal <- list(signal = if (is.null(signal)) 1 else signal, f_mean = mean(f),
                  f_sd = stats::sd(f), intercept = intercept)
    } else {
      cal <- calibrateSim(family = family, linear = linear, rel_p = rel_p,
                          time_points = time_points, rho_W = rho_W, rho_Z = rho_Z,
                          random_sd_intercept = random_sd_intercept,
                          random_sd_slope = random_sd_slope,
                          target = if (family == "binomial") target_auc else target_c,
                          prevalence = prevalence, pilot_n = pilot_n, pilot_seed = pilot_seed)
    }
    fs <- (f - cal$f_mean) / cal$f_sd
    lp <- cal$signal * fs + random_effects
    if (family == "binomial") {
      c0 <- if (!is.null(intercept)) intercept else if (!is.null(cal$intercept)) cal$intercept else
        stats::uniroot(function(c) mean(stats::plogis(c + lp)) - prevalence,
                       interval = c(-50, 50))$root
      cal$intercept <- c0
      eta <- c0 + lp
      Y <- stats::rbinom(N, 1, stats::plogis(eta))
    } else {
      eta <- lp
      Tev <- (-log(stats::runif(N)) / (weibull_scale * exp(eta)))^(1 / weibull_shape)
      if (censoring > 0) {
        E <- stats::rexp(N)
        rate <- exp(stats::uniroot(function(lr) mean(E / exp(lr) < Tev) - censoring,
                                   interval = c(-30, 30), extendInt = "yes")$root)
        Cens <- E / rate
      } else Cens <- rep(Inf, N)
      Y <- pmin(Tev, Cens)
      status <- as.numeric(Tev <= Cens)
      cal$censoring_rate <- if (censoring > 0) rate else 0
    }
  }

  data <- data.frame(
    id = id,
    time = tt,
    Y = as.vector(Y),
    RandomIntercept = Z[, 1],
    RandomSlope = Z[, 2],
    as.data.frame(X)
  )
  if (family == "cox") data$status <- status
  attr(data, "f") <- f
  attr(data, "eta") <- eta
  attr(data, "b") <- b
  attr(data, "calibration") <- cal
  return(data)
}


#' Calibrate the signal strength (and intercept) of the binary / survival simulation.
#'
#' Uses a large pilot sample (only the active predictors need to be generated) to find the
#' multiplier \eqn{\kappa} of the standardised fixed effect \eqn{f^*} such that the population
#' (fixed-effect) discrimination equals \code{target}:
#' \itemize{
#' \item binomial: the expected AUC of \eqn{f^*} for the outcome
#' \eqn{y\sim Bernoulli(expit(c_0+\kappa f^*+Z b))}, computed exactly from the event
#' probabilities (no sampling of \eqn{y}); for each \eqn{\kappa} the intercept \eqn{c_0} is solved so that
#' the mean event probability equals \code{prevalence}. The prevalence therefore also enters the AUC.
#' \item cox: the expected concordance index of \eqn{f^*} for event times from the frailty model,
#' \eqn{P(T_i<T_j)=h_i/(h_i+h_j)} with \eqn{h=\exp(\kappa f^*+Zb)} (independent of the baseline hazard).
#' }
#' The target is solved on the finite pilot sample using \code{uniroot}; achieved
#' discrimination describes that pilot, not an exact population guarantee. Unattainable
#' targets issue a warning. Independent simulation replicates need not attain the target exactly.
#'
#' @param family \code{"binomial"} or \code{"cox"}.
#' @param target Target AUC (binomial) or concordance (cox), in [0.5, 1).
#' @param prevalence binomial target event rate.
#' @param linear,rel_p,time_points,rho_W,rho_Z,random_sd_intercept,random_sd_slope as in \code{\link{simLong}}.
#' @param pilot_n number of pilot individuals.
#' @param pilot_seed seed of the pilot sample (the global random-number stream is left untouched).
#' @param max_signal upper bound of the search for \eqn{\kappa}.
#'
#' @return list with \code{signal} (\eqn{\kappa}), \code{intercept} (\eqn{c_0}, binomial),
#'   \code{f_mean}, \code{f_sd} (standardisation of \eqn{f}), \code{achieved} (discrimination at the
#'   solution) and \code{target}.
#' @export
#' @examples
#' calibrateSim("binomial", target = 0.85, prevalence = 0.5, linear = FALSE, pilot_n = 50)
calibrateSim <- function(family = c("binomial", "cox"), target, prevalence = 0.5,
                         linear = TRUE, rel_p = 6, time_points = 10, rho_W = 0.6, rho_Z = 0.6,
                         random_sd_intercept = sqrt(0.5), random_sd_slope = sqrt(3),
                         pilot_n = 300, pilot_seed = 1, max_signal = 50){
  family <- match.arg(family)
  mexgb_sim_inputs(linear, rel_p, time_points, rho_W, rho_Z,
                   random_sd_intercept, random_sd_slope, pilot_n, pilot_seed)
  mexgb_number(target, "target", 0.5, 1, open_upper = TRUE)
  mexgb_number(prevalence, "prevalence", 0, 1, open_lower = TRUE, open_upper = TRUE)
  mexgb_number(max_signal, "max_signal", 1e-3, open_lower = TRUE)
  ## pilot sample drawn under a private seed; the caller's RNG state is restored
  old_seed <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = globalenv()) else
    rm(".Random.seed", envir = globalenv()))
  set.seed(pilot_seed)
  N <- pilot_n * time_points
  idp <- rep(1:pilot_n, each = time_points)
  ttp <- rep(seq(0, 1, length.out = time_points), pilot_n)
  Xp <- sim_X(N, max(6, rel_p), rho_W)
  fp <- sim_f(Xp, linear, rel_p)
  bp <- sim_b(pilot_n, random_sd_intercept, random_sd_slope, rho_Z)
  rep_ <- bp[idp, 1] + bp[idp, 2] * ttp
  f_mean <- mean(fp); f_sd <- stats::sd(fp)
  fs <- (fp - f_mean) / f_sd

  if (family == "binomial") {
    c0_of <- function(k) stats::uniroot(function(c) mean(stats::plogis(c + k * fs + rep_)) - prevalence,
                                        interval = c(-100, 100))$root
    auc_of <- function(k) expected_auc(fs, stats::plogis(c0_of(k) + k * fs + rep_))
    disc <- auc_of
  } else {
    if (N > 3000) {               # O(N^2) pairs: use a random subset of the pilot
      sub <- sample.int(N, 3000); fs_c <- fs[sub]; re_c <- rep_[sub]
    } else { fs_c <- fs; re_c <- rep_ }
    disc <- function(k) expected_cindex(fs_c, k * fs_c + re_c)
  }
  lo <- 1e-3; hi <- max_signal
  d_lo <- disc(lo); d_hi <- disc(hi)
  if (target <= d_lo) {
    warning("target discrimination is below the minimum attainable; using signal = ", lo)
    k <- lo
  } else if (target >= d_hi) {
    warning(sprintf("target %.3f exceeds the maximum attainable (%.3f) for these random-effect variances; using signal = %g",
                    target, d_hi, hi))
    k <- hi
  } else {
    k <- stats::uniroot(function(k) disc(k) - target, interval = c(lo, hi), tol = 1e-6)$root
  }
  out <- list(signal = k, f_mean = f_mean, f_sd = f_sd, achieved = disc(k), target = target)
  if (family == "binomial") { out$intercept <- c0_of(k); out$prevalence <- prevalence }
  out
}

#' Expected AUC of a score for Bernoulli outcomes with known probabilities (internal).
#'
#' \eqn{E[AUC]\approx\sum_{i\ne j}p_i(1-p_j)1(s_i>s_j)/\sum_{i\ne j}p_i(1-p_j)}.
#' @keywords internal
expected_auc <- function(s, prob){
  o <- order(s); s <- s[o]; p <- prob[o]; q <- 1 - p
  grp <- cumsum(c(TRUE, diff(s) != 0))
  ps <- as.numeric(rowsum(p, grp)); qs <- as.numeric(rowsum(q, grp))
  pq <- as.numeric(rowsum(p * q, grp))
  below <- c(0, cumsum(qs)[-length(qs)])
  num <- sum(ps * below + 0.5 * (ps * qs - pq))
  den <- sum(p) * sum(q) - sum(p * q)
  if (den <= 0) return(NA_real_)
  num / den
}

#' Expected concordance of a score under a proportional-hazards model (internal).
#'
#' \eqn{E[C]=\sum_{s_i>s_j}h_i/(h_i+h_j)/\#\{s_i>s_j\}} with \eqn{h=\exp(\eta)}.
#' @keywords internal
expected_cindex <- function(s, eta){
  h <- exp(eta - max(eta))
  num <- 0; den <- 0
  n <- length(s)
  chunk <- 500
  for (a in seq(1, n, by = chunk)) {
    ia <- a:min(n, a + chunk - 1)
    gt <- outer(s[ia], s, ">")
    pr <- outer(h[ia], h, function(x, y) x / (x + y))   # MEXGB 0.5.0 arithmetic
    bad <- !is.finite(pr)                                # 0/0 under extreme predictors only
    if (any(bad)) pr[bad] <- stats::plogis(outer(eta[ia], eta, "-"))[bad]
    tied <- outer(s[ia], s, "==")
    tied[cbind(seq_along(ia), ia)] <- FALSE
    num <- num + sum(pr[gt]) + 0.25 * sum(tied)
    den <- den + sum(gt) + 0.5 * sum(tied)
  }
  num / den
}

#' @keywords internal
sim_X <- function(N, p, rho_W){
  X <- matrix(0, N, p)
  X[, 1] <- stats::rnorm(N)
  if (p > 1) for (g in 2:p) X[, g] <- rho_W * X[, g - 1] + sqrt(1 - rho_W^2) * stats::rnorm(N)
  colnames(X) <- paste0("Var", 1:p)
  X
}

#' @keywords internal
sim_f <- function(X, linear, rel_p){
  if (linear) {
    beta <- rep(c(2, -2, 1.5, -1.5, 1, -1), length.out = rel_p)
    as.numeric(X[, 1:rel_p, drop = FALSE] %*% beta)
  } else {
    2 * sin(pi * X[, 1] * X[, 2]) + 4 * (X[, 3] - 0.5)^2 + 2 * X[, 4] +
      exp(X[, 5]) + X[, 6] * X[, 1]
  }
}

#' @keywords internal
sim_b <- function(n, sd0, sd1, rho){
  SigmaZ <- matrix(c(sd0^2, rho * sd0 * sd1, rho * sd0 * sd1, sd1^2), 2, 2)
  matrix(MASS::mvrnorm(n, mu = rep(0, 2), Sigma = SigmaZ), n, 2)
}

`%||%` <- function(a, b) if (is.null(a)) b else a


#' -2 x Laplace approximation of the marginal log-likelihood (internal).
#'
#' \eqn{\ell_{La}=\sum_i[\log p(y_i\mid\eta_i)-\frac12\hat b_i^\top D^{-1}\hat b_i
#' -\frac12\log|I+DZ_i^\top W_iZ_i|]} with \eqn{\eta=\hat f+Z\hat b} at the posterior modes
#' \eqn{\hat b_i}; binomial: Bernoulli log-likelihood, \eqn{W=\mu(1-\mu)}; cox: Breslow
#' partial log-likelihood, \eqn{W=\hat\Lambda_0(t)e^\eta} (Poisson representation).
#' @keywords internal
laplace_crit <- function(family, Y, status, f, zb, b, D, Z, indiv_list, eps = 1e-6) {
  eta <- pmin(pmax(f + zb, -30), 30)
  if (family == "binomial") {
    mu <- stats::plogis(eta); w <- pmax(mu * (1 - mu), eps)
    ll <- sum(Y * eta - log1p(exp(eta)))
  } else {
    w <- pmax(breslow(Y, status, eta)$cumhaz_at * exp(eta), eps)
    ll <- cox_partial_loglik(Y, status, eta)
  }
  q <- ncol(Z)
  Dr <- D + diag(1e-8, q)
  Di <- tryCatch(solve(Dr), error = function(e) MASS::ginv(Dr))
  pen <- 0
  for (k in seq_along(indiv_list)) {
    ii <- indiv_list[[k]]; Zk <- Z[ii, , drop = FALSE]
    bk <- b[k, ]
    M <- diag(q) + D %*% crossprod(Zk * w[ii], Zk)
    pen <- pen + 0.5 * sum(bk * (Di %*% bk)) +
      0.5 * as.numeric(determinant(M, logarithm = TRUE)$modulus)
  }
  -2 * (ll - pen)
}


#' Posterior modes of the random effects given D (internal).
#'
#' Newton iterations on the penalised log-likelihood
#' \eqn{\sum_j\log p(y_{ij}\mid f_{ij}+z_{ij}^\top b_i)-\frac12 b_i^\top D^{-1}b_i}, written as
#' BLUPs of the working variate (binomial: logistic; cox: Poisson representation with the
#' Breslow baseline recomputed at every step).
#' @keywords internal
re_modes <- function(family, Y, status, f, b, D, Z, indiv_list, eps = 1e-6, centre = FALSE,
                     int_col = integer(0), maxit = 50, tol = 1e-7) {
  N <- length(Y)
  for (it in seq_len(maxit)) {
    zb <- numeric(N)
    for (k in seq_along(indiv_list)) { ii <- indiv_list[[k]]; zb[ii] <- Z[ii, , drop = FALSE] %*% b[k, ] }
    eta <- pmin(pmax(f + zb, -30), 30)
    if (family == "binomial") {
      mu <- stats::plogis(eta); w <- pmax(mu * (1 - mu), eps); yt <- eta + (Y - mu) / w
    } else {
      mu <- breslow(Y, status, eta)$cumhaz_at * exp(eta); w <- pmax(mu, eps); yt <- eta + (status - mu) / w
    }
    b_old <- b
    for (k in seq_along(indiv_list)) {
      ii <- indiv_list[[k]]; Zk <- Z[ii, , drop = FALSE]
      ## (D^-1 + Z'WZ)^-1 Z'W (yt - f) written without D^-1:  D (I + Z'WZ D)^-1 Z'W r
      A <- diag(ncol(Z)) + crossprod(Zk * w[ii], Zk) %*% D
      b[k, ] <- D %*% solve(A, crossprod(Zk * w[ii], yt[ii] - f[ii]))
    }
    if (centre && length(int_col) == 1) b[, int_col] <- b[, int_col] - mean(b[, int_col])
    if (max(abs(b - b_old)) < tol) break
  }
  b
}

#' Laplace E-step: D maximising the Laplace-approximate marginal likelihood (internal).
#' @keywords internal
laplace_estep <- function(family, Y, status, f, b, D, Z, indiv_list, eps = 1e-6,
                          centre = FALSE, int_col = integer(0)) {
  q <- ncol(Z)
  toD <- function(th) {
    if (q == 1) return(matrix(exp(2 * th), 1, 1))
    L <- matrix(0, q, q); L[lower.tri(L, diag = TRUE)] <- th
    diag(L) <- exp(diag(L)); L %*% t(L)
  }
  fromD <- function(D) {
    D <- D + diag(1e-6, q)
    if (q == 1) return(0.5 * log(D[1, 1]))
    L <- t(chol(D)); diag(L) <- log(diag(L)); L[lower.tri(L, diag = TRUE)]
  }
  bcur <- b
  obj <- function(th) {
    Dt <- toD(th)
    bm <- re_modes(family, Y, status, f, bcur, Dt, Z, indiv_list, eps, centre, int_col)
    bcur <<- bm
    zb <- numeric(length(Y))
    for (k in seq_along(indiv_list)) { ii <- indiv_list[[k]]; zb[ii] <- Z[ii, , drop = FALSE] %*% bm[k, ] }
    v <- laplace_crit(family, Y, status, f, zb, bm, Dt, Z, indiv_list, eps)
    if (!is.finite(v)) 1e100 else v
  }
  th0 <- tryCatch(fromD(D), error = function(e) NULL)
  if (is.null(th0)) return(list(D = NULL, converged = FALSE, code = NA_integer_, message = "Invalid starting covariance."))
  op <- tryCatch(stats::optim(th0, obj, method = if (length(th0) == 1) "BFGS" else "Nelder-Mead",
                              control = list(maxit = 300, reltol = 1e-8)), error = function(e) NULL)
  if (is.null(op) || !is.finite(op$value))
    return(list(D = NULL, converged = FALSE, code = NA_integer_,
                message = if (is.null(op)) "Optimizer failed." else "Non-finite criterion."))
  Dn <- toD(op$par)
  list(D = Dn, b = re_modes(family, Y, status, f, bcur, Dn, Z, indiv_list, eps, centre, int_col),
       crit = op$value, converged = op$convergence == 0L, code = op$convergence,
       message = if (op$convergence == 0L) "Converged." else
         "Iteration limit reached; optimizer result used.")
}
