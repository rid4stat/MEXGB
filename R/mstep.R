## =============================================================================
## MEXGB 0.5 -- M-step: fit of the fixed effect f(x) = a + x'beta + T(x) given the offset
## o = Z b (random-effect contribution), on all training rows and on the V cross-fitting
## complements.
##   * a + x'beta : optional penalised linear part (glmnet, elastic-net mixing 'lin_alpha'),
##                  lambda chosen on the cross-fitting folds (out-of-fold deviance);
##   * T(x)       : xgboost trees grown on top of the offset + linear part (base_margin);
##                  the number of trees is chosen on the same folds (out-of-fold loss at
##                  every boosting round, 0 trees allowed) when ntree_select = "cv".
## Everything is evaluated out of fold with the SAME folds as the cross-fitted E-step, so
## the selection costs no extra fits (the fold models are needed for the E-step anyway).
## =============================================================================

#' Loss of a linear predictor on a set of rows (internal).
#' gaussian: sum of squared errors; binomial: sum of log-losses; cox: minus the log partial
#' likelihood (Breslow) of the rows (risk sets within the rows).
#' @keywords internal
mexgb_loss <- function(family, y, status, eta) {
  if (family == "gaussian") return(sum((y - eta)^2))
  if (family == "binomial") {
    eta <- pmin(pmax(eta, -30), 30)
    return(sum(log1p(exp(eta)) - y * eta))
  }
  -cox_partial_loglik(y, status, eta)
}

#' Penalised linear part fitted with glmnet on the cross-fitting folds (internal).
#' Returns the full-data and fold-specific linear predictors (intercept included for
#' gaussian / binomial) at the lambda with the smallest out-of-fold loss.
#' @keywords internal
mexgb_linear <- function(X, Y, status, offset, family, cf_fold, alpha = 1, nlambda = 40, offset_oof = NULL,
                         lambda_min_ratio = 0.01, relax = TRUE, gammas = c(0, 0.5, 1),
                         relax_df_max = 30) {
  N <- nrow(X); V <- max(cf_fold)
  offo <- if (is.null(offset_oof)) offset else offset_oof
  fam <- switch(family, gaussian = "gaussian", binomial = "binomial", cox = "cox")
  yy <- if (family == "cox") cbind(time = Y, status = status) else Y
  full <- mexgb_glmnet(X, yy, family = fam, offset = offset, alpha = alpha,
                         nlambda = nlambda, lambda.min.ratio = lambda_min_ratio)
  lam <- full$lambda
  ## coefficient matrix (rows: intercept (not cox) + p; columns: lambda) of a fit, with the
  ## relaxed versions: gamma * lasso + (1 - gamma) * unpenalised refit on the active set
  ## unpenalised refit on an active set (intercept first for gaussian / binomial)
  unpen_fit <- function(Xa, rows) {
    o <- offset[rows]
    if (family == "gaussian") return(unname(stats::lm.fit(cbind(1, Xa), Y[rows] - o)$coefficients))
    if (family == "binomial")
      return(unname(suppressWarnings(stats::glm.fit(cbind(1, Xa), Y[rows], offset = o,
                                                    family = stats::binomial()))$coefficients))
    unname(stats::coef(survival::coxph(survival::Surv(Y[rows], status[rows]) ~ Xa + offset(o),
                                       ties = "breslow")))
  }
  coef_mat <- function(fit, rows) {
    B <- as.matrix(stats::coef(fit, s = lam))
    if (!relax) return(list(B))
    p_off <- if (family == "cox") 0 else 1
    Bu <- B
    for (k in seq_along(lam)) {
      act <- which(B[p_off + seq_len(ncol(X)), k] != 0)
      if (length(act) == 0 || length(act) > relax_df_max) next
      bu <- tryCatch(unpen_fit(X[rows, act, drop = FALSE], rows), error = function(e) NULL)
      if (is.null(bu) || any(!is.finite(bu))) next
      Bu[, k] <- 0
      if (family == "cox") Bu[act, k] <- bu else { Bu[1, k] <- bu[1]; Bu[1 + act, k] <- bu[-1] }
    }
    lapply(gammas, function(g) g * B + (1 - g) * Bu)
  }
  lin_pred <- function(B, newx) {
    if (family == "cox") as.matrix(newx %*% B) else
      sweep(as.matrix(newx %*% B[-1, , drop = FALSE]), 2, B[1, ], "+")
  }
  G <- if (relax) length(gammas) else 1
  fold_B <- vector("list", V)
  loss <- array(0, c(V, length(lam), G))
  for (v in seq_len(V)) {
    te <- which(cf_fold == v); tr <- setdiff(seq_len(N), te)
    fit <- tryCatch(mexgb_glmnet(X[tr, , drop = FALSE],
                                   if (family == "cox") yy[tr, , drop = FALSE] else yy[tr],
                                   family = fam, offset = offset[tr], alpha = alpha,
                                   lambda = lam), error = function(e) NULL)
    if (is.null(fit)) { loss[v, , ] <- Inf; next }
    Bl <- coef_mat(fit, tr)
    fold_B[[v]] <- Bl
    for (g in seq_len(G)) {
      L <- lin_pred(Bl[[g]], X[te, , drop = FALSE])
      for (k in seq_along(lam))
        loss[v, k, g] <- mexgb_loss(family, Y[te], status[te], offo[te] + L[, k])
    }
  }
  tot <- apply(loss, c(2, 3), sum)
  if (all(!is.finite(tot))) { k <- 1; g <- G } else {
    idx <- which(tot == min(tot[is.finite(tot)]), arr.ind = TRUE)[1, ]
    k <- idx[1]; g <- idx[2]
  }
  Bfull <- coef_mat(full, seq_len(N))[[g]][, k, drop = FALSE]
  g_full <- as.numeric(lin_pred(Bfull, X))
  g_fold <- matrix(0, N, V)
  for (v in seq_len(V))
    g_fold[, v] <- if (is.null(fold_B[[v]])) g_full else
      as.numeric(lin_pred(fold_B[[v]][[g]][, k, drop = FALSE], X))
  list(g_full = g_full, g_fold = g_fold, lambda = lam[k], gamma = if (relax) gammas[g] else 1,
       intercept = if (family == "cox") 0 else Bfull[1, 1],
       beta = if (family == "cox") Bfull[, 1] else Bfull[-1, 1],
       oof_loss = tot)
}

#' The M-step of MEXGB 0.5 (internal).
#'
#' @return list with \code{fixed_full} (fitted fixed effect on the link scale, offset
#'   excluded, all rows), \code{fixed_cf} (cross-fitted fixed effect), \code{model} (what
#'   \code{fixedMEXGB} needs to predict) and \code{nrounds}.
#' @keywords internal
mexgb_mstep <- function(X, Y, status, offset, family, params, ntree, cf_fold, f0 = 0,
                        linear = FALSE, lin_alpha = 1, ntree_select = c("cv", "fixed"),
                        min_trees = 0, grid_step = 10, stall_rounds = 200,
                        ntree_scale = NULL, lin_relax = FALSE, screen = "none",
                        screen_sets = NULL, screen_sizes = c(5, 10, 20, 40), offset_oof = NULL) {
  ntree_select <- match.arg(ntree_select)
  ## offset used for the held-out rows of the out-of-fold loss: the random effects predicted
  ## WITHOUT the held-out rows, otherwise the in-sample offset
  offo <- if (is.null(offset_oof)) offset else offset_oof
  if (identical(screen, "auto") && is.null(screen_sets) && !is.null(cf_fold) &&
      ntree_select == "cv" && ncol(X) > min(screen_sizes)) {
    ## feature screening for the trees: pass 1 on all predictors; pass 2 on the predictors
    ## the K top-ranked predictors (by split gain) of each fold model, K chosen among screen_sizes
    ## out-of-fold comparison of the two passes excludes the held-out rows); the pass with the smaller
    ## out-of-fold loss is kept
    args <- list(X = X, Y = Y, status = status, offset = offset, family = family,
                 params = params, ntree = ntree, cf_fold = cf_fold, f0 = f0, linear = linear,
                 lin_alpha = lin_alpha, ntree_select = ntree_select, min_trees = min_trees,
                 grid_step = grid_step, stall_rounds = stall_rounds, ntree_scale = ntree_scale,
                 lin_relax = lin_relax, screen = "none", screen_sizes = screen_sizes,
                 offset_oof = offset_oof)
    r_all <- do.call(mexgb_mstep, args)
    ## linear = "auto": the structure is decided on all predictors, and the screened candidates
    ## are fitted with the structure chosen there (deciding it separately for every candidate
    ## set would compare a linear-plus-trees fit on all predictors with a trees-only fit on few)
    lin_info <- list(linear_chosen = r_all$linear_chosen, cv_loss_both = r_all$cv_loss_both)
    if (identical(linear, "auto") && !is.null(r_all$linear_chosen)) args$linear <- isTRUE(r_all$linear_chosen)
    rk <- r_all$screen_candidates
    if (is.null(rk)) { r_all$screen_sets <- "none"; return(r_all) }
    ## candidate sizes K: the K top-ranked predictors of every fold model (and of the full model)
    Ks <- screen_sizes[screen_sizes < ncol(X)]
    best <- r_all; best_sets <- "none"; losses <- c(all = r_all$cv_loss)
    for (K in Ks) {
      sets <- list(full = utils::head(rk$full, K), folds = lapply(rk$folds, utils::head, K))
      r <- do.call(mexgb_mstep, c(args[setdiff(names(args), "screen")],
                                  list(screen = "none", screen_sets = sets)))
      losses[paste0("K", K)] <- r$cv_loss
      if (is.finite(r$cv_loss) && r$cv_loss < best$cv_loss) { best <- r; best_sets <- sets }
    }
    ## Gaussian: the screened set replaces all predictors only when it is significantly better
    ## out of fold (paired comparison of the held-out squared errors, z > 2), as for the linear
    ## part and the scale of D: a near-tie is resolved in favour of all predictors
    if (family == "gaussian" && is.list(best_sets)) {
      dl <- (Y - offo - r_all$fixed_cf)^2 - (Y - offo - best$fixed_cf)^2
      z <- mean(dl) / (stats::sd(dl) / sqrt(length(dl)))
      if (!is.finite(z) || z <= 2) { best <- r_all; best_sets <- "none" }
    }
    best$screen_sets <- best_sets
    best$cv_loss_screen <- losses
    best$linear_chosen <- lin_info$linear_chosen; best$cv_loss_both <- lin_info$cv_loss_both
    return(best)
  }
  if (identical(linear, "auto")) {
    ## structure selection: linear part + trees versus trees only, by out-of-fold loss
    if (is.null(cf_fold) || ntree_select != "cv") linear <- TRUE else {
      args <- list(X = X, Y = Y, status = status, offset = offset, family = family,
                   params = params, ntree = ntree, cf_fold = cf_fold, f0 = f0,
                   lin_alpha = lin_alpha, ntree_select = ntree_select, min_trees = min_trees,
                   grid_step = grid_step, stall_rounds = stall_rounds, ntree_scale = ntree_scale,
                   lin_relax = lin_relax, screen = screen, screen_sets = screen_sets,
                   offset_oof = offset_oof)
      r1 <- do.call(mexgb_mstep, c(args, list(linear = TRUE)))
      r0 <- do.call(mexgb_mstep, c(args, list(linear = FALSE)))
      ## the two structures are alternatives of equal standing (neither is a departure from a
      ## default), so the one with the smaller out-of-fold loss is kept
      keep_lin <- r1$cv_loss <= r0$cv_loss
      out <- if (keep_lin) r1 else r0
      out$linear_chosen <- keep_lin
      out$cv_loss_both <- c(linear = r1$cv_loss, trees = r0$cv_loss)
      return(out)
    }
  }
  N <- nrow(X)
  curve <- NULL
  allc <- seq_len(ncol(X))
  cols_full <- if (is.list(screen_sets)) screen_sets$full else allc
  cols_fold <- function(v) if (is.list(screen_sets)) screen_sets$folds[[v]] else allc
  use_cf <- !is.null(cf_fold)
  V <- if (use_cf) max(cf_fold) else 0
  if (family == "cox") label <- ifelse(status == 1, Y, -Y) else label <- Y
  params$base_score <- switch(family, gaussian = 0, binomial = 0.5, cox = 1)
  if (family == "gaussian") params$objective <- "reg:squarederror"
  if (family == "binomial") params$objective <- "binary:logistic"
  if (family == "cox") params$objective <- "survival:cox"

  ## ---- linear part --------------------------------------------------------------
  lin <- NULL
  if (linear && use_cf) {
    lin <- tryCatch(mexgb_linear(X, Y, status, offset, family, cf_fold, alpha = lin_alpha, relax = lin_relax,
                                 offset_oof = offset_oof),
                    error = function(e) NULL)
  }
  if (!is.null(lin)) {
    base_full <- lin$g_full
    base_fold <- lin$g_fold
    a0 <- lin$intercept
  } else {
    base_full <- rep(f0, N)
    base_fold <- if (use_cf) matrix(f0, N, V) else NULL
    a0 <- f0
  }

  ## ---- trees: number of rounds chosen out of fold ------------------------------------
  fold_models <- vector("list", V)
  M <- ntree
  if (use_cf) {
    ## fold models grown in chunks of 'grid_step' rounds; the out-of-fold loss of the sum
    ## over folds is evaluated after every chunk (0 trees included), and growth stops when
    ## the best number of rounds is 'stall_rounds' rounds behind (or at ntree)
    dtr <- vector("list", V); te_rows <- vector("list", V)
    for (v in seq_len(V)) {
      te <- which(cf_fold == v); tr <- setdiff(seq_len(N), te); te_rows[[v]] <- te
      dtr[[v]] <- xgboost::xgb.DMatrix(X[tr, cols_fold(v), drop = FALSE], label = label[tr],
                                       base_margin = offset[tr] + base_fold[tr, v])
    }
    grid <- 0
    curve <- sum(vapply(seq_len(V), function(v) {
      te <- te_rows[[v]]
      mexgb_loss(family, Y[te], status[te], offo[te] + base_fold[te, v])
    }, numeric(1)))
    cur <- 0
    if (ntree_select == "fixed") {
      for (v in seq_len(V))
        fold_models[[v]] <- xgboost::xgb.train(params = params, data = dtr[[v]], nrounds = ntree, verbose = 0)
    } else {
      acc <- lapply(te_rows, function(te) numeric(length(te)))
      repeat {
        ## chunks grow geometrically (few continuation calls: xgboost copies the model at
        ## every call), the loss is still evaluated every 'grid_step' rounds inside a chunk
        chunk <- min(max(grid_step, cur), 500)
        nxt <- min(cur + chunk, ntree)
        gpts <- if (cur + grid_step <= nxt) seq(cur + grid_step, nxt, by = grid_step) else numeric(0)
        gpts <- sort(unique(c(gpts, nxt)))
        gpts <- gpts[gpts > cur]
        lv <- numeric(length(gpts))
        for (v in seq_len(V)) {
          fold_models[[v]] <- xgboost::xgb.train(params = params, data = dtr[[v]],
                                                 nrounds = nxt - cur, verbose = 0,
                                                 xgb_model = fold_models[[v]])
          te <- te_rows[[v]]; Xte <- X[te, cols_fold(v), drop = FALSE]
          b_te <- offo[te] + base_fold[te, v]
          prev <- cur
          for (g in seq_along(gpts)) {
            ## add only the new trees to the running out-of-fold margin (linear cost)
            acc[[v]] <- acc[[v]] + as.numeric(predict(fold_models[[v]], Xte,
                                                      iterationrange = xgb_range(prev + 1, gpts[g]),
                                                      outputmargin = TRUE))
            lv[g] <- lv[g] + mexgb_loss(family, Y[te], status[te], b_te + acc[[v]])
            prev <- gpts[g]
          }
        }
        cur <- nxt; grid <- c(grid, gpts); curve <- c(curve, lv)
        best <- grid[which.min(curve)]
        if (cur >= ntree || cur - best >= stall_rounds) break
      }
      M <- max(grid[which.min(curve)], min_trees)
    }
  }

  ## ---- full-data fit ------------------------------------------------------------------
  booster <- NULL
  T_full <- rep(0, N)
  ## the full-data booster sees V/(V-1) times more rows than the fold models on which M was
  ## chosen; 'ntree_scale' (default V/(V-1)) rescales the number of trees accordingly
  M_full <- if (use_cf && ntree_select == "cv") round(M * (ntree_scale %||% (V / (V - 1)))) else M
  if (M > 0) {
    dfull <- xgboost::xgb.DMatrix(X[, cols_full, drop = FALSE], label = label,
                                  base_margin = offset + base_full)
    booster <- xgboost::xgb.train(params = params, data = dfull, nrounds = M_full, verbose = 0)
    T_full <- as.numeric(predict(booster, X[, cols_full, drop = FALSE], outputmargin = TRUE))
  }
  fixed_full <- base_full + T_full

  ## ---- cross-fitted fixed effect ----------------------------------------------------------
  fixed_cf <- fixed_full
  if (use_cf) {
    fixed_cf <- numeric(N)
    for (v in seq_len(V)) {
      te <- which(cf_fold == v); tr <- setdiff(seq_len(N), te)
      Tv_te <- if (M > 0) as.numeric(predict(fold_models[[v]], X[te, cols_fold(v), drop = FALSE],
                                             iterationrange = xgb_range(1, M), outputmargin = TRUE)) else 0
      out <- base_fold[te, v] + Tv_te
      if (family == "cox") {      # log-hazard identified up to a constant: align with full fit
        Tv_tr <- if (M > 0) as.numeric(predict(fold_models[[v]], X[tr, cols_fold(v), drop = FALSE],
                                               iterationrange = xgb_range(1, M), outputmargin = TRUE)) else 0
        out <- out - mean(base_fold[tr, v] + Tv_tr) + mean(fixed_full[tr])
      }
      fixed_cf[te] <- out
    }
  }
  ## candidate predictor sets for screening (fold-specific and full), from the split gain
  screen_candidates <- NULL
  if (use_cf && M > 0 && !is.list(screen_sets) && ncol(X) > min(screen_sizes)) {
    ## predictors ranked by their split gain (unused predictors last, in their original order)
    top_cols <- function(bst, cols, ntrees) {
      tab <- tryCatch(as.data.frame(xgboost::xgb.importance(model = bst, trees = xgb_trees(ntrees))),
                      error = function(e) NULL)
      if (is.null(tab) || nrow(tab) == 0) return(cols)
      idx <- match(tab$Feature, colnames(X)[cols])
      used <- cols[idx[!is.na(idx)]]
      c(used, setdiff(cols, used))
    }
    screen_candidates <- list(full = top_cols(booster, cols_full, M_full),
                              folds = lapply(seq_len(V), function(v) top_cols(fold_models[[v]], cols_fold(v), M)))
  }
  list(fixed_full = fixed_full, fixed_cf = fixed_cf, nrounds = M_full, nrounds_cv = M,
       screen_candidates = screen_candidates,
       cv_loss = if (length(curve) > 0) min(curve) else NA_real_,
       linear_chosen = !is.null(lin),
       model = list(booster = booster, intercept = a0, tree_cols = cols_full,
                    lin_range = if (is.null(lin)) NULL else range(lin$g_full),
                    beta = if (is.null(lin)) NULL else lin$beta,
                    lambda = if (is.null(lin)) NULL else lin$lambda,
                    f0 = if (is.null(lin)) f0 else NULL),
       linear_part = base_full, tree_part = T_full)
}

#' Fixed effect (link scale, before calibration) of a fitted M-step model (internal).
#' @keywords internal
mexgb_fixed_predict <- function(model, X) {
  out <- if (!is.null(model$beta)) {
    ## the linear part is clamped to the range it takes on the training data: unlike the
    ## trees it would otherwise extrapolate without bound for unusual predictor values
    g <- model$intercept + as.numeric(X %*% model$beta)
    if (!is.null(model$lin_range)) pmin(pmax(g, model$lin_range[1]), model$lin_range[2]) else g
  } else rep(model$f0 %||% model$intercept, nrow(X))
  if (!is.null(model$booster)) {
    Xt <- if (!is.null(model$tree_cols)) X[, model$tree_cols, drop = FALSE] else X
    out <- out + as.numeric(predict(model$booster, Xt, outputmargin = TRUE))
  }
  out
}

#' Importance of the predictors: mean absolute contribution to the fixed effect (internal).
#' Tree part: SHAP contributions of xgboost; linear part: beta_j (x_j - mean x_j).
#' @keywords internal
mexgb_importance <- function(model, X) {
  imp <- stats::setNames(numeric(ncol(X)), colnames(X))
  contrib <- matrix(0, nrow(X), ncol(X))
  if (!is.null(model$booster)) {
    tc <- model$tree_cols %||% seq_len(ncol(X))
    sh <- predict(model$booster, X[, tc, drop = FALSE], predcontrib = TRUE)
    contrib[, tc] <- contrib[, tc] + sh[, seq_along(tc), drop = FALSE]
  }
  if (!is.null(model$beta))
    contrib <- contrib + sweep(sweep(X, 2, colMeans(X)), 2, model$beta, "*")
  imp[] <- colMeans(abs(contrib))
  if (sum(imp) > 0) imp <- 100 * imp / sum(imp)
  imp
}

## ---- xgboost version compatibility --------------------------------------------------------
## The R package changed two index conventions in 3.0: predict(iterationrange = c(a, b)) is
## 1-based and INCLUSIVE from 3.0 on but 1-based and half-open [a, b) in 1.7 (where c(1, 1)
## means "all trees"), and xgb.importance(trees = ) is 1-based from 3.0 on but 0-based in 1.7
## (an index past the last tree crashes 1.7). Both are wrapped here so MEXGB runs on either.
.mexgb_xgb <- new.env(parent = emptyenv())
xgb_v3 <- function() {
  if (is.null(.mexgb_xgb$v3)) .mexgb_xgb$v3 <- utils::packageVersion("xgboost") >= "3.0.0"
  .mexgb_xgb$v3
}
## trees from..to (1-based, inclusive) for predict(iterationrange = )
xgb_range <- function(from, to) if (xgb_v3()) c(from, to) else c(from, to + 1)
## trees 1..n (1-based) for xgb.importance(trees = )
xgb_trees <- function(n) if (xgb_v3()) seq_len(n) else seq_len(n) - 1L

