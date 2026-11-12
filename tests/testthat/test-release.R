small_design <- function() {
  set.seed(101)
  d <- simLong(8, 6, time_points = 4)
  list(X = as.matrix(d[paste0("Var", 1:6)]), Y = d$Y, id = d$id,
       Z = as.matrix(d[c("RandomIntercept", "RandomSlope")]), time = d$time,
       iter = 2, inner = 2, crossfit = 2, linear = FALSE,
       screen = "none", verbose = FALSE)
}

test_that("CV accepts short and partial final tree grids", {
  a <- small_design()
  for (n in c(1, 5, 25, 30)) {
    f <- do.call(MEXGB, c(a, list(ntree = n)))
    expect_s3_class(f, "MEXGB")
    expect_true(all(is.finite(predict(f, a$X, a$Z, a$id, a$time))))
  }
})

test_that("unsupported survival predictions are rejected before family returns", {
  a <- small_design()
  f <- do.call(MEXGB, c(a, list(ntree = 5)))
  for (family in c("gaussian", "binomial")) {
    f$family <- family
    expect_error(predict(f, a$X, a$Z, a$id, a$time, type = "survival"), "only available")
  }
})

test_that("prediction aligns named designs and retains population Z effects", {
  a <- small_design()
  f <- do.call(MEXGB, c(a, list(ntree = 5)))
  f$beta_z <- c(1, 2)
  unseen <- a$id + 100
  expected <- MEXGB:::fixedMEXGB(f, a$X) + as.numeric(a$Z %*% f$beta_z)
  expect_equal(predict(f, a$X, a$Z, unseen, a$time), expected)
  expect_equal(predict(f, a$X[, 6:1], a$Z[, 2:1], unseen, a$time), expected)
  expect_error(predict(f, a$X[, -1], a$Z, unseen, a$time), "same number")
  expect_error(predict(f, a$X, a$Z[, 1, drop = FALSE], unseen, a$time), "training number")
  expect_error(predict(f, a$X, a$Z, unseen[-1], a$time), "identifier")
})

test_that("invalid fit inputs fail with informative errors", {
  a <- small_design()
  bad <- function(name, value, pattern) {
    b <- a; b[[name]] <- value
    expect_error(do.call(MEXGB, b), pattern)
  }
  bad("iter", 0, "iter")
  bad("ntree", 1.5, "ntree")
  bad("crossfit", 1, "crossfit")
  bad("crossfit", 100, "crossfit")
  bad("time", a$time[-1], "time")
  bad("id", replace(a$id, 1, NA), "identifier")
  bad("Y", replace(a$Y, 1, Inf), "finite")
  bad("X", replace(a$X, 1, NA), "finite numeric")
  b <- a; b$family <- "binomial"; b$Y <- rep(0, length(a$Y))
  expect_error(do.call(MEXGB, b), "Both binary")
  b <- a; b$family <- "cox"
  b$Y <- survival::Surv(rep(1, length(a$Y)), rep(2, length(a$Y)), rep(1, length(a$Y)))
  expect_error(do.call(MEXGB, b), "Only right-censored")
  b$Y <- cbind(1, 2, rep(1, length(a$Y)))
  expect_error(do.call(MEXGB, b), "two-column")
  b$Y <- cbind(seq_along(a$Y), 0)
  expect_error(do.call(MEXGB, b), "observed event")
})

test_that("simulation and calibration parameters are checked", {
  expect_error(simLong(0, 6), "n must")
  expect_error(simLong(8, 6, rho_W = 1), "rho_W")
  expect_error(simLong(8, 6, prevalence = 1), "prevalence")
  expect_error(simLong(8, 6, censoring = 1), "censoring")
  expect_error(simLong(8, 6, calibration = list(signal = 1)), "calibration")
  expect_error(calibrateSim("binomial", target = 1), "target")
  expect_error(calibrateSim("cox", target = 0.7, max_signal = 0), "max_signal")
})

test_that("unusable folds cannot silently supply in-sample predictions", {
  a <- small_design()
  folds <- MEXGB:::cf_make_folds(a$id, 2, 1)
  a$family <- "binomial"
  a$Y <- as.numeric(folds == 1)
  expect_error(do.call(MEXGB, a), "both binary classes")
  a$family <- "cox"
  a$Y <- cbind(seq_along(a$id), as.numeric(folds == 1))
  expect_error(do.call(MEXGB, a), "must contain an event")
})

test_that("zero-contribution models have a valid empty summary", {
  f <- structure(list(importance = c(a = 0, b = 0)), class = "MEXGB")
  expect_equal(nrow(summary(f)), 0)
  expect_named(summary(f), c("var", "rel.inf"))
  expect_error(summary(f, top = -1), "top")
})

test_that("cluster comparison accounts for within-subject replication", {
  d <- c(1, 2, 3, 4)
  z <- MEXGB:::mexgb_cluster_score(d, 1:4)
  expect_equal(z, MEXGB:::mexgb_cluster_score(rep(d, each = 3), rep(1:4, each = 3)))
  expect_equal(z, mean(d) / (sd(d) / sqrt(4)))
})

test_that("discrimination helpers handle ties and extreme predictors", {
  expect_equal(MEXGB:::expected_auc(rep(0, 4), c(.1, .2, .6, .8)), .5)
  expect_equal(MEXGB:::expected_cindex(rep(0, 4), c(-1000, 0, 0, 1000)), .5)
  expect_true(is.finite(MEXGB:::expected_cindex(1:4, c(-2000, -1000, 0, 1000))))
})

test_that("Laplace failure is explicit rather than silently accepted", {
  d <- MEXGB:::laplace_estep("binomial", c(0, 1), c(1, 1), c(0, 0),
                           matrix(0, 1, 2), matrix(c(1, 2, 2, 1), 2),
                           cbind(1, c(0, 1)), list(1:2))
  expect_false(d$converged)
  expect_match(d$message, "covariance")
})

test_that("prediction range helpers select the intended boosted trees", {
  set.seed(12)
  x <- matrix(rnorm(80), 40, 2)
  y <- x[, 1] + rnorm(40, sd = .1)
  bst <- xgboost::xgb.train(list(objective = "reg:squarederror", base_score = 0,
                                max_depth = 1, nthread = 1),
                           xgboost::xgb.DMatrix(x, label = y), nrounds = 5, verbose = 0)
  all <- predict(bst, x, outputmargin = TRUE)
  first <- predict(bst, x, outputmargin = TRUE, iterationrange = MEXGB:::xgb_range(1, 2))
  last <- predict(bst, x, outputmargin = TRUE, iterationrange = MEXGB:::xgb_range(3, 5))
  expect_equal(all, first + last, tolerance = 1e-6)
  expect_s3_class(xgboost::xgb.importance(model = bst, trees = MEXGB:::xgb_trees(5)), "data.frame")
})
