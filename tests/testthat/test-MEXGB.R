test_that("simLong returns the documented structure", {
  set.seed(1)
  d <- simLong(n = 10, p = 8, time_points = 5, linear = FALSE)
  expect_equal(dim(d), c(50, 13))
  expect_equal(range(d$time), c(0, 1))
  db <- simLong(n = 10, p = 8, time_points = 5, linear = TRUE, family = "binomial")
  expect_true(all(db$Y %in% c(0, 1)))
})

test_that("MEXGB fits and predicts a continuous outcome", {
  set.seed(1)
  d <- simLong(n = 12, p = 8, time_points = 6, linear = TRUE)
  X <- as.matrix(d[, -1:-5]); Z <- as.matrix(d[, 4:5])
  fit <- MEXGB(X = X, Y = d$Y, id = d$id, Z = Z, time = d$time, ntree = 30,
               interaction.depth = 2, iter = 15, verbose = FALSE)
  expect_s3_class(fit, "MEXGB")
  expect_equal(dim(fit$random_effects), c(12, 2))
  p <- predict(fit, X = X, Z = Z, id = d$id, time = d$time)
  expect_length(p, nrow(d))
  expect_lt(mean((p - d$Y)^2), stats::var(d$Y))
  # unseen subjects get the fixed-effect prediction only
  pnew <- predict(fit, X = X, Z = Z, id = d$id + 1000, time = d$time)
  expect_equal(pnew, MEXGB:::fixedMEXGB(fit, X) + as.numeric(Z %*% fit$beta_z))
  expect_equal(sum(fit$importance), 100)
})

test_that("MEXGB fits and predicts a binary outcome", {
  set.seed(2)
  d <- simLong(n = 12, p = 8, time_points = 6, linear = TRUE, family = "binomial")
  X <- as.matrix(d[, -1:-5]); Z <- as.matrix(d[, 4:5])
  fit <- MEXGB(X = X, Y = d$Y, id = d$id, Z = Z, time = d$time, ntree = 30,
               interaction.depth = 2, iter = 15, family = "binomial", verbose = FALSE)
  p <- predict(fit, X = X, Z = Z, id = d$id, time = d$time)
  expect_true(all(p > 0 & p < 1))
  expect_true(all(is.finite(fit$LL)))
})

test_that("binary calibration reaches the target AUC and prevalence", {
  cal <- calibrateSim("binomial", target = 0.85, prevalence = 0.3, linear = FALSE, pilot_n = 200)
  expect_equal(cal$achieved, 0.85, tolerance = 1e-3)
  set.seed(3)
  d <- simLong(n = 200, p = 6, linear = FALSE, family = "binomial", target_auc = 0.85,
               prevalence = 0.3, pilot_n = 200)
  expect_lt(abs(mean(d$Y) - 0.3), 0.05)
})

test_that("MEXGB fits and predicts a clustered survival outcome", {
  set.seed(4)
  d <- simLong(n = 15, p = 8, time_points = 6, linear = TRUE, family = "cox",
               random_sd_slope = sqrt(0.5), target_c = 0.75, censoring = 0.3, pilot_n = 100)
  expect_lt(abs(mean(d$status) - 0.7), 0.02)
  X <- as.matrix(d[, grep("^Var", names(d))]); Z <- as.matrix(d[, 4:5])
  fit <- MEXGB(X = X, Y = cbind(d$Y, d$status), id = d$id, Z = Z, time = d$time,
               ntree = 30, interaction.depth = 2, iter = 10, family = "cox", verbose = FALSE)
  lp <- predict(fit, X = X, Z = Z, id = d$id, time = d$time, type = "link")
  expect_length(lp, nrow(d))
  S <- predict(fit, X = X[1:4, ], Z = Z[1:4, ], id = d$id[1:4], time = d$time[1:4],
               type = "survival", times = c(0.1, 1, 5))
  expect_equal(dim(S), c(4, 3))
  expect_true(all(S >= 0 & S <= 1) && all(S[, 1] >= S[, 3]))
})

test_that("the cross-fitted E-step keeps the random effects (no collapse to plain XGBoost)", {
  set.seed(5)
  d <- simLong(n = 20, p = 100, time_points = 10, linear = TRUE)  # true D_22 = 3, sigma^2 = 0.25
  X <- as.matrix(d[, grep("^Var", names(d))]); Z <- as.matrix(d[, 4:5])
  args <- list(X = X, Y = d$Y, id = d$id, Z = Z, time = d$time, ntree = 300,
               interaction.depth = 3, iter = 20, verbose = FALSE)
  old <- do.call(MEXGB, c(args, list(crossfit = 0, init = "zero")))
  new <- do.call(MEXGB, c(args, list(crossfit = 5)))
  expect_equal(new$crossfit, 5)
  # in-sample E-step: the boosting interpolates Y and D collapses towards 0
  expect_lt(old$var_random_effects[2, 2], 1)
  expect_gt(new$var_random_effects[2, 2], 1)
  # folds: every individual spread over the folds, caller's RNG untouched
  set.seed(9); r1 <- runif(1); set.seed(9); f <- MEXGB:::cf_make_folds(d$id, 5, 1); r2 <- runif(1)
  expect_equal(r1, r2)
  expect_true(all(tapply(f, d$id, function(x) length(unique(x))) == 5))
})

test_that("PQL dispersion is finite (cox) and fixed at 1 (binary)", {
  set.seed(6)
  d <- simLong(n = 15, p = 8, time_points = 6, linear = TRUE, family = "cox",
               random_sd_slope = sqrt(0.5), target_c = 0.75, censoring = 0.3, pilot_n = 100)
  X <- as.matrix(d[, grep("^Var", names(d))]); Z <- as.matrix(d[, 4:5])
  fc <- MEXGB(X = X, Y = cbind(d$Y, d$status), id = d$id, Z = Z, time = d$time, ntree = 30,
              interaction.depth = 2, iter = 8, family = "cox", verbose = FALSE)
  expect_true(is.finite(fc$sigma) && fc$sigma > 0 && !fc$diverged)
  db <- simLong(n = 12, p = 8, time_points = 6, linear = TRUE, family = "binomial")
  fb <- MEXGB(X = as.matrix(db[, -1:-5]), Y = db$Y, id = db$id, Z = as.matrix(db[, 4:5]),
              time = db$time, ntree = 30, interaction.depth = 2, iter = 8, family = "binomial",
              verbose = FALSE)
  expect_equal(as.numeric(fb$sigma), 1)
})

test_that("v0.4: inner E-step, centred Cox frailties and calibration slope", {
  set.seed(8)
  d <- simLong(n = 20, p = 8, time_points = 10, linear = TRUE, family = "cox",
               random_sd_slope = sqrt(0.5), target_c = 0.75, censoring = 0.3, pilot_n = 100)
  X <- as.matrix(d[, grep("^Var", names(d))]); Z <- as.matrix(d[, 4:5])
  fc <- MEXGB(X = X, Y = cbind(d$Y, d$status), id = d$id, Z = Z, time = d$time, ntree = 40,
              interaction.depth = 2, n.minobsinnode = 2, iter = 6, family = "cox", verbose = FALSE)
  expect_equal(mean(fc$random_effects[, 1]), 0, tolerance = 1e-8)      # identifiability
  expect_true(is.finite(fc$cal_slope) && fc$cal_slope > 0)
  expect_true(all(is.finite(fc$var_random_effects)))
  # inner = 1 (MEXGB 0.3 E-step) still runs and gives a valid fit
  f1 <- MEXGB(X = X, Y = cbind(d$Y, d$status), id = d$id, Z = Z, time = d$time, ntree = 40,
              interaction.depth = 2, iter = 4, family = "cox", inner = 1, calibrate = FALSE,
              verbose = FALSE)
  expect_equal(f1$cal_slope, 1)
  # gaussian: with the inner loop the variance components are at their fixed point
  g <- simLong(n = 20, p = 6, time_points = 10, linear = TRUE)
  fg <- MEXGB(X = as.matrix(g[, grep("^Var", names(g))]), Y = g$Y, id = g$id, Z = as.matrix(g[, 4:5]),
              time = g$time, ntree = 60, interaction.depth = 2, iter = 5, verbose = FALSE)
  expect_true(fg$var_random_effects[2, 2] > 0.5)
})
