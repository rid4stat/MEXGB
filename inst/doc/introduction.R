## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----gaussian-----------------------------------------------------------------
library(MEXGB)
set.seed(10)
d <- simLong(8, 6, time_points = 4, rho_W = 0)
X <- as.matrix(d[paste0("Var", 1:6)])
Z <- as.matrix(d[c("RandomIntercept", "RandomSlope")])
fit <- MEXGB(X, d$Y, d$id, Z, time = d$time,
             ntree = 10, iter = 2, inner = 2, crossfit = 2,
             linear = FALSE, screen = "none", verbose = FALSE)
fit
head(predict(fit, X, Z, d$id, d$time))

## ----new-subject--------------------------------------------------------------
head(predict(fit, X, Z, d$id + 100, d$time))
summary(fit)

