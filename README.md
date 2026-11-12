# MEXGB 1.0.0

Mixed-effects extreme gradient boosting for longitudinal and clustered data.
Supports Gaussian, binary and clustered right-censored survival outcomes, with
subject-specific random effects and optional penalised linear fixed effects.

## Installation

From the source archive, after installing dependencies:

```r
install.packages(c("xgboost", "MASS", "survival", "glmnet"))
install.packages("MEXGB_1.0.0.tar.gz", repos = NULL, type = "source")
```

## Example

```r
library(MEXGB)
set.seed(1)
d <- simLong(12, 6, time_points = 5)
X <- as.matrix(d[paste0("Var", 1:6)])
Z <- as.matrix(d[c("RandomIntercept", "RandomSlope")])
fit <- MEXGB(X, d$Y, d$id, Z, time = d$time, ntree = 30,
             iter = 5, crossfit = 2, verbose = FALSE)
predict(fit, X, Z, d$id, d$time)
summary(fit)
```

This is an API example, not an estimate of out-of-sample performance. Short
iteration limits illustrate usage and need not yield a converged fit.

## Prediction and validation

- Existing subjects use their fitted random effects. Unseen identifiers use zero
  random effects while retaining the population contribution of `Z`.
- Binary and survival predictions for unseen subjects are conditional at zero
  random effects; they are not random-effect-integrated marginal predictions.
- Internal folds split observations within subjects. For new-subject evaluation,
  hold out whole subjects; for forecasting, respect temporal order. Use an
  independent test set or nested validation for performance claims.
- Cox inputs are two-column `(time, status)` matrices or right-censored `Surv`
  objects. Start-stop, interval censoring and longitudinal time-varying-covariate
  survival records are not supported.
- The Cox variance update uses a Poisson working approximation. The internal
  selection criterion is not a general-purpose inferential likelihood.
- Gaussian model fitting assumes conditionally independent residual errors.
  `simLong()` can generate additional residual AR(1) correlation; such runs assess
  robustness to residual misspecification, rather than a correctly specified fit.
- Simulation discrimination targets are calibrated on a finite pilot. Survival
  censoring is calibrated on each realised sample, making the rate sample-dependent.
- Inspect `converged`, `diverged`, `best_iter` and `laplace_diagnostics`.
  A failed Laplace optimization warns and retains the PQL update.

## Development

Run `Rscript tools/check-release.R` from the source root. The GitHub workflow
checks Linux, Windows, macOS and R-devel. See `RELEASE_GUIDE.md` for release steps
and remaining maintainer-specific actions. Documentation is generated from
roxygen comments in `R/`.

## Attribution and license

GPL-2. The algorithm builds on XGBoost and mixed-effects tree ensembles; see
the package manual and `inst/CITATION`. The package describes its relationship
to MEGB and credits the authors of the adapted Gaussian update code in
`inst/NOTICE`. Maintainers should confirm any additional source provenance.

For support, contact the maintainer listed in `DESCRIPTION`. After the repository
is created, add its verified URL and issue tracker to `DESCRIPTION`.
