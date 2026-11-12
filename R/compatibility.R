## Keep the Cox loss consistent with MEXGB's Breslow risk-set calculations.
## Earlier glmnet releases used Breslow without a configurable cox.ties argument.
mexgb_glmnet <- function(x, y, family, ...) {
  extra <- list(...)
  if (family == "cox" && "cox.ties" %in% names(formals(glmnet::glmnet)))
    extra$cox.ties <- "breslow"
  do.call(glmnet::glmnet, c(list(x = x, y = y, family = family), extra))
}
