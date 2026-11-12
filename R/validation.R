## Shared input checks; kept internal to the package.
mexgb_number <- function(x, name, lower = -Inf, upper = Inf,
                         integer = FALSE, open_lower = FALSE, open_upper = FALSE,
                         allow_inf = FALSE) {
  ok <- is.numeric(x) && length(x) == 1L && !is.na(x) &&
    (is.finite(x) || (allow_inf && identical(x, Inf)))
  if (ok) ok <- (if (open_lower) x > lower else x >= lower) &&
    (if (open_upper) x < upper else x <= upper) &&
    (!integer || !is.finite(x) || x == floor(x))
  if (!ok) stop(name, " must be a valid ", if (integer) "integer" else "number",
                " in the documented range.", call. = FALSE)
  invisible(x)
}

mexgb_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE.", call. = FALSE)
}

mexgb_matrix <- function(x, name) {
  x <- as.matrix(x)
  if (!is.numeric(x) || nrow(x) < 1L || ncol(x) < 1L || any(!is.finite(x)))
    stop(name, " must be a non-empty, finite numeric matrix.", call. = FALSE)
  nm <- colnames(x)
  if (!is.null(nm) && (anyNA(nm) || any(!nzchar(nm)) || anyDuplicated(nm)))
    stop(name, " must have unique, non-empty column names.", call. = FALSE)
  x
}

mexgb_rows <- function(id, time, n) {
  if (!is.atomic(id) || !is.null(dim(id)) || length(id) != n || anyNA(id) ||
      any(!nzchar(as.character(id))) || (is.numeric(id) && any(!is.finite(id))))
    stop("id must contain one non-missing identifier per observation.", call. = FALSE)
  if (!is.numeric(time) || !is.null(dim(time)) || length(time) != n || any(!is.finite(time)))
    stop("time must contain one finite numeric measurement time per observation.", call. = FALSE)
}

mexgb_sim_inputs <- function(linear, rel_p, time_points, rho_W, rho_Z,
                             random_sd_intercept, random_sd_slope, pilot_n, pilot_seed) {
  mexgb_flag(linear, "linear")
  mexgb_number(rel_p, "rel_p", 1, integer = TRUE)
  mexgb_number(time_points, "time_points", 2, integer = TRUE)
  mexgb_number(rho_W, "rho_W", -1, 1, open_lower = TRUE, open_upper = TRUE)
  mexgb_number(rho_Z, "rho_Z", -1, 1)
  mexgb_number(random_sd_intercept, "random_sd_intercept", 0)
  mexgb_number(random_sd_slope, "random_sd_slope", 0)
  mexgb_number(pilot_n, "pilot_n", 2, integer = TRUE)
  mexgb_number(pilot_seed, "pilot_seed", 0, .Machine$integer.max, integer = TRUE)
}

