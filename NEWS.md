# MEXGB 1.0.0

First CRAN release. Fitted models are identical to MEXGB 0.5.0 for the same
inputs, seed and xgboost 3.x; the changes below are interface, validation,
documentation and robustness changes.

## Renamed arguments

* `honest`, `honest_d`, `honest_offset` are now `heldout_blup`,
  `heldout_dscale`, `heldout_offset`; the returned `honest_loss` is now
  `heldout_loss`.
* `patience` is now `stall_iter` in `MEXGB()` and `stall_rounds` in
  `mexgb_mstep()`.

## Other changes

* Input validation for `MEXGB()`, `predict()`, `simLong()` and `calibrateSim()`;
  infeasible cross-fitting folds are rejected with an informative error.
* `predict()` aligns `X` and `Z` by column name and rejects `type = "survival"`
  for non-Cox fits and non-right-censored `Surv` responses at fitting.
* With `crossfit = 0`, `linear = TRUE` now warns that the linear part is not
  fitted (previously it was ignored silently).
* Laplace E-step optimizer diagnostics are returned in `laplace_diagnostics`.
* The returned `linear` and `screened` entries now describe the selected
  iteration.
* Fixed the cross-validated tree grid when fewer than `grid_step` rounds remain.
* Compatibility with xgboost 1.7 tree indexing (`iterationrange`, `trees`).
* Tied scores handled in the simulation discrimination helpers.
* Rewritten manual, a vignette, and additional regression tests.
* Attribution for Gaussian update code adapted from MEGB 0.2.
