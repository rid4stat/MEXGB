# MEXGB 1.0.0

* Fixed CV tree-grid construction for small and partial final chunks.
* Corrected the unseen-subject prediction regression test to include population Z effects.
* Reject unsupported survival predictions and unsupported survival response types.
* Added fit, prediction and simulation validation, and explicit fold-feasibility checks.
* Removed full-data substitution for failed cross-validation linear fits.
* Added Laplace optimizer convergence diagnostics and explicit PQL fallback warnings.
* Replaced row-independent selection standard errors with subject-clustered heuristics.
* Preserved selected-iteration screening metadata; improved Gaussian log-determinant stability.
* Stabilised simulated discrimination calculations and handled tied scores.
* Updated the manual to describe the implemented model, returned objects and limitations.
* Added regression tests, a vignette, release instructions and cross-platform CI.
* Removed the duplicate survival dependency declaration.
* Explicitly request Breslow Cox tie handling when supported by glmnet.
* Added attribution for Gaussian update code adapted from MEGB 0.2.

These changes can alter fitted models and selected variance components relative
to 0.5.1. Re-run scientific benchmarks before using results interchangeably.
