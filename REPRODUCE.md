# Reproducing the results

1. `pip install -r requirements.lock.txt` — exact `==` pins for the environment in which
   the cached predictions were last regenerated and checked. `requirements.txt` is the
   same dependency list without versions, kept for readers who want it.
2. Obtain the raw data and place it as described in [`data/README.md`](data/README.md).
   The filenames matter — the preprocessing scripts glob for them.
3. Run `bash reproduce.sh` (or the steps below individually). Outputs land in `results/`.

`code/utils/config.py` detects whether it is running in this release layout (`data/`,
`results/`) or in the authors' working tree (`02_data/`, `04_results/`), so nothing needs
configuring.

Long runs are structured so they can be interrupted and resumed: the feature frame is cached
once, each horizon runs in its own process, and the fitters `_j2_fit.py`, `_j5_fit.py` and
`r1/r1_fit_cache.py` write incrementally and can simply be re-run until they report `ALL_DONE`.

**Paths in the two phase lists below are relative to `code/`**, which is the directory
`reproduce.sh` runs them from.

## Phases 1–7 — the first analysis pass

Produces the unprefixed result files (`j2_*`, `j5_*`, `p2_*`, …).

1. `preprocessing/p2_clean.py` → `preprocessing/p3_clearsky_regimes.py` (Yulara) and `preprocessing/dkasc_prepare.py` (DKASC)
2. `run_phase2_baselines.py` (point benchmark) — deep models optional
3. `_j2_fit.py` + `_j2_one_horizon.py` (per horizon) → `_j2_aggregate.py` (conformal)
4. `_j3_one_horizon.py` → `_j3_aggregate.py` (external site)
5. `_j4_one_horizon.py` → `_j4_aggregate.py` (multivariate)
6. `_j5_fit.py` + `dispatch/dispatch_soc.py` → `_j5_aggregate.py` (reserve scheduling + CVaR)
7. `_j6_drift.py` (per year) + `_j6_ablations.py` → `_j6_aggregate.py` (robustness)

## Phase 8 — the final analysis pass

Produces the `r1_`-prefixed files, which are what the article reports.
Run with `bash reproduce.sh r1` if phases 1–3 have already run. The final pass reads
the cleaned frames and cached predictions that phases 1–3 produce, so an `r1`-only
rerun requires those earlier outputs to exist, generated under the same preprocessing
mode (see the gap-fill note below); it does not rebuild them from raw data.

| step | command | what it answers |
|---|---|---|
| 8a | `bash code/r1/r1_restore_env.sh`, then `r1_fit_extra_taus.py <site> <h>` | rebuilds the feature frames, the GHI→PV map and the cached GBM point + quantile predictions for both sites × 4 horizons |
| 8b | `r1_p2_point_causal.py`, `r1_p2_cv_anova.py`, `r1_p2_deep_vs_gbm.py`, `r1_p2_table2_build.py` | point benchmark under the causal clear-sky regime, and the gradient-boosting-vs-deep comparison on common support with its p-values |
| 8c | `r1_j2_delayed.py <site> <h>`, `r1_j2_stats.py`, `r1_j2_aggregate.py`, `r1_j2_figures.py` | adaptive conformal inference with an **h-step feedback delay** (the first-pass implementation updated with a target not yet observable at issue time), quantile-crossing rate, upper-bound capping, Diebold–Mariano tests and day-block bootstrap intervals |
| 8d | `r1_dispatch.py <site> <h> all`, `r1_j5_aggregate.py`, `r1_j5_vc_stats.py` | reserve dispatch with the reserve level selected on a set that **excludes the test year**, plus the mean–CVaR frontier, battery and cost-ratio sweeps and bootstrap intervals on value captured |
| 8e | `r1_regen_phase_reports.py [--verify]` | generates the J2 and J5 phase reports from the current tables and machine-checks all 390 numbers |
| 8f | `r1_s9_regimes.py`, `r1_s9_stats.py`, `r1_s9_dkasc_mirror.py`, `r1_j6_aggregate.py`, `r1_s9_causal_rescore.py` | the weather-regime split on one stated denominator (and Figure 1), multiplicity control on the designated primary analysis families, selection regret, the multi-day block bootstrap, the repeated-measures refit, the external-site mirror table, the regenerated robustness layer, and the measurement of how much the interpolated-input exposure is worth |
| 8g | `r1_s10_verify_runlength_guard.py` | checks the cleaner's run-length guard, and the count of long-run interpolated cells it withholds, against the shipped data |

`code/r1/r1_deep_causal.py` requires PyTorch and is the only step that is not CPU-cheap; it
is optional, and its outputs are shipped in `results/tables/r1_p2_deep_causal*.csv`. See
`code/models/HOW_TO_RUN_DEEP_MODEL.md`.

## Known reproducibility limits, stated rather than buried

- The reserve-dispatch tables are **not** bit-for-bit reproducible against the first-pass
  run. Re-running the identical protocol moves the mean daily cost while preserving
  every ranking at every reserve level; the size of the movement and the unresolved
  356-versus-357-day provenance are in
  `results/reports/2026-08-16-J5-Dispatch-SoC-CVaR-Report.md`.
- The robustness layer — both ablation tables, the drift summaries and the drift figure —
  **was regenerated in the current stack** and ships as `results/tables/r1_j6_*` and
  `results/figures/r1_j6_drift_picp_by_year.png`. The first-pass `j6_*` files are kept
  beside them, unmodified, so the two can be compared. An earlier version of this file
  said the tables had not been regenerated; that was true when written and is not now.
- **The shipped results predate the cleaner's run-length guard.** They were produced when
  `code/preprocessing/p2_clean.py` filled the first six steps of a gap of any length,
  which affected 4,014 of 13,925 imputed GHI cells. The guard is the default now, so a fresh
  run from raw data yields slightly different absolute levels; for bit-for-bit
  article reproduction, `reproduce.sh` passes `--legacy-gapfill` to
  `code/preprocessing/p2_clean.py`, which applies the exact pre-guard fill that
  produced the shipped artifacts. The two modes must never be mixed in one run. That difference was
  measured, not assumed: see the provenance note at the end of `MANIFEST.md`,
  `code/r1/r1_s9_causal_rescore.py` and `code/r1/r1_s10_verify_runlength_guard.py`.
- **One figure has no generating script here.** Figure 7 of the article is an
  illustrative example-day plot from the first pass; the file ships, but the script that
  drew it was not retained. It is the only exhibit for which this is true and no reported
  number is read from it. `MANIFEST.md` says so in the row itself.
- `results/metrics/j2_summary.json` carries a regime coverage-error statistic pooled over
  all four horizons under a heading that reads as 5-minute. It is a first-pass
  file and is therefore not rewritten; the corrected 5-minute values are in
  `results/tables/r1_j2_ace_rms_5min.csv` and in the regenerated J2 report.
