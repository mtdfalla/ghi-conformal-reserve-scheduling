# Phase 3 — Calibrated Uncertainty via Conformal Prediction

**Date:** 2026-06-22  **Status:** Complete (core novelty of the project).

Goal: turn the GBM point forecaster into **calibrated, regime-aware prediction
intervals**, and show that *how* uncertainty is expressed matters operationally —
especially in the transitional (broken-cloud) regime where point errors are largest.

## Setup
- Base model: gradient boosting in clear-sky-index space (from Phase 2).
- **Split conformal:** fit on 2016–2022, **calibrate on 2023**, **test on 2024**.
- Nominal coverages 80/90/95%; horizons 5/15/30/60 min; intervals in GHI (W/m²).
- Methods: **ICP** (marginal), **ICP-normalized** (width ∝ clear-sky), **Mondrian**
  (regime-conditional — the novelty), **CQR** (conformalized quantile regression).
- Metrics: PICP (coverage), ACE, PINAW (sharpness), Winkler, CRPS — overall and per regime.

## Headline finding
A single global (marginal) interval is **badly miscalibrated across regimes**.
At nominal 90%, 5-min (PICP, target 0.90):

| Method | clear | transitional | cloudy | overall |
|---|---|---|---|---|
| ICP (marginal) | 0.989 | **0.696** | 0.990 | 0.878 |
| ICP-normalized | 0.950 | 0.783 | 0.999 | 0.891 |
| CQR | 0.914 | 0.857 | 0.970 | 0.898 |
| **Mondrian (regime)** | **0.894** | **0.885** | **0.904** | **0.892** |

Marginal ICP **over-covers** stable conditions (clear/cloudy ≈ 0.99) while **severely
under-covering the transitional regime (0.70)** — i.e., it is most wrong exactly when
uncertainty matters most. **Regime-conditional (Mondrian) conformal prediction restores
calibration in every regime** (all ≈ 0.88–0.90).

## Sharpness (PINAW at 90%, 5-min; lower = tighter)
| Method | clear | transitional | cloudy |
|---|---|---|---|
| Mondrian | **0.049** | 0.290 | **0.124** |
| CQR | 0.082 | 0.248 | 0.294 |
| ICP | 0.166 | 0.144* | 0.258 |

Mondrian is **tightest in stable conditions** and appropriately **widest in transitional**
— it puts interval width where the risk is. (*ICP's narrow transitional width is why it
under-covers there.)

## Probabilistic accuracy (CRPS, W/m², lower better)
| Method | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| **CQR** | **19.9** | **25.9** | **29.1** | **33.5** |
| Mondrian | 21.2 | 27.6 | 31.8 | 37.1 |
| ICP / ICP-norm | ~26 | ~35 | ~38–39 | ~43 |

CQR gives the best overall CRPS (adaptive, full-distribution sharpness); Mondrian is a
close second and is the best-calibrated per regime. Both clearly beat plain ICP.

## Takeaways for the paper
1. **Regime-conditional conformal prediction is the contribution:** it fixes the
   per-regime miscalibration of marginal CP and is sharpest in stable conditions. This is
   under-explored for ultra-short-term GHI and directly operationally relevant.
2. **CQR is the strongest single-distribution method (CRPS)** and a natural companion;
   the paper can present Mondrian (calibration story) + CQR (accuracy story).
3. The transitional regime needs ~2–6× wider intervals than clear — consistent with the
   Phase-2 point-error structure. Uncertainty, not point accuracy, is the lever.
4. All methods are **post-hoc and CPU-cheap** (no GPU), reinforcing the practical angle.

## Outputs
Tables: `04_results/tables/p3_interval_metrics.csv`, `p3_crps.csv`.
Figures: `p3_reliability.png`, `p3_picp_by_regime_90.png`, `p3_pinaw_by_regime_90.png`,
`p3_crps_by_horizon.png`, `p3_example_day_bands.png`.
Code: `03_code/conformal/conformal.py`, `03_code/_p3_one_horizon.py`.
Reproducibility: `02_data/interim/p3_h1_bands.parquet` (test bands for plotting).

## Next (Phase 4)
Translate these intervals into operational value: GHI→PV mapping (mind Q5),
reserve/dispatch simulation, and the cost of forecast uncertainty (interval-aware vs
deterministic vs perfect foresight). Needs the battery/diesel assumptions (Q1).
