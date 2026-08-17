# Phase 2 (interim) — Baselines & Gradient Boosting

**Date:** 2026-06-22  **Status:** Baselines + GBM done; compact deep model + ANOVA pending.
**Setup:** Predict clear-sky index kt(t+h) → reconstruct GHI = kt·GHI_clearsky(t+h).
Univariate (GHI-derived) features, all 9 years. Split: train 2016–2022, calib 2023
(reserved for Phase-3 conformal), **test 2024**. Horizons 5/15/30/60 min. Daytime,
non-interpolated targets only.

## Headline (test 2024, GHI W/m²)
| Horizon | Best model | RMSE | MAE | Skill vs smart-persistence |
|---|---|---|---|---|
| 5 min | GBM | 88.6 | 41.6 | +9.5% |
| 15 min | GBM | 113.8 | 60.5 | +15.0% |
| 30 min | GBM | 125.2 | 70.5 | +15.6% |
| 60 min | GBM | 138.2 | 82.7 | +14.9% |

Ranking at every horizon: **GBM > linear AR > smart persistence > naive persistence.**
All pairwise gains are statistically significant (Diebold–Mariano p < 1e-40).

## Key findings
- **Smart (clear-sky) persistence is a strong baseline**; naive persistence degrades
  badly with horizon (skill −42% at 60 min). Both are therefore reported.
- **Learned models help and the gain is significant**, but the increment from linear AR
  to GBM is modest — consistent with the view that architecture
  differences are often small; our contribution is *uncertainty + decision value*, not
  squeezing point accuracy.
- **Regime stratification is the real story:** transitional (broken-cloud) RMSE is
  4–6× clear-sky RMSE (e.g., 60-min GBM: clear 74, cloudy 150, transitional 197 W/m²).
  This is precisely why calibrated, regime-aware uncertainty (Phase 3) matters
  operationally.
- At **5 min in stable (clear/cloudy) conditions**, persistence is essentially
  unbeatable; GBM's advantage comes from transitional periods and longer horizons —
  an honest, nuanced result.

## Outputs
Tables: `04_results/tables/p2_point_metrics.csv`, `_by_regime.csv`, `p2_dm_tests.csv`.
Figures: `p2_skill_vs_horizon.png`, `p2_rmse_by_regime_60min.png`.
Test predictions (for Phase 3 conformal): `02_data/interim/p2_test_pred_h{1,3,6,12}.parquet`.

## Remaining for Phase 2
Compact GRU / GRU-TCN (links to base paper) + ANOVA across years/regimes; then Phase 3.
