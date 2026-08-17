# Phase 2 Summary — Point Forecasting Benchmark (with rigour)

**Date:** 2026-06-22  **Status:** COMPLETE (incl. compact GRU/GRU-TCN, run locally).

Goal: establish honest, statistically-tested point-forecast baselines so the
project's real contribution (calibrated uncertainty + decision value) builds on
solid ground — and to stand on simple baselines, multiple
conditions, significance, no overclaiming).

## Setup
- Work in clear-sky-index space: predict kt(t+h) → GHI = kt·GHI_clearsky(t+h).
- Univariate (GHI-derived) features → all 9 years usable.
- Horizons: 5/15/30/60 min. Daytime, non-interpolated targets only.
- Two evaluation regimes:
  1. **Holdout:** train 2016–2022, calibrate 2023 (reserved for Phase-3 conformal), **test 2024**.
  2. **Expanding-window CV:** test each of 2019–2024, train on all prior years (6 folds).

## Models
Naive persistence · smart (clear-sky) persistence · linear AR (Ridge) · gradient
boosting (HistGBM). Compact GRU & GRU-TCN script ready (`03_code/models/deep_gru_tcn.py`).

## Headline results (holdout, test 2024, GHI W/m²)
| Horizon | GBM RMSE | GBM MAE | Skill vs smart-persistence |
|---|---|---|---|
| 5 min | 88.6 | 41.6 | +9.5% |
| 15 min | 113.8 | 60.5 | +15.0% |
| 30 min | 125.2 | 70.5 | +15.6% |
| 60 min | 138.2 | 82.7 | +14.9% |

Ranking at every horizon: **GBM > linear AR > smart persistence > naive persistence**,
all pairwise differences significant (Diebold–Mariano p < 1e-40).

## Statistical rigour
- **ANOVA on per-day RMSE** (factors: model, weather regime, year as block):
  - **Weather regime dominates** error (F = 166–206; p ≈ 1e-71…1e-87).
  - **Model effect significant** at all horizons (p = 5e-6 at 5 min → ~0 at 60 min).
  - **Year significant** (F ≈ 30–38) → year-to-year variability is real; multi-year
    validation is warranted.
  - Model×regime interaction negligible at 5 min, growing with horizon.
- **Robustness:** across the 6 CV years, GBM's skill over smart persistence is
  **always positive** — mean 7.7% (5 min) to 15.7% (60 min); std ≈ 1–2%; min ≥ 6.6%.

## Key takeaways for the paper
1. **Smart (clear-sky) persistence is a strong, must-report baseline;** naive
   persistence collapses with horizon (skill −42% at 60 min).
2. **Learned models give significant but modest point-accuracy gains** (linear AR →
   GBM is a small step). This supports the thesis: the contribution is *uncertainty
   and decision value*, not architecture.
3. **Regime is the dominant error driver** (transitional RMSE 4–6× clear-sky). This is
   the operational motivation for calibrated, regime-aware uncertainty (Phase 3).
4. At **5 min in stable conditions persistence is ~unbeatable**; model advantage comes
   from transitional periods and longer horizons.

## Outputs
Tables: `p2_point_metrics.csv`, `p2_point_metrics_by_regime.csv`, `p2_dm_tests.csv`,
`p2_anova.csv`, `p2_cv_summary_by_year.csv`, `p2_cv_perday_errors.csv`.
Figures: `p2_skill_vs_horizon.png`, `p2_rmse_by_regime_60min.png`, `p2_cv_skill_by_year.png`.
Code: `03_code/run_phase2_baselines.py`, `_p2_cv_one.py`, `models/deep_gru_tcn.py`,
`utils/datasets.py`, `evaluation/metrics.py`. Test predictions for Phase 3:
`02_data/interim/p2_test_pred_h{1,3,6,12}.parquet`.

## Compact GRU / GRU-TCN — COMPLETE (run locally on CPU, ~5 min)
Confirmatory deep models connecting to the base paper's GRU-TCN. Evaluated on the same
test 2024, with Diebold-Mariano aligned to GBM on common timestamps (~39,380).

| Model | RMSE 5 / 15 / 30 / 60 min (W/m²) |
|---|---|
| GRU | 92.0 / 118.6 / 130.7 / 146.3 |
| GRU-TCN | 91.3 / 119.4 / 130.0 / 144.3 |
| **GBM (workhorse)** | **88.6 / 113.8 / 125.2 / 138.2** |

**GBM significantly outperforms both deep models at every horizon** (Diebold-Mariano
p from 2.6e-38 at 5 min to 1.2e-10 at 60 min). GRU ≈ GRU-TCN (the TCN block does not
clearly help). The neural models cluster with linear AR — all below GBM.

**Conclusion (reinforces the thesis):** even the base paper's GRU-TCN architecture does
not beat a well-tuned gradient-boosting model on this task. Point-forecast architecture
is not the lever; the contribution is calibrated, regime-aware uncertainty (Phase 3) and
its decision value (Phase 4). GBM is confirmed as the base model for Phases 3-4.

Full comparison: `04_results/tables/p2_all_models_comparison.csv`,
figure `p2_all_models_rmse.png`; deep metrics `p2_deep_metrics.csv`.
