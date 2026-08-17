# J4 — Multivariate Feature Study (2023–24)

_Date: 2026-06-23. Question: does adding **valid meteorology** (temp, wind, pressure, rain) improve point / interval performance over the univariate clear-sky-index model? Honest, leakage-free, same train/test for every feature set._

## Design
Auxiliary met sensors are reliable only in **2023–24** (P1 finding; verified here: 100% non-null daytime). To isolate the *feature-set* effect (not training-period effect):
- **train** = 2023 first 80% (chronological), **calib** = 2023 last 20%, **test** = full 2024.
- Feature sets: **uni** (kt lags/rolling + cyclical + cosz), **+temp** (+air & module temp), **+wind** (+speed, gust, direction sin/cos), **all** (+pressure, rain).
- Base learner: GBM in kt-space (same as the project). Point RMSE/MAE by horizon & regime; intervals via **Mondrian-CQR** @90% (the J2-recommended method); **Diebold–Mariano** each multivariate set vs univariate; CRPS.
- Code: `03_code/_j4_one_horizon.py`, `_j4_aggregate.py`. Outputs `04_results/tables/j4_*`, `figures/j4_rmse_by_featureset.png`, `metrics/j4_summary.json`.

## Point-forecast results — RMSE (W/m², all conditions)
| set | 5 | 15 | 30 | 60 |
|---|---|---|---|---|
| uni | 89.32 | 117.62 | 133.86 | 153.07 |
| +temp | 89.30 | 117.16 | 134.71 | 153.66 |
| +wind | 89.38 | 117.56 | 133.15 | 152.53 |
| all | 89.44 | 116.94 | **130.00** | 157.76 |

**% change vs univariate** (negative = better): differences are within ±0.6% at 5/15/60 min for +temp/+wind. The only material move is **all** at **30-min: −2.88%**, offset by **all** *overfitting* at 60-min (+3.07%).

### Significance (Diebold–Mariano vs univariate; positive stat ⇒ multivariate better)
- 5-min: no set significantly differs from univariate (p = 0.36–0.87).
- 30-min: **all** significantly better (DM +5.46, **p = 4.8e-8**); +wind borderline (p = 0.058).
- 60-min: **all** significantly *worse* (DM −2.88, p = 0.004) — extra features overfit the thin 2023 training window at long range.

## Interval / probabilistic results (Mondrian-CQR @90%, all conditions)
PICP near nominal at 5–30 min for all sets (0.85–0.89); PINAW and CRPS differ only marginally (CRPS@5min 24.4–24.7; @30min ~39.3 for every set). At 60-min the intervals are unstable (uni PICP 0.61) because the calibration set is only ~6,800 points (last 20% of one year) — a small-sample artifact, not a feature effect; +temp happens to widen and recover coverage. The interval comparison is therefore **inconclusive at 60-min** and **flat (no meaningful gain) at 5–30 min**.

## Conclusion (honest, and it supports the thesis)
Adding meteorology yields **no benefit at the ultra-short (5-min) horizon**, a **small (~3%) but significant point-RMSE gain only at 30-min** (full feature set), and **slight harm at 60-min** from overfitting the limited multivariate window. Interval/decision-relevant metrics are essentially **unchanged**. This reproduces and generalizes the base paper's finding (Elmousaid et al. 2024: multivariate did not help the 1-step GHI forecast) across horizons and regimes, and reinforces the project's central claim: **the operational lever is calibrated, regime-aware uncertainty and its decision value — not richer inputs or architecture.**

## Caveats
- Absolute RMSE here (89.3 @5min) is marginally higher than the main benchmark (88.6) because training uses 2023 only (≈38k vs ≈700k rows); the *within-J4* comparison is the valid one.
- The met benefit at 30-min is real but modest; a defensible framing is "meteorology offers limited, horizon-specific value and does not change the uncertainty story."
- A multivariate × decision-value test (does met change reserve cost?) is deferred to J5, where the dispatch layer is built; given the flat interval metrics, no large effect is expected.
