# J3 — External-Site Validation: DKASC Alice Springs

_Date: 2026-06-23. Cross-site generalization test. Second arid-desert site:
**DKASC Alice Springs** Class-A weather station, 5-min, 2020–2024. GHI =
`101_DKA_WeatherStation_Global_Horizontal_Radiation` (verified ~100% non-null).
Pipeline mirrors Yulara exactly: clean → calibrated clear-sky index → regimes →
GBM (kt-space) → conformal → decision. Train 2020–22 / calib 2023 / test 2024._

## Why this site
DKASC Alice Springs is ~460 km from Yulara, same arid-desert (Köppen BWh) climate,
independent instrument and operator. A clean test of whether the project's calibration
+ decision-value story is site-specific or a transferable principle.

## Site characterisation transfers
- Clear-sky alignment: per-day correlation **0.983** (Yulara 0.98); clear-day calibration factor 1.091 (Yulara 1.049) — both pristine-desert envelopes.
- **Regime distribution (daytime): clear 52.9% / transitional 36.5% / cloudy 8.3%** vs **Yulara 52% / 36% / 7%** — almost identical. The weather-regime structure the method exploits is a property of the climate, not the single site.

## Point forecasting transfers
DKASC GBM test-2024 RMSE (W/m²): **83.8 / 110.3 / 123.9 / 138.4** @ 5/15/30/60 min — on par with (slightly better than) Yulara (88.6/113.8/125.2/138.2). The lightweight kt-space GBM generalizes.

## Calibration story transfers (headline)
Per-regime PICP @90%, 5-min:
| method | all | clear | transitional | cloudy |
|---|---|---|---|---|
| ICP (marginal) | 0.876 | 0.988 | **0.694** | 0.978 |
| Mondrian | 0.894 | 0.891 | 0.896 | 0.898 |
| CQR | 0.903 | 0.928 | 0.860 | 0.934 |
| Mondrian-CQR | 0.905 | 0.909 | 0.899 | 0.910 |
| ACI | 0.926 | 0.961 | 0.868 | 0.962 |

The **identical failure mode** recurs at DKASC: marginal ICP collapses to **0.69** coverage in the transitional regime while over-covering (0.99) in clear/cloudy — matching Yulara's 0.70. The **identical fix** works: Mondrian and Mondrian-CQR restore ~0.89–0.91 in every regime.

### Cross-site per-regime calibration (ACE-RMS @90%, 5-min; lower = better)
| method | Yulara | DKASC |
|---|---|---|
| ICP | 0.139 | 0.137 |
| Mondrian | 0.0094 | **0.0056** |
| CQR | 0.048 | 0.034 |
| Mondrian-CQR | 0.0124 | **0.0079** |
| ACI | 0.048 | 0.053 |

Regime-aware methods are as well-calibrated (or better) at the unseen site. CRPS ordering also transfers: **Mondrian-CQR best (15.75)** ≈ CQR (15.81) < Mondrian (19.0) < ICP (24.6).

## Decision value transfers
Load-independent operating-reserve newsvendor in GHI space, cost ratio r=10, value captured = (det−mondrian)/(det−oracle):
| horizon | 5 | 15 | 30 | 60 |
|---|---|---|---|---|
| value captured (Mondrian) | **39.3%** | 48.2% | 50.6% | 53.5% |

Yulara captured 36.9% @5min and 44.6% @30min — DKASC is comparable and if anything slightly stronger. Reserve cost falls from 200.8 (deterministic) → 129.5 (Mondrian) toward the 19.3 oracle floor at 5-min.

## Verdict (J3 verification criteria)
- **Same data-quality checks pass:** GHI clean, per-day clear-sky corr >0.90, sane regime distribution. ✓
- **Results qualitatively consistent:** point, calibration, CRPS ordering, and decision value all transfer; the transitional miscalibration of marginal CP and the Mondrian/Mondrian-CQR fix reproduce on an independent site. ✓
- **Conclusion:** the contribution is a **transferable principle for arid-desert ultra-short-term GHI**, not a Yulara artifact. (Generalization beyond arid climates is future work — a temperate/tropical site would be the natural next test; NREL remains the documented option.)

## Artifacts
- Data: `02_data/DKASC/raw/Alice_Springs_2020..2024.csv` (source), `02_data/DKASC/cleaned/asp_clean_5min.parquet`, `02_data/DKASC/regime_labels/asp_regimes_5min.parquet`.
- Code: `03_code/preprocessing/dkasc_prepare.py`, `03_code/_j3_one_horizon.py`, `_j3_aggregate.py`.
- Tables: `04_results/tables/j3_*` (point, interval, crps, decision, cross-site comparison). Figures: `j3_crosssite_calibration.png`, `j3_dkasc_value_captured.png`. Metrics: `j3_summary.json`, `j3_asp_clearsky_alignment.json`.

## Notes / limitations
- DKASC WeatherStation wind channel is empty in these files; not needed for GHI forecasting (multivariate value already studied at Yulara in J4). Temp/humidity present if a DKASC multivariate cross-check is later wanted.
- Same-climate validation by design (strongest, unconfounded comparison). A different-climate stress test is optional future work.
