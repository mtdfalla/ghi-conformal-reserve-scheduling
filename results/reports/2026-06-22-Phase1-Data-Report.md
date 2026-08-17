# Phase 1 Report — Data Pipeline & Exploratory Analysis

**Date:** 2026-06-22  **Status:** Complete
**Scope:** Clean the 9-year Yulara record, establish a reliable GHI target, build a
clear-sky index and weather-regime labels, and characterise data quality so that
later modelling rests on a solid foundation.

---

## 1. Headline findings (read these first)

1. **The reliable GHI source is `Pyranometer_1`, not the channel named
   `Global_Horizontal_Radiation`.** The latter is *faulty in 2016–2022* (mostly
   near-zero during daytime; 2020 ~60% near-zero) and only becomes reliable in
   2023–2024. `Pyranometer_1` is clean and consistent across **all** years
   (daytime median ≈ 685–765 W/m², ~0% spurious lows). Where both are healthy
   (2024) they agree: corr 0.94, ratio 1.02. **We adopt `Pyranometer_1` as the GHI
   target for the whole record.** (the design decision.)

2. **Auxiliary meteorological sensors are heavily corrupted in 2016–2022** (air
   temperature, pressure, daily rainfall, module-temp-1, max-wind): long blocks of
   sentinel/garbage values (e.g. temperature reading 548,003; up to ~7-day blocks).
   They become reliable only in **2023–2024** (~92% valid). Wind speed/direction
   and module-temp-2 are usable across most years. The corruption is
   sensor-specific — GHI is unaffected in those rows (verified: 100% valid GHI
   where temperature is corrupt).

3. **PV power (`Total Site PV Generation`) is anomalous in 2016–2022** — daytime
   median ~955 kW, nearly flat, weak GHI correlation (0.25–0.45), consistent with a
   grid-export cap / clipping. In **2023–2024** it behaves physically (median
   ~570–630 kW, GHI corr 0.79–0.82). The metering appears to have been corrected
   around 2023. **Implication:** the GHI→PV mapping for the decision layer (Phase 4)
   should rely on 2023–2024 or explicitly model the cap. (Open question Q5.)

4. **Timezone is local ACST (UTC+9:30), confirmed empirically:** measured GHI peaks
   at 12:45 local (matches solar noon for this longitude); per-day correlation with
   modelled clear-sky has median **0.98**.

---

## 2. What was done (pipeline)

All steps are reproducible scripts in `03_code/preprocessing/` driven by
`03_code/utils/config.py`.

| Step | Script | Output |
|---|---|---|
| Audit raw schema/quality | (inline) | confirmed 9 files, 5-min, key cols consistent |
| Clean & consolidate | `p2_clean.py` | `02_data/cleaned/yulara_clean_5min.parquet` (+ flags) |
| Clear-sky & regimes | `p3_clearsky_regimes.py` | `02_data/regime_labels/yulara_regimes_5min.parquet` |
| EDA & figures | `p4_eda.py` | `04_results/figures/*.png`, tables, metrics |

**Cleaning operations:** parsed timestamps; removed **2,197** duplicate rows from
yearly-file overlaps; built a continuous 5-min grid (**920,451** rows,
2016-04-01 → 2024-12-31; 1,623 missing slots inserted); removed sentinels
(|x| ≥ 9e4) and out-of-physical-bounds values; interpolated short gaps (≤ 30 min);
clipped night-time negative irradiance/PV to 0.

**Clear-sky index:** pvlib Ineichen clear-sky, **calibrated to the clear-day
envelope** (factor 1.049 — the Ineichen model under-predicts this pristine-desert
site) so the clear-sky index `kt` centres clear points at ~1.0. A model-free
clearness index (`kt = GHI / extraterrestrial-horizontal`) is also stored.
(the design decision.)

---

## 3. GHI completeness (the usable target)

After cleaning, `Pyranometer_1` GHI is **91.9–98.5% complete** every year:

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| GHI valid % | 91.9 | 97.2 | 93.8 | 93.0 | 98.5 | 97.6 | 96.1 | 92.2 | 93.2 |

→ **All nine years are usable for GHI forecasting.** (See
`04_results/figures/p1_validity_heatmap.png` for the full variable × year matrix.)

---

## 4. Weather regimes (daytime)

Data-driven thresholds on the calibrated clear-sky index + 1-hour variability:

| Regime | Definition | % of daytime |
|---|---|---|
| Clear | kt ≥ 0.85, low variability | **51.8%** |
| Transitional | high variability, or 0.5 ≤ kt < 0.85 | **36.4%** |
| Cloudy/overcast | kt < 0.5, stable | **6.8%** |
| (daytime, GHI missing) | — | 5.0% |

Daily classes: 254 clear, 2,736 mixed, 206 overcast days (of 3,196). The desert
climate shows clear conditions about half the time, with substantial transitional
(ramp) periods — exactly the conditions that make ultra-short-term forecasting and
its uncertainty operationally important.

---

## 5. Outputs

**Data:** `02_data/cleaned/yulara_clean_5min.parquet`,
`02_data/cleaned/yulara_quality_flags.parquet`,
`02_data/regime_labels/yulara_regimes_5min.parquet`

**Figures (`04_results/figures/`):** validity_heatmap, ghi_diurnal_by_season,
regime_distribution, kt_distribution, example_days, ghi_vs_pv

**Tables (`04_results/tables/`):** cleaning_summary, ghi_completeness_by_year,
validity_by_year, regime_distribution
**Metrics (`04_results/metrics/`):** cleaning_summary, clearsky_alignment, eda_summary (JSON)

---

## 6. Implications for modelling

- **GHI forecasting (Phases 2–3): use all 9 years** with `Pyranometer_1`. The
  univariate-GHI path (also best in the base paper) is well supported.
- **Multivariate inputs:** auxiliary meteorology is only fully reliable in
  2023–2024; use validity-aware feature handling or restrict multivariate
  experiments to 2023–2024. Wind speed/direction and module-temp-2 are broader.
- **Decision layer (Phase 4):** prefer 2023–2024 for the GHI→PV mapping, or model
  the apparent export cap; flagged as Q5.
- **Splits:** chronological train / calibration (for conformal) / test, preserving
  order; candidate: train 2016–2022, calibrate 2023, test 2024 — to be finalised in
  Phase 2 with the multivariate-availability caveat in mind.

---

## 7. Next steps (Phase 2)
Build forecasting baselines (persistence, smart persistence, ARIMA, gradient
boosting, compact GRU/GRU-TCN) with point metrics and Diebold-Mariano/ANOVA
significance, stratified by horizon and regime.
