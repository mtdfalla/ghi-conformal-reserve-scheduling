# Data access

The raw solar-monitoring data are **third-party** (Desert Knowledge Australia Solar
Centre, DKASC) and are **not redistributed** in this repository. They are free and
publicly available; download them as below and place them where the scripts expect.

## 1. Yulara (primary site, 2016–2024, 5-min)
Source: DKASC Data Download, Yulara location — https://dkasolarcentre.com.au/download?location=yulara
Download the per-year "with weather data" CSV exports (irradiance + weather). The GHI
target is the pyranometer channel (`Pyranometer_1`); the channel literally named
"Global_Horizontal_Radiation" is faulty for 2016–2022 and must not be used as the target.
Place the raw CSVs **directly in `data/raw/`, named `Yulara_<year>.csv`** — that is the
exact glob `code/preprocessing/p2_clean.py` uses (`C.DATA_RAW.glob("Yulara_*.csv")`), so a
nested subdirectory or a different prefix will silently match nothing. Then run
`code/preprocessing/p2_clean.py` followed by `code/preprocessing/p3_clearsky_regimes.py`.

## 2. DKASC Alice Springs (external validation, 2020–2024, 5-min)
Source: https://dkasolarcentre.com.au/download?location=alice-springs
Use the **"All Individual Technologies (yearly data set with weather data)"** export and
download years 2020–2024 (these per-year CSVs include the weather-station channels,
including `101_DKA_WeatherStation_Global_Horizontal_Radiation`). Note: the page's
interactive "Specific Data Download" form may error; the yearly export works.
Place the CSVs as `data/DKASC/raw/Alice_Springs_{2020..2024}.csv` and run
`code/preprocessing/dkasc_prepare.py`.

## Where outputs go
`code/utils/config.py` detects which layout it is running in. In this release checkout it
resolves `data/raw`, `data/cleaned`, `data/regime_labels`, `data/interim`, `data/DKASC`
and writes every table, figure and metric under `results/`. Nothing needs configuring; if
you see a `04_results/` directory appear, you are running an older copy of `config.py`.

## Notes
- Timestamps are ACST (UTC+9:30) for both sites.
- The cleaning pipeline removes overlap duplicates, hardware sentinels, and out-of-bounds
  values, builds a regular 5-min grid, clips night to zero, and interpolates gaps ≤30 min.
  **"≤30 min" is measured on the full length of each missing run**, so no part of a longer
  gap is filled; `code/preprocessing/p2_clean.py` computes run lengths before interpolating.
  Every interpolated cell is flagged in `data/cleaned/yulara_quality_flags.parquet`.
- **What the flags are and are not used for, stated precisely.** A sample is dropped from
  training and evaluation when its **target** was interpolated (`code/utils/datasets.py`,
  `make_xy`). Interpolated values are **not** excluded from the **issue-time inputs**: a
  lagged or rolling feature may still read one. The article quantifies that exposure and
  reports what correcting for it does; `code/r1/r1_s9_causal_rescore.py` is the measurement.
  Earlier versions of this file said interpolated values were "excluded from training/eval"
  without that distinction, which was true of targets only.
- **The results shipped in `results/` predate the run-length guard** described above;
  `p2_clean.py --legacy-gapfill` reproduces that pre-guard behaviour bit-for-bit and is
  what `reproduce.sh` uses for article reproduction, while the guarded cleaner (no flag)
  is the default for new work. See
  the provenance note at the end of `MANIFEST.md`. See `code/preprocessing/` and the article
  Appendix for exact rules.
- Please cite DKASC when using the data, e.g.: *Desert Knowledge Australia Centre,
  "Download Data," Alice Springs / Yulara, https://dkasolarcentre.com.au/download,
  accessed dd/mm/yyyy.*
