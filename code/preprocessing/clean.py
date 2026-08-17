"""
clean.py — Cleaning pipeline for the Yulara dataset (Phase 1).

Reads raw CSVs from 02_data/Original Dataset/ (READ-ONLY) and writes cleaned,
grid-aligned per-year + combined parquet files to 02_data/cleaned/, plus a
data-quality manifest (JSON + Markdown) to 04_results/metrics/.

Evidence-based rules (from raw profiling, 2026-06-22):
  * Timestamps are reindexed to a regular 5-minute grid (per calendar year).
  * Implausible values -> NaN using physical plausibility ranges (handles the
    counter-like sensor/logger corruption found in Temp/Pressure/Wind in
    2016-2022, and GHI/Pyranometer high spikes > 1600 W/m^2).
  * GHI/PV small negatives (night sensor offset) are clipped to 0 in *_clean.
  * Short gaps (<= MAX_INTERP_STEPS) in GHI are linearly interpolated and flagged;
    long gaps are left as NaN (never fabricated).
  * Nothing in the raw folder is modified.

Run:  python3 clean.py
"""
import os, json, glob, datetime
import numpy as np
import pandas as pd

BASE = os.environ.get("GHI_BASE", "/sessions/nice-hopeful-fermi/mnt/GHI Forecasting")
RAW = os.path.join(BASE, "02_data", "Original Dataset")
OUT = os.path.join(BASE, "02_data", "cleaned")
OUT_METRICS = os.path.join(BASE, "04_results", "metrics")
os.makedirs(OUT, exist_ok=True); os.makedirs(OUT_METRICS, exist_ok=True)

# raw column -> clean name
COLMAP = {
    "3052_Environment_DG_Weather_Station_Global_Horizontal_Radiation": "GHI",
    "3052_Environment_DG_Weather_Station_Pyranometer_1": "Pyranometer",
    "3052_Environment_DG_Weather_Station_Weather_Temperature_Celsius": "AirTemp",
    "3052_Environment_DG_Weather_Station_Wind_Speed": "WindSpeed",
    "3052_Environment_DG_Weather_Station_Air_Pressure": "AirPressure",
    "3050_Total_Site_PV_Generation_Active_Power": "PV_power",
}
# physical plausibility ranges (outside -> NaN)
RANGES = {
    "GHI": (-50, 1600), "Pyranometer": (-50, 1600), "AirTemp": (-20, 60),
    "WindSpeed": (0, 70), "AirPressure": (850, 1050), "PV_power": (-5, 2000),
}
CLIP_NEG_ZERO = ["GHI", "Pyranometer", "PV_power"]  # clip small negatives to 0 in *_clean
MAX_INTERP_STEPS = 6  # interpolate GHI gaps up to 30 min; longer left as NaN
FREQ = "5min"

def clean_year(path):
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
    raw = raw.drop_duplicates(subset="_ts", keep="first").set_index("_ts")

    year = int(pd.to_datetime(raw.index.min()).year)
    grid = pd.date_range(raw.index.min().floor("5min"),
                         raw.index.max().ceil("5min"), freq=FREQ)
    out = pd.DataFrame(index=grid); out.index.name = "timestamp"

    qc = {"year": year, "grid_rows": int(len(grid)), "columns": {}}
    for rawcol, name in COLMAP.items():
        if rawcol not in raw.columns:
            qc["columns"][name] = {"present": False}; continue
        s = pd.to_numeric(raw[rawcol], errors="coerce").reindex(grid)
        lo, hi = RANGES[name]
        n_present = int(s.notna().sum())
        implausible = ((s < lo) | (s > hi))
        n_implausible = int(implausible.sum())
        s = s.mask(implausible, np.nan)
        if name in CLIP_NEG_ZERO:
            s = s.clip(lower=0)
        valid = int(s.notna().sum())
        out[name] = s
        qc["columns"][name] = {
            "present": True,
            "valid_rows": valid,
            "valid_pct": round(100*valid/len(grid), 2),
            "implausible_to_nan": n_implausible,
            "raw_present_rows": n_present,
        }

    # GHI gap interpolation (short only) + provenance flag
    ghi = out["GHI"].copy()
    isnan = ghi.isna()
    # identify run lengths of NaN
    grp = (isnan != isnan.shift()).cumsum()
    runlen = isnan.groupby(grp).transform("sum").where(isnan, 0)
    short_gap = isnan & (runlen <= MAX_INTERP_STEPS)
    ghi_interp = ghi.interpolate(method="time", limit=MAX_INTERP_STEPS, limit_area="inside")
    out["GHI_filled"] = ghi.where(~short_gap, ghi_interp)
    out["GHI_was_interpolated"] = short_gap.astype(int)
    qc["ghi_short_gaps_interpolated"] = int(short_gap.sum())
    qc["ghi_long_gap_nan_remaining"] = int(out["GHI_filled"].isna().sum())

    return year, out.reset_index(), qc

def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    manifest = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                "rule_set": {"ranges": RANGES, "max_interp_steps": MAX_INTERP_STEPS,
                             "clip_neg_zero": CLIP_NEG_ZERO}, "years": []}
    allparts = []
    for f in files:
        year, df, qc = clean_year(f)
        df.to_parquet(os.path.join(OUT, f"Yulara_{year}_clean.parquet"), index=False)
        manifest["years"].append(qc)
        allparts.append(df.assign(year=year))
        print(f"{year}: grid={qc['grid_rows']:>7}  GHIvalid%={qc['columns']['GHI']['valid_pct']:>6}"
              f"  Tvalid%={qc['columns']['AirTemp']['valid_pct']:>6}"
              f"  Pvalid%={qc['columns']['AirPressure']['valid_pct']:>6}"
              f"  Wvalid%={qc['columns']['WindSpeed']['valid_pct']:>6}"
              f"  PVvalid%={qc['columns']['PV_power']['valid_pct']:>6}"
              f"  shortGapsFilled={qc['ghi_short_gaps_interpolated']:>5}")
    allc = pd.concat(allparts, ignore_index=True)
    allc.to_parquet(os.path.join(OUT, "yulara_all_clean.parquet"), index=False)
    with open(os.path.join(OUT_METRICS, "2026-06-22-clean-manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCombined: {len(allc):,} rows -> {os.path.join(OUT,'yulara_all_clean.parquet')}")
    print(f"Manifest -> {os.path.join(OUT_METRICS,'2026-06-22-clean-manifest.json')}")

if __name__ == "__main__":
    main()
