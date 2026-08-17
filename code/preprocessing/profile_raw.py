"""
profile_raw.py — Deep profiling of the raw Yulara CSVs (Phase 1, evidence-gathering).
Reads 02_data/Original Dataset/*.csv (READ-ONLY) and writes a profile report
(JSON + Markdown) to 04_results/metrics/ and 04_results/. No data is modified.

Run:  python3 profile_raw.py
"""
import os, json, glob, datetime
import numpy as np
import pandas as pd

BASE = os.environ.get("GHI_BASE", "/sessions/nice-hopeful-fermi/mnt/GHI Forecasting")
RAW = os.path.join(BASE, "02_data", "Original Dataset")
OUT_METRICS = os.path.join(BASE, "04_results", "metrics")
os.makedirs(OUT_METRICS, exist_ok=True)

GHI = "3052_Environment_DG_Weather_Station_Global_Horizontal_Radiation"
PYR = "3052_Environment_DG_Weather_Station_Pyranometer_1"
TEMP = "3052_Environment_DG_Weather_Station_Weather_Temperature_Celsius"
WIND = "3052_Environment_DG_Weather_Station_Wind_Speed"
PRES = "3052_Environment_DG_Weather_Station_Air_Pressure"
PVP = "3050_Total_Site_PV_Generation_Active_Power"
KEY = {"GHI": GHI, "Pyranometer": PYR, "AirTemp": TEMP, "WindSpeed": WIND,
       "AirPressure": PRES, "PV_power": PVP}
SENTINEL = 99999.9

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def profile_file(path):
    df = pd.read_csv(path)
    n = len(df)
    ts = parse_ts(df["timestamp"])
    rep = {"file": os.path.basename(path), "rows": int(n),
           "n_columns": int(df.shape[1])}
    # timestamps
    rep["ts_unparseable"] = int(ts.isna().sum())
    valid = ts.dropna().sort_values()
    rep["ts_min"] = str(valid.min()); rep["ts_max"] = str(valid.max())
    rep["ts_duplicates"] = int(valid.duplicated().sum())
    # spacing
    diffs = valid.diff().dropna().dt.total_seconds()
    rep["dt_median_s"] = float(diffs.median()) if len(diffs) else None
    rep["dt_eq_300_pct"] = float((diffs == 300).mean()*100) if len(diffs) else None
    # gap analysis on a regular grid
    full = pd.date_range(valid.min(), valid.max(), freq="5min")
    present = pd.Index(valid.unique())
    missing_slots = len(full) - len(present.intersection(full))
    rep["grid_expected_rows"] = int(len(full))
    rep["grid_missing_slots"] = int(missing_slots)
    rep["grid_missing_pct"] = float(100*missing_slots/len(full)) if len(full) else None
    # largest contiguous gap (in steps)
    if len(diffs):
        max_gap_steps = int(round(diffs.max()/300))
        rep["max_gap_steps"] = max_gap_steps
        rep["max_gap_hours"] = round(diffs.max()/3600, 2)
    # per key column stats (treat sentinel as NaN)
    cols = {}
    for name, col in KEY.items():
        if col not in df.columns:
            cols[name] = {"present": False}; continue
        s = pd.to_numeric(df[col], errors="coerce")
        sent = int((s == SENTINEL).sum())
        s2 = s.replace(SENTINEL, np.nan)
        cols[name] = {
            "present": True,
            "nan_raw_pct": round(float(s.isna().mean()*100), 3),
            "sentinel_count": sent,
            "nan_after_sentinel_pct": round(float(s2.isna().mean()*100), 3),
            "min": None if s2.dropna().empty else round(float(s2.min()), 2),
            "max": None if s2.dropna().empty else round(float(s2.max()), 2),
            "mean": None if s2.dropna().empty else round(float(s2.mean()), 2),
            "neg_count": int((s2 < 0).sum()),
            "neg_lt_-5_count": int((s2 < -5).sum()),
        }
    rep["columns"] = cols
    return rep

def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    reports = [profile_file(f) for f in files]
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "raw_dir": RAW, "n_files": len(files), "files": reports}
    with open(os.path.join(OUT_METRICS, "2026-06-22-raw-profile.json"), "w") as f:
        json.dump(out, f, indent=2)

    # console summary
    print(f"{'file':<18}{'rows':>8}{'dt=300%':>9}{'gridMiss%':>10}{'maxGap_h':>9}"
          f"{'GHI_nan%':>9}{'GHI_sent':>9}{'GHI_neg':>8}")
    for r in reports:
        g = r["columns"]["GHI"]
        print(f"{r['file']:<18}{r['rows']:>8}{r['dt_eq_300_pct']:>9.1f}"
              f"{r['grid_missing_pct']:>10.2f}{r['max_gap_hours']:>9.1f}"
              f"{g['nan_after_sentinel_pct']:>9.2f}{g['sentinel_count']:>9}{g['neg_count']:>8}")
    # key column presence + global ranges
    print("\nKey columns (after sentinel removal) — min / max across years:")
    for name in KEY:
        mins = [r['columns'][name]['min'] for r in reports if r['columns'][name].get('present') and r['columns'][name]['min'] is not None]
        maxs = [r['columns'][name]['max'] for r in reports if r['columns'][name].get('present') and r['columns'][name]['max'] is not None]
        if mins:
            print(f"  {name:<14} min={min(mins):>10.2f}  max={max(maxs):>10.2f}")
    print(f"\nSaved JSON -> {os.path.join(OUT_METRICS, '2026-06-22-raw-profile.json')}")

if __name__ == "__main__":
    main()
