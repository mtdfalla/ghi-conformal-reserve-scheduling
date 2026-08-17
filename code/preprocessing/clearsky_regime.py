"""
clearsky_regime.py — Clear-sky modelling + weather-regime labels (Phase 1).

For each cleaned year:
  * localize timestamps to Australia/Darwin (UTC+9:30, no DST) -- verified
    empirically (GHI peak aligns with solar noon to ~15 min).
  * compute clear-sky GHI with pvlib (Ineichen + climatological Linke turbidity).
  * compute clear-sky index kt = GHI / GHI_cs over daytime.
  * compute a short-term variability index (rolling std of kt, ~1 h centered).
  * label each daytime step as: clear / cloudy / transitional.

Outputs per-year + combined parquet to 02_data/regime_labels/ and a summary
(JSON + console) to 04_results/metrics/.

Regime rule (configurable):
  clear        : kt >= KT_CLEAR  and var <  VAR_STABLE
  cloudy        : kt <= KT_CLOUDY and var <  VAR_STABLE   (stable overcast)
  transitional : otherwise (broken cloud / volatile) -- the hard-to-forecast regime
Night (clear-sky GHI <= DAY_MIN_CS) is labelled 'night' and excluded from modelling.

Run:  python3 clearsky_regime.py
"""
import os, json, glob, datetime
import numpy as np
import pandas as pd
import pvlib

BASE = os.environ.get("GHI_BASE", "/sessions/nice-hopeful-fermi/mnt/GHI Forecasting")
CLEAN = os.path.join(BASE, "02_data", "cleaned")
OUT = os.path.join(BASE, "02_data", "regime_labels")
OUT_METRICS = os.path.join(BASE, "04_results", "metrics")
os.makedirs(OUT, exist_ok=True); os.makedirs(OUT_METRICS, exist_ok=True)

LAT, LON, ALT, TZ = -25.2406, 130.9889, 492, "Australia/Darwin"
DAY_MIN_CS = 20.0       # W/m^2: clear-sky GHI above this = daytime
KT_CLIP = 1.3
KT_CLEAR = 0.85
KT_CLOUDY = 0.50
VAR_STABLE = 0.08
VAR_WIN = 13            # ~1 h centered (13 x 5min)

def label_year(path):
    df = pd.read_parquet(path)[["timestamp", "GHI", "GHI_filled"]].copy()
    t = pd.DatetimeIndex(df["timestamp"])
    t_local = t.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    keep = ~t_local.isna()
    df, t_local = df[keep].reset_index(drop=True), t_local[keep]

    loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    cs = loc.get_clearsky(t_local, model="ineichen")          # GHI/DNI/DHI
    ghi_cs = cs["ghi"].to_numpy()

    ghi = df["GHI_filled"].fillna(df["GHI"]).to_numpy()
    is_day = ghi_cs > DAY_MIN_CS
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = np.where(is_day, ghi / ghi_cs, np.nan)
    kt = np.clip(kt, 0, KT_CLIP)

    kt_s = pd.Series(kt)
    var = kt_s.rolling(VAR_WIN, center=True, min_periods=5).std().to_numpy()

    regime = np.full(len(df), "night", dtype=object)
    day = is_day & ~np.isnan(kt)
    clear = day & (kt >= KT_CLEAR) & (var < VAR_STABLE)
    cloudy = day & (kt <= KT_CLOUDY) & (var < VAR_STABLE)
    trans = day & ~clear & ~cloudy
    regime[clear] = "clear"; regime[cloudy] = "cloudy"; regime[trans] = "transitional"

    res = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "GHI": df["GHI"].values,
        "GHI_clearsky": ghi_cs,
        "kt": kt,
        "kt_variability": var,
        "is_day": is_day,
        "regime": regime,
    })
    year = int(pd.to_datetime(df["timestamp"].iloc[0]).year)
    counts = res.loc[res["is_day"], "regime"].value_counts().to_dict()
    return year, res, counts

def main():
    files = sorted(glob.glob(os.path.join(CLEAN, "Yulara_*_clean.parquet")))
    summary = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "location": {"lat": LAT, "lon": LON, "alt_m": ALT, "tz": TZ},
               "rule": {"kt_clear": KT_CLEAR, "kt_cloudy": KT_CLOUDY,
                        "var_stable": VAR_STABLE, "var_win": VAR_WIN,
                        "day_min_clearsky": DAY_MIN_CS}, "years": []}
    parts = []
    print(f"{'year':<6}{'day_steps':>10}{'clear%':>9}{'transit%':>10}{'cloudy%':>9}{'kt_med':>8}")
    for f in files:
        year, res, counts = label_year(f)
        res.to_parquet(os.path.join(OUT, f"Yulara_{year}_regime.parquet"), index=False)
        parts.append(res.assign(year=year))
        day = res[res["is_day"]]
        nd = len(day)
        c = lambda k: 100*counts.get(k, 0)/nd if nd else 0
        summary["years"].append({"year": year, "day_steps": int(nd),
            "clear_pct": round(c("clear"), 2), "transitional_pct": round(c("transitional"), 2),
            "cloudy_pct": round(c("cloudy"), 2), "kt_median": round(float(day["kt"].median()), 3)})
        print(f"{year:<6}{nd:>10}{c('clear'):>9.1f}{c('transitional'):>10.1f}"
              f"{c('cloudy'):>9.1f}{day['kt'].median():>8.3f}")
    allr = pd.concat(parts, ignore_index=True)
    allr.to_parquet(os.path.join(OUT, "yulara_all_regime.parquet"), index=False)
    with open(os.path.join(OUT_METRICS, "2026-06-22-regime-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    # overall
    day = allr[allr["is_day"]]
    tot = day["regime"].value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"\nOVERALL daytime regime mix: {tot}")
    print(f"Saved -> {OUT}  and summary -> 2026-06-22-regime-summary.json")

if __name__ == "__main__":
    main()
