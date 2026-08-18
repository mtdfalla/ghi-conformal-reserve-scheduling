"""Phase 1 - Step 2: Clean & consolidate the raw Yulara data.

Pipeline
--------
1. Load all 9 yearly CSVs, keep key columns only.
2. Parse timestamps; drop unparseable rows.
3. Concatenate; drop duplicate timestamps (yearly files overlap by ~1 day); sort.
4. Reindex onto a continuous regular 5-min grid (min..max).
5. Replace sentinels (|x|>=9e4) and out-of-physical-bounds values with NaN.
6. Record raw missingness; interpolate SHORT gaps (<=6 steps / 30 min);
   leave LONG gaps as NaN and flag them. "Short" is measured on the FULL run
   length of each NaN run, so no part of a long gap is ever filled (see the
   run-length guard at step 6 below, added 2026-08-18).
7. Clip small negative GHI/PV (night offset) to 0.
8. Save cleaned dataset (parquet) + per-variable imputation flags + quality report.

Outputs
-------
02_data/cleaned/yulara_clean_5min.parquet
02_data/cleaned/yulara_quality_flags.parquet
results/tables/p1_cleaning_summary.csv
results/metrics/p1_cleaning_summary.json
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd

def log(m): print(f"[clean] {m}", flush=True)

# ---- 1-3. Load, parse, concat, dedupe ----
keep_raw = [C.TS] + list(C.COLS.values())
ren = {v: k for k, v in C.COLS.items()}
frames = []
for f in sorted(C.DATA_RAW.glob("Yulara_*.csv")):
    df = pd.read_csv(f, usecols=lambda c: c in keep_raw)
    df = df.rename(columns=ren)
    df["ts"] = pd.to_datetime(df[C.TS], errors="coerce")
    bad = df["ts"].isna().sum()
    if bad: log(f"{f.name}: dropped {bad} unparseable timestamps")
    df = df.dropna(subset=["ts"]).drop(columns=[C.TS])
    frames.append(df)
    log(f"loaded {f.name}: {len(df)} rows")

if not frames:
    raise SystemExit(
        "[clean] no input files found.\n"
        "        Looked for:  %s\n"
        "        The raw data are third-party and are not redistributed with this code.\n"
        "        Download them and place the per-year CSVs DIRECTLY in that directory,\n"
        "        named Yulara_<year>.csv (no nested subdirectory). See data/README.md."
        % (C.DATA_RAW / "Yulara_*.csv"))

raw = pd.concat(frames, ignore_index=True)
dup = raw.duplicated(subset="ts").sum()
raw = raw.sort_values("ts").drop_duplicates(subset="ts", keep="first").set_index("ts")
log(f"combined {len(raw)} unique rows (removed {dup} duplicate timestamps from file overlaps)")

# ---- 4. Regular 5-min grid ----
full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq=C.FREQ)
grid = raw.reindex(full_idx)
grid.index.name = "ts"
n_inserted = len(grid) - len(raw)
log(f"reindexed to regular grid: {len(grid)} rows ({n_inserted} missing slots inserted)")

cols = list(C.COLS.keys())

# ---- 5. Sentinels + physical bounds -> NaN ----
sent_counts, bound_counts = {}, {}
for col in cols:
    s = grid[col]
    sent = (s.abs() >= C.SENTINEL_ABS)
    sent_counts[col] = int(sent.sum())
    s = s.mask(sent)
    lo, hi = C.BOUNDS[col]
    oob = pd.Series(False, index=s.index)
    if lo is not None: oob |= (s < lo)
    if hi is not None: oob |= (s > hi)
    bound_counts[col] = int(oob.sum())
    grid[col] = s.mask(oob)

# ---- 6. Missingness before interpolation ----
miss_before = grid[cols].isna().sum()

# Interpolate SHORT gaps only (time-based), flag what was filled.
#
# RUN-LENGTH GUARD (added 2026-08-18).
# ---------------------------------------------------------------------------
# `Series.interpolate(limit=n, limit_area="inside")` does NOT do what the
# docstring above describes. `limit` caps the number of CONSECUTIVE values
# filled going forward, so it fills the first n steps of a gap of ANY length
# and leaves the remainder NaN. A 3,025-step outage therefore had its first
# six steps fabricated from the values either side of a ten-day hole.
#
# Measured on the shipped Yulara data before this guard existed: 4,014 of the
# 13,925 imputed GHI cells (28.8 %) lay inside original NaN runs longer than
# SHORT_GAP_STEPS. Downstream, `utils/datasets.py:make_xy` excludes imputed
# TARGETS but not imputed issue-time FEATURES, so those cells reached the
# supervised frame as inputs.
#
# The guard below computes each NaN run's length first and fills a cell only
# when its ENTIRE run is <= SHORT_GAP_STEPS, which is the documented rule and
# is the same idiom `code/preprocessing/clean.py` has always used.
#
# PROVENANCE, STATED SO IT CANNOT BE MISREAD: every result shipped with the
# accompanying article was produced by the pre-guard behaviour. Re-running this
# script now yields 9,911 imputed GHI cells where the archived quality flags
# record 13,925: exactly the 4,014 long-run cells above are withheld. That
# difference is verified by `code/r1/r1_s10_verify_runlength_guard.py`, which
# reconstructs the original NaN pattern from the shipped artifacts and also
# proves the guard on constructed series including the boundary at the limit.
# The size of the difference to the RESULTS was measured rather than assumed --
# see `code/r1/r1_s9_causal_rescore.py`, which re-scores the shipped test
# predictions on the subset of rows carrying no interpolated issue-time input:
# the largest change in any per-regime coverage figure is +0.0142 and point
# RMSE moves by -1.65 % to +0.30 %. No conclusion in the article depends on it.
flags = pd.DataFrame(index=grid.index)
withheld_counts = {}
for col in cols:
    s = grid[col]
    isnan = s.isna()
    # run length of each NaN run, broadcast back onto every cell in that run
    runid = (isnan != isnan.shift()).cumsum()
    runlen = isnan.groupby(runid).transform("sum").where(isnan, 0)
    short_gap = isnan & (runlen <= C.SHORT_GAP_STEPS)
    interp = s.interpolate(method="time", limit=C.SHORT_GAP_STEPS,
                           limit_area="inside")
    filled = s.where(~short_gap, interp)
    flags[col + "_imputed"] = isnan & filled.notna()
    # cells the old behaviour would have fabricated and this guard withholds
    withheld_counts[col] = int((isnan & interp.notna() & ~short_gap).sum())
    grid[col] = filled
miss_after = grid[cols].isna().sum()
imputed_counts = {c: int(flags[c + "_imputed"].sum()) for c in cols}
log("run-length guard: withheld "
    + ", ".join(f"{c}={withheld_counts[c]}" for c in cols)
    + f" cells that lay inside NaN runs longer than {C.SHORT_GAP_STEPS} steps")

# ---- 7. Clip night negatives for irradiance/power ----
for col in ["ghi", "ghi_alt", "pv_total"]:
    neg = (grid[col] < 0)
    grid.loc[neg, col] = 0.0
log("clipped negative GHI/PV (night offset) to 0")

# ---- 8. Save ----
clean_path = C.DATA_CLEAN / "yulara_clean_5min.parquet"
flags_path = C.DATA_CLEAN / "yulara_quality_flags.parquet"
grid.to_parquet(clean_path)
flags.to_parquet(flags_path)
log(f"saved cleaned data -> {clean_path.name}")

# ---- Quality report (overall + per year) ----
n = len(grid)
overall = {
    "rows_total_grid": n,
    "span_start": str(grid.index.min()),
    "span_end": str(grid.index.max()),
    "duplicate_ts_removed": int(dup),
    "missing_slots_inserted": int(n_inserted),
}
per_var = []
for col in cols:
    per_var.append(dict(
        variable=col,
        sentinels=sent_counts[col],
        out_of_bounds=bound_counts[col],
        missing_before_interp=int(miss_before[col]),
        pct_missing_before=round(miss_before[col] / n * 100, 3),
        interpolated=imputed_counts[col],
        withheld_by_runlength_guard=withheld_counts[col],
        missing_after_interp=int(miss_after[col]),
        pct_missing_after=round(miss_after[col] / n * 100, 3),
    ))
per_var_df = pd.DataFrame(per_var)
per_var_df.to_csv(C.TAB / "p1_cleaning_summary.csv", index=False)

# Per-year GHI completeness (post-clean, daytime proxy not yet available)
gy = grid["ghi"].notna().groupby(grid.index.year).mean().mul(100).round(2)
year_tbl = gy.rename("ghi_complete_pct_after").to_frame()
year_tbl.to_csv(C.TAB / "p1_ghi_completeness_by_year.csv")

with open(C.MET / "p1_cleaning_summary.json", "w") as fh:
    json.dump({"overall": overall, "per_variable": per_var,
               "ghi_complete_pct_by_year": gy.to_dict()}, fh, indent=2)

log("=== SUMMARY ===")
print(per_var_df.to_string(index=False))
print("\nGHI completeness after clean, by year (%):")
print(year_tbl.to_string())
print("\noverall:", json.dumps(overall, indent=2))
log("done.")
