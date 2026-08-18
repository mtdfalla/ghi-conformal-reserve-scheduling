"""Verify the run-length guard in `p2_clean.py`.

This does two independent things, and neither takes anything on assertion:

  A. REPRODUCE THE DEFECT from the shipped artifacts. The original pre-interpolation
     NaN pattern of GHI is recoverable exactly, because
     `data/cleaned/yulara_clean_5min.parquet` holds the post-fill series and
     `data/cleaned/yulara_quality_flags.parquet` holds the per-cell imputation
     flag: original_NaN = imputed | still_NaN. Run lengths are then computed on
     that reconstructed mask, and we count how many imputed cells lie inside runs
     longer than SHORT_GAP_STEPS. The expected answer, recorded when the defect was
     found, is 4,014 of 13,925 (28.8 %) with a longest run of 3,025 steps.

  B. PROVE THE GUARD IS CORRECT on constructed series where the answer is known by
     construction, including the exact failure mode (a long gap whose leading
     steps the old code fabricates) and the boundary cases at the limit.

Writes results/metrics/r1_s10_runlength_guard.json. Reads only; overwrites nothing.

Usage:  python3 code/r1/r1_s10_verify_runlength_guard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# config.py detects the directory layout (02_data/04_results in the authors' tree,
# data/results in a release checkout) and must be used rather than hard-coded paths --
# hard-coding them is exactly the defect that once made a clean checkout write to the
# wrong directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C  # noqa: E402

CLEAN = C.DATA_CLEAN / "yulara_clean_5min.parquet"
FLAGS = C.DATA_CLEAN / "yulara_quality_flags.parquet"
OUT = C.MET / "r1_s10_runlength_guard.json"

SHORT_GAP_STEPS = int(getattr(C, "SHORT_GAP_STEPS", 6))


def run_lengths(isnan: pd.Series) -> pd.Series:
    """Length of each NaN run, broadcast back onto every cell in that run; 0 elsewhere."""
    runid = (isnan != isnan.shift()).cumsum()
    return isnan.groupby(runid).transform("sum").where(isnan, 0)


def old_fill(s: pd.Series, limit: int) -> pd.Series:
    """The pre-guard behaviour, verbatim."""
    return s.interpolate(method="time", limit=limit, limit_area="inside")


def new_fill(s: pd.Series, limit: int) -> pd.Series:
    """The guarded behaviour, verbatim from the patched p2_clean.py."""
    isnan = s.isna()
    short_gap = isnan & (run_lengths(isnan) <= limit)
    return s.where(~short_gap, old_fill(s, limit))


# ----------------------------------------------------------------- B. unit proofs
def unit_proofs():
    idx = pd.date_range("2020-01-01", periods=40, freq="5min")
    checks = []

    def case(name, nan_slice, expect_old_filled, expect_new_filled):
        s = pd.Series(np.arange(40, dtype=float), index=idx)
        s.iloc[nan_slice] = np.nan
        o, n = old_fill(s, SHORT_GAP_STEPS), new_fill(s, SHORT_GAP_STEPS)
        got_old = int((s.isna() & o.notna()).sum())
        got_new = int((s.isna() & n.notna()).sum())
        ok = (got_old == expect_old_filled) and (got_new == expect_new_filled)
        checks.append(dict(case=name, old_filled=got_old, new_filled=got_new,
                           expected_old=expect_old_filled,
                           expected_new=expect_new_filled, pass_=bool(ok)))
        return ok

    # a 20-step interior gap: the old code fabricates its first 6 steps, the guard none
    case("interior gap of 20 (the defect)", slice(10, 30), 6, 0)
    # a gap of exactly the limit: both fill all 6 -- the guard must NOT be stricter
    case("interior gap of 6 (at the limit)", slice(10, 16), 6, 6)
    # a gap one over the limit: old fills 6, guard fills none
    case("interior gap of 7 (one over)", slice(10, 17), 6, 0)
    # a single missing step
    case("interior gap of 1", slice(10, 11), 1, 1)
    # leading gap: limit_area='inside' fills neither, guard must not change that
    case("leading gap of 3 (no left anchor)", slice(0, 3), 0, 0)
    # trailing gap: same
    case("trailing gap of 3 (no right anchor)", slice(37, 40), 0, 0)

    # two gaps in one series: one short, one long. Only the short one may be filled.
    s = pd.Series(np.arange(40, dtype=float), index=idx)
    s.iloc[5:8] = np.nan       # short, 3 steps
    s.iloc[20:35] = np.nan     # long, 15 steps
    n = new_fill(s, SHORT_GAP_STEPS)
    ok = (n.iloc[5:8].notna().all() and n.iloc[20:35].isna().all())
    checks.append(dict(case="short and long gap in one series", old_filled=None,
                       new_filled=int((s.isna() & n.notna()).sum()),
                       expected_old=None, expected_new=3, pass_=bool(ok)))

    # the guard may only ever REMOVE fills, never add or alter a value
    rng = np.random.default_rng(20260818)
    monotone_ok, subset_ok = True, True
    for _ in range(200):
        v = pd.Series(rng.normal(size=200).cumsum(), index=pd.date_range(
            "2020-01-01", periods=200, freq="5min"))
        mask = rng.random(200) < 0.35
        v[mask] = np.nan
        o, n = old_fill(v, SHORT_GAP_STEPS), new_fill(v, SHORT_GAP_STEPS)
        of, nf = v.isna() & o.notna(), v.isna() & n.notna()
        if not (nf & ~of).sum() == 0:
            subset_ok = False                       # new fills must be a subset of old
        if not np.allclose(o[nf], n[nf], equal_nan=True):
            monotone_ok = False                     # where both fill, values must agree
        if not v.notna().equals(n.notna() & v.notna()):
            pass
        if not np.allclose(v.dropna().to_numpy(), n[v.notna()].to_numpy()):
            monotone_ok = False                     # observed values untouched
    checks.append(dict(case="200 random series: new fills are a subset of old",
                       old_filled=None, new_filled=None, expected_old=None,
                       expected_new=None, pass_=bool(subset_ok)))
    checks.append(dict(case="200 random series: shared fills and observations identical",
                       old_filled=None, new_filled=None, expected_old=None,
                       expected_new=None, pass_=bool(monotone_ok)))
    return checks


# ------------------------------------------------- A. reproduce against real data
def reproduce():
    if not (CLEAN.exists() and FLAGS.exists()):
        return {"available": False,
                "note": f"missing {CLEAN.name} or {FLAGS.name}; run p2_clean.py first"}
    clean = pd.read_parquet(CLEAN, columns=["ghi"])
    flags = pd.read_parquet(FLAGS, columns=["ghi_imputed"])
    df = clean.join(flags, how="left")
    imputed = df["ghi_imputed"].fillna(False).astype(bool)
    still_nan = df["ghi"].isna()
    original_nan = imputed | still_nan
    rl = run_lengths(original_nan)

    n_imputed = int(imputed.sum())
    inside_long = int((imputed & (rl > SHORT_GAP_STEPS)).sum())
    longest = int(rl.max())
    return {
        "available": True,
        "rows": int(len(df)),
        "imputed_ghi_cells": n_imputed,
        "imputed_cells_inside_runs_longer_than_limit": inside_long,
        "pct_of_imputed": round(inside_long / n_imputed * 100, 2) if n_imputed else None,
        "longest_original_nan_run_steps": longest,
        "short_gap_steps": SHORT_GAP_STEPS,
        "expected": {"inside_long": 4014, "n_imputed": 13925,
                     "pct": 28.8, "longest": 3025},
    }


def main():
    proofs = unit_proofs()
    repro = reproduce()

    print("=== B. guard correctness, on constructed series ===")
    for c in proofs:
        print(f"  [{'PASS' if c['pass_'] else 'FAIL'}] {c['case']}"
              + (f"  old_filled={c['old_filled']} new_filled={c['new_filled']}"
                 if c["old_filled"] is not None else ""))
    all_pass = all(c["pass_"] for c in proofs)

    print("\n=== A. the defect, reproduced from the shipped artifacts ===")
    match = None
    if repro["available"]:
        t = repro["expected"]
        print(f"  imputed GHI cells                  {repro['imputed_ghi_cells']:>8,}"
              f"   (expected: {t['n_imputed']:,})")
        print(f"  of those, inside runs > {SHORT_GAP_STEPS} steps    "
              f"{repro['imputed_cells_inside_runs_longer_than_limit']:>8,}"
              f"   (expected: {t['inside_long']:,})")
        print(f"  as a percentage                    {repro['pct_of_imputed']:>8}"
              f" %  (expected: {t['pct']} %)")
        print(f"  longest original NaN run           "
              f"{repro['longest_original_nan_run_steps']:>8,}"
              f"   (expected: {t['longest']:,})")
        match = (repro["imputed_ghi_cells"] == t["n_imputed"]
                 and repro["imputed_cells_inside_runs_longer_than_limit"] == t["inside_long"]
                 and repro["longest_original_nan_run_steps"] == t["longest"])
        print(f"  -> expected values {'CONFIRMED' if match else 'DO NOT MATCH'}")
    else:
        print("  " + repro["note"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"unit_proofs": proofs, "reproduction": repro,
                   "all_unit_proofs_pass": all_pass,
                   "expected_values_confirmed": match}, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0 if (all_pass and (match is not False)) else 1


if __name__ == "__main__":
    sys.exit(main())
