"""Final pass - paired significance tests behind Table 2's "GBM beats the deep
models" claim, run against the POST-CAUSAL GBM only.

WHY THIS FILE EXISTS
--------------------
the first-pass table asserts GBM is significantly more accurate than the compact GRU and
GRU-TCN "at all horizons (p from 2.6e-38 to 1.2e-10)". Those tests were run against
PRE- GBM predictions (`02_data/interim/p2_test_pred_h*.parquet`). forbids
reusing them: comparing pre-causal GBM to post-causal deep predictions would
reintroduce exactly the inconsistency the final pass exists to remove.

This file therefore reads ONLY:
    04_results/tables/r1_p2_point_pred_h{h}.parquet        post-causal GBM ( a)
    04_results/tables/r1_p2_deep_pred_h{h}_full.parquet    run C (primary)
    04_results/tables/r1_p2_deep_pred_h{h}.parquet         run A (secondary)
and refuses to open the pre- interim files at all.

RUN C IS PRIMARY. Run A is the smaller-budget point on the same curve and is
reported alongside; quoting run A while run C sits on disk would be selective
reporting.

COMMON SUPPORT
--------------
The classical frame (`datasets.make_xy`) and the deep windower
(`deep_gru_tcn.build_windows`) do not admit exactly the same test instants:
make_xy needs kt at lags {0..6, 9, 12} plus 6-step rolling statistics and a non-null
base GHI, while build_windows needs 12 contiguous non-null kt. Each model's headline
Table 2 row is therefore computed on its own valid set (as the deep run did), but every
PAIRED test here runs on the INTERSECTION of timestamps, and the intersection size
is reported in every row so the reader can see what was compared.

TESTS (the same two the interval layer used in S3, for consistency)
    * Diebold-Mariano on the squared-error differential with the
      Harvey-Leybourne-Newbold small-sample correction, at
        (i)  HAC lag h-1   - the textbook convention, reported for continuity
        (ii) HAC lag = median daytime observations per day - far more conservative,
             and the one to quote
    * Paired day-block bootstrap, B = 10,000, resampling whole operating days, on
      the difference in RMSE. Assumes nothing about the autocorrelation structure.

OUTPUTS (r1_-prefixed, nothing overwritten)
    04_results/tables/r1_p2_deep_vs_gbm.csv      one row per (run, horizon, pair)
    04_results/tables/r1_p2_common_support.csv   RMSE/MAE/R2 of all three models on
                                                 the common index, per horizon+run
    04_results/metrics/r1_p2_deep_vs_gbm.json    settings + provenance

Run from the code directory (03_code/ in the working tree, code/ in a release checkout):  python3 r1/r1_p2_deep_vs_gbm.py
Requires r1_p2_point_causal.py to have run first.
"""
from __future__ import annotations

import json
import platform
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parents[1]   # 03_code/ in the working tree, code/ in a release checkout
sys.path.insert(0, str(CODE / "utils"))
sys.path.insert(0, str(CODE / "evaluation"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy import stats                  # noqa: E402
import config as CFG                     # noqa: E402
import metrics as M                      # noqa: E402

HORIZONS = [1, 3, 6, 12]
B_BOOT = 10_000
SEED = 42
R1_PREFIX = "r1_"

FORBIDDEN = "p2_test_pred_h"   # the pre- GBM predictions. Never opened here.


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1_PREFIX):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_pred(path: Path) -> pd.DataFrame:
    if FORBIDDEN in path.name:
        raise SystemExit(f"REFUSING to read {path.name}: pre- GBM predictions.")
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_parquet(path)


def dm_hln(d: np.ndarray, hac_lag: int, h_forecast: int) -> tuple[float, float]:
    """Diebold-Mariano on a pre-formed loss differential d. Long-run variance by a
    Newey-West (Bartlett-kernel) estimator at the given HAC truncation lag (PSD by
    construction); HLN small-sample correction at the true forecast horizon. Negative statistic => model A better.
    Identical machinery to evaluation/metrics.diebold_mariano, but taking d directly
    so the same differential feeds the bootstrap."""
    n = len(d)
    dbar = float(d.mean())
    var = float(np.mean((d - dbar) ** 2))
    L = max(hac_lag, 1)
    for k in range(1, L):
        ck = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        var += 2 * (1.0 - k / L) * ck        # Bartlett weight: PSD by construction
    if var <= 0 or n < 3:
        return float("nan"), float("nan")
    hh = int(max(h_forecast, 1))             # HLN uses the TRUE forecast horizon
    DM = dbar / np.sqrt(var / n)
    corr = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    if corr <= 0:
        return float("nan"), float("nan")
    DM *= np.sqrt(corr)
    return float(DM), float(2 * stats.t.cdf(-abs(DM), df=n - 1))


def day_block_bootstrap(y, pa, pb, days, B=B_BOOT, seed=SEED):
    """Paired day-block bootstrap on the RMSE difference RMSE(A) - RMSE(B).
    Resamples whole operating days with replacement; both models see the same day
    draw, so the difference is paired.

    RMSE over a resample of whole days is sqrt(sum of day SSEs / sum of day counts),
    so only the per-day sum of squared errors and the per-day count are needed. That
    makes each draw O(n_days) instead of O(n_obs) and is exact, not an approximation."""
    rng = np.random.default_rng(seed)
    udays, inv = np.unique(days, return_inverse=True)
    n_days = len(udays)
    sa = (y - pa) ** 2
    sb = (y - pb) ** 2
    sse_a = np.bincount(inv, weights=sa, minlength=n_days)
    sse_b = np.bincount(inv, weights=sb, minlength=n_days)
    cnt = np.bincount(inv, minlength=n_days).astype(float)
    obs = float(np.sqrt(sa.mean()) - np.sqrt(sb.mean()))
    pick = rng.integers(0, n_days, size=(B, n_days))
    n_b = cnt[pick].sum(axis=1)
    draws = np.sqrt(sse_a[pick].sum(axis=1) / n_b) - np.sqrt(sse_b[pick].sum(axis=1) / n_b)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # two-sided bootstrap p: proportion of draws on the other side of zero, doubled
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return obs, float(lo), float(hi), float(min(p, 1.0)), n_days


def main() -> None:
    force = "--force" in sys.argv or os.environ.get("R1_REBUILD") == "1"   # reproduce.sh sets R1_REBUILD=1
    t0 = time.time()
    print(f"python : {platform.python_version()}   numpy {np.__version__}   pandas {pd.__version__}")

    planned = [CFG.TAB / f"{R1_PREFIX}p2_deep_vs_gbm.csv",
               CFG.TAB / f"{R1_PREFIX}p2_common_support.csv",
               CFG.MET / f"{R1_PREFIX}p2_deep_vs_gbm.json"]
    blocked = []
    for p in planned:
        try:
            guarded(p, force)
        except WriteGuard as e:
            blocked.append(str(e))
    if blocked:
        raise SystemExit("Pre-flight FAILED:\n\n" + "\n\n".join(blocked))

    runs = {"C_full": "_full", "A": ""}      # run C first: it is the primary source
    rows, support = [], []

    for run_name, suffix in runs.items():
        for h in HORIZONS:
            gb = read_pred(CFG.TAB / f"{R1_PREFIX}p2_point_pred_h{h}.parquet")
            dp = read_pred(CFG.TAB / f"{R1_PREFIX}p2_deep_pred_h{h}{suffix}.parquet")
            common = gb.index.intersection(dp.index)
            g = gb.loc[common]
            d = dp.loc[common]

            # the two frames must agree on the target on the shared instants, else the
            # two halves of Table 2 are not describing the same experiment
            tgt_max = float(np.max(np.abs(g["y_ghi"].values - d["y_ghi"].values)))
            if tgt_max > 1e-3:
                raise SystemExit(f"Target mismatch on common index at h={h}, run {run_name}: "
                                 f"max |Delta y| = {tgt_max}")

            y = g["y_ghi"].values.astype(float)
            days = pd.DatetimeIndex(common).normalize().values
            med_obs_per_day = int(np.median(pd.Series(1, index=pd.DatetimeIndex(common))
.groupby(pd.DatetimeIndex(common).normalize()).sum()))
            preds = {"gbm": g["gbm"].values.astype(float),
                     "gru": d["pred_gru"].values.astype(float),
                     "gru_tcn": d["pred_gru_tcn"].values.astype(float)}

            for name, p in preds.items():
                m = M.all_metrics(y, p)
                m.update(model=name, horizon_min=h * 5, run=run_name,
                         n_common=int(len(common)),
                         n_gbm_own=int(len(gb)), n_deep_own=int(len(dp)),
                         scope="common_support")
                support.append(m)

            for a, b in [("gbm", "gru"), ("gbm", "gru_tcn")]:
                dd = (y - preds[a]) ** 2 - (y - preds[b]) ** 2
                dm_h, p_h = dm_hln(dd, hac_lag=h - 1, h_forecast=h)
                dm_day, p_day = dm_hln(dd, hac_lag=med_obs_per_day, h_forecast=h)
                obs, lo, hi, p_boot, n_days = day_block_bootstrap(y, preds[a], preds[b], days)
                rmse_a = M.rmse(y, preds[a])
                rmse_b = M.rmse(y, preds[b])
                rows.append(dict(
                    run=run_name, horizon_min=h * 5, model_A=a, model_B=b,
                    n_common=int(len(common)), n_days=int(n_days),
                    RMSE_A=round(rmse_a, 4), RMSE_B=round(rmse_b, 4),
                    dRMSE=round(obs, 4),
                    pct_A_better_than_B=round(100 * (rmse_b - rmse_a) / rmse_b, 4),
                    boot_lo=round(lo, 4), boot_hi=round(hi, 4),
                    boot_p=float(f"{p_boot:.4g}"),
                    boot_excludes_zero=bool(lo < 0 and hi < 0) or bool(lo > 0 and hi > 0),
                    DM_hac_hminus1=round(dm_h, 4), p_hac_hminus1=float(f"{p_h:.3e}"),
                    DM_hac_day=round(dm_day, 4), p_hac_day=float(f"{p_day:.3e}"),
                    hac_lag_day=med_obs_per_day, A_better=bool(obs < 0),
                ))
                print(f"{run_name:>6} h={h*5:>2}min {a} vs {b}: dRMSE {obs:+.3f} "
                      f"CI [{lo:+.3f},{hi:+.3f}] p_boot {p_boot:.2g} "
                      f"p_DM(day) {p_day:.2e}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_deep_vs_gbm.csv", force), index=False)
    sup = pd.DataFrame(support)[["run", "horizon_min", "model", "n_common", "n_gbm_own",
                                 "n_deep_own", "MAE", "RMSE", "nRMSE", "R2", "scope"]].round(4)
    sup.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_common_support.csv", force), index=False)

    json.dump(dict(
        run_id="(b) paired DM + day-block bootstrap, deep vs post-causal GBM",
        finished_utc=pd.Timestamp.utcnow().isoformat(),
        elapsed_s=round(time.time() - t0, 1),
        B_bootstrap=B_BOOT, seed=SEED,
        primary_run="C_full (run C, 30 epochs, all training windows)",
        secondary_run="A (15 epochs, 150,000 windows)",
        forbidden_inputs=["02_data/interim/p2_test_pred_h*.parquet (pre-)"],
        note="Headline Table 2 rows are each model on its own valid set; every paired "
             "test here is on the intersection, whose size is in n_common.",
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
    ), open(guarded(CFG.MET / f"{R1_PREFIX}p2_deep_vs_gbm.json", force), "w"), indent=2, default=str)

    print("\n=== common-support metrics ===")
    print(sup.to_string(index=False))
    print("\n=== paired tests ===")
    print(res.to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
