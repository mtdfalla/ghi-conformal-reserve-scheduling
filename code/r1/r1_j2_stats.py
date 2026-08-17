"""Final pass — statistical evidence behind every "best" / "lowest" claim.

Consumes the per-observation dumps written by `r1_j2_delayed.py` and produces:

  (1) Paired Diebold-Mariano tests on the CRPS (mean pinball-loss) differential,
      with the Harvey-Leybourne-Newbold small-sample correction, reported at
      * HAC lag h-1  (the textbook choice for an h-step forecast, and the one the
        a fair reader would ask for), and
      * HAC lag = the median number of daytime observations per day, which is the
        honest bandwidth for 5-minute data whose losses are correlated all day.
      The second is always the more conservative of the two and is what we quote.

  (2) Paired day-block bootstrap confidence intervals (B = 10,000, resampling whole
      days) on: CRPS levels, CRPS differences, PICP, and per-regime ACE. The
      day-block bootstrap is the primary evidence: it makes no assumption about the
      autocorrelation structure and respects the diurnal blocking.

Usage (from 03_code):  python3 r1/r1_j2_stats.py [B]
Writes  04_results/tables/r1_j2_stats.csv          (DM tests, CRPS CIs, differences)
        04_results/tables/r1_j2_coverage_ci.csv    (PICP / ACE CIs per method x scope)
"""
import sys, os, glob, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils")
import numpy as np, pandas as pd
from scipy import stats
import config as CFG

B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED = 42
OUT = "/tmp/r1j2"
COVS = [0.80, 0.90, 0.95]
QMETHODS = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr"]


# ------------------------------------------------------------------ DM test
def dm_test(d, hac_lag):
    """Paired DM on a loss differential d = loss_A - loss_B (negative => A better).
    HLN small-sample correction with `hac_lag` autocovariances (lag 0 = none)."""
    d = np.asarray(d, float); n = len(d); dbar = d.mean()
    var = np.mean((d - dbar) ** 2)
    L = int(max(hac_lag, 0))
    for k in range(1, L + 1):
        ck = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * ck
    if var <= 0 or n < 3:
        return np.nan, np.nan
    hh = L + 1                                    # forecast horizon implied by the lag
    DM = dbar / np.sqrt(var / n)
    corr = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    if corr <= 0:
        return float(DM), float(2 * stats.t.cdf(-abs(DM), df=n - 1))
    DM *= np.sqrt(corr)
    return float(DM), float(2 * stats.t.cdf(-abs(DM), df=n - 1))


# ------------------------------------------------- day-block bootstrap engine
class DayBoot:
    def __init__(self, day_code, n_days, B, seed):
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, n_days, size=(B, n_days))
        self.day_code = day_code
        self.nd = n_days

    def _per_day(self, v, mask):
        dc = self.day_code[mask]
        S = np.bincount(dc, weights=np.asarray(v, float)[mask], minlength=self.nd)
        C = np.bincount(dc, minlength=self.nd).astype(float)
        return S, C

    def mean_ci(self, v, mask, alpha=0.05):
        """Bootstrap distribution of the mask-restricted mean of v."""
        S, C = self._per_day(v, mask)
        num = S[self.idx].sum(axis=1); den = C[self.idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = num / den
        b = b[np.isfinite(b)]
        pt = float(S.sum() / C.sum()) if C.sum() > 0 else np.nan
        return pt, float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))), b


def pct(b, a):
    return float(np.percentile(b, a))


rows_stats = []; rows_cov = []
files = sorted(glob.glob(f"{OUT}/*_obs.npz"))
print(f"{len(files)} observation dumps, B={B}", flush=True)

for f in files:
    z = np.load(f, allow_pickle=True)
    site = str(z["site"]); hmin = int(z["horizon_min"]); h_steps = hmin // 5
    day_code = z["day_code"]; nd = int(z["n_days"])
    reg_code = z["reg_code"]; reg_names = [str(x) for x in z["reg_names"]]
    n = len(day_code)
    hac_day = int(np.median(np.bincount(day_code)))
    boot = DayBoot(day_code, nd, B, SEED)
    scopes = [("all", np.ones(n, bool))] + [(r, reg_code == i) for i, r in enumerate(reg_names)]
    t0 = time.time()

    # ---------------- CRPS levels + paired differences ----------------
    crps = {k[len("crps__"):]: z[k] for k in z.files if k.startswith("crps__")}
    for scope, mask in scopes:
        if mask.sum() < 30:
            continue
        bdist = {}
        for key, v in crps.items():
            p, lo, hi, b = boot.mean_ci(v, mask)
            bdist[key] = b
            rows_stats.append(dict(site=site, horizon_min=hmin, scope=scope, kind="CRPS_level",
                                   method_a=key, method_b="", n=int(mask.sum()),
                                   value=round(p, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                   dm_stat=np.nan, dm_p_hac_h=np.nan, dm_p_hac_day=np.nan,
                                   significant_95=""))
        keys = [f"{m}|static" for m in QMETHODS if f"{m}|static" in crps]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                A, Bk = keys[i], keys[j]
                dv = crps[A] - crps[Bk]
                dm1, p1 = dm_test(dv[mask], h_steps - 1)
                dm2, p2 = dm_test(dv[mask], hac_day)
                _, _, _, bd = boot.mean_ci(dv, mask)
                dlo, dhi = pct(bd, 2.5), pct(bd, 97.5)
                sig = "yes" if (dlo > 0) == (dhi > 0) else "no"
                rows_stats.append(dict(site=site, horizon_min=hmin, scope=scope, kind="CRPS_diff",
                                       method_a=A.split("|")[0], method_b=Bk.split("|")[0],
                                       n=int(mask.sum()),
                                       value=round(float(dv[mask].mean()), 4),
                                       ci_lo=round(dlo, 4), ci_hi=round(dhi, 4),
                                       dm_stat=round(dm2, 3), dm_p_hac_h=round(p1, 6),
                                       dm_p_hac_day=round(p2, 6), significant_95=sig))
        # rearranged vs static ( effect on CRPS)
        for m in ["cqr", "mondrian_cqr"]:
            A, Bk = f"{m}|rearranged", f"{m}|static"
            if A in crps and Bk in crps:
                dv = crps[A] - crps[Bk]
                dm2, p2 = dm_test(dv[mask], hac_day)
                _, _, _, bd = boot.mean_ci(dv, mask)
                dlo, dhi = pct(bd, 2.5), pct(bd, 97.5)
                rows_stats.append(dict(site=site, horizon_min=hmin, scope=scope,
                                       kind="CRPS_diff_rearranged", method_a=A, method_b=Bk,
                                       n=int(mask.sum()), value=round(float(dv[mask].mean()), 6),
                                       ci_lo=round(dlo, 6), ci_hi=round(dhi, 6),
                                       dm_stat=round(dm2, 3), dm_p_hac_h=np.nan,
                                       dm_p_hac_day=round(p2, 6),
                                       significant_95="yes" if (dlo > 0) == (dhi > 0) else "no"))

    # ---------------- PICP / ACE confidence intervals ----------------
    covs = {k[len("cov__"):]: z[k] for k in z.files if k.startswith("cov__")}
    for key, v in covs.items():
        m, variant, cov = key.split("|"); cov = float(cov)
        for scope, mask in scopes:
            if mask.sum() < 30:
                continue
            p, lo, hi, b = boot.mean_ci(v.astype(float), mask)
            rows_cov.append(dict(site=site, horizon_min=hmin, method=m, variant=variant,
                                 nominal=cov, scope=scope, n=int(mask.sum()),
                                 PICP=round(p, 4), PICP_ci_lo=round(lo, 4), PICP_ci_hi=round(hi, 4),
                                 ACE=round(p - cov, 4), ACE_ci_lo=round(lo - cov, 4),
                                 ACE_ci_hi=round(hi - cov, 4),
                                 covers_nominal="yes" if (lo <= cov <= hi) else "no"))
    print(f"  {site} h={hmin}min done in {time.time()-t0:.0f}s (hac_day={hac_day}, days={nd})",
          flush=True)

st = pd.DataFrame(rows_stats); cv = pd.DataFrame(rows_cov)
st.to_csv(CFG.TAB / "r1_j2_stats.csv", index=False)
cv.to_csv(CFG.TAB / "r1_j2_coverage_ci.csv", index=False)
print(f"wrote r1_j2_stats.csv ({len(st)} rows) and r1_j2_coverage_ci.csv ({len(cv)} rows)")
