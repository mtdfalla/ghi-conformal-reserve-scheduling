"""Final pass — one (site, horizon) re-run of the J2 interval layer.

Produces, from the S0 prediction cache (`/tmp/r1cache/`), for one site and horizon:

   ACI and ACI-regime with **h-step delayed feedback**, side by side with the
         anticipative update computed from the *same* predictions, so the
         difference table isolates the delay and not a refit.
  the effect of capping the upper bound at 1.5 * G^cs.
  the quantile-crossing rate, before and after rearrangement.
   per-observation CRPS and coverage indicators, dumped for the day-block
         bootstrap and the Diebold-Mariano tests in `r1_j2_stats.py`.
   per-horizon records so ACE-RMS can be computed at 5 min only.

Usage (from 03_code):  python3 r1/r1_j2_delayed.py <yulara|asp> <h_steps>
Writes /tmp/r1j2/<site>_h<h>.json  and  /tmp/r1j2/<site>_h<h>_obs.npz
"""
import sys, os, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "evaluation"); sys.path.insert(0, "conformal")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP
import conformal_adaptive as CA
import r1_conformal_delayed as CD

site = sys.argv[1]; h = int(sys.argv[2])
COVS = [0.80, 0.90, 0.95]
GAMMA = 0.05
GAMMA_SWEEP = [0.01, 0.02, 0.05, 0.10]
TAUS_Q = [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]
BASEP = "/tmp/base.parquet" if site == "yulara" else "/tmp/base_asp.parquet"
PRE = f"/tmp/r1cache/{site}_h{h}"
OUT = "/tmp/r1j2"; os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data + cache
base = pd.read_parquet(BASEP)
d = D.make_xy(base, h).sort_index()
ca = d[d.year == 2023]; te = d[d.year == 2024]

yc = ca["y_ghi"].values; yt = te["y_ghi"].values
csc = np.maximum(ca["y_ghi_cs"].values, 50.0); cst = np.maximum(te["y_ghi_cs"].values, 50.0)
cs_raw_t = te["y_ghi_cs"].values                     # un-floored, for the 1.5*G^cs cap
gc = ca["base_regime"].values.astype(str); gt = te["base_regime"].values.astype(str)
day_id = te.index.normalize().values
month = te.index.to_period("M").astype(str).values

pc = np.load(f"{PRE}_point_c.npy"); pt = np.load(f"{PRE}_point_t.npy")
qc = {t_: np.load(f"{PRE}_q{t_}_c.npy") for t_ in TAUS_Q}
qt = {t_: np.load(f"{PRE}_q{t_}_t.npy") for t_ in TAUS_Q}
assert len(pt) == len(yt) and len(pc) == len(yc), "cache / frame length mismatch"
print(f"[{site} h={h*5}min] n_cal={len(yc)} n_test={len(yt)} days={len(np.unique(day_id))}", flush=True)

# ---------------------------------------------------------- crossing
def crossing_stats(qd, taus):
    """Fraction of test points where the quantile sequence is non-monotone in tau,
    the magnitude of the violation, and a breakdown by adjacent tau pair."""
    M = np.column_stack([qd[t_] for t_ in taus])          # (n, k)
    dif = np.diff(M, axis=1)
    viol = dif < 0
    any_v = viol.any(axis=1)
    n_pts = int(any_v.sum())
    worst = np.where(any_v, -dif.min(axis=1), 0.0)
    pairs = {}
    for i in range(viol.shape[1]):
        cnt = int(viol[:, i].sum())
        mag = float(-dif[viol[:, i], i].mean()) if cnt else 0.0
        pairs[f"{taus[i]}->{taus[i+1]}"] = dict(n=cnt, rate=round(cnt / M.shape[0], 6),
                                                mean_violation_Wm2=round(mag, 4))
    return dict(n=int(M.shape[0]), n_crossing=n_pts,
                rate=round(float(n_pts / M.shape[0]), 6),
                mean_violation_Wm2=round(float(worst[worst > 0].mean()) if n_pts else 0.0, 4),
                median_violation_Wm2=round(float(np.median(worst[worst > 0])) if n_pts else 0.0, 4),
                p99_violation_Wm2=round(float(np.percentile(worst[worst > 0], 99)) if n_pts else 0.0, 4),
                max_violation_Wm2=round(float(worst.max()), 4),
                mean_violation_pct_of_mean_GHI=round(
                    float(worst[worst > 0].mean() / np.mean(yt) * 100) if n_pts else 0.0, 4),
                by_adjacent_pair=pairs)

taus_all = sorted(TAUS_Q + [0.5])
qt_full = dict(qt); qt_full[0.5] = pt
qc_full = dict(qc); qc_full[0.5] = pc
crossing = {
    "test_quantiles_only": crossing_stats(qt, TAUS_Q),
    "test_with_point_as_median": crossing_stats(qt_full, taus_all),
    "calib_quantiles_only": crossing_stats(qc, TAUS_Q),
}
# the crossings that actually matter for an interval: the CQR band edges
for cov_ in COVS:
    a_ = 1 - cov_; lt = round(a_ / 2, 3); ht = round(1 - a_ / 2, 3)
    bad = qt[lt] > qt[ht]
    crossing[f"band_edges_{int(cov_*100)}"] = dict(
        tau_lo=lt, tau_hi=ht, n=int(len(yt)), n_crossing=int(bad.sum()),
        rate=round(float(bad.mean()), 6),
        mean_violation_Wm2=round(float((qt[lt] - qt[ht])[bad].mean()) if bad.any() else 0.0, 4),
        max_violation_Wm2=round(float((qt[lt] - qt[ht])[bad].max()) if bad.any() else 0.0, 4))

def rearrange(qd, taus):
    M = np.column_stack([qd[t_] for t_ in taus])
    M = np.sort(M, axis=1)
    return {t_: M[:, i] for i, t_ in enumerate(taus)}

qt_rear = rearrange(qt_full, taus_all)
qc_rear = rearrange(qc_full, taus_all)

# ------------------------------------------------------------------- helpers
SCOPES = [("all", np.ones(len(yt), bool)), ("clear", gt == "clear"),
          ("transitional", gt == "transitional"), ("cloudy", gt == "cloudy")]

def pinball_obs(y, q, tau):
    dd = y - q
    return np.maximum(tau * dd, (tau - 1) * dd)

def crps_obs_from(qd, taus):
    """Per-observation CRPS approximation: 2 * mean over tau of the pinball loss."""
    return 2.0 * np.mean(np.column_stack([pinball_obs(yt, qd[t_], t_) for t_ in taus]), axis=1)

def add_recs(recs, method, cov, lo, hi, variant):
    for rgname, mask in SCOPES:
        if mask.sum() < 30:
            continue
        p = CP.picp(yt[mask], lo[mask], hi[mask])
        recs.append(dict(site=site, method=method, variant=variant, horizon_min=h * 5,
                         nominal=cov, scope=rgname, n=int(mask.sum()),
                         PICP=round(p, 4), ACE=round(p - cov, 4),
                         PINAW=round(CP.pinaw(yt[mask], lo[mask], hi[mask]), 4),
                         Winkler=round(CP.winkler(yt[mask], lo[mask], hi[mask], cov), 2)))

# ------------------------------------------------------------- static methods
icp_s = CP.icp_fit(yc, pc)
icpn_s = CP.icpn_fit(yc, pc, csc)
mon_s = CP.mondrian_fit(yc, pc, gc)

recs = []; rel = []; crps_recs = []; capr = []
qstore = {m: {} for m in ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr"]}
qstore_rear = {m: {} for m in ["cqr", "mondrian_cqr"]}
cov_obs = {}          # (method, variant, cov) -> bool array
alpha_traces = {}
t0 = time.time()

for cov in COVS:
    a = 1 - cov
    lo_tau = round(a / 2, 3); hi_tau = round(1 - a / 2, 3)

    iv = {}
    iv["icp"] = CP.icp_interval(pt, icp_s, cov)
    iv["icp_norm"] = CP.icpn_interval(pt, cst, icpn_s, cov)
    iv["mondrian"] = CP.mondrian_interval(pt, gt, mon_s, cov, icp_s)
    cqr_s = CP.cqr_fit(yc, qc[lo_tau], qc[hi_tau])
    iv["cqr"] = CP.cqr_interval(qt[lo_tau], qt[hi_tau], cqr_s, cov)
    mcqr_s = CA.mondrian_cqr_fit(yc, qc[lo_tau], qc[hi_tau], gc)
    iv["mondrian_cqr"] = CA.mondrian_cqr_interval(qt[lo_tau], qt[hi_tau], gt, mcqr_s, cov, cqr_s)

    for m, (lo, hi) in iv.items():
        add_recs(recs, m, cov, lo, hi, "static")
        qstore[m][lo_tau] = lo; qstore[m][hi_tau] = hi
        cov_obs[f"{m}|static|{cov}"] = (yt >= lo) & (yt <= hi)
        if abs(cov - 0.90) < 1e-9:
            for r in CA.reliability_over_time(yt, lo, hi, month):
                r.update(site=site, method=m, variant="static", horizon_min=h * 5); rel.append(r)
        # ---- cap the upper bound at 1.5 * G^cs (same clip as the point rec.)
        hi_c = np.minimum(hi, 1.5 * np.maximum(cs_raw_t, 0.0))
        n_capped = int((hi_c < hi - 1e-9).sum())
        capr.append(dict(site=site, method=m, horizon_min=h * 5, nominal=cov,
                         n=int(len(yt)), n_capped=n_capped,
                         frac_capped=round(n_capped / len(yt), 6),
                         PICP=round(CP.picp(yt, lo, hi), 4),
                         PICP_capped=round(CP.picp(yt, lo, hi_c), 4),
                         PINAW=round(CP.pinaw(yt, lo, hi), 4),
                         PINAW_capped=round(CP.pinaw(yt, lo, hi_c), 4),
                         Winkler=round(CP.winkler(yt, lo, hi, cov), 2),
                         Winkler_capped=round(CP.winkler(yt, lo, hi_c, cov), 2)))

    # ---- the same CQR-family intervals built on REARRANGED quantiles
    cqr_s_r = CP.cqr_fit(yc, qc_rear[lo_tau], qc_rear[hi_tau])
    lo_r, hi_r = CP.cqr_interval(qt_rear[lo_tau], qt_rear[hi_tau], cqr_s_r, cov)
    add_recs(recs, "cqr", cov, lo_r, hi_r, "rearranged")
    qstore_rear["cqr"][lo_tau] = lo_r; qstore_rear["cqr"][hi_tau] = hi_r
    mcqr_s_r = CA.mondrian_cqr_fit(yc, qc_rear[lo_tau], qc_rear[hi_tau], gc)
    lo_r2, hi_r2 = CA.mondrian_cqr_interval(qt_rear[lo_tau], qt_rear[hi_tau], gt,
                                            mcqr_s_r, cov, cqr_s_r)
    add_recs(recs, "mondrian_cqr", cov, lo_r2, hi_r2, "rearranged")
    qstore_rear["mondrian_cqr"][lo_tau] = lo_r2; qstore_rear["mondrian_cqr"][hi_tau] = hi_r2

    # ------------------------------------------------- adaptive: anticipative vs delayed
    for variant in ("anticipative", "delayed"):
        if variant == "anticipative":
            lo_a, hi_a, atr = CA.aci_run(pt, yt, icp_s, a, GAMMA, day_id)
            lo_ar, hi_ar, atr_r = CA.aci_regime_run(pt, yt, gt, mon_s, a, GAMMA, day_id, icp_s)
        else:
            lo_a, hi_a, atr = CD.aci_run_delayed(pt, yt, icp_s, a, GAMMA, day_id, h)
            lo_ar, hi_ar, atr_r = CD.aci_regime_run_delayed(pt, yt, gt, mon_s, a, GAMMA,
                                                            day_id, icp_s, h)
        for m, (lo, hi) in (("aci", (lo_a, hi_a)), ("aci_regime", (lo_ar, hi_ar))):
            add_recs(recs, m, cov, lo, hi, variant)
            cov_obs[f"{m}|{variant}|{cov}"] = (yt >= lo) & (yt <= hi)
            if abs(cov - 0.90) < 1e-9:
                for r in CA.reliability_over_time(yt, lo, hi, month):
                    r.update(site=site, method=m, variant=variant, horizon_min=h * 5); rel.append(r)
        if abs(cov - 0.90) < 1e-9:
            alpha_traces[f"aci_{variant}"] = atr
            alpha_traces[f"aci_regime_{variant}"] = atr_r

        # ---- gamma sweep (all-scope)
        for g in GAMMA_SWEEP:
            if variant == "anticipative":
                lo_g, hi_g, _ = CA.aci_run(pt, yt, icp_s, a, g, day_id)
            else:
                lo_g, hi_g, _ = CD.aci_run_delayed(pt, yt, icp_s, a, g, day_id, h)
            p = CP.picp(yt, lo_g, hi_g)
            recs.append(dict(site=site, method=f"aci_g{g}", variant=variant, horizon_min=h * 5,
                             nominal=cov, scope="all", n=int(len(yt)), PICP=round(p, 4),
                             ACE=round(p - cov, 4),
                             PINAW=round(CP.pinaw(yt, lo_g, hi_g), 4),
                             Winkler=round(CP.winkler(yt, lo_g, hi_g, cov), 2)))
    print(f"  cov={cov} done @{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------- CRPS
crps_obs = {}
for m in qstore:
    qd = dict(qstore[m]); qd[0.5] = pt
    tt = sorted(qd)
    co = crps_obs_from(qd, tt)
    crps_obs[f"{m}|static"] = co
    for rgname, mask in SCOPES:
        if mask.sum() < 30:
            continue
        crps_recs.append(dict(site=site, method=m, variant="static", horizon_min=h * 5,
                              scope=rgname, n=int(mask.sum()), CRPS=round(float(co[mask].mean()), 3)))
for m in qstore_rear:
    qd = dict(qstore_rear[m]); qd[0.5] = pt
    tt = sorted(qd)
    co = crps_obs_from(qd, tt)
    crps_obs[f"{m}|rearranged"] = co
    for rgname, mask in SCOPES:
        if mask.sum() < 30:
            continue
        crps_recs.append(dict(site=site, method=m, variant="rearranged", horizon_min=h * 5,
                              scope=rgname, n=int(mask.sum()), CRPS=round(float(co[mask].mean()), 3)))

# --------------------------------------------------------------- h=1 identity
identity = None
if h == 1:
    ok = []
    for cov in COVS:
        a = 1 - cov
        l1, h1_, _ = CA.aci_run(pt, yt, icp_s, a, GAMMA, day_id)
        l2, h2_, _ = CD.aci_run_delayed(pt, yt, icp_s, a, GAMMA, day_id, 1)
        l3, h3_, _ = CA.aci_regime_run(pt, yt, gt, mon_s, a, GAMMA, day_id, icp_s)
        l4, h4_, _ = CD.aci_regime_run_delayed(pt, yt, gt, mon_s, a, GAMMA, day_id, icp_s, 1)
        ok.append(bool(np.array_equal(l1, l2) and np.array_equal(h1_, h2_) and
                       np.array_equal(l3, l4) and np.array_equal(h3_, h4_)))
    identity = dict(all_bitwise_identical=bool(all(ok)), per_cov=dict(zip(map(str, COVS), ok)))
    print(f"  h=1 delayed==anticipative bitwise: {identity}", flush=True)

json.dump(dict(intervals=recs, crps=crps_recs, reliability=rel, crossing=crossing,
               bound_cap=capr, h1_identity=identity),
          open(f"{OUT}/{site}_h{h}.json", "w"), default=str)

# per-observation dump for the bootstrap / DM stage
udays, day_code = np.unique(day_id, return_inverse=True)
uregs, reg_code = np.unique(gt, return_inverse=True)
np.savez_compressed(f"{OUT}/{site}_h{h}_obs.npz",
                    y=yt, pred=pt, day_code=day_code.astype(np.int32),
                    reg_code=reg_code.astype(np.int8), reg_names=uregs,
                    n_days=len(udays), horizon_min=h * 5, site=site,
                    **{f"crps__{k}": v for k, v in crps_obs.items()},
                    **{f"cov__{k}": v for k, v in cov_obs.items()},
                    **{f"alpha__{k}": v for k, v in alpha_traces.items()})
print(f"[{site} h={h*5}min] wrote {len(recs)} interval recs, {len(crps_recs)} crps recs "
      f"in {time.time()-t0:.0f}s", flush=True)
