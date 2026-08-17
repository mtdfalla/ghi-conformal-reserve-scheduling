"""J3 — DKASC Alice Springs cross-site forecasting + conformal, ONE horizon.

Mirrors the Yulara J2 path on DKASC: GBM in kt-space, train 2020-22 / calib 2023 /
test 2024; conformal methods icp/mondrian/cqr/mondrian_cqr/aci; metrics by horizon
& regime; plus a load-independent reserve newsvendor (value captured) in GHI space.

Run from 03_code: python3 _j3_one_horizon.py <h_steps>  (1,3,6,12)
Caches /tmp/asp_base.parquet; writes /tmp/j3out/h{h}.json.
"""
import sys, json, time, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "evaluation"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP, conformal_adaptive as CA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

h = int(sys.argv[1]); COVS = [0.80, 0.90, 0.95]; GAMMA = 0.05
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)
REG = CFG.BASE / "02_data" / "DKASC" / "regime_labels" / "asp_regimes_5min.parquet"
BASEP = "/tmp/asp_base.parquet"

def build_base():
    reg = pd.read_parquet(REG)
    df = pd.DataFrame(index=reg.index)
    df["ghi"] = reg["ghi"]; df["ghi_cs"] = reg["ghi_cs"]; df["kt"] = reg["kt"]
    df["zenith"] = reg["zenith"]; df["is_day"] = reg["is_day"]; df["regime"] = reg["regime"]
    df["ghi_imputed"] = reg["ghi_imputed"]; df["year"] = df.index.year
    hod = df.index.hour + df.index.minute/60
    df["hod_sin"] = np.sin(2*np.pi*hod/24); df["hod_cos"] = np.cos(2*np.pi*hod/24)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365); df["doy_cos"] = np.cos(2*np.pi*doy/365)
    df["cosz"] = np.cos(np.radians(df["zenith"].clip(0, 90)))
    for L in D.LAGS: df[f"kt_l{L}"] = df["kt"].shift(L)
    df["kt_rmean"] = df["kt"].rolling(D.ROLL, min_periods=D.ROLL).mean()
    df["kt_rstd"] = df["kt"].rolling(D.ROLL, min_periods=D.ROLL).std()
    return df

if not os.path.exists(BASEP):
    build_base().to_parquet(BASEP)
base = pd.read_parquet(BASEP); F = D.FEATURES

def make_xy(df, h):
    d = df.copy()
    d["y_kt"] = d["kt"].shift(-h); d["y_ghi"] = d["ghi"].shift(-h)
    d["y_ghi_cs"] = d["ghi_cs"].shift(-h); d["y_is_day"] = d["is_day"].shift(-h)
    d["y_imputed"] = d["ghi_imputed"].shift(-h); d["base_regime"] = d["regime"]
    need = F + ["y_kt", "y_ghi", "y_ghi_cs"]
    valid = (d["is_day"] & d["y_is_day"].fillna(False) & (~d["y_imputed"].fillna(True)) & d[need].notna().all(axis=1))
    return d[valid]

d = make_xy(base, h).sort_index()
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]

def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)
t0 = time.time()
gp = gbm().fit(tr[F].values, tr["y_kt"].values)
pc = rec(gp.predict(ca[F].values), ca["y_ghi_cs"].values)
pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
yc = ca["y_ghi"].values; yt = te["y_ghi"].values
gc = ca["base_regime"].values; gt = te["base_regime"].values
csc = np.maximum(ca["y_ghi_cs"].values, 50.0); cst = np.maximum(te["y_ghi_cs"].values, 50.0)
taus = sorted({0.025, 0.05, 0.10, 0.90, 0.95, 0.975})
qc = {}; qt = {}
for tau in taus:
    gq = gbm(loss="quantile", quantile=tau).fit(tr[F].values, tr["y_kt"].values)
    qc[tau] = rec(gq.predict(ca[F].values), ca["y_ghi_cs"].values)
    qt[tau] = rec(gq.predict(te[F].values), te["y_ghi_cs"].values)
print(f"h={h*5} fitted in {time.time()-t0:.1f}s | tr={len(tr)} ca={len(ca)} te={len(te)}", flush=True)

icp_s = CP.icp_fit(yc, pc); mon_s = CP.mondrian_fit(yc, pc, gc)
day_id = te.index.normalize().values
SCOPES = [("all", np.ones(len(yt), bool)), ("clear", gt == "clear"),
          ("transitional", gt == "transitional"), ("cloudy", gt == "cloudy")]

# point metrics
pts = []
for rgname, mask in SCOPES:
    if mask.sum() < 30: continue
    e = yt[mask] - pt[mask]
    pts.append(dict(horizon_min=h*5, scope=rgname, RMSE=round(float(np.sqrt(np.mean(e**2))), 2),
                    MAE=round(float(np.mean(np.abs(e))), 2), n=int(mask.sum())))

recs = []; crps = []; qstore = {m: {} for m in ["icp", "mondrian", "cqr", "mondrian_cqr"]}
for cov in COVS:
    a = 1-cov; lo_tau = round(a/2, 3); hi_tau = round(1-a/2, 3)
    iv = {}
    iv["icp"] = CP.icp_interval(pt, icp_s, cov)
    iv["mondrian"] = CP.mondrian_interval(pt, gt, mon_s, cov, icp_s)
    iv["cqr"] = CP.cqr_interval(qt[lo_tau], qt[hi_tau], CP.cqr_fit(yc, qc[lo_tau], qc[hi_tau]), cov)
    iv["mondrian_cqr"] = CA.mondrian_cqr_interval(qt[lo_tau], qt[hi_tau], gt,
        CA.mondrian_cqr_fit(yc, qc[lo_tau], qc[hi_tau], gc), cov, CP.cqr_fit(yc, qc[lo_tau], qc[hi_tau]))
    lo_a, hi_a, _ = CA.aci_run(pt, yt, icp_s, a, GAMMA, day_id); iv["aci"] = (lo_a, hi_a)
    for m, (lo, hi) in iv.items():
        if m in qstore: qstore[m][lo_tau] = lo; qstore[m][hi_tau] = hi
        for rgname, mask in SCOPES:
            if mask.sum() < 30: continue
            p = CP.picp(yt[mask], lo[mask], hi[mask])
            recs.append(dict(method=m, horizon_min=h*5, nominal=cov, scope=rgname, n=int(mask.sum()),
                             PICP=round(p, 4), ACE=round(p-cov, 4),
                             PINAW=round(CP.pinaw(yt[mask], lo[mask], hi[mask]), 4),
                             Winkler=round(CP.winkler(yt[mask], lo[mask], hi[mask], cov), 2)))
for m in qstore:
    qd = dict(qstore[m]); qd[0.5] = pt
    for rgname, mask in SCOPES:
        if mask.sum() < 30: continue
        crps.append(dict(method=m, horizon_min=h*5, scope=rgname,
                         CRPS=round(CP.crps_from_quantiles(yt[mask], {t: qd[t][mask] for t in qd}), 3)))

# ---- decision value: load-independent reserve newsvendor in GHI space ----
# cost ratio r = c_u/c_o; optimal reserve = (r/(1+r))-quantile of GHI forecast error.
# det: zero reserve (point). method: conformal upper offset (Mondrian). oracle: perfect.
def reserve_cost(y, pred, reserve, r, c_o=1.0):
    short = np.maximum(y - (pred + reserve), 0.0)   # under-provision
    over = np.maximum((pred + reserve) - y, 0.0)    # over-provision
    return float(np.mean(r*c_o*short + c_o*over))
dec = []
for r in [10.0]:
    # Mondrian reserve = per-regime conformal offset at coverage r/(1+r)
    cov_r = r/(1+r)
    _, hi_m = CP.mondrian_interval(pt, gt, mon_s, 2*cov_r-1 if cov_r > 0.5 else cov_r, icp_s)
    res_m = np.maximum(hi_m - pt, 0.0)
    # simpler: use one-sided Mondrian quantile of residuals at cov_r
    res_m = np.empty(len(pt))
    for g in np.unique(gt):
        s = mon_s.get(g, icp_s); q = CP.conformal_q(np.abs(s), cov_r)
        res_m[gt == g] = q
    res_det = np.zeros(len(pt))
    res_or = np.maximum(yt - pt, 0.0)  # oracle perfectly covers shortfall
    c_det = reserve_cost(yt, pt, res_det, r); c_m = reserve_cost(yt, pt, res_m, r); c_or = reserve_cost(yt, pt, res_or, r)
    vc = (c_det - c_m)/(c_det - c_or) if (c_det - c_or) > 0 else np.nan
    dec.append(dict(horizon_min=h*5, ratio=r, cost_det=round(c_det, 2), cost_mondrian=round(c_m, 2),
                    cost_oracle=round(c_or, 2), value_captured=round(float(vc), 4)))

os.makedirs("/tmp/j3out", exist_ok=True)
json.dump(dict(point=pts, intervals=recs, crps=crps, decision=dec), open(f"/tmp/j3out/h{h}.json", "w"), default=str)
print(f"h={h*5} done: point+{len(recs)} interval recs; value_captured(mondrian,r10)={dec[0]['value_captured']}", flush=True)
