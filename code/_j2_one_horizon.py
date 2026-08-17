"""J2 — multi-horizon, multi-method conformal (extends P3) for ONE horizon.

Adds to the P3 method set: ACI (online, day-reset), regime-conditional ACI, and
Mondrian-CQR; plus reliability-over-time (monthly coverage) and ACI alpha traces.

Run from 03_code with: python3 _j2_one_horizon.py <h_steps>   (h in 1,3,6,12)
Reads /tmp/base.parquet, writes /tmp/j2out/h{h}.json (+ bands/alpha for h==1).
"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "evaluation"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP
import conformal_adaptive as CA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

h = int(sys.argv[1]); COVS = [0.80, 0.90, 0.95]
GAMMA = 0.05                       # primary ACI learning rate
GAMMA_SWEEP = [0.01, 0.02, 0.05, 0.10]
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)

base = pd.read_parquet("/tmp/base.parquet")
d = D.make_xy(base, h); F = D.FEATURES
d = d.sort_index()                 # chronological (ACI needs time order)
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]

def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)

t0 = time.time()
yc = ca["y_ghi"].values; yt = te["y_ghi"].values
csc = np.maximum(ca["y_ghi_cs"].values, 50.0); cst = np.maximum(te["y_ghi_cs"].values, 50.0)
gc = ca["base_regime"].values; gt = te["base_regime"].values
taus = sorted({0.025, 0.05, 0.10, 0.90, 0.95, 0.975})
import os as _os
_pf = f"/tmp/j2pred_h{h}.npz"
if _os.path.exists(_pf):
    _z = np.load(_pf, allow_pickle=True); pc=_z["pc"]; pt=_z["pt"]; qc=_z["qc"].item(); qt=_z["qt"].item()
    print(f"h={h*5} loaded cached preds", flush=True)
else:
    gp = gbm().fit(tr[F].values, tr["y_kt"].values)
    pc = rec(gp.predict(ca[F].values), ca["y_ghi_cs"].values)
    pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
    qc = {}; qt = {}
    for tau in taus:
        gq = gbm(loss="quantile", quantile=tau).fit(tr[F].values, tr["y_kt"].values)
        qc[tau] = rec(gq.predict(ca[F].values), ca["y_ghi_cs"].values)
        qt[tau] = rec(gq.predict(te[F].values), te["y_ghi_cs"].values)
    np.savez(_pf, pc=pc, pt=pt, qc=qc, qt=qt)
    print(f"h={h*5} fitted+cached in {time.time()-t0:.1f}s", flush=True)

# time keys for ACI / reliability (test = 2024)
day_id = te.index.normalize().values
month  = te.index.to_period("M").astype(str).values

# fixed-calibration nonconformity scores
icp_s  = CP.icp_fit(yc, pc)
icpn_s = CP.icpn_fit(yc, pc, csc)
mon_s  = CP.mondrian_fit(yc, pc, gc)

SCOPES = [("all", np.ones(len(yt), bool)), ("clear", gt == "clear"),
          ("transitional", gt == "transitional"), ("cloudy", gt == "cloudy")]
recs = []; rel = []; qstore = {m: {} for m in ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr"]}
alpha_traces = {}

for cov in COVS:
    a = 1 - cov; lo_tau = round(a/2, 3); hi_tau = round(1-a/2, 3)
    iv = {}
    iv["icp"]          = CP.icp_interval(pt, icp_s, cov)
    iv["icp_norm"]     = CP.icpn_interval(pt, cst, icpn_s, cov)
    iv["mondrian"]     = CP.mondrian_interval(pt, gt, mon_s, cov, icp_s)
    iv["cqr"]          = CP.cqr_interval(qt[lo_tau], qt[hi_tau], CP.cqr_fit(yc, qc[lo_tau], qc[hi_tau]), cov)
    mcqr_s             = CA.mondrian_cqr_fit(yc, qc[lo_tau], qc[hi_tau], gc)
    iv["mondrian_cqr"] = CA.mondrian_cqr_interval(qt[lo_tau], qt[hi_tau], gt, mcqr_s, cov, CP.cqr_fit(yc, qc[lo_tau], qc[hi_tau]))
    # --- adaptive (online) methods ---
    lo_a, hi_a, atr   = CA.aci_run(pt, yt, icp_s, a, GAMMA, day_id)
    iv["aci"]         = (lo_a, hi_a)
    lo_ar, hi_ar, atr_r = CA.aci_regime_run(pt, yt, gt, mon_s, a, GAMMA, day_id, icp_s)
    iv["aci_regime"]  = (lo_ar, hi_ar)
    if abs(cov-0.90) < 1e-9:
        alpha_traces["aci"] = atr; alpha_traces["aci_regime"] = atr_r

    for m, (lo, hi) in iv.items():
        if m in qstore:
            qstore[m][lo_tau] = lo; qstore[m][hi_tau] = hi
        for rgname, mask in SCOPES:
            if mask.sum() < 30: continue
            p = CP.picp(yt[mask], lo[mask], hi[mask])
            recs.append(dict(method=m, horizon_min=h*5, nominal=cov, scope=rgname, n=int(mask.sum()),
                             PICP=round(p, 4), ACE=round(p-cov, 4),
                             PINAW=round(CP.pinaw(yt[mask], lo[mask], hi[mask]), 4),
                             Winkler=round(CP.winkler(yt[mask], lo[mask], hi[mask], cov), 2)))
        # reliability over time at 90% only
        if abs(cov-0.90) < 1e-9:
            for r in CA.reliability_over_time(yt, lo, hi, month):
                r.update(method=m, horizon_min=h*5); rel.append(r)

    # ACI gamma sensitivity (all scope, this cov)
    for g in GAMMA_SWEEP:
        lo_g, hi_g, _ = CA.aci_run(pt, yt, icp_s, a, g, day_id)
        recs.append(dict(method=f"aci_g{g}", horizon_min=h*5, nominal=cov, scope="all",
                         n=int(len(yt)), PICP=round(CP.picp(yt, lo_g, hi_g), 4),
                         ACE=round(CP.picp(yt, lo_g, hi_g)-cov, 4),
                         PINAW=round(CP.pinaw(yt, lo_g, hi_g), 4),
                         Winkler=round(CP.winkler(yt, lo_g, hi_g, cov), 2)))

# CRPS for quantile-based methods (ACI excluded: it is an interval-coverage method)
crps = []
for m in qstore:
    qd = dict(qstore[m]); qd[0.5] = pt
    for rgname, mask in SCOPES:
        if mask.sum() < 30: continue
        crps.append(dict(method=m, horizon_min=h*5, scope=rgname,
                         CRPS=round(CP.crps_from_quantiles(yt[mask], {t: qd[t][mask] for t in qd}), 3)))

import os; os.makedirs("/tmp/j2out", exist_ok=True)
json.dump(dict(intervals=recs, crps=crps, reliability=rel), open(f"/tmp/j2out/h{h}.json", "w"), default=str)
if h == 1:
    # save bands + alpha traces for example-day / drift figures
    dfp = pd.DataFrame({"y": yt, "pred": pt, "regime": gt,
                        "aci_alpha": alpha_traces["aci"], "aci_regime_alpha": alpha_traces["aci_regime"]},
                       index=te.index)
    lo, hi, _ = CA.aci_run(pt, yt, icp_s, 0.10, GAMMA, day_id); dfp["aci_lo"], dfp["aci_hi"] = lo, hi
    lo, hi = CA.mondrian_cqr_interval(qt[0.05], qt[0.95], gt, CA.mondrian_cqr_fit(yc, qc[0.05], qc[0.95], gc), 0.90, CP.cqr_fit(yc, qc[0.05], qc[0.95]))
    dfp["mcqr_lo"], dfp["mcqr_hi"] = lo, hi
    dfp.to_parquet("/tmp/j2out/h1_bands.parquet")
print(f"h={h*5} done: {len(recs)} interval recs, {len(rel)} reliability recs", flush=True)
