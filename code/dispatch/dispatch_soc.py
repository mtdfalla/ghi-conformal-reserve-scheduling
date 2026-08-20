"""J5 — Rolling cost-of-uncertainty reserve dispatch with battery SoC + CVaR.

WHAT THIS MODEL IS, AND WHAT IT IS NOT
--------------------------------------
This is a **load-independent** reserve dispatch (, extending the conference
per-step newsvendor of). It is NOT a unit-commitment model and it does not serve
a site load: no load series is modelled, assumed or required anywhere in the cost. A
unit commitment would need a real site-load series, which this project does not have,
and assuming one would make the reported cost a property of that assumption rather than
of the forecast.

What is modelled is the **cost of forecast uncertainty alone**. The operator pre-commits
backup against a PV quantity it counts on, PV = f(GHI) through the binned-median GHI->PV
map; the realized PV then differs from the counted PV, and only that imbalance is priced:

  realized < counted  ->  deficit, buffered by the battery within its SoC/power limits,
                          the remainder charged at the fast-diesel/VOLL premium c_u;
  realized > counted  ->  surplus, stored in the battery within its remaining room,
                          the remainder charged as over-procured reserve at c_o.

The battery (SoC dynamics, one-way efficiency eta, power and energy limits) is the only
state carried through time. SoC resets each morning, so each day is an independent cost
draw and the DAILY operating cost is the CVaR unit. The oracle (count on the realized
value) costs exactly zero, which is what makes "value captured" well posed.

The single decision under uncertainty is therefore HOW MUCH PV TO COUNT ON: a one-sided
lower conformal bound at reserve level rho. Calibrated, regime-aware quantiles set that
bound correctly within each weather regime -> lower expected cost AND lower tail risk.

Policies: deterministic (point), ICP, Mondrian, Mondrian-CQR, oracle (perfect).
For each policy we sweep the reserve level rho -> (E[daily cost], CVaR_0.95) frontier.

NOTE ON RESERVE-LEVEL SELECTION. This Phase-6 module selects the reported rho by
`idxmin()` over the 2024 TEST-year cost (headline block below), so its headline numbers
contain test-set tuning. The final pass corrects this. The
The final pass's `code/r1/r1_dispatch.py` re-implements the same physical dispatch with rho
selected on a set that excludes 2024, and every dispatch number in the article
comes from that module, not from this one. This file is kept unchanged in
substance as the record of what was originally submitted.

Run from the code directory (03_code/ in the working tree, code/ in a release checkout): python3 dispatch/dispatch_soc.py <h_steps>   (1 or 6)
Needs /tmp/base.parquet and /tmp/ghi_pv_map.npz. Writes /tmp/j5out/h{h}.json.
"""
import sys, json, time, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP
import conformal_adaptive as CA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

h = int(sys.argv[1]); DT = 5/60.0
C_O = 0.30; RATIO = 10.0; C_U = C_O*RATIO           # committed-diesel vs fast/VOLL premium
C_CURTAIL = 0.0
RHO_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]          # reserve levels for the frontier
# battery configs: (E_max kWh, P_max kW); first is the default (-scale ~1 MW PV)
BATT = {"default": (600.0, 400.0), "small": (250.0, 250.0), "large": (1500.0, 600.0)}
ETA = 0.949                                          # one-way (round-trip ~0.90)
SOC_MIN, SOC_MAX = 0.10, 1.0
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5)*cs, 0, None)

m = np.load("/tmp/ghi_pv_map.npz"); CENT, VALS, CAP = m["centers"], m["vals"], float(m["cap"])
f = lambda ghi: np.clip(np.interp(ghi, CENT, VALS, left=0, right=CAP), 0, CAP)

base = pd.read_parquet("/tmp/base.parquet")
d = D.make_xy(base, h).sort_index()
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]

def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)
LQTAUS = [0.05, 0.1, 0.2, 0.3, 0.5]
t0 = time.time()
PRED = f"/tmp/j5_pred_h{h}.npz"
yc = ca["y_ghi"].values; yt = te["y_ghi"].values
gc = ca["base_regime"].values; gt = te["base_regime"].values
if os.path.exists(PRED):
    z = np.load(PRED, allow_pickle=True)
    pc, pt = z["pc"], z["pt"]; qc = z["qc"].item(); qt = z["qt"].item()
    print(f"h={h*5} loaded cached preds | te={len(te)}", flush=True)
else:
    gp = gbm().fit(tr[D.FEATURES].values, tr["y_kt"].values)
    pc = rec(gp.predict(ca[D.FEATURES].values), ca["y_ghi_cs"].values)
    pt = rec(gp.predict(te[D.FEATURES].values), te["y_ghi_cs"].values)
    qc = {}; qt = {}
    for tau in LQTAUS:
        gq = gbm(loss="quantile", quantile=tau).fit(tr[D.FEATURES].values, tr["y_kt"].values)
        qc[tau] = rec(gq.predict(ca[D.FEATURES].values), ca["y_ghi_cs"].values)
        qt[tau] = rec(gq.predict(te[D.FEATURES].values), te["y_ghi_cs"].values)
    np.savez(PRED, pc=pc, pt=pt, qc=qc, qt=qt)
    print(f"h={h*5} fitted+cached in {time.time()-t0:.1f}s | te={len(te)}", flush=True)

# signed calibration residuals (e = y - pred) for one-sided conformal lower bounds
e_all = yc - pc
e_by_grp = {g: e_all[gc == g] for g in np.unique(gc)}

def counted_ghi(policy, rho):
    """GHI the operator counts on (lower bound) for reserve level rho."""
    if policy == "deterministic":
        return pt.copy()
    if policy == "oracle":
        return yt.copy()
    if policy == "icp":
        off = np.quantile(e_all, 1-rho)             # negative -> conservative
        return np.clip(pt + off, 0, None)
    if policy == "mondrian":
        out = np.empty(len(pt))
        for g in np.unique(gt):
            off = np.quantile(e_by_grp.get(g, e_all), 1-rho)
            out[gt == g] = np.clip(pt[gt == g] + off, 0, None)
        return out
    if policy in ("cqr", "mondrian_cqr"):
        tau = round(1-rho, 3); tau = min(LQTAUS, key=lambda z: abs(z-tau))
        qlo_c = qc[tau]; qlo_t = qt[tau]
        if policy == "cqr":
            s = qlo_c - yc                          # one-sided CQR score (lower)
            adj = np.quantile(s, rho)
            return np.clip(qlo_t - adj, 0, None)
        out = np.empty(len(pt))
        for g in np.unique(gt):
            s = (qlo_c - yc)[gc == g]
            adj = np.quantile(s, rho) if len(s) > 30 else np.quantile(qlo_c-yc, rho)
            out[gt == g] = np.clip(qlo_t[gt == g] - adj, 0, None)
        return out
    raise ValueError(policy)

# day grouping (SoC resets each morning); precompute day index groups ONCE
day = te.index.normalize().values
pv_act = f(yt)
_udays = np.unique(day)
DAY_GROUPS = [np.where(day == dv)[0] for dv in _udays]

def simulate(counted_pv, E_max, P_max):
    """Cost-of-forecast-uncertainty reserve dispatch with a FINITE battery (SoC).

    The operator counts on PV = counted_pv and schedules backup accordingly. Realized
    PV = pv_act. Battery stores PV surplus (pva>pvq) to cover later deficits (pva<pvq);
    deficits beyond the battery -> fast diesel/VOLL (c_u); surplus beyond battery room
    -> wasted over-procured reserve (c_o). Oracle (counted=actual) => zero cost (lower
    bound). SoC carries across the day (resets each morning); daily cost is the CVaR unit.
    """
    pv_c = f(counted_pv)
    Pstep = P_max
    daily = []; soc_lo = 1e9; soc_hi = -1e9
    for idxs in DAY_GROUPS:
        soc = SOC_MIN*E_max          # start each day empty (charged only by PV surplus)
        cost = 0.0
        for i in idxs:
            diff = pv_act[i] - pv_c[i]      # >0 surplus vs counted, <0 deficit
            if diff < 0:                    # PV underdelivered vs counted -> deficit
                deficit = -diff
                dis = min(deficit, Pstep, (soc - SOC_MIN*E_max)*ETA/DT); dis = max(dis, 0.0)
                soc -= dis*DT/ETA
                unserved = deficit - dis
                cost += C_U*unserved*DT
            else:                            # PV over-delivered -> store, else over-procured
                surplus = diff
                chg = min(surplus, Pstep, (SOC_MAX*E_max - soc)/(ETA*DT)); chg = max(chg, 0.0)
                soc += chg*ETA*DT
                wasted = surplus - chg
                cost += C_O*wasted*DT
            soc_lo = min(soc_lo, soc); soc_hi = max(soc_hi, soc)
        daily.append(cost)
    dc = np.array(daily)
    return dc, soc_lo/E_max, soc_hi/E_max

def cvar(x, alpha=0.95):
    k = max(1, int(np.ceil((1-alpha)*len(x))))
    return float(np.mean(np.sort(x)[::-1][:k]))

POLICIES = ["deterministic", "icp", "mondrian", "mondrian_cqr", "oracle"]
results = []; frontier = []; soc_check = []
E0, P0 = BATT["default"]
# oracle/deterministic are single points; CP policies sweep rho
for pol in POLICIES:
    rhos = [None] if pol in ("deterministic", "oracle") else RHO_GRID
    for rho in rhos:
        cp = counted_ghi(pol, rho if rho is not None else 0.5)
        dc, slo, shi = simulate(cp, E0, P0)
        rowf = dict(policy=pol, rho=rho, horizon_min=h*5, E_max=E0, P_max=P0,
                    mean_daily=round(float(dc.mean()), 2), cvar95_daily=round(cvar(dc), 2),
                    total=round(float(dc.sum()), 1), soc_min=round(slo, 3), soc_max=round(shi, 3))
        frontier.append(rowf)

# headline: best rho per CP policy by mean cost; value captured vs det/oracle
fr = pd.DataFrame(frontier)
det_mean = fr[fr.policy == "deterministic"]["mean_daily"].iloc[0]
det_cvar = fr[fr.policy == "deterministic"]["cvar95_daily"].iloc[0]
ora_mean = fr[fr.policy == "oracle"]["mean_daily"].iloc[0]
ora_cvar = fr[fr.policy == "oracle"]["cvar95_daily"].iloc[0]
for pol in ["icp", "mondrian", "mondrian_cqr"]:
    sub = fr[fr.policy == pol]
    best = sub.loc[sub["mean_daily"].idxmin()]
    vc = (det_mean - best["mean_daily"])/(det_mean - ora_mean) if det_mean > ora_mean else np.nan
    vc_cvar = (det_cvar - best["cvar95_daily"])/(det_cvar - ora_cvar) if det_cvar > ora_cvar else np.nan
    results.append(dict(policy=pol, horizon_min=h*5, best_rho=float(best["rho"]),
                        mean_daily=best["mean_daily"], cvar95_daily=best["cvar95_daily"],
                        value_captured_mean=round(float(vc), 3), value_captured_cvar=round(float(vc_cvar), 3),
                        soc_min=best["soc_min"], soc_max=best["soc_max"]))

# battery sensitivity at headline policy (mondrian, best rho) and deterministic
sens = []
mon_best_rho = fr[fr.policy == "mondrian"].loc[fr[fr.policy == "mondrian"]["mean_daily"].idxmin(), "rho"]
for name, (E, P) in BATT.items():
    for pol, rho in [("deterministic", None), ("mondrian", mon_best_rho), ("oracle", None)]:
        cp = counted_ghi(pol, rho if rho is not None else 0.5)
        dc, slo, shi = simulate(cp, E, P)
        sens.append(dict(batt=name, E_max=E, P_max=P, policy=pol, horizon_min=h*5,
                         mean_daily=round(float(dc.mean()), 2), cvar95_daily=round(cvar(dc), 2),
                         soc_min=round(slo, 3), soc_max=round(shi, 3)))

os.makedirs("/tmp/j5out", exist_ok=True)
json.dump(dict(frontier=frontier, headline=results, sensitivity=sens,
               params=dict(C_O=C_O, RATIO=RATIO, ETA=ETA, SOC_MIN=SOC_MIN, CAP=CAP)),
          open(f"/tmp/j5out/h{h}.json", "w"), default=str)
print(f"h={h*5}: det mean/cvar={det_mean}/{det_cvar}, oracle={ora_mean}/{ora_cvar}")
for r in results:
    print("  ", r["policy"], "VC_mean=%.3f VC_cvar=%.3f best_rho=%.2f soc[%.2f,%.2f]" % (
        r["value_captured_mean"], r["value_captured_cvar"], r["best_rho"], r["soc_min"], r["soc_max"]))
print(f"done in {time.time()-t0:.1f}s", flush=True)
