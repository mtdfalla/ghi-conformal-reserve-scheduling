"""J6 — ablations: calibration-set size and feature-set. Train <=2022 / test 2024, h=1.

Modes:
  python3 _j6_ablations.py calib            # calibration-set-size ablation (1 GBM)
  python3 _j6_ablations.py feat <setname>   # one feature set (point + interval)
                                            # sets: full, no_roll, lags_only, minimal
Writes /tmp/j6out/calib.json or /tmp/j6out/feat_<set>.json.
"""
import sys, json, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "evaluation"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP, conformal_adaptive as CA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

mode = sys.argv[1]; h = 1; COV = 0.90
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5)*cs, 0, None)
os.makedirs("/tmp/j6out", exist_ok=True)
base = pd.read_parquet("/tmp/base.parquet")
d = D.make_xy(base, h).sort_index()
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]
yt = te["y_ghi"].values; gt = te["base_regime"].values
gc = ca["base_regime"].values
def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)

if mode == "calib":
    F = D.FEATURES
    gp = gbm().fit(tr[F].values, tr["y_kt"].values)
    pc = rec(gp.predict(ca[F].values), ca["y_ghi_cs"].values)
    pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
    yc = ca["y_ghi"].values
    rows = []
    last = ca.index.max()
    for months in [0.5, 1, 2, 3, 6, 9, 12]:
        cutoff = last - pd.DateOffset(days=int(months*30))
        msk = ca.index >= cutoff
        if msk.sum() < 100: continue
        icp_s = CP.icp_fit(yc[msk], pc[msk]); mon_s = CP.mondrian_fit(yc[msk], pc[msk], gc[msk])
        for m, (lo, hi) in {"icp": CP.icp_interval(pt, icp_s, COV),
                             "mondrian": CP.mondrian_interval(pt, gt, mon_s, COV, icp_s)}.items():
            rows.append(dict(calib_months=months, calib_n=int(msk.sum()), method=m,
                             PICP=round(CP.picp(yt, lo, hi), 4), PINAW=round(CP.pinaw(yt, lo, hi), 4)))
    json.dump(rows, open("/tmp/j6out/calib.json", "w"))
    print(pd.DataFrame(rows).to_string(index=False))

elif mode == "feat":
    setname = sys.argv[2]
    lags = [f"kt_l{L}" for L in D.LAGS]
    SETS = {"full": D.FEATURES,
            "no_roll": [c for c in D.FEATURES if c not in ("kt_rmean", "kt_rstd")],
            "lags_only": lags,
            "minimal": ["kt_l0", "kt_l1", "cosz"]}
    F = SETS[setname]
    t0 = time.time()
    gp = gbm().fit(tr[F].values, tr["y_kt"].values)
    pc = rec(gp.predict(ca[F].values), ca["y_ghi_cs"].values)
    pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
    yc = ca["y_ghi"].values
    e = yt - pt
    rmse = float(np.sqrt(np.mean(e**2)))
    mon_s = CP.mondrian_fit(yc, pc, gc)
    lo, hi = CP.mondrian_interval(pt, gt, mon_s, COV, CP.icp_fit(yc, pc))
    out = dict(set=setname, n_features=len(F), RMSE=round(rmse, 2),
               PICP_mondrian=round(CP.picp(yt, lo, hi), 4), PINAW_mondrian=round(CP.pinaw(yt, lo, hi), 4),
               time_s=round(time.time()-t0, 1))
    json.dump(out, open(f"/tmp/j6out/feat_{setname}.json", "w"))
    print(out)
