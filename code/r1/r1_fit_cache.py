"""Resumable GBM prediction cache for the final pass.
Fits the point model + all quantile models needed by S1/S2/S3, for one (site, horizon)
per invocation-chunk, caching each component to /tmp/r1cache so it survives restarts.
Usage: python3 r1_fit_cache.py <site: yulara|asp> <h_steps>
"""
import sys, os, time, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils")
import numpy as np, pandas as pd, config as CFG, datasets as D
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

site = sys.argv[1]; h = int(sys.argv[2])
BASEP = "/tmp/base.parquet" if site == "yulara" else "/tmp/base_asp.parquet"
YRS = dict(yulara=(2022, 2023, 2024), asp=(2022, 2023, 2024))[site]
# tau grid: conformal (0.025,0.05,0.10,0.90,0.95,0.975) U dispatch lower bounds
TAUS = [0.025, 0.05, 0.075, 0.0909, 0.10, 0.125, 0.15, 0.1667, 0.20, 0.25, 0.90, 0.95, 0.975]
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)

base = pd.read_parquet(BASEP); d = D.make_xy(base, h).sort_index(); F = D.FEATURES
tr = d[d.year <= YRS[0]]; ca = d[d.year == YRS[1]]; te = d[d.year == YRS[2]]
PRE = f"/tmp/r1cache/{site}_h{h}"; os.makedirs("/tmp/r1cache", exist_ok=True)

def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)

jobs = [("point", None)] + [(f"q{t}", t) for t in TAUS]
t0 = time.time()
for name, tau in jobs:
    fc, ft = f"{PRE}_{name}_c.npy", f"{PRE}_{name}_t.npy"
    if os.path.exists(fc) and os.path.exists(ft):
        continue
    m = gbm() if tau is None else gbm(loss="quantile", quantile=tau)
    m.fit(tr[F].values, tr["y_kt"].values)
    np.save(fc, rec(m.predict(ca[F].values), ca["y_ghi_cs"].values))
    np.save(ft, rec(m.predict(te[F].values), te["y_ghi_cs"].values))
    print(f"  {site} h={h*5}min {name} done @{time.time()-t0:.0f}s", flush=True)

done = all(os.path.exists(f"{PRE}_{n}_c.npy") and os.path.exists(f"{PRE}_{n}_t.npy") for n, _ in jobs)
if done:
    meta = dict(site=site, h_steps=h, h_min=h*5, taus=TAUS,
                n_train=len(tr), n_calib=len(ca), n_test=len(te),
                calib_year=int(YRS[1]), test_year=int(YRS[2]))
    # sanity: point-forecast RMSE on test (must match the first-pass point-forecast table)
    pt = np.load(f"{PRE}_point_t.npy")
    meta["test_rmse"] = float(np.sqrt(np.mean((te["y_ghi"].values - pt) ** 2)))
    json.dump(meta, open(f"{PRE}_meta.json", "w"), indent=1)
    print(f"ALL_DONE {site} h={h*5}min  test RMSE={meta['test_rmse']:.2f} W/m2", flush=True)
else:
    print("PARTIAL", flush=True)
