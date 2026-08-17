"""Final pass — fit the two quantile levels the S0 cache is missing.

The theta grid is 0.75:0.025:0.95, so the CQR policies need lower quantiles at
tau = 1-theta in {0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25}. The S0 cache
(`r1_fit_cache.py`) holds every one of these except 0.175 and 0.225. Rather than snap
those two grid points to a neighbouring tau (which would make the grid uneven across
policies), we fit them here with identical hyper-parameters and cache them alongside.

Usage: python3 r1/r1_fit_extra_taus.py <site: yulara|asp> <h_steps>
"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils")
import numpy as np, pandas as pd, config as CFG, datasets as D
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

site = sys.argv[1]; h = int(sys.argv[2])
BASEP = "/tmp/base.parquet" if site == "yulara" else "/tmp/base_asp.parquet"
TAUS = [0.175, 0.225]
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)

base = pd.read_parquet(BASEP); d = D.make_xy(base, h).sort_index(); F = D.FEATURES
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]
os.makedirs("/tmp/r1cache_extra", exist_ok=True)
PRE = f"/tmp/r1cache_extra/{site}_h{h}"
t0 = time.time()
for tau in TAUS:
    fc, ft = f"{PRE}_q{tau}_c.npy", f"{PRE}_q{tau}_t.npy"
    if os.path.exists(fc) and os.path.exists(ft):
        continue
    m = HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED,
            loss="quantile", quantile=tau)
    m.fit(tr[F].values, tr["y_kt"].values)
    np.save(fc, rec(m.predict(ca[F].values), ca["y_ghi_cs"].values))
    np.save(ft, rec(m.predict(te[F].values), te["y_ghi_cs"].values))
    print(f"  {site} h={h*5}min q{tau} done @{time.time()-t0:.0f}s", flush=True)
print(f"EXTRA_DONE {site} h={h*5}min", flush=True)
