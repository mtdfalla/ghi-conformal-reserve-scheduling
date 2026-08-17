"""Incremental, resumable GBM fitter for J2 (and reused by J3/J5 style runs).
Fits point + 6 quantile GBMs for horizon h, caching each component immediately so
the work survives an interrupted run. Run repeatedly until 'ALL_DONE';
on completion it assembles /tmp/j2pred_h{h}.npz for the evaluation driver.

Usage: python3 _j2_fit.py <h_steps>
"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils")
import numpy as np, pandas as pd, config as CFG, datasets as D
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

h = int(sys.argv[1]); rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5)*cs, 0, None)
base = pd.read_parquet("/tmp/base.parquet"); d = D.make_xy(base, h).sort_index(); F = D.FEATURES
tr = d[d.year <= 2022]; ca = d[d.year == 2023]; te = d[d.year == 2024]
PRE = f"/tmp/j2c_h{h}"
def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)
jobs = [("point", None)] + [(f"q{t}", t) for t in [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]]
t0 = time.time(); BUDGET = 30
for name, tau in jobs:
    fc, ft = f"{PRE}_{name}_c.npy", f"{PRE}_{name}_t.npy"
    if os.path.exists(fc) and os.path.exists(ft): continue
    if time.time() - t0 > BUDGET:
        print("budget hit; rerun to continue", flush=True); break
    m = gbm() if tau is None else gbm(loss="quantile", quantile=tau)
    m.fit(tr[F].values, tr["y_kt"].values)
    np.save(fc, rec(m.predict(ca[F].values), ca["y_ghi_cs"].values))
    np.save(ft, rec(m.predict(te[F].values), te["y_ghi_cs"].values))
    print(f"{name} done @{time.time()-t0:.0f}s", flush=True)
done = all(os.path.exists(f"{PRE}_{n}_c.npy") and os.path.exists(f"{PRE}_{n}_t.npy") for n, _ in jobs)
if done:
    pc = np.load(f"{PRE}_point_c.npy"); pt = np.load(f"{PRE}_point_t.npy")
    qc = {t: np.load(f"{PRE}_q{t}_c.npy") for _, t in jobs if t is not None}
    qt = {t: np.load(f"{PRE}_q{t}_t.npy") for _, t in jobs if t is not None}
    np.savez(f"/tmp/j2pred_h{h}.npz", pc=pc, pt=pt, qc=qc, qt=qt)
    print("ALL_DONE -> assembled npz", flush=True)
else:
    print("PARTIAL", flush=True)
