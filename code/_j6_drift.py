"""J6 — distribution drift over the 9-year record (expanding-window coverage).

For each test year Y: train on years <= Y-2, calibrate on Y-1, test on Y (realistic
rolling deployment). Measures whether conformal coverage holds year-to-year for
marginal ICP, Mondrian (regime), and ACI (online, day-reset). h=1 (5-min), 90%.

Run from the code directory (03_code/ in the working tree, code/ in a release checkout): python3 _j6_drift.py <test_year>   (2019..2024)
Caches /tmp/j6out/drift_{year}.json (skips if present).
"""
import sys, json, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP, conformal_adaptive as CA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

Y = int(sys.argv[1]); h = 1; COV = 0.90
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5)*cs, 0, None)
os.makedirs("/tmp/j6out", exist_ok=True)
outp = f"/tmp/j6out/drift_{Y}.json"
if os.path.exists(outp):
    print(f"{Y} cached"); sys.exit()

base = pd.read_parquet("/tmp/base.parquet")
d = D.make_xy(base, h).sort_index(); F = D.FEATURES
tr = d[d.year <= Y-2]; ca = d[d.year == Y-1]; te = d[d.year == Y]
if len(tr) < 5000 or len(ca) < 1000 or len(te) < 1000:
    json.dump(dict(year=Y, skipped=True), open(outp, "w")); print(f"{Y} insufficient data"); sys.exit()

def gbm(**kw):
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED, **kw)
t0 = time.time()
gp = gbm().fit(tr[F].values, tr["y_kt"].values)
pc = rec(gp.predict(ca[F].values), ca["y_ghi_cs"].values)
pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
yc = ca["y_ghi"].values; yt = te["y_ghi"].values
gc = ca["base_regime"].values; gt = te["base_regime"].values
icp_s = CP.icp_fit(yc, pc); mon_s = CP.mondrian_fit(yc, pc, gc)
day_id = te.index.normalize().values

rows = []
iv = {"icp": CP.icp_interval(pt, icp_s, COV),
      "mondrian": CP.mondrian_interval(pt, gt, mon_s, COV, icp_s),
      "aci": CA.aci_run(pt, yt, icp_s, 1-COV, 0.05, day_id)[:2]}
for m, (lo, hi) in iv.items():
    for scope, mask in [("all", np.ones(len(yt), bool)), ("clear", gt == "clear"),
                        ("transitional", gt == "transitional"), ("cloudy", gt == "cloudy")]:
        if mask.sum() < 30: continue
        rows.append(dict(year=Y, method=m, scope=scope, n=int(mask.sum()),
                         PICP=round(CP.picp(yt[mask], lo[mask], hi[mask]), 4),
                         PINAW=round(CP.pinaw(yt[mask], lo[mask], hi[mask]), 4)))
json.dump(dict(year=Y, n_train=int(len(tr)), n_test=int(len(te)), rows=rows), open(outp, "w"))
print(f"{Y} done in {time.time()-t0:.1f}s (train={len(tr)}, test={len(te)})", flush=True)
