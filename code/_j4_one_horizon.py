"""J4 — multivariate feature study (2023-24) for ONE horizon.

Question: does adding VALID meteorology (temp, wind, pressure, rain) improve
point / interval performance over the univariate kt-space model? Honest, leakage-
free, same train/test for every feature set (only the feature columns differ).

Window: aux met sensors are reliable only 2023-24 (P1 finding). So:
  train = 2023 first 80% (chronological), calib = 2023 last 20%, test = 2024.
This isolates the *feature-set* effect (not the training-period effect).

Run from 03_code: python3 _j4_one_horizon.py <h_steps>   (h in 1,3,6,12)
Writes /tmp/j4out/h{h}.json. Aggregated by _j4_aggregate.py.
"""
import sys, json, time, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils"); sys.path.insert(0, "evaluation"); sys.path.insert(0, "conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP, conformal_adaptive as CA
from metrics import diebold_mariano
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

h = int(sys.argv[1])
rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)

# ---- build met-augmented frame for 2023-24 ----
base = pd.read_parquet("/tmp/base.parquet")
clean = pd.read_parquet(CFG.DATA_CLEAN / "yulara_clean_5min.parquet")
met_cols = ["temp_air", "wind_spd", "wind_max", "wind_dir", "pressure", "rain_day", "temp_mod1"]
df = base.join(clean[met_cols], how="left")
df = df[df["year"].isin([2023, 2024])].copy()
df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))

UNI = list(D.FEATURES)
SETS = {
    "uni":   UNI,
    "+temp": UNI + ["temp_air", "temp_mod1"],
    "+wind": UNI + ["wind_spd", "wind_max", "wind_dir_sin", "wind_dir_cos"],
    "all":   UNI + ["temp_air", "temp_mod1", "wind_spd", "wind_max", "wind_dir_sin", "wind_dir_cos", "pressure", "rain_day"],
}
ALLF = SETS["all"]

# supervised frame (validity requires ALL candidate features present)
d = df.copy()
d["y_kt"] = d["kt"].shift(-h); d["y_ghi"] = d["ghi"].shift(-h)
d["y_ghi_cs"] = d["ghi_cs"].shift(-h); d["y_is_day"] = d["is_day"].shift(-h)
d["y_imputed"] = d["ghi_imputed"].shift(-h); d["base_regime"] = d["regime"]
need = ALLF + ["y_kt", "y_ghi", "y_ghi_cs"]
valid = (d["is_day"] & d["y_is_day"].fillna(False) & (~d["y_imputed"].fillna(True)) & d[need].notna().all(axis=1))
d = d[valid].sort_index()

d23 = d[d.year == 2023]; te = d[d.year == 2024]
cut = d23.index[int(len(d23) * 0.8)]
tr = d23[d23.index < cut]; ca = d23[d23.index >= cut]
yc = ca["y_ghi"].values; yt = te["y_ghi"].values
gc = ca["base_regime"].values; gt = te["base_regime"].values
csref = np.maximum(te["y_ghi_cs"].values, 50.0)

def gbm(**kw):
    return HGB(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=12, random_state=CFG.SEED, **kw)

SCOPES = [("all", np.ones(len(yt), bool)), ("clear", gt == "clear"),
          ("transitional", gt == "transitional"), ("cloudy", gt == "cloudy")]
t0 = time.time()
point = {}; recs = []; crps = []; pin = {}
preds_te = {}
for name, F in SETS.items():
    gp = gbm().fit(tr[F].values, tr["y_kt"].values)
    pt = rec(gp.predict(te[F].values), te["y_ghi_cs"].values)
    preds_te[name] = pt
    # point RMSE by scope
    for rgname, mask in SCOPES:
        if mask.sum() < 30: continue
        e = yt[mask] - pt[mask]
        recs.append(dict(set=name, horizon_min=h*5, scope=rgname, metric="RMSE",
                         value=round(float(np.sqrt(np.mean(e**2))), 3), n=int(mask.sum())))
        recs.append(dict(set=name, horizon_min=h*5, scope=rgname, metric="MAE",
                         value=round(float(np.mean(np.abs(e))), 3), n=int(mask.sum())))
    # interval via Mondrian-CQR @90%
    qlo_c = {}; qhi_c = {}; qlo_t = {}; qhi_t = {}
    for tau, store_c, store_t in [(0.05, qlo_c, qlo_t), (0.95, qhi_c, qhi_t)]:
        gq = gbm(loss="quantile", quantile=tau).fit(tr[F].values, tr["y_kt"].values)
        store_c[tau] = rec(gq.predict(ca[F].values), ca["y_ghi_cs"].values)
        store_t[tau] = rec(gq.predict(te[F].values), te["y_ghi_cs"].values)
    mcqr_s = CA.mondrian_cqr_fit(yc, qlo_c[0.05], qhi_c[0.95], gc)
    fb = CP.cqr_fit(yc, qlo_c[0.05], qhi_c[0.95])
    lo, hi = CA.mondrian_cqr_interval(qlo_t[0.05], qhi_t[0.95], gt, mcqr_s, 0.90, fb)
    for rgname, mask in SCOPES:
        if mask.sum() < 30: continue
        recs.append(dict(set=name, horizon_min=h*5, scope=rgname, metric="PICP90",
                         value=round(CP.picp(yt[mask], lo[mask], hi[mask]), 4), n=int(mask.sum())))
        recs.append(dict(set=name, horizon_min=h*5, scope=rgname, metric="PINAW90",
                         value=round(CP.pinaw(yt[mask], lo[mask], hi[mask]), 4), n=int(mask.sum())))
    qd = {0.05: lo, 0.95: hi, 0.5: pt}
    crps.append(dict(set=name, horizon_min=h*5, scope="all",
                     CRPS=round(CP.crps_from_quantiles(yt, qd), 3)))

# DM tests: each multivariate set vs univariate (aligned on test index)
dm = []
for name in ["+temp", "+wind", "all"]:
    stat, p = diebold_mariano(yt, preds_te["uni"], preds_te[name], h=h)
    dm.append(dict(set=name, horizon_min=h*5, dm_stat=round(float(stat), 3), p_value=float(p)))

os.makedirs("/tmp/j4out", exist_ok=True)
json.dump(dict(metrics=recs, crps=crps, dm=dm), open(f"/tmp/j4out/h{h}.json", "w"), default=str)
print(f"h={h*5} done in {time.time()-t0:.1f}s | train={len(tr)} calib={len(ca)} test={len(te)}", flush=True)
