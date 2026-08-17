"""Phase 2: persistence, smart persistence, gradient boosting.

Predict kt(t+h) -> reconstruct GHI = kt_hat * GHI_clearsky(t+h).
Evaluate on test (2024), overall and by base-time weather regime.
Saves point-metric tables, DM tests, and test predictions (for Phase 3).
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent/"utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent/"evaluation"))
import numpy as np, pandas as pd
import config as C, datasets as D, metrics as M
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONS = [1,3,6,12]   # 5,15,30,60 min
def log(m): print(f"[p2] {m}", flush=True)

def reconstruct(kt_pred, ghi_cs):
    return np.clip(np.clip(kt_pred,0,1.5)*ghi_cs, 0, None)

base = D.build_base()
rows=[]; rows_reg=[]; dm_rows=[]
for h in HORIZONS:
    d = D.make_xy(base, h); tr,ca,te = D.split_years(d)
    Xtr,ytr = tr[D.FEATURES].values, tr["y_kt"].values
    Xte = te[D.FEATURES].values
    y = te["y_ghi"].values; cs = te["y_ghi_cs"].values; reg = te["base_regime"].values

    # --- models -> GHI predictions on test ---
    pred = {}
    pred["persistence"]       = te["base_ghi"].values
    pred["smart_persistence"] = reconstruct(te["base_kt"].values, cs)
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
            max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1,
            random_state=C.SEED)
    gbm.fit(Xtr, ytr)
    pred["gbm"] = reconstruct(gbm.predict(Xte), cs)
    log(f"h={h}: trained GBM ({gbm.n_iter_} iters); test n={len(te)}")

    ref = pred["smart_persistence"]
    for name,p in pred.items():
        m = M.all_metrics(y,p,ref=ref); m.update(model=name, horizon_min=h*5)
        rows.append(m)
        for rg in ["clear","transitional","cloudy"]:
            mask = reg==rg
            if mask.sum()>30:
                mm=M.all_metrics(y[mask],p[mask],ref=ref[mask])
                mm.update(model=name,horizon_min=h*5,regime=rg); rows_reg.append(mm)
    # DM tests
    for a,b in [("gbm","smart_persistence"),("smart_persistence","persistence")]:
        dm,pv=M.diebold_mariano(y,pred[a],pred[b],h=h)
        dm_rows.append(dict(horizon_min=h*5,model_A=a,model_B=b,DM=round(dm,3),
                            p_value=float(f"{pv:.2e}"),A_better=bool(dm<0)))
    # save test predictions for Phase 3 (conformal)
    out=pd.DataFrame({"y_ghi":y,"ghi_cs":cs,"regime":reg,
        "persistence":pred["persistence"],"smart_persistence":pred["smart_persistence"],
        "gbm":pred["gbm"]}, index=te.index)
    out.to_parquet(C.DATA_INTERIM/f"p2_test_pred_h{h}.parquet")

res=pd.DataFrame(rows)[["model","horizon_min","n","MAE","RMSE","nRMSE","R2","skill_vs_ref"]]
res=res.round(4); res.to_csv(C.TAB/"p2_point_metrics.csv",index=False)
resr=pd.DataFrame(rows_reg)[["model","horizon_min","regime","n","MAE","RMSE","R2","skill_vs_ref"]].round(4)
resr.to_csv(C.TAB/"p2_point_metrics_by_regime.csv",index=False)
dmt=pd.DataFrame(dm_rows); dmt.to_csv(C.TAB/"p2_dm_tests.csv",index=False)
json.dump({"overall":rows,"by_regime":rows_reg,"dm":dm_rows},
          open(C.MET/"p2_point_metrics.json","w"),indent=2,default=str)

log("=== OVERALL (test 2024), GHI W/m^2 ===")
print(res.to_string(index=False))
log("=== BY REGIME (RMSE) ===")
piv=resr.pivot_table(index=["model","horizon_min"],columns="regime",values="RMSE")
print(piv.round(2).to_string())
log("=== DIEBOLD-MARIANO ===")
print(dmt.to_string(index=False))
log("done.")
