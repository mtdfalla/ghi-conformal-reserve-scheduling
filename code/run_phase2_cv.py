"""Phase 2: expanding-window cross-validation across years + per-day errors.
For test_year in 2019..2024: train GBM on all prior years (>=2016), evaluate on
test_year. Produces per-day RMSE/MAE per (year, day, day_class, model, horizon)
-> tidy CSV for ANOVA (multiple conditions + significance).
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/"utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent/"evaluation"))
import numpy as np, pandas as pd
import config as C, datasets as D, metrics as M
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

base = pd.read_parquet("/tmp/base.parquet")
reg = pd.read_parquet(C.DATA_REGIME/"yulara_regimes_5min.parquet")
dayclass = reg["day_class"]; dayclass.index = reg.index
rec = lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
TEST_YEARS=[2019,2020,2021,2022,2023,2024]; HOR=[1,3,6,12]
recs=[]
for h in HOR:
    d=D.make_xy(base,h)
    d["date"]=d.index.date; d["dclass"]=dayclass.reindex(d.index).values
    for ty in TEST_YEARS:
        tr=d[d["year"]<ty]; te=d[d["year"]==ty]
        if len(tr)<5000 or len(te)<1000: continue
        g=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.07,max_leaf_nodes=31,
            early_stopping=True,validation_fraction=0.1,n_iter_no_change=12,random_state=C.SEED)
        g.fit(tr[D.FEATURES].values,tr["y_kt"].values)
        y=te["y_ghi"].values; cs=te["y_ghi_cs"].values
        preds={"persistence":te["base_ghi"].values,
               "smart_persistence":rec(te["base_kt"].values,cs),
               "gbm":rec(g.predict(te[D.FEATURES].values),cs)}
        tmp=pd.DataFrame({"date":te["date"].values,"dclass":te["dclass"].values,"y":y})
        for nm,p in preds.items(): tmp[nm]=p
        # per-day errors
        for (dt,dc),grp in tmp.groupby(["date","dclass"]):
            if len(grp)<10 or dc is None: continue
            for nm in preds:
                e=grp[nm].values-grp["y"].values
                recs.append(dict(year=ty,date=str(dt),day_class=dc,model=nm,horizon_min=h*5,
                                 n=len(grp),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e)))))
        print(f"h={h*5} test={ty}: train={len(tr)} test={len(te)} gbmRMSE={M.rmse(y,preds['gbm']):.1f}",flush=True)
df=pd.DataFrame(recs)
df.to_csv("/tmp/p2out/p2_cv_perday_errors.csv",index=False)
print("TOTAL per-day records:",len(df))
