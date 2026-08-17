import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"utils"); sys.path.insert(0,"evaluation")
import numpy as np, pandas as pd
import config as C, datasets as D
from sklearn.ensemble import HistGradientBoostingRegressor
h=int(sys.argv[1])
base=pd.read_parquet("/tmp/base.parquet")
reg=pd.read_parquet(C.DATA_REGIME/"yulara_regimes_5min.parquet")
dc_series=reg["day_class"]
rec=lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
d=D.make_xy(base,h)
d["date"]=d.index.date; d["dclass"]=dc_series.reindex(d.index).values
recs=[]
for ty in [2019,2020,2021,2022,2023,2024]:
    tr=d[d["year"]<ty]; te=d[d["year"]==ty]
    if len(tr)<5000 or len(te)<1000: continue
    g=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.07,max_leaf_nodes=31,
        early_stopping=True,validation_fraction=0.1,n_iter_no_change=12,random_state=C.SEED)
    g.fit(tr[D.FEATURES].values,tr["y_kt"].values)
    cs=te["y_ghi_cs"].values
    tmp=pd.DataFrame({"date":te["date"].values,"dclass":te["dclass"].values,"y":te["y_ghi"].values,
        "persistence":te["base_ghi"].values,"smart_persistence":rec(te["base_kt"].values,cs),
        "gbm":rec(g.predict(te[D.FEATURES].values),cs)})
    for (dt,dcl),grp in tmp.groupby(["date","dclass"]):
        if len(grp)<10 or dcl is None: continue
        for nm in ["persistence","smart_persistence","gbm"]:
            e=grp[nm].values-grp["y"].values
            recs.append(dict(year=ty,date=str(dt),day_class=dcl,model=nm,horizon_min=h*5,
                n=len(grp),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e)))))
    print(f"h={h*5} ty={ty} done",flush=True)
pd.DataFrame(recs).to_csv(f"/tmp/p2out/cv_h{h}.csv",index=False)
print(f"saved cv_h{h}.csv rows={len(recs)}",flush=True)
