import sys,json,time,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"utils"); sys.path.insert(0,"evaluation")
import numpy as np, pandas as pd
import config as C, datasets as D, metrics as M
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
h=int(sys.argv[1])
base=pd.read_parquet("/tmp/base.parquet")
d=D.make_xy(base,h); tr,ca,te=D.split_years(d)
rec=lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
y=te["y_ghi"].values; cs=te["y_ghi_cs"].values; reg=te["base_regime"].values
pred={"persistence":te["base_ghi"].values,
      "smart_persistence":rec(te["base_kt"].values,cs)}
t0=time.time()
g=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.07,max_leaf_nodes=31,
   early_stopping=True,validation_fraction=0.1,n_iter_no_change=12,random_state=C.SEED)
g.fit(tr[D.FEATURES].values,tr["y_kt"].values)
pred["gbm"]=rec(g.predict(te[D.FEATURES].values),cs)
ar=make_pipeline(StandardScaler(),Ridge(alpha=1.0)); ar.fit(tr[D.FEATURES].values,tr["y_kt"].values)
pred["linear_ar"]=rec(ar.predict(te[D.FEATURES].values),cs)
ref=pred["smart_persistence"]
rows=[];rows_reg=[];dm=[]
for nm,p in pred.items():
    m=M.all_metrics(y,p,ref=ref); m.update(model=nm,horizon_min=h*5); rows.append(m)
    for rg in ["clear","transitional","cloudy"]:
        mk=reg==rg
        if mk.sum()>30:
            mm=M.all_metrics(y[mk],p[mk],ref=ref[mk]); mm.update(model=nm,horizon_min=h*5,regime=rg); rows_reg.append(mm)
for a,b in [("gbm","smart_persistence"),("smart_persistence","persistence")]:
    s,pv=M.diebold_mariano(y,pred[a],pred[b],h=h)
    dm.append(dict(horizon_min=h*5,model_A=a,model_B=b,DM=round(s,3),p_value=float(f"{pv:.2e}"),A_better=bool(s<0)))
pd.DataFrame({"y_ghi":y,"ghi_cs":cs,"regime":reg,**{k:pred[k] for k in pred}},index=te.index).to_parquet(f"/tmp/p2out/interim/p2_test_pred_h{h}.parquet")
json.dump(dict(rows=rows,rows_reg=rows_reg,dm=dm),open(f"/tmp/p2out/h{h}.json","w"),default=str)
print(f"h={h} done in {time.time()-t0:.1f}s iters={g.n_iter_} | GBM RMSE={M.rmse(y,pred['gbm']):.2f} skill={M.skill(y,pred['gbm'],ref):.3f}",flush=True)
