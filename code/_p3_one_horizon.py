import sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"utils"); sys.path.insert(0,"evaluation"); sys.path.insert(0,"conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
h=int(sys.argv[1]); COVS=[0.80,0.90,0.95]
rec=lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
base=pd.read_parquet("/tmp/base.parquet")
d=D.make_xy(base,h); F=D.FEATURES
tr=d[d.year<=2022]; ca=d[d.year==2023]; te=d[d.year==2024]
def gbm(**kw): return HGB(max_iter=150,learning_rate=0.08,max_leaf_nodes=31,
    early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=CFG.SEED,**kw)
t0=time.time()
gp=gbm().fit(tr[F].values,tr["y_kt"].values)
pc=rec(gp.predict(ca[F].values),ca["y_ghi_cs"].values)
pt=rec(gp.predict(te[F].values),te["y_ghi_cs"].values)
yc=ca["y_ghi"].values; yt=te["y_ghi"].values
csc=np.maximum(ca["y_ghi_cs"].values,50.0); cst=np.maximum(te["y_ghi_cs"].values,50.0)
gc=ca["base_regime"].values; gt=te["base_regime"].values
taus=sorted({0.025,0.05,0.10,0.90,0.95,0.975})
qc={};qt={}
for tau in taus:
    gq=gbm(loss="quantile",quantile=tau).fit(tr[F].values,tr["y_kt"].values)
    qc[tau]=rec(gq.predict(ca[F].values),ca["y_ghi_cs"].values)
    qt[tau]=rec(gq.predict(te[F].values),te["y_ghi_cs"].values)
print(f"h={h*5} fitted point+{len(taus)}q GBMs in {time.time()-t0:.1f}s",flush=True)

icp_s=CP.icp_fit(yc,pc); icpn_s=CP.icpn_fit(yc,pc,csc); mon_s=CP.mondrian_fit(yc,pc,gc)
SCOPES=[("all",np.ones(len(yt),bool)),("clear",gt=="clear"),
        ("transitional",gt=="transitional"),("cloudy",gt=="cloudy")]
recs=[]; qstore={m:{} for m in ["icp","icp_norm","mondrian","cqr"]}
for cov in COVS:
    a=1-cov; lo_tau=round(a/2,3); hi_tau=round(1-a/2,3)
    iv={}
    iv["icp"]=CP.icp_interval(pt,icp_s,cov)
    iv["icp_norm"]=CP.icpn_interval(pt,cst,icpn_s,cov)
    iv["mondrian"]=CP.mondrian_interval(pt,gt,mon_s,cov,icp_s)
    iv["cqr"]=CP.cqr_interval(qt[lo_tau],qt[hi_tau],CP.cqr_fit(yc,qc[lo_tau],qc[hi_tau]),cov)
    for m,(lo,hi) in iv.items():
        qstore[m][lo_tau]=lo; qstore[m][hi_tau]=hi
        for rgname,mask in SCOPES:
            if mask.sum()<30: continue
            recs.append(dict(method=m,horizon_min=h*5,nominal=cov,scope=rgname,n=int(mask.sum()),
                PICP=round(CP.picp(yt[mask],lo[mask],hi[mask]),4),
                ACE=round(CP.picp(yt[mask],lo[mask],hi[mask])-cov,4),
                PINAW=round(CP.pinaw(yt[mask],lo[mask],hi[mask]),4),
                Winkler=round(CP.winkler(yt[mask],lo[mask],hi[mask],cov),2)))
crps=[]
for m in qstore:
    qd=dict(qstore[m]); qd[0.5]=pt
    for rgname,mask in SCOPES:
        if mask.sum()<30: continue
        crps.append(dict(method=m,horizon_min=h*5,scope=rgname,
            CRPS=round(CP.crps_from_quantiles(yt[mask],{t:qd[t][mask] for t in qd}),3)))
json.dump(dict(intervals=recs,crps=crps),open(f"/tmp/p3out/h{h}.json","w"),default=str)
if h==1:
    dfp=pd.DataFrame({"y":yt,"pred":pt,"regime":gt},index=te.index)
    lo,hi=CP.cqr_interval(qt[0.05],qt[0.95],CP.cqr_fit(yc,qc[0.05],qc[0.95]),0.90); dfp["cqr_lo"],dfp["cqr_hi"]=lo,hi
    lo,hi=CP.icp_interval(pt,icp_s,0.90); dfp["icp_lo"],dfp["icp_hi"]=lo,hi
    dfp.to_parquet("/tmp/p3out/h1_bands.parquet")
print(f"h={h*5} done: {len(recs)} interval recs",flush=True)
