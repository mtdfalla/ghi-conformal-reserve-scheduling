"""Phase 4: operating-reserve cost-of-uncertainty simulation (test 2024).
Policies: deterministic, ICP (marginal), Mondrian (regime), perfect foresight.
Cost depends only on PV forecast-quantile error (load-independent)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"utils"); sys.path.insert(0,"conformal")
import numpy as np, pandas as pd
import config as CFG, datasets as D
import conformal as CP
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

m=np.load("/tmp/ghi_pv_map.npz"); CENT,VALS,CAP=m["centers"],m["vals"],float(m["cap"])
f=lambda ghi: np.clip(np.interp(ghi,CENT,VALS,left=0,right=CAP),0,CAP)
DT=5/60.0  # hours per step
C_O=0.30; RATIOS=[3,5,10,19]; P_B_LIST=[0,300]; HOR=[1,6]
rec=lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
def gbm(): return HGB(max_iter=200,learning_rate=0.07,max_leaf_nodes=31,early_stopping=True,
    validation_fraction=0.1,n_iter_no_change=12,random_state=CFG.SEED)
base=pd.read_parquet("/tmp/base.parquet")
rows=[]
for h in HOR:
    d=D.make_xy(base,h); tr=d[d.year<=2022]; ca=d[d.year==2023]; te=d[d.year==2024]
    g=gbm().fit(tr[D.FEATURES].values,tr["y_kt"].values)
    ghi_pt_ca=rec(g.predict(ca[D.FEATURES].values),ca["y_ghi_cs"].values)
    ghi_pt_te=rec(g.predict(te[D.FEATURES].values),te["y_ghi_cs"].values)
    ya_ca=ca["y_ghi"].values; ya_te=te["y_ghi"].values
    gc=ca["base_regime"].values; gt=te["base_regime"].values
    pv_act=f(ya_te)                      # actual PV supply
    icp_s=CP.icp_fit(ya_ca,ghi_pt_ca); mon_s=CP.mondrian_fit(ya_ca,ghi_pt_ca,gc)
    def cost(pv_q,cu,co,pb):
        short=np.maximum(pv_q-pv_act-pb,0); over=np.maximum(pv_act-pv_q-pb,0)
        return (cu*short+co*over)*DT   # $ per step (per-unit cost * energy)
    for r in RATIOS:
        cu=C_O*r; tau=co_tau=1.0/(1+r); cov=1-2*tau   # symmetric coverage giving lower quantile=tau
        # PV_q per policy (GHI lower quantile -> f)
        q_icp=CP.conformal_q(icp_s,cov)
        ghi_lo_icp=np.clip(ghi_pt_te-q_icp,0,None)
        ghi_lo_mon=np.empty(len(ghi_pt_te))
        for grp in np.unique(gt):
            qg=CP.conformal_q(mon_s.get(grp,icp_s),cov); ghi_lo_mon[gt==grp]=np.clip(ghi_pt_te[gt==grp]-qg,0,None)
        pvq={"deterministic":f(ghi_pt_te),"icp":f(ghi_lo_icp),"mondrian":f(ghi_lo_mon),"oracle":pv_act}
        for pb in P_B_LIST:
            costs={k:cost(v,cu,C_O,pb) for k,v in pvq.items()}
            det=costs["deterministic"].sum(); ora=costs["oracle"].sum()
            for k in pvq:
                tot=costs[k].sum()
                vc = (det-tot)/(det-ora) if (det-ora)>0 else np.nan
                rows.append(dict(horizon_min=h*5,cost_ratio=r,P_b_kW=pb,policy=k,scope="all",
                    total_cost=round(tot,1),per_day=round(tot/ (len(np.unique(te.index.date))),2),
                    value_captured=round(vc,3) if k not in("deterministic","oracle") else (0.0 if k=="deterministic" else 1.0)))
            # by-regime value captured at base ratio later; store regime breakdown for r=10,pb=0
            if r==10 and pb==0:
                for grp in ["clear","transitional","cloudy"]:
                    mk=gt==grp
                    if mk.sum()<50: continue
                    detg=costs["deterministic"][mk].sum(); orag=costs["oracle"][mk].sum()
                    for k in ["icp","mondrian"]:
                        tg=costs[k][mk].sum(); vcg=(detg-tg)/(detg-orag) if (detg-orag)>0 else np.nan
                        rows.append(dict(horizon_min=h*5,cost_ratio=r,P_b_kW=pb,policy=k,scope=grp,
                            total_cost=round(tg,1),per_day=np.nan,value_captured=round(vcg,3)))
    print(f"h={h*5} done",flush=True)
df=pd.DataFrame(rows)
df.to_csv("/tmp/p4_results.csv",index=False)
print("=== Value captured (% of det->oracle gap closed), scope=all, P_b=0 ===")
piv=df[(df.scope=='all')&(df.P_b_kW==0)&(df.policy.isin(['icp','mondrian']))].pivot_table(
    index=['horizon_min','cost_ratio'],columns='policy',values='value_captured')
print((piv*100).round(1).to_string())
print("\n=== Cost ($, scope=all, ratio=10, P_b=0) by policy ===")
print(df[(df.scope=='all')&(df.cost_ratio==10)&(df.P_b_kW==0)][['horizon_min','policy','total_cost','per_day']].to_string(index=False))
print("\n=== Value captured by regime (ratio=10,P_b=0,5min&30min) ===")
print(df[(df.scope!='all')][['horizon_min','scope','policy','value_captured']].to_string(index=False))
