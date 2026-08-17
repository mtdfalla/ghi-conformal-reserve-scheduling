"""Phase 1 - Step 4: EDA + data-quality visualisation.

Generates figures (04_results/figures) and tables (04_results/tables):
  - per-year x per-variable validity heatmap (the key data-quality story)
  - GHI diurnal profile by season
  - regime distribution (overall + monthly stacked)
  - calibrated clear-sky index distribution
  - example clear / transitional / cloudy days
  - GHI vs PV power relationship (decision-layer feasibility)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def log(m): print(f"[eda] {m}", flush=True)
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,"grid.alpha":.3})

clean = pd.read_parquet(C.DATA_CLEAN/"yulara_clean_5min.parquet")
reg   = pd.read_parquet(C.DATA_REGIME/"yulara_regimes_5min.parquet")
df = clean.join(reg[["is_day","kt","kt_var","regime","ghi_cs","day_class","zenith"]])
df["year"]=df.index.year; df["month"]=df.index.month
COLS=list(C.COLS.keys())

# ---- 1. Validity heatmap (per-year % non-missing, after cleaning) ----
val = df.groupby("year")[COLS].apply(lambda g: g.notna().mean()*100)
val.to_csv(C.TAB/"p1_validity_by_year.csv")
fig,ax=plt.subplots(figsize=(10,5))
im=ax.imshow(val.T.values,aspect="auto",cmap="RdYlGn",vmin=0,vmax=100)
ax.set_xticks(range(len(val.index))); ax.set_xticklabels(val.index)
ax.set_yticks(range(len(COLS))); ax.set_yticklabels(COLS)
for i in range(len(COLS)):
    for j in range(len(val.index)):
        ax.text(j,i,f"{val.T.values[i,j]:.0f}",ha="center",va="center",fontsize=7)
ax.set_title("Data validity (% non-missing after cleaning) by year and variable")
plt.colorbar(im,label="% valid"); plt.tight_layout()
plt.savefig(C.FIG/"p1_validity_heatmap.png"); plt.close()
log("fig: validity heatmap")

# ---- 2. GHI diurnal profile by season ----
df["mod"]=df.index.hour+df.index.minute/60
seas={12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
df["season"]=df["month"].map(seas)
fig,ax=plt.subplots(figsize=(9,5))
for s in ["DJF","MAM","JJA","SON"]:
    prof=df[df.season==s].groupby("mod")["ghi"].mean()
    ax.plot(prof.index,prof.values,label=s)
ax.set_xlabel("hour of day (ACST)"); ax.set_ylabel("mean GHI (W/m²)")
ax.set_title("Mean GHI diurnal profile by season (Pyranometer_1)"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG/"p1_ghi_diurnal_by_season.png"); plt.close()
log("fig: diurnal profile")

# ---- 3. Regime distribution overall + monthly ----
dday=df[df.is_day & df.regime.isin(["clear","transitional","cloudy"])]
order=["clear","transitional","cloudy"]; colors={"clear":"#f4a300","transitional":"#7fb0d0","cloudy":"#5a6b7b"}
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
vc=dday.regime.value_counts(normalize=True).reindex(order)*100
axes[0].bar(order,vc.values,color=[colors[o] for o in order])
for i,v in enumerate(vc.values): axes[0].text(i,v+0.5,f"{v:.1f}%",ha="center")
axes[0].set_ylabel("% of daytime steps"); axes[0].set_title("Weather-regime distribution (daytime)")
mon=dday.groupby(["month","regime"]).size().unstack().reindex(columns=order).fillna(0)
mon=mon.div(mon.sum(1),axis=0)*100
bottom=np.zeros(12)
for o in order:
    axes[1].bar(mon.index,mon[o].values,bottom=bottom,label=o,color=colors[o]); bottom+=mon[o].values
axes[1].set_xlabel("month"); axes[1].set_ylabel("% of daytime"); axes[1].set_title("Regime mix by month"); axes[1].legend()
plt.tight_layout(); plt.savefig(C.FIG/"p1_regime_distribution.png"); plt.close()
log("fig: regime distribution")

# ---- 4. clear-sky index distribution ----
fig,ax=plt.subplots(figsize=(8,4.5))
ax.hist(df.loc[df.is_day,"kt"].dropna(),bins=80,color="#3a7ca5")
ax.axvline(0.85,color="k",ls="--",lw=1,label="clear≥0.85"); ax.axvline(0.5,color="r",ls="--",lw=1,label="cloudy<0.5")
ax.set_xlabel("calibrated clear-sky index kt"); ax.set_ylabel("count")
ax.set_title("Distribution of clear-sky index (daytime)"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG/"p1_kt_distribution.png"); plt.close()
log("fig: kt distribution")

# ---- 5. example days (clear / transitional / cloudy) ----
def find_day(cls):
    cand=reg.loc[reg.day_class==cls]
    if len(cand)==0: return None
    return pd.Timestamp(cand.index[len(cand)//2].date())
fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
for ax,cls,t in zip(axes,["clear","mixed","overcast"],["clear","transitional","overcast"]):
    d0=find_day(cls)
    if d0 is None: continue
    day=df.loc[str(d0.date())]
    ax.plot(day.index,day.ghi,label="GHI (meas)",color="#e67300")
    ax.plot(day.index,day.ghi_cs,label="clear-sky",color="k",ls="--",lw=1)
    ax.set_title(f"{t} day: {d0.date()}"); ax.tick_params(axis="x",rotation=45)
axes[0].set_ylabel("GHI (W/m²)"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(C.FIG/"p1_example_days.png"); plt.close()
log("fig: example days")

# ---- 6. GHI vs PV power (decision-layer feasibility) ----
sub=df[df.is_day].dropna(subset=["ghi","pv_total"]).sample(min(8000,df.is_day.sum()),random_state=C.SEED)
fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(sub.ghi,sub.pv_total,s=4,alpha=.25,color="#2a9d8f")
ax.set_xlabel("GHI (W/m²)"); ax.set_ylabel("Total site PV power (kW)")
corr=df[df.is_day][["ghi","pv_total"]].corr().iloc[0,1]
ax.set_title(f"GHI vs PV power (daytime), corr={corr:.3f}")
plt.tight_layout(); plt.savefig(C.FIG/"p1_ghi_vs_pv.png"); plt.close()
log(f"fig: GHI vs PV (corr={corr:.3f})")

# ---- summary stats table ----
stats=dict(
  n_rows=int(len(df)), span=[str(df.index.min()),str(df.index.max())],
  daytime_steps=int(df.is_day.sum()),
  ghi_valid_overall_pct=round(df.ghi.notna().mean()*100,2),
  ghi_daytime_valid_pct=round(df.loc[df.is_day,"ghi"].notna().mean()*100,2),
  ghi_pv_corr_daytime=round(float(corr),4),
  regime_daytime_pct=(dday.regime.value_counts(normalize=True)*100).round(2).to_dict(),
  validity_by_year=val.round(1).to_dict(),
)
with open(C.MET/"p1_eda_summary.json","w") as fh: json.dump(stats,fh,indent=2)
log("=== VALIDITY BY YEAR (% non-missing) ===")
print(val.round(1).to_string())
log("done.")
