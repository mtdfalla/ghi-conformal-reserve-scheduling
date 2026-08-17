"""Fast DKASC causal regime build from cached per-year weather parquets (/tmp/aspw).
Identical logic to dkasc_prepare.py (causal trailing window + train-only clear-sky
scalar) but reads the small cache instead of the 200 MB CSVs. Run from 03_code."""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd, pvlib

SITE = dict(name="DKASC_AliceSprings", latitude=-23.7621, longitude=133.8745, altitude=545, tz="Australia/Darwin")
W = "101_DKA_WeatherStation_"
COLS = {"ghi": W+"Global_Horizontal_Radiation", "ghi_diffuse": W+"Diffuse_Horizontal_Radiation",
        "temp_air": W+"Weather_Temperature_Celsius", "humidity": W+"Weather_Relative_Humidity",
        "wind_dir": W+"Wind_Direction", "rain_day": W+"Weather_Daily_Rainfall"}
CLEAN = C.BASE/"02_data"/"DKASC"/"cleaned"; REG = C.BASE/"02_data"/"DKASC"/"regime_labels"
def log(m): print(f"[dkasc] {m}", flush=True)

raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/tmp/aspw/*.parquet"))], ignore_index=True)
raw = raw.rename(columns={v: k for k, v in COLS.items()})
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = raw.drop_duplicates("timestamp").set_index("timestamp").sort_index()
full = pd.date_range(raw.index.min().floor("D"), raw.index.max().ceil("D"), freq="5min", inclusive="left")
df = raw.reindex(full)
for c in df.columns: df.loc[df[c].abs() >= 9e4, c] = np.nan
for c,(lo,hi) in {"ghi":(-50,1600),"ghi_diffuse":(-50,1600),"temp_air":(-15,60),"humidity":(0,100),"wind_dir":(0,360),"rain_day":(0,500)}.items():
    if c in df: df.loc[(df[c]<lo)|(df[c]>hi),c]=np.nan
df["ghi_imputed"]=df["ghi"].isna(); df["ghi"]=df["ghi"].interpolate(limit=6,limit_area="inside")
df["ghi_imputed"]=df["ghi_imputed"]&df["ghi"].notna(); df.loc[df["ghi"]<0,"ghi"]=0.0
df.to_parquet(CLEAN/"asp_clean_5min.parquet")

KT_CLEAR,KT_OVERCAST,VAR_WIN,VAR_STABLE,CS_DAY_MIN=0.85,0.50,12,0.10,20.0
loc=pvlib.location.Location(SITE["latitude"],SITE["longitude"],tz=SITE["tz"],altitude=SITE["altitude"],name=SITE["name"])
idx=df.index.tz_localize(SITE["tz"],ambiguous="NaT",nonexistent="NaT")
ok=idx.notna(); df=df[ok]; idx=idx[ok]; meas=df["ghi"].values
zenith=loc.get_solarposition(idx)["zenith"].values; cosz=np.cos(np.radians(zenith))
ghi_cs_raw=loc.get_clearsky(idx,model="ineichen")["ghi"].values
i0h=np.clip(pvlib.irradiance.get_extra_radiation(idx).values*cosz,0,None)
day=ghi_cs_raw>CS_DAY_MIN
dd=pd.DataFrame({"meas":meas,"cs":ghi_cs_raw,"day":day},index=df.index)
def daycorr(x):
    x=x[x.day]
    if len(x)<20 or x["meas"].isna().all() or x["meas"].std()==0: return np.nan
    return x["meas"].corr(x["cs"])
median_daycorr=float(dd.groupby(dd.index.date).apply(daycorr).median())
kt_raw=pd.Series(np.where(day,meas/np.where(ghi_cs_raw>0,ghi_cs_raw,np.nan),np.nan),index=df.index)
kt_raw_var=kt_raw.rolling(VAR_WIN,min_periods=VAR_WIN//2).std()          # trailing (causal)
is_train=(df.index.year<=2022)
band=day&kt_raw.between(0.85,1.15).values&(kt_raw_var.values<0.05)&is_train
cal=float(np.nanmedian(kt_raw.values[band]))
log(f"clear-day calibration factor = {cal:.4f} (n_band={int(band.sum())}, train-only)")
ghi_cs=ghi_cs_raw*cal
kt=np.clip(np.where(day,meas/np.where(ghi_cs>0,ghi_cs,np.nan),np.nan),0,1.5)
out=pd.DataFrame(index=df.index)
out["ghi"]=meas; out["ghi_cs"]=ghi_cs; out["i0h"]=i0h; out["zenith"]=zenith
out["is_day"]=day; out["kt"]=kt
out["kt_clearness"]=np.clip(np.where(i0h>CS_DAY_MIN,meas/np.where(i0h>0,i0h,np.nan),np.nan),0,1.2)
out["kt_var"]=pd.Series(kt,index=df.index).rolling(VAR_WIN,min_periods=VAR_WIN//2).std()  # trailing
out["ghi_imputed"]=df["ghi_imputed"].values
for c in ["temp_air","humidity","wind_dir","rain_day","ghi_diffuse"]: out[c]=df[c].values
ktv=out["kt_var"].values; reg=np.full(len(out),"night",dtype=object)
d=out["is_day"].values&~np.isnan(kt)
trans=d&(ktv>=VAR_STABLE); clear=d&~trans&(kt>=KT_CLEAR); cloudy=d&~trans&(kt<KT_OVERCAST); mid=d&~trans&~clear&~cloudy
reg[trans]="transitional"; reg[clear]="clear"; reg[cloudy]="cloudy"; reg[mid]="transitional"
out["regime"]=reg; out.to_parquet(REG/"asp_regimes_5min.parquet")
rd=out.loc[out["is_day"],"regime"].value_counts(); rdp=(rd/rd.sum()*100).round(2)
json.dump(dict(site=SITE["name"],clearsky_model="ineichen_clearband_calibrated_causal_trainonly",
    calibration_factor=round(cal,4),median_per_day_corr=round(median_daycorr,4),
    daytime_steps=int(out["is_day"].sum()),regime_daytime_pct=rdp.to_dict(),years="2020-2024",
    n_rows=int(len(out))),open(C.MET/"j3_asp_clearsky_alignment.json","w"),indent=2)
log(f"per-day corr median={median_daycorr:.3f}")
print(pd.DataFrame({"count":rd,"pct":rdp}).to_string()); log("done.")
