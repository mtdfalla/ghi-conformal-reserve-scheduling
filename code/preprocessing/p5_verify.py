"""Phase 1 - Step 5: integrity & readiness verification (run before Phase 2)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd
ok=True
def check(name, cond, detail=""):
    global ok; ok = ok and cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))

clean=pd.read_parquet(C.DATA_CLEAN/"yulara_clean_5min.parquet")
reg=pd.read_parquet(C.DATA_REGIME/"yulara_regimes_5min.parquet")
flags=pd.read_parquet(C.DATA_CLEAN/"yulara_quality_flags.parquet")

check("cleaned/regime row counts equal", len(clean)==len(reg), f"{len(clean)} vs {len(reg)}")
check("indexes identical", clean.index.equals(reg.index))
check("regular 5-min grid (no gaps in index)",
      (pd.infer_freq(clean.index) in ("5min","5T")) or (clean.index.to_series().diff().dropna().eq(pd.Timedelta('5min')).all()))
g=clean["ghi"]
check("GHI no values >1500 remain", not (g>1500).any(), f"max={g.max():.1f}")
check("GHI no negatives remain", not (g<0).any(), f"min={g.min():.1f}")
check("GHI no sentinels remain", not (g.abs()>=9e4).any())
check("regime categories valid", set(reg["regime"].unique())<= {"clear","transitional","cloudy","night"},
      str(sorted(reg['regime'].unique())))
# clear-band sanity
band_med = reg.loc[reg.is_day & reg.kt.between(0.95,1.05),"kt"].median()
check("clear-sky index centred ~1 on clear pts", 0.9<band_med<1.1, f"median≈{band_med:.3f}")

# Candidate split readiness (daytime, valid GHI)
day=reg["is_day"]; valid=clean["ghi"].notna()
dd=clean.assign(regime=reg["regime"], yr=clean.index.year)[day & valid]
splits={"train(2016-2022)":range(2016,2023),"calib(2023)":[2023],"test(2024)":[2024]}
print("\nDaytime, valid-GHI sample counts by split:")
for nm,yrs in splits.items():
    sub=dd[dd.yr.isin(list(yrs))]
    rc=sub.regime.value_counts().to_dict()
    print(f"  {nm:18s} n={len(sub):>7}  regimes={ {k:int(v) for k,v in rc.items()} }")
test=dd[dd.yr==2024]
check("test set has all 3 daytime regimes", {"clear","transitional","cloudy"}<=set(test.regime.unique()))
check("test set sufficiently large", len(test)>20000, f"n={len(test)}")
# interpolation footprint (for leakage awareness in Phase 2)
imp=flags["ghi_imputed"].mean()*100
print(f"\nNote: {imp:.2f}% of GHI rows are short-gap interpolated (will be excluded from target rows in modelling).")
print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
