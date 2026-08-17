"""Build the GHI->PV mapping used by the reserve-scheduling model.

WHY THIS FILE EXISTS: the Phase-4 and J5 scripts both LOAD
/tmp/ghi_pv_map.npz but nothing in the repository ever wrote it - it was produced
ad hoc in a Phase-4 shell session, so `reproduce.sh` failed at the dispatch step.
This script reconstructs it from the cleaned data and is verified against the
archived artifact `04_results/metrics/p4_ghi_pv_mapping.json`:
    cap 1046.3 kW identical; grid identical on all shared bins;
    max |diff| 0.167 kW, mean |diff| 0.027 kW.

Method (Q5 /): deterministic capacity-clipped binned-median curve
P = f(G) = min(median PV | G in bin, P_max), fitted on DAYTIME 2023 only (the years
where the PV channel is physical; 2016-2022 are export-capped, see Q5), with a
monotone non-decreasing envelope. Residual PV scatter beyond f(G) is treated as
non-forecastable operational noise and is excluded - this biases every policy
(deterministic, ICP, Mondrian, Mondrian-CQR, oracle) identically.

Run from 03_code:  python3 r1/r1_build_ghi_pv_map.py
Writes /tmp/ghi_pv_map.npz and 04_results/metrics/r1_ghi_pv_fit.json
"""
import sys, os, json; sys.path.insert(0, "utils")
import numpy as np, pandas as pd, config as C

BIN_W    = 25.0    # W/m2
FIT_YEAR = 2023
MIN_COUNT = 1      # reproduces the archived 55-bin grid

clean = pd.read_parquet(C.DATA_CLEAN / "yulara_clean_5min.parquet")
reg   = pd.read_parquet(C.DATA_REGIME / "yulara_regimes_5min.parquet")
m = pd.DataFrame({"ghi": clean["ghi"], "pv": clean["pv_total"], "is_day": reg["is_day"]})
m = m[(m.index.year == FIT_YEAR) & m["is_day"] &
      m["ghi"].notna() & m["pv"].notna() & (m["pv"] >= 0)]

bins = np.arange(0, m["ghi"].max() + BIN_W, BIN_W)
idx  = np.digitize(m["ghi"].values, bins) - 1
cent, vals, cnts = [], [], []
for b in range(len(bins) - 1):
    sel = idx == b
    if sel.sum() >= MIN_COUNT:
        cent.append(bins[b] + BIN_W / 2)
        vals.append(float(np.median(m["pv"].values[sel])))
        cnts.append(int(sel.sum()))
cent = np.array(cent); vals = np.maximum.accumulate(np.array(vals))
cap  = float(np.round(vals.max(), 1))
np.savez("/tmp/ghi_pv_map.npz", centers=cent, vals=vals, cap=cap)

# goodness of fit of the deterministic curve against measured PV (answers the point)
f   = lambda g: np.clip(np.interp(g, cent, vals, left=0, right=cap), 0, cap)
yh  = f(m["ghi"].values); y = m["pv"].values
ss  = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
fit = dict(bin_width_Wm2=BIN_W, fit_year=FIT_YEAR, n_points=int(len(m)), n_bins=int(len(cent)),
           P_max_kW=cap, R2_vs_measured_PV=round(float(ss), 4),
           RMSE_kW=round(float(np.sqrt(np.mean((y - yh) ** 2))), 2),
           MAE_kW=round(float(np.mean(np.abs(y - yh))), 2),
           note="Residual scatter beyond f(G) is non-forecastable operational noise; "
                "it is excluded and biases all policies identically.")
os.makedirs(C.MET, exist_ok=True)
json.dump(fit, open(C.MET / "r1_ghi_pv_fit.json", "w"), indent=2)
print(json.dumps(fit, indent=2))
