"""J4 aggregation: per-horizon JSON -> tables, figure, summary.
Run from the code directory (03_code/ in the working tree, code/ in a release checkout): python3 _j4_aggregate.py
"""
import sys, json, glob; sys.path.insert(0, "utils")
import config as CFG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/tmp/j4out"; TAB = CFG.TAB; FIG = CFG.FIG
M = []; CR = []; DM = []
for f in sorted(glob.glob(f"{OUT}/h*.json")):
    j = json.load(open(f)); M += j["metrics"]; CR += j["crps"]; DM += j["dm"]
M = pd.DataFrame(M); CR = pd.DataFrame(CR); DM = pd.DataFrame(DM)
M.to_csv(TAB / "j4_metrics_long.csv", index=False)
CR.to_csv(TAB / "j4_crps.csv", index=False)
DM.to_csv(TAB / "j4_dm_vs_univariate.csv", index=False)

SETS = ["uni", "+temp", "+wind", "all"]
# RMSE all-conditions by set x horizon
rmse = M[(M.metric == "RMSE") & (M.scope == "all")].pivot(index="set", columns="horizon_min", values="value").reindex(SETS)
rmse.to_csv(TAB / "j4_rmse_all_set_x_horizon.csv")
# RMSE transitional (where variability lives)
rmse_tr = M[(M.metric == "RMSE") & (M.scope == "transitional")].pivot(index="set", columns="horizon_min", values="value").reindex(SETS)
rmse_tr.to_csv(TAB / "j4_rmse_transitional_set_x_horizon.csv")
# interval @90
picp = M[(M.metric == "PICP90") & (M.scope == "all")].pivot(index="set", columns="horizon_min", values="value").reindex(SETS)
pinaw = M[(M.metric == "PINAW90") & (M.scope == "all")].pivot(index="set", columns="horizon_min", values="value").reindex(SETS)
crps = CR.pivot(index="set", columns="horizon_min", values="CRPS").reindex(SETS)
picp.to_csv(TAB / "j4_picp90_set_x_horizon.csv"); pinaw.to_csv(TAB / "j4_pinaw90_set_x_horizon.csv")

# % RMSE change vs uni
rmse_pct = (rmse.div(rmse.loc["uni"]) - 1) * 100

# figure: RMSE vs horizon by feature set (all + transitional)
hz = [5, 15, 30, 60]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for s in SETS:
    ax[0].plot(hz, [rmse.loc[s, h] for h in hz], "-o", label=s)
    ax[1].plot(hz, [rmse_tr.loc[s, h] for h in hz], "-o", label=s)
ax[0].set_title("RMSE — all conditions"); ax[1].set_title("RMSE — transitional regime")
for a in ax:
    a.set_xlabel("horizon (min)"); a.set_ylabel("RMSE (W/m²)"); a.legend(fontsize=9)
# Figure-level title deliberately not drawn: the caption in the article carries the
# description, and a title above a caption is redundant. Per-panel titles are kept.
# plt.suptitle("J4: multivariate vs univariate (train 2023 → test 2024)")
plt.tight_layout(); plt.savefig(FIG / "j4_rmse_by_featureset.png", dpi=130); plt.close()

summary = dict(rmse_all=rmse.round(3).to_dict(), rmse_pct_vs_uni=rmse_pct.round(2).to_dict(),
               picp90=picp.round(3).to_dict(), crps=crps.round(2).to_dict(),
               dm=DM.to_dict("records"))
json.dump(summary, open(CFG.MET / "j4_summary.json", "w"), indent=2, default=str)

print("== RMSE (all) by set x horizon =="); print(rmse.round(2).to_string())
print("\n== RMSE % change vs univariate (negative = better) =="); print(rmse_pct.round(2).to_string())
print("\n== RMSE (transitional) =="); print(rmse_tr.round(2).to_string())
print("\n== Interval @90: PICP / PINAW / CRPS (all) ==");
print("PICP:\n", picp.round(3).to_string()); print("PINAW:\n", pinaw.round(3).to_string()); print("CRPS:\n", crps.round(2).to_string())
print("\n== DM vs univariate (negative stat => multivariate better; p<0.05 sig) ==")
print(DM.to_string(index=False))
print("\nWrote j4_* tables, figure, summary.")
