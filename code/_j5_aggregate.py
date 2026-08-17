"""J5 aggregation: dispatch JSON -> tables, risk-cost frontier + figures.
Run from 03_code: python3 _j5_aggregate.py
"""
import sys, json, glob; sys.path.insert(0, "utils")
import config as CFG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'lines.linewidth':2.2,'lines.markersize':7,'savefig.dpi':200,'figure.dpi':200,'savefig.bbox':'tight'})

OUT = "/tmp/j5out"; TAB = CFG.TAB; FIG = CFG.FIG
fr = []; hd = []; sn = []; params = {}
for f in sorted(glob.glob(f"{OUT}/h*.json")):
    j = json.load(open(f)); fr += j["frontier"]; hd += j["headline"]; sn += j["sensitivity"]; params = j["params"]
fr = pd.DataFrame(fr); hd = pd.DataFrame(hd); sn = pd.DataFrame(sn)
fr.to_csv(TAB / "j5_frontier.csv", index=False)
hd.to_csv(TAB / "j5_headline_value_captured.csv", index=False)
sn.to_csv(TAB / "j5_battery_sensitivity.csv", index=False)

PRETTY = {"deterministic": "Deterministic", "icp": "ICP", "mondrian": "Mondrian",
          "mondrian_cqr": "Mondrian-CQR", "oracle": "Oracle"}
HUES = {"deterministic": "#444", "icp": "#888", "mondrian": "#1b7", "mondrian_cqr": "#d22", "oracle": "#27c"}

print("== Headline value captured (fraction of det->oracle gap) ==")
print(hd[["horizon_min", "policy", "best_rho", "mean_daily", "cvar95_daily",
          "value_captured_mean", "value_captured_cvar"]].to_string(index=False))
print("\n== Battery sensitivity (mean / CVaR95 daily cost) ==")
print(sn[["horizon_min", "batt", "E_max", "P_max", "policy", "mean_daily", "cvar95_daily", "soc_min", "soc_max"]].to_string(index=False))

# Fig 1: risk-cost frontier (mean vs CVaR), h=5, CP policies sweep rho
for hmin in sorted(fr.horizon_min.unique()):
    sub = fr[fr.horizon_min == hmin]
    plt.figure(figsize=(7, 5))
    for pol in ["icp", "mondrian", "mondrian_cqr"]:
        s = sub[sub.policy == pol].sort_values("mean_daily")
        plt.plot(s["mean_daily"], s["cvar95_daily"], "-o", color=HUES[pol], label=PRETTY[pol])
        for _, r in s.iterrows():
            plt.annotate(f"{r['rho']:.2f}", (r["mean_daily"], r["cvar95_daily"]), fontsize=6, alpha=0.6)
    for pol in ["deterministic", "oracle"]:
        r = sub[sub.policy == pol].iloc[0]
        plt.scatter([r["mean_daily"]], [r["cvar95_daily"]], color=HUES[pol], s=90, marker="*", label=PRETTY[pol], zorder=5)
    plt.xlabel("expected daily reserve cost (USD)"); plt.ylabel("CVaR$_{0.95}$ daily cost (USD)")
    plt.title(f"J5: risk–cost frontier ({hmin}-min, reserve level labelled)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / f"j5_risk_cost_frontier_{hmin}min.png", dpi=200); plt.close()

# Fig 2: value captured (mean & CVaR) by method x horizon
methods = ["icp", "mondrian", "mondrian_cqr"]; hs = sorted(hd.horizon_min.unique())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
x = np.arange(len(methods)); w = 0.35
for k, metric in enumerate(["value_captured_mean", "value_captured_cvar"]):
    for j, hm in enumerate(hs):
        vals = [hd[(hd.policy == m) & (hd.horizon_min == hm)][metric].iloc[0]*100 for m in methods]
        ax[k].bar(x + j*w, vals, w, label=f"{hm}-min")
    ax[k].set_xticks(x + w/2); ax[k].set_xticklabels([PRETTY[m] for m in methods])
    ax[k].set_title("value captured — " + ("expected cost" if k == 0 else "tail risk (CVaR$_{0.95}$)"))
    ax[k].set_ylim(0, 100); ax[k].legend(fontsize=9); ax[k].grid(alpha=0.3, axis="y")
ax[0].set_ylabel("% of deterministic→oracle gap closed")
plt.suptitle("J5: decision value of calibrated regime-aware uncertainty (full SoC dispatch)")
plt.tight_layout(); plt.savefig(FIG / "j5_value_captured.png", dpi=200); plt.close()

# Fig 3: battery sensitivity (Mondrian value captured vs battery size), h=5
s5 = sn[sn.horizon_min == 5]
det5 = s5[s5.policy == "deterministic"].set_index("batt")["mean_daily"]
mon5 = s5[s5.policy == "mondrian"].set_index("batt")["mean_daily"]
ora5 = s5[s5.policy == "oracle"].set_index("batt")["mean_daily"]
order = ["small", "default", "large"]
vc = [(det5[b]-mon5[b])/(det5[b]-ora5[b]) if (det5[b]-ora5[b]) > 0 else np.nan for b in order]
emax = [s5[s5.batt == b]["E_max"].iloc[0] for b in order]
plt.figure(figsize=(6.5, 4))
plt.plot([f"{b}\n({e:.0f} kWh)" for b, e in zip(order, emax)], [v*100 for v in vc], "-o", color="#1b7")
plt.ylabel("Mondrian value captured (%)"); plt.xlabel("battery size")
plt.title("J5: value of forecast calibration vs battery size (5-min)")
plt.ylim(0, 100); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "j5_battery_sensitivity.png", dpi=200); plt.close()

# SoC feasibility check
soc_ok = bool((fr["soc_min"] >= 0.099).all() and (fr["soc_max"] <= 1.001).all())
summary = dict(params=params, soc_feasible=soc_ok,
               headline=hd.to_dict("records"))
json.dump(summary, open(CFG.MET / "j5_summary.json", "w"), indent=2, default=str)
print(f"\nSoC feasibility (all sims within [{0.10},1.0]): {soc_ok}")
print("Wrote j5_* tables, figures, summary.")
