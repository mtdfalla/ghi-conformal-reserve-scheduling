"""J3 aggregation + cross-site (DKASC vs Yulara) comparison tables/figures.
Run from 03_code: python3 _j3_aggregate.py
"""
import sys, json, glob; sys.path.insert(0, "utils")
import config as CFG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'lines.linewidth':2.2,'lines.markersize':7,'savefig.dpi':200,'figure.dpi':200,'savefig.bbox':'tight'})

OUT = "/tmp/j3out"; TAB = CFG.TAB; FIG = CFG.FIG
pt = []; iv = []; cr = []; dec = []
for f in sorted(glob.glob(f"{OUT}/h*.json")):
    j = json.load(open(f)); pt += j["point"]; iv += j["intervals"]; cr += j["crps"]; dec += j["decision"]
pt = pd.DataFrame(pt); iv = pd.DataFrame(iv); cr = pd.DataFrame(cr); dec = pd.DataFrame(dec)
pt.to_csv(TAB / "j3_point_metrics.csv", index=False)
iv.to_csv(TAB / "j3_interval_metrics.csv", index=False)
cr.to_csv(TAB / "j3_crps.csv", index=False)
dec.to_csv(TAB / "j3_decision_value.csv", index=False)

MAIN = ["icp", "mondrian", "cqr", "mondrian_cqr", "aci"]
# DKASC per-regime PICP @90 5-min
d5 = iv[(iv.nominal == 0.90) & (iv.horizon_min == 5) & (iv.method.isin(MAIN))]
d5p = d5.pivot(index="method", columns="scope", values="PICP").reindex(MAIN)[["all", "clear", "transitional", "cloudy"]]
d5p.to_csv(TAB / "j3_dkasc_picp90_5min_by_regime.csv")

# ---- cross-site comparison vs Yulara (from J2 tables) ----
def load_yulara():
    y_iv = pd.read_csv(TAB / "j2_interval_metrics.csv")
    y5 = y_iv[(y_iv.nominal == 0.90) & (y_iv.horizon_min == 5) & (y_iv.method.isin(MAIN))]
    y5p = y5.pivot(index="method", columns="scope", values="PICP").reindex(MAIN)[["all", "clear", "transitional", "cloudy"]]
    y_cr = pd.read_csv(TAB / "j2_crps.csv")
    y_cr5 = y_cr[(y_cr.horizon_min == 5) & (y_cr.scope == "all")].set_index("method")["CRPS"]
    return y5p, y_cr5
try:
    y5p, y_cr5 = load_yulara(); have_y = True
except Exception as e:
    have_y = False; print("Yulara J2 tables not found:", e)

# DKASC point RMSE all by horizon
rmse = pt[pt.scope == "all"].pivot(index="scope", columns="horizon_min", values="RMSE")
dcr5 = cr[(cr.horizon_min == 5) & (cr.scope == "all")].set_index("method")["CRPS"]

print("== DKASC point RMSE (all) by horizon ==")
print(pt[pt.scope == "all"][["horizon_min", "RMSE", "MAE"]].to_string(index=False))
print("\n== DKASC per-regime PICP @90 (5-min) ==")
print(d5p.round(3).to_string())
print("\n== DKASC CRPS @5min (W/m2) =="); print(dcr5.round(2).to_string())
print("\n== DKASC decision value (reserve, r=10) ==")
print(dec[["horizon_min", "cost_det", "cost_mondrian", "cost_oracle", "value_captured"]].to_string(index=False))

if have_y:
    cmp = pd.concat({"Yulara": y5p["all"], "DKASC": d5p["all"]}, axis=1)
    cmp.to_csv(TAB / "j3_crosssite_picp90_5min_all.csv")
    # regime ACE-RMS per site
    def ace_rms(p): return np.sqrt(np.mean((p[["clear", "transitional", "cloudy"]].values - 0.90) ** 2, axis=1))
    cal = pd.DataFrame({"Yulara_regimeACErms": ace_rms(y5p), "DKASC_regimeACErms": ace_rms(d5p)}, index=MAIN).round(4)
    cal.to_csv(TAB / "j3_crosssite_regime_calibration.csv")
    print("\n== CROSS-SITE per-regime ACE-RMS @90 5-min (lower=better) ==")
    print(cal.to_string())

    # Figure: per-regime PICP, Yulara vs DKASC, key methods
    regs = ["clear", "transitional", "cloudy"]; methods = ["icp", "mondrian", "mondrian_cqr"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for k, (site, P) in enumerate([("Yulara", y5p), ("DKASC", d5p)]):
        x = np.arange(len(regs)); w = 0.25
        for i, m in enumerate(methods):
            ax[k].bar(x + i*w, [P.loc[m, r] for r in regs], w, label=m)
        ax[k].axhline(0.90, ls="--", c="k", lw=1); ax[k].set_xticks(x + w); ax[k].set_xticklabels(regs)
        ax[k].set_title(site); ax[k].set_ylim(0.5, 1.02)
    ax[0].set_ylabel("PICP (5-min, 90% target)"); ax[1].legend(fontsize=9)
    plt.suptitle("J3: per-regime calibration transfers across sites (Yulara → DKASC)")
    plt.tight_layout(); plt.savefig(FIG / "j3_crosssite_calibration.png", dpi=200); plt.close()

# Figure: DKASC value captured vs horizon
plt.figure(figsize=(6.5, 4))
plt.plot(dec["horizon_min"], dec["value_captured"]*100, "-o", color="#d22")
plt.xlabel("horizon (min)"); plt.ylabel("value captured by Mondrian (%)")
plt.title("J3 (DKASC): decision value of regime-aware uncertainty (r=10)")
plt.ylim(0, 70); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "j3_dkasc_value_captured.png", dpi=200); plt.close()

summary = dict(dkasc_rmse_all=pt[pt.scope == "all"].set_index("horizon_min")["RMSE"].to_dict(),
               dkasc_picp90_5min_regime=d5p.round(3).to_dict(),
               dkasc_value_captured=dec.set_index("horizon_min")["value_captured"].to_dict())
json.dump(summary, open(CFG.MET / "j3_summary.json", "w"), indent=2, default=str)
print("\nWrote j3_* tables, figures, summary.")
