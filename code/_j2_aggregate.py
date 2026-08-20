"""J2 aggregation: per-horizon JSON -> tables, figures, and a headline summary.
Run from the code directory (03_code/ in the working tree, code/ in a release checkout): python3 _j2_aggregate.py
"""
import sys, json, glob; sys.path.insert(0, "utils")
import config as CFG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'lines.linewidth':2.2,'lines.markersize':7,'savefig.dpi':200,'figure.dpi':200,'savefig.bbox':'tight'})

OUT = "/tmp/j2out"
TAB = CFG.TAB; FIG = CFG.FIG
iv = []; cr = []; rel = []
for f in sorted(glob.glob(f"{OUT}/h*.json")):
    j = json.load(open(f))
    iv += j["intervals"]; cr += j["crps"]; rel += j["reliability"]
iv = pd.DataFrame(iv); cr = pd.DataFrame(cr); rel = pd.DataFrame(rel)

# ---- tables ----
iv.to_csv(TAB / "j2_interval_metrics.csv", index=False)
cr.to_csv(TAB / "j2_crps.csv", index=False)
rel.to_csv(TAB / "j2_reliability_over_time.csv", index=False)

MAIN = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "aci", "aci_regime"]
PRETTY = {"icp": "ICP", "icp_norm": "ICP-norm", "mondrian": "Mondrian", "cqr": "CQR",
          "mondrian_cqr": "Mondrian-CQR", "aci": "ACI", "aci_regime": "ACI-regime"}

# headline pivot: PICP @90% all-scope by method x horizon
m90 = iv[(iv.nominal == 0.90) & (iv.scope == "all") & (iv.method.isin(MAIN))]
picp_piv = m90.pivot(index="method", columns="horizon_min", values="PICP").reindex(MAIN)
pinaw_piv = m90.pivot(index="method", columns="horizon_min", values="PINAW").reindex(MAIN)
wink_piv = m90.pivot(index="method", columns="horizon_min", values="Winkler").reindex(MAIN)
picp_piv.to_csv(TAB / "j2_picp90_method_x_horizon.csv")
pinaw_piv.to_csv(TAB / "j2_pinaw90_method_x_horizon.csv")

# per-regime PICP @90% 5-min (the calibration story)
reg5 = iv[(iv.nominal == 0.90) & (iv.horizon_min == 5) & (iv.method.isin(MAIN))]
reg5_piv = reg5.pivot(index="method", columns="scope", values="PICP").reindex(MAIN)
reg5_piv = reg5_piv[["all", "clear", "transitional", "cloudy"]]
reg5_piv.to_csv(TAB / "j2_picp90_5min_by_regime.csv")

# CRPS @5min by method (quantile methods)
cr5 = cr[(cr.horizon_min == 5) & (cr.scope == "all")].set_index("method")["CRPS"]
cr5.to_csv(TAB / "j2_crps5_by_method.csv")

# ---- figures ----
HUES = {"icp": "#888", "icp_norm": "#b0a", "mondrian": "#1b7", "cqr": "#e8a",
        "mondrian_cqr": "#d22", "aci": "#27c", "aci_regime": "#063"}
hz = [5, 15, 30, 60]

# Fig 1: PICP vs horizon (target 0.90)
plt.figure(figsize=(7, 4.2))
for m in MAIN:
    if m in picp_piv.index:
        plt.plot(hz, [picp_piv.loc[m, h] for h in hz], "-o", color=HUES[m], label=PRETTY[m])
plt.axhline(0.90, ls="--", c="k", lw=1, label="nominal 0.90")
plt.xlabel("horizon (min)"); plt.ylabel("PICP (all conditions)"); plt.ylim(0.80, 1.005)
plt.title("J2: marginal coverage vs horizon (90% target)"); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "j2_picp_vs_horizon.png", dpi=200); plt.close()

# Fig 2: per-regime PICP @5min (bar)
plt.figure(figsize=(7.5, 4.2))
regs = ["clear", "transitional", "cloudy"]; x = np.arange(len(regs)); w = 0.12
for i, m in enumerate(MAIN):
    if m in reg5_piv.index:
        plt.bar(x + i*w, [reg5_piv.loc[m, r] for r in regs], w, color=HUES[m], label=PRETTY[m])
plt.axhline(0.90, ls="--", c="k", lw=1)
plt.xticks(x + 3*w, regs); plt.ylabel("PICP (5-min, 90% target)"); plt.ylim(0.4, 1.02)
plt.title("J2: per-regime calibration (5-min)"); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "j2_picp_by_regime_5min.png", dpi=200); plt.close()

# Fig 3: reliability over time (monthly PICP @90%, 5-min)
r5 = rel[(rel.horizon_min == 5)].copy()
plt.figure(figsize=(8, 4.2))
for m in MAIN:
    sub = r5[r5.method == m].sort_values("bucket")
    if len(sub):
        plt.plot(sub["bucket"], sub["PICP"], "-o", ms=3, color=HUES[m], label=PRETTY[m])
plt.axhline(0.90, ls="--", c="k", lw=1)
plt.xticks(rotation=45, fontsize=7); plt.ylabel("monthly PICP (5-min)"); plt.ylim(0.6, 1.02)
plt.title("J2: reliability over time across 2024 (90% target)"); plt.legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(FIG / "j2_reliability_over_time.png", dpi=200); plt.close()

# Fig 4: ACI gamma sensitivity (PICP @90% all vs horizon)
gam = iv[(iv.nominal == 0.90) & (iv.scope == "all") & (iv.method.str.startswith("aci_g"))]
plt.figure(figsize=(7, 4.2))
for g, sub in gam.groupby("method"):
    sub = sub.sort_values("horizon_min")
    plt.plot(sub["horizon_min"], sub["PICP"], "-o", label=g.replace("aci_g", "γ="))
plt.axhline(0.90, ls="--", c="k", lw=1)
plt.xlabel("horizon (min)"); plt.ylabel("PICP (all, 90% target)")
plt.title("J2: ACI learning-rate (γ) sensitivity"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "j2_aci_gamma_sensitivity.png", dpi=200); plt.close()

# ---- headline numbers ----
def ace_rms(df):
    return float(np.sqrt(np.mean(df["ACE"].values**2)))
summary = {}
for m in MAIN:
    sub = iv[(iv.method == m) & (iv.scope.isin(["clear", "transitional", "cloudy"])) & (iv.nominal == 0.90)]
    summary[m] = dict(
        picp90_all_5min=float(picp_piv.loc[m, 5]) if m in picp_piv.index else None,
        pinaw90_all_5min=float(pinaw_piv.loc[m, 5]) if m in pinaw_piv.index else None,
        regime_ACE_rms_90=round(ace_rms(sub), 4),
        crps_5min=float(cr5.get(m, np.nan)) if m in cr5.index else None,
    )
json.dump(summary, open(CFG.MET / "j2_summary.json", "w"), indent=2)

print("== PICP @90% all-conditions (method x horizon) ==")
print(picp_piv.round(3).to_string())
print("\n== PICP @90% 5-min by regime ==")
print(reg5_piv.round(3).to_string())
print("\n== regime ACE RMS @90% (lower=better calibrated across regimes) ==")
for m in MAIN: print(f"  {PRETTY[m]:14s} {summary[m]['regime_ACE_rms_90']:.4f}   CRPS5={summary[m]['crps_5min']}")
print("\n== ACI gamma sweep PICP@90 all ==")
print(gam.pivot(index="method", columns="horizon_min", values="PICP").round(3).to_string())
print("\nWrote j2_* tables, figures, metrics.")
