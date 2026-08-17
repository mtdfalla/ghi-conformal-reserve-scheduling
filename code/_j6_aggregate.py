"""J6 aggregation: drift + ablations -> tables, figures, summary.
Run from 03_code: python3 _j6_aggregate.py
"""
import sys, json, glob; sys.path.insert(0, "utils")
import config as CFG
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'lines.linewidth':2.2,'lines.markersize':7,'savefig.dpi':200,'figure.dpi':200,'savefig.bbox':'tight'})

OUT = "/tmp/j6out"; TAB = CFG.TAB; FIG = CFG.FIG

# ---- drift ----
drift = []
for f in sorted(glob.glob(f"{OUT}/drift_*.json")):
    j = json.load(open(f))
    if j.get("skipped"): continue
    drift += j["rows"]
drift = pd.DataFrame(drift)
drift.to_csv(TAB / "j6_drift_coverage.csv", index=False)
HUES = {"icp": "#888", "mondrian": "#1b7", "aci": "#27c"}
PRETTY = {"icp": "ICP", "mondrian": "Mondrian", "aci": "ACI"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for k, scope in enumerate(["all", "transitional"]):
    sub = drift[drift.scope == scope]
    for m in ["icp", "mondrian", "aci"]:
        s = sub[sub.method == m].sort_values("year")
        ax[k].plot(s["year"], s["PICP"], "-o", color=HUES[m], label=PRETTY[m])
    ax[k].axhline(0.90, ls="--", c="k", lw=1); ax[k].set_title(f"{scope}")
    ax[k].set_xlabel("test year"); ax[k].set_ylim(0.55, 1.02); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("PICP (5-min, 90% target)"); ax[1].legend(fontsize=9)
# Figure-level title deliberately not drawn: the caption in the article carries the
# description, and a title above a caption is redundant. Per-panel titles are kept.
# plt.suptitle("J6: coverage stability across years (expanding-window deployment)")
plt.tight_layout(); plt.savefig(FIG / "j6_drift_picp_by_year.png", dpi=200); plt.close()

# drift summary stats: mean & std of PICP across years (all scope)
dstat = drift[drift.scope == "all"].groupby("method")["PICP"].agg(["mean", "std", "min", "max"]).round(4)
dstat_tr = drift[drift.scope == "transitional"].groupby("method")["PICP"].agg(["mean", "std", "min", "max"]).round(4)
dstat.to_csv(TAB / "j6_drift_summary_all.csv"); dstat_tr.to_csv(TAB / "j6_drift_summary_transitional.csv")

# ---- calibration-set size ----
calib = pd.DataFrame(json.load(open(f"{OUT}/calib.json")))
calib.to_csv(TAB / "j6_calib_size.csv", index=False)
plt.figure(figsize=(7, 4.3))
for m in ["icp", "mondrian"]:
    s = calib[calib.method == m].sort_values("calib_months")
    plt.plot(s["calib_months"], s["PICP"], "-o", color=HUES[m], label=PRETTY[m])
plt.axhline(0.90, ls="--", c="k", lw=1)
plt.xlabel("calibration window (months of 2023)"); plt.ylabel("PICP (test 2024, 90% target)")
# Figure-level title deliberately not drawn; see note above.
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "j6_calib_size.png", dpi=200); plt.close()

# ---- feature ablation ----
feats = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/feat_*.json"))]
feats = pd.DataFrame(feats).sort_values("n_features", ascending=False)
feats.to_csv(TAB / "j6_feature_ablation.csv", index=False)

print("== DRIFT: PICP across years (all) ==")
print(drift[drift.scope == "all"].pivot(index="year", columns="method", values="PICP").round(3).to_string())
print("\n== DRIFT summary (all-scope PICP across years) =="); print(dstat.to_string())
print("\n== DRIFT (transitional) PICP across years =="); print(dstat_tr.to_string())
print("\n== FEATURE ablation ==")
print(feats[["set", "n_features", "RMSE", "PICP_mondrian", "PINAW_mondrian"]].to_string(index=False))

json.dump(dict(drift_all=dstat.to_dict(), drift_transitional=dstat_tr.to_dict(),
               feature_ablation=feats.to_dict("records")),
          open(CFG.MET / "j6_summary.json", "w"), indent=2, default=str)
print("\nWrote j6_* tables, figures, summary.")
