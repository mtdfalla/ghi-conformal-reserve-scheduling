"""Final pass, aggregation — turns the per-horizon JSONs into the r1_ tables and figures.

Run from 03_code:  python3 r1/r1_j2_aggregate.py

Writes (all under 04_results, all `r1_`-prefixed; no Phase-6 file is touched):
  tables/r1_j2_interval_metrics.csv   every method x variant x horizon x nominal x scope
  tables/r1_j2_aci_delayed.csv difference table: first pass | anticipative | delayed
  tables/r1_j2_aci_gamma.csv gamma sweep, both variants
  tables/r1_j2_reliability.csv        reliability over time, both variants
  tables/r1_j2_ace_rms_5min.csv corrected ACE-RMS (5 min only) vs the pooled bug
  tables/r1_j2_crossing.csv  metrics/r1_crossing.json
  tables/r1_j2_bound_cap.csv metrics/r1_bound_cap.json
  figures/r1_j2_aci_delayed_picp.png, r1_j2_aci_gamma_delayed.png,
          r1_j2_reliability_delayed.png
"""
import sys, os, json, glob; sys.path.insert(0, "utils")
import numpy as np, pandas as pd
import config as CFG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
                     'lines.linewidth': 2.0, 'lines.markersize': 6, 'savefig.dpi': 200,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})

OUT = "/tmp/r1j2"
MAIN = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "aci", "aci_regime"]
PRETTY = {"icp": "ICP", "icp_norm": "ICP-norm", "mondrian": "Mondrian", "cqr": "CQR",
          "mondrian_cqr": "Mondrian-CQR", "aci": "ACI", "aci_regime": "ACI-regime"}
# first-pass ACE-RMS column (captioned "5 min", actually pooled over horizons)
PAPER_ACE_RMS = {"icp": 0.114, "icp_norm": 0.069, "cqr": 0.033, "mondrian": 0.010,
                 "mondrian_cqr": 0.009, "aci": 0.026, "aci_regime": 0.020}

iv = []; cr = []; rel = []; cross = {}; cap = []; ident = {}
for f in sorted(glob.glob(f"{OUT}/*.json")):
    j = json.load(open(f))
    iv += j["intervals"]; cr += j["crps"]; rel += j["reliability"]; cap += j["bound_cap"]
    key = os.path.basename(f).replace(".json", "")
    cross[key] = j["crossing"]
    if j.get("h1_identity"):
        ident[key] = j["h1_identity"]
iv = pd.DataFrame(iv); cr = pd.DataFrame(cr); rel = pd.DataFrame(rel); cap = pd.DataFrame(cap)

iv.to_csv(CFG.TAB / "r1_j2_interval_metrics.csv", index=False)
cr.to_csv(CFG.TAB / "r1_j2_crps.csv", index=False)
rel.to_csv(CFG.TAB / "r1_j2_reliability.csv", index=False)
cap.to_csv(CFG.TAB / "r1_j2_bound_cap.csv", index=False)
json.dump(dict(crossing=cross, note="rate = fraction of test points whose quantile "
               "sequence is non-monotone in tau"), open(CFG.MET / "r1_crossing.json", "w"), indent=2)
json.dump(dict(bound_cap=cap.to_dict("records"),
               note="upper bound capped at 1.5*G^cs, the same clip applied to the point "
                    "reconstruction; lower bound already clipped at 0 in all methods"),
          open(CFG.MET / "r1_bound_cap.json", "w"), indent=2)

rowsx = []; rowsp = []
for k, v in cross.items():
    site, hh = k.rsplit("_h", 1)
    for which, d in v.items():
        d = dict(d); pairs = d.pop("by_adjacent_pair", {})
        rowsx.append(dict(site=site, horizon_min=int(hh) * 5, set=which, **d))
        for pname, pv in pairs.items():
            rowsp.append(dict(site=site, horizon_min=int(hh) * 5, set=which, pair=pname, **pv))
pd.DataFrame(rowsx).to_csv(CFG.TAB / "r1_j2_crossing.csv", index=False)
pd.DataFrame(rowsp).to_csv(CFG.TAB / "r1_j2_crossing_by_pair.csv", index=False)

# ------------------------------------------------------- difference table
r0 = pd.read_csv(CFG.TAB / "j2_interval_metrics.csv")   # Phase-6 published values
r0 = r0[r0.method.isin(["aci", "aci_regime"])].rename(
    columns={"PICP": "PICP_R0", "ACE": "ACE_R0", "PINAW": "PINAW_R0", "Winkler": "Winkler_R0"})
r0 = r0[["method", "horizon_min", "nominal", "scope", "PICP_R0", "ACE_R0", "PINAW_R0", "Winkler_R0"]]

aci = iv[iv.method.isin(["aci", "aci_regime"])]
ant = aci[aci.variant == "anticipative"].rename(
    columns={"PICP": "PICP_ant", "PINAW": "PINAW_ant", "Winkler": "Winkler_ant", "ACE": "ACE_ant"})
dly = aci[aci.variant == "delayed"].rename(
    columns={"PICP": "PICP_delayed", "PINAW": "PINAW_delayed", "Winkler": "Winkler_delayed",
             "ACE": "ACE_delayed"})
keys = ["site", "method", "horizon_min", "nominal", "scope"]
tab = ant[keys + ["n", "PICP_ant", "ACE_ant", "PINAW_ant", "Winkler_ant"]].merge(
    dly[keys + ["PICP_delayed", "ACE_delayed", "PINAW_delayed", "Winkler_delayed"]], on=keys)
tab = tab.merge(r0, how="left", left_on=["method", "horizon_min", "nominal", "scope"],
                right_on=["method", "horizon_min", "nominal", "scope"])
tab.loc[tab.site != "yulara", ["PICP_R0", "ACE_R0", "PINAW_R0", "Winkler_R0"]] = np.nan
tab["d_PICP_delay"] = (tab.PICP_delayed - tab.PICP_ant).round(4)          # effect of the fix
tab["d_PICP_refit"] = (tab.PICP_ant - tab.PICP_R0).round(4)               # effect of the re-fit
tab["d_PICP_total_vs_R0"] = (tab.PICP_delayed - tab.PICP_R0).round(4)
tab["d_PINAW_delay"] = (tab.PINAW_delayed - tab.PINAW_ant).round(4)
tab["d_Winkler_delay"] = (tab.Winkler_delayed - tab.Winkler_ant).round(2)
tab = tab.sort_values(["site", "method", "nominal", "horizon_min", "scope"])
tab.to_csv(CFG.TAB / "r1_j2_aci_delayed.csv", index=False)

gam = iv[iv.method.str.startswith("aci_g")]
gam.to_csv(CFG.TAB / "r1_j2_aci_gamma.csv", index=False)

# --------------------------------------------------------------- ACE-RMS
def ace_rms(df):
    return float(np.sqrt(np.mean(np.asarray(df["ACE"].values, float) ** 2)))

rows = []
for site in sorted(iv.site.unique()):
    for m in MAIN:
        variant = "delayed" if m.startswith("aci") else "static"
        base = iv[(iv.site == site) & (iv.method == m) & (iv.variant == variant) &
                  (iv.scope.isin(["clear", "transitional", "cloudy"])) & (iv.nominal == 0.90)]
        five = base[base.horizon_min == 5]
        anti = iv[(iv.site == site) & (iv.method == m) &
                  (iv.variant == ("anticipative" if m.startswith("aci") else "static")) &
                  (iv.scope.isin(["clear", "transitional", "cloudy"])) & (iv.nominal == 0.90)]
        rows.append(dict(
            site=site, method=m, method_pretty=PRETTY[m],
            ace_rms_5min=round(ace_rms(five), 4) if len(five) else np.nan,
            ace_rms_pooled_all_horizons=round(ace_rms(base), 4) if len(base) else np.nan,
            ace_rms_pooled_anticipative=round(ace_rms(anti), 4) if len(anti) else np.nan,
            paper_value=PAPER_ACE_RMS.get(m) if site == "yulara" else np.nan,
            n_regime_rows_5min=int(len(five)), n_regime_rows_pooled=int(len(base))))
sf1 = pd.DataFrame(rows)
sf1["delta_vs_paper"] = (sf1.ace_rms_5min - sf1.paper_value).round(4)
sf1.to_csv(CFG.TAB / "r1_j2_ace_rms_5min.csv", index=False)

# ------------------------------------------------------------------- figures
HUES = {"anticipative": "#c33", "delayed": "#27c"}
for site in sorted(iv.site.unique()):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, m in zip(axes, ["aci", "aci_regime"]):
        for var in ["anticipative", "delayed"]:
            sub = iv[(iv.site == site) & (iv.method == m) & (iv.variant == var) &
                     (iv.nominal == 0.90) & (iv.scope == "all")].sort_values("horizon_min")
            ax.plot(sub.horizon_min, sub.PICP, marker="o", color=HUES[var],
                    ls="--" if var == "anticipative" else "-",
                    label=f"{var} feedback")
        sub0 = r0[(r0.method == m) & (r0.nominal == 0.90) & (r0.scope == "all")].sort_values("horizon_min")
        if site == "yulara" and len(sub0):
            ax.plot(sub0.horizon_min, sub0.PICP_R0, "s:", color="#888", label="R0 published")
        ax.axhline(0.90, ls="--", c="k", lw=1)
        ax.set_xlabel("horizon (min)"); ax.set_title(PRETTY[m]); ax.legend()
    axes[0].set_ylabel("PICP (all conditions, 90% target)")
    # Figure-level title deliberately not drawn: the caption in the article carries the
    # description, and a title above a caption is redundant. Per-panel titles are kept.
    # fig.suptitle(f"effect of h-step delayed ACI feedback ({site})")
    fig.tight_layout(); fig.savefig(CFG.FIG / f"r1_j2_aci_delayed_picp_{site}.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, var in zip(axes, ["anticipative", "delayed"]):
        sub = gam[(gam.site == site) & (gam.variant == var) & (gam.nominal == 0.90)]
        for g, s in sub.groupby("method"):
            s = s.sort_values("horizon_min")
            ax.plot(s.horizon_min, s.PICP, "-o", label=g.replace("aci_g", "γ="))
        ax.axhline(0.90, ls="--", c="k", lw=1)
        ax.set_xlabel("horizon (min)"); ax.set_title(f"{var} feedback"); ax.legend()
    axes[0].set_ylabel("PICP (all, 90% target)")
    # Figure-level title deliberately not drawn; see note above.
    # fig.suptitle(f"ACI learning-rate sensitivity ({site})")
    fig.tight_layout(); fig.savefig(CFG.FIG / f"r1_j2_aci_gamma_delayed_{site}.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, hm in zip(axes, [5, 60]):
        for m in ["aci", "aci_regime"]:
            for var in ["anticipative", "delayed"]:
                s = rel[(rel.site == site) & (rel.method == m) & (rel.variant == var) &
                        (rel.horizon_min == hm)].sort_values("bucket")
                if len(s):
                    ax.plot(s.bucket, s.PICP, marker="o", ms=3,
                            ls="--" if var == "anticipative" else "-",
                            label=f"{PRETTY[m]} ({var[:4]})")
        ax.axhline(0.90, ls="--", c="k", lw=1); ax.set_title(f"{hm} min")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    axes[0].set_ylabel("monthly PICP (90% target)"); axes[0].legend(fontsize=7)
    # Figure-level title deliberately not drawn; see note above.
    # fig.suptitle(f"reliability over time, anticipative vs delayed ({site})")
    fig.tight_layout(); fig.savefig(CFG.FIG / f"r1_j2_reliability_delayed_{site}.png"); plt.close(fig)

json.dump(ident, open(CFG.MET / "r1_j2_h1_identity.json", "w"), indent=2)

print("== h=1 delayed == anticipative (bitwise) ==")
print(json.dumps(ident, indent=1))
print("\n== regime ACE-RMS @90% ==")
print(sf1[sf1.site == "yulara"][["method_pretty", "paper_value", "ace_rms_5min",
                                 "ace_rms_pooled_all_horizons", "delta_vs_paper"]].to_string(index=False))
print("\n== ACI PICP @90% all-scope, Yulara ==")
p = tab[(tab.site == "yulara") & (tab.nominal == 0.90) & (tab.scope == "all")]
print(p[["method", "horizon_min", "PICP_R0", "PICP_ant", "PICP_delayed",
         "d_PICP_delay", "PINAW_ant", "PINAW_delayed"]].to_string(index=False))
print("\n== quantile crossing ==")
print(pd.DataFrame(rowsx)[["site", "horizon_min", "set", "n_crossing", "rate",
                           "mean_violation_Wm2"]].to_string(index=False))
print("\nWrote r1_j2_* tables and figures.")
