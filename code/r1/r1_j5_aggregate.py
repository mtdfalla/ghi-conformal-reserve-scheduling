"""Final pass — aggregate the honest-theta dispatch runs into tables and figures.

Reads  /tmp/r1j5out/{site}_h{h}_all.json  +  {site}_h{h}_daily.npz
Writes 04_results/tables/r1_j5_*.csv, r1_j3_dispatch.csv
       04_results/figures/r1_j5_*.png, r1_ghi_pv_map.png
       04_results/metrics/r1_j5_summary.json
       04_results/tables/r1_j5_daily_costs.npz

Run from 03_code:  python3 r1/r1_j5_aggregate.py
"""
import sys, os, json, glob; sys.path.insert(0, "utils")
import numpy as np, pandas as pd, config as CFG
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
                     'lines.linewidth': 2.2, 'lines.markersize': 7, 'savefig.dpi': 200,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})
TAB, FIG, MET = CFG.TAB, CFG.FIG, CFG.MET
PRETTY = {"deterministic": "Deterministic", "icp": "ICP", "mondrian": "Mondrian",
          "cqr": "CQR", "mondrian_cqr": "Mondrian-CQR", "oracle": "Oracle"}
HUES = {"deterministic": "#444", "icp": "#888", "mondrian": "#1b7", "cqr": "#e8a",
        "mondrian_cqr": "#d22", "oracle": "#27c"}
SITEN = {"yulara": "Yulara", "asp": "DKASC Alice Springs"}

rows, theta_log, cis, front, batt, cost, daily = [], [], [], [], [], [], {}
for f in sorted(glob.glob("/tmp/r1j5out/*_all.json")):
    j = json.load(open(f))
    m = j["main"]
    rows += m["rows"]; theta_log += m["theta_log"]; cis += m["cis"]; front += m["frontier"]
    batt += j["battery"]; cost += j["costratio"]
    tag = os.path.basename(f).replace("_all.json", "")
    z = np.load(f"/tmp/r1j5out/{tag}_daily.npz")
    for k in z.files:
        daily[f"{tag}|{k}"] = z[k]

R = pd.DataFrame(rows); TL = pd.DataFrame(theta_log); CI = pd.DataFrame(cis)
FR = pd.DataFrame(front); BA = pd.DataFrame(batt); CO = pd.DataFrame(cost)
for df in (R, TL, CI, FR, BA, CO):
    df.sort_values([c for c in ["site", "horizon_min", "policy"] if c in df.columns], inplace=True)

# ---------------------------------------------------------------- tables
TL.to_csv(TAB / "r1_j5_theta_selection.csv", index=False)
R.to_csv(TAB / "r1_j5_protocols.csv", index=False)                 # / full grid
CI.to_csv(TAB / "r1_j5_cvar_ci.csv", index=False)
FR.to_csv(TAB / "r1_j5_frontier.csv", index=False)                 # efficient set
BA.to_csv(TAB / "r1_j5_battery.csv", index=False)
CO.to_csv(TAB / "r1_j5_costratio.csv", index=False)
R[R.site == "asp"].to_csv(TAB / "r1_j3_dispatch.csv", index=False)
R[R.protocol.str.startswith("A_meancvar") | (R.protocol == "A")].to_csv(
    TAB / "r1_j5_meancvar.csv", index=False)
np.savez_compressed(TAB / "r1_j5_daily_costs.npz", **daily)         # raw vectors

# ---- THE before/after table (the S1 exit gate) ----
KEEP = ["R0_test_argmin_coarse", "R0_test_argmin_fine", "A", "A_swapped",
        "A_meancvar_lam0.5", "B_critical_fractile"]
# The "R0_*" keys are protocol identifiers stored in the result CSVs and in the published
# figure legends, so they are left as written; they denote the test-tuned selection of the
# first analysis pass.
LABEL = {"R0_test_argmin_coarse": "R0 (theta = argmin on TEST 2024, coarse grid)",
         "R0_test_argmin_fine":   "R0 protocol on the fine grid (still TEST-tuned)",
         "A":                     "A: theta selected on 2023-H2 (mean cost)",
         "A_swapped":             "A-swapped: theta selected on 2023-H1",
         "A_meancvar_lam0.5":     "A: theta selected on 2023-H2 (mean-CVaR, lambda=0.5)",
         "B_critical_fractile":   "B: theta = c_u/(c_u+c_o) = 0.909, zero tuning"}
BAF = R[R.protocol.isin(KEEP)].copy()
BAF["protocol_label"] = BAF["protocol"].map(LABEL)
ci_key = CI.set_index(["site", "horizon_min", "protocol", "policy"])
def attach(r):
    k = (r["site"], r["horizon_min"], r["protocol"], r["policy"])
    if k in ci_key.index:
        c = ci_key.loc[k]
        return pd.Series({"cvar_ci_lo": c["cvar_ci"][0], "cvar_ci_hi": c["cvar_ci"][1],
                          "vc_cvar_ci_lo": c["vc_cvar_ci"][0], "vc_cvar_ci_hi": c["vc_cvar_ci"][1],
                          "vc_mean_ci_lo": c["vc_mean_ci"][0], "vc_mean_ci_hi": c["vc_mean_ci"][1]})
    return pd.Series({k2: np.nan for k2 in ["cvar_ci_lo", "cvar_ci_hi", "vc_cvar_ci_lo",
                                            "vc_cvar_ci_hi", "vc_mean_ci_lo", "vc_mean_ci_hi"]})
BAF = pd.concat([BAF, BAF.apply(attach, axis=1)], axis=1)
BAF = BAF[["site", "horizon_min", "policy", "protocol", "protocol_label", "theta",
           "mean_daily", "cvar95_daily", "value_captured_mean", "value_captured_cvar",
           "vc_mean_ci_lo", "vc_mean_ci_hi", "vc_cvar_ci_lo", "vc_cvar_ci_hi",
           "cvar_ci_lo", "cvar_ci_hi"]]
BAF.to_csv(TAB / "r1_j5_before_after.csv", index=False)

# ---- reproduction delta vs the first-pass tables ----
try:
    r0 = pd.read_csv(TAB / "j5_frontier.csv")
    new = FR[(FR.site == "yulara") & (FR.set == "test2024")]
    j = r0.merge(new, left_on=["policy", "rho", "horizon_min"],
                 right_on=["policy", "theta", "horizon_min"], suffixes=("_R0", "_R1"))
    j["d_mean"] = (j.mean_daily_R1 - j.mean_daily_R0).round(2)
    j["d_cvar"] = (j.cvar95_daily_R1 - j.cvar95_daily_R0).round(2)
    j["rel_mean_pct"] = (100 * (j.mean_daily_R1 - j.mean_daily_R0) / j.mean_daily_R0).round(1)
    j[["horizon_min", "policy", "rho", "mean_daily_R0", "mean_daily_R1", "d_mean",
       "rel_mean_pct", "cvar95_daily_R0", "cvar95_daily_R1", "d_cvar"]].to_csv(
        TAB / "r1_j5_r0_reproduction.csv", index=False)
except Exception as e:
    print("reproduction table skipped:", e)

# ---------------------------------------------------------------- figures
# F1 risk-cost frontier, test year, per site x horizon ( efficient set)
for site in FR.site.unique():
    hs = sorted(FR[FR.site == site].horizon_min.unique())
    fig, ax = plt.subplots(1, len(hs), figsize=(4.6 * len(hs), 4.2))
    ax = np.atleast_1d(ax)
    for a, hm in zip(ax, hs):
        s = FR[(FR.site == site) & (FR.horizon_min == hm) & (FR.set == "test2024")]
        for pol in ["icp", "mondrian", "cqr", "mondrian_cqr"]:
            ss = s[s.policy == pol].sort_values("theta")
            a.plot(ss["mean_daily"], ss["cvar95_daily"], "-o", color=HUES[pol], label=PRETTY[pol],
                   ms=4.5, lw=1.8)
        d = R[(R.site == site) & (R.horizon_min == hm) & (R.policy == "deterministic")].iloc[0]
        a.scatter([d.mean_daily], [d.cvar95_daily], color="#444", marker="*", s=120,
                  label="Deterministic", zorder=5)
        a.scatter([0], [0], color="#27c", marker="*", s=120, label="Oracle", zorder=5)
        for pol in ["icp", "mondrian_cqr"]:
            t = TL[(TL.site == site) & (TL.horizon_min == hm) & (TL.policy == pol)]
            if len(t):
                th = float(t.theta_protocolA.iloc[0])
                pt = s[(s.policy == pol) & (np.isclose(s.theta, th))]
                if len(pt):
                    a.scatter(pt.mean_daily, pt.cvar95_daily, facecolors="none",
                              edgecolors="k", s=160, lw=1.6, zorder=6)
        a.set_title(f"{hm} min"); a.set_xlabel("E[daily reserve cost] (USD)"); a.grid(alpha=0.3)
    ax[0].set_ylabel(r"CVaR$_{0.95}$ daily cost (USD)")
    ax[-1].legend(fontsize=8)
    # Figure-level title deliberately not drawn: the caption in the article carries the
    # description, and a title above a caption is redundant. Per-panel titles are kept.
    # plt.suptitle(f"{SITEN[site]}: risk-cost frontier; circled = reserve level chosen "
    #              f"without the test year (Protocol A)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / f"r1_j5_frontier_{site}.png"); plt.close()

# F2 theta stability: selected theta by protocol
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, site in zip(ax, ["yulara", "asp"]):
    t = TL[TL.site == site]
    for pol in ["icp", "mondrian", "cqr", "mondrian_cqr"]:
        s = t[t.policy == pol].sort_values("horizon_min")
        if not len(s): continue
        a.plot(s.horizon_min, s.theta_protocolA, "-o", color=HUES[pol], label=PRETTY[pol])
        a.plot(s.horizon_min, s.theta_R0_test_argmin_fine, "--s", color=HUES[pol], alpha=0.55, ms=5)
    a.axhline(0.909, color="k", ls=":", lw=1.4)
    a.text(30, 0.9115, r"Protocol B: $\theta=c_u/(c_u+c_o)=0.909$", fontsize=9)
    a.set_xlabel("horizon (min)"); a.set_title(SITEN[site]); a.grid(alpha=0.3)
    a.set_xticks([5, 15, 30, 60]); a.set_ylim(0.74, 0.945)
ax[0].set_ylabel(r"selected reserve level $\theta$")
ax[0].legend(fontsize=9, ncol=2, loc="lower right")
# Figure-level title deliberately not drawn; see note above.
# plt.suptitle(r"Selected $\theta$: solid = Protocol A (2023 only); dashed = R0 (argmin on test 2024)",
#              fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "r1_j5_theta_stability.png"); plt.close()

# F3 value captured under each protocol (the before/after picture)
for metric, lab in [("value_captured_mean", "expected cost"),
                    ("value_captured_cvar", r"tail risk (CVaR$_{0.95}$)")]:
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
    hs = [5, 15, 30, 60]
    pols = ["icp", "mondrian", "cqr", "mondrian_cqr"]
    protos = ["R0_test_argmin_fine", "A", "B_critical_fractile"]
    plab = ["R0 (test-tuned)", "A (honest)", "B (zero-tuning)"]
    for a, hm in zip(ax, hs):
        x = np.arange(len(pols)); w = 0.26
        for k, pr in enumerate(protos):
            v = [float(R[(R.site == "yulara") & (R.horizon_min == hm) & (R.policy == p) &
                         (R.protocol == pr)][metric].iloc[0]) * 100 for p in pols]
            a.bar(x + k * w, v, w, label=plab[k])
        a.set_xticks(x + w); a.set_xticklabels([PRETTY[p] for p in pols], rotation=20, fontsize=9)
        a.set_title(f"{hm} min"); a.grid(alpha=0.3, axis="y"); a.set_ylim(0, 100)
    ax[0].set_ylabel(r"% of deterministic$\to$oracle gap closed"); ax[0].legend(fontsize=9)
    # Figure-level title deliberately not drawn; see note above.
    # plt.suptitle(f"Yulara: value captured, {lab} - by reserve-level selection protocol", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG / f"r1_j5_value_captured_{'mean' if 'expected' in lab else 'cvar'}.png")
    plt.close()

# F4 battery sensitivity, all four horizons
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
order = ["small", "default", "large"]
for a, metric in zip(ax, ["value_captured_mean", "value_captured_cvar"]):
    for hm in [5, 15, 30, 60]:
        s = BA[(BA.site == "yulara") & (BA.horizon_min == hm) & (BA.policy == "mondrian_cqr")]
        v = [float(s[s.batt == b][metric].iloc[0]) * 100 for b in order if len(s[s.batt == b])]
        a.plot(range(len(v)), v, "-o", label=f"{hm} min")
    a.set_xticks(range(3))
    a.set_xticklabels([f"{b}\n({BA[BA.batt==b].E_max.iloc[0]:.0f} kWh /"
                       f" {BA[BA.batt==b].P_max.iloc[0]:.0f} kW)" for b in order], fontsize=9)
    a.grid(alpha=0.3); a.set_ylim(0, 100)
    a.set_title("expected cost" if "mean" in metric else r"tail risk (CVaR$_{0.95}$)")
ax[0].set_ylabel("Mondrian-CQR value captured (%)"); ax[0].legend(fontsize=9)
# Figure-level title deliberately not drawn; see note above.
# plt.suptitle(r"Yulara: battery sensitivity, $\theta$ selected by Protocol A (never on the test year)",
#              fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "r1_j5_battery.png"); plt.close()

# F5 cost-ratio sensitivity
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for a, metric, lab in zip(ax, ["value_captured_mean", "value_captured_cvar"],
                          ["expected cost", r"tail risk (CVaR$_{0.95}$)"]):
    for pol in ["icp", "mondrian", "cqr", "mondrian_cqr"]:
        s = CO[(CO.site == "yulara") & (CO.horizon_min == 5) & (CO.policy == pol) &
               (CO.protocol == "A")].sort_values("ratio")
        a.plot(s.ratio, s[metric].astype(float) * 100, "-o", color=HUES[pol], label=PRETTY[pol])
    a.set_xlabel(r"cost ratio $c_u/c_o$"); a.set_xticks([3, 5, 10, 19])
    a.grid(alpha=0.3); a.set_title(lab); a.set_ylim(0, 100)
ax[0].set_ylabel("value captured (%)"); ax[0].legend(fontsize=9)
# Figure-level title deliberately not drawn; see note above.
# plt.suptitle(r"Yulara 5 min: cost-ratio sensitivity, $\theta$ re-selected per ratio (Protocol A)",
#              fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "r1_j5_costratio.png"); plt.close()

# F6 GHI->PV mapping validation
m = np.load("/tmp/ghi_pv_map.npz")
clean = pd.read_parquet(CFG.DATA_CLEAN / "yulara_clean_5min.parquet")
reg = pd.read_parquet(CFG.DATA_REGIME / "yulara_regimes_5min.parquet")
mm = pd.DataFrame({"ghi": clean["ghi"], "pv": clean["pv_total"], "is_day": reg["is_day"]})
mm = mm[(mm.index.year == 2023) & mm["is_day"] & mm["ghi"].notna() & mm["pv"].notna() & (mm["pv"] >= 0)]
fit = json.load(open(MET / "r1_ghi_pv_fit.json"))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].hexbin(mm["ghi"], mm["pv"], gridsize=70, mincnt=1, cmap="Blues", bins="log")
ax[0].plot(m["centers"], m["vals"], "-", color="#d22", lw=2.4, label=r"$f(G)$: binned median, capped")
ax[0].axhline(float(m["cap"]), color="k", ls=":", lw=1.3)
ax[0].text(20, float(m["cap"]) * 1.02, rf"$P_{{\max}}={float(m['cap']):.1f}$ kW", fontsize=10)
ax[0].set_xlabel(r"measured GHI $G$ (W/m$^2$)"); ax[0].set_ylabel("site PV active power (kW)")
ax[0].set_title(f"Yulara 2023 daytime (n={fit['n_points']:,})"); ax[0].legend(fontsize=9)
f = lambda g: np.clip(np.interp(g, m["centers"], m["vals"], left=0, right=float(m["cap"])), 0, float(m["cap"]))
res = mm["pv"].values - f(mm["ghi"].values)
ax[1].hexbin(f(mm["ghi"].values), res, gridsize=70, mincnt=1, cmap="Blues", bins="log")
ax[1].axhline(0, color="#d22", lw=2)
ax[1].set_xlabel(r"$f(G)$ (kW)"); ax[1].set_ylabel("residual PV (kW)")
ax[1].set_title(rf"$R^2$={fit['R2_vs_measured_PV']:.4f}, RMSE={fit['RMSE_kW']:.1f} kW, "
                rf"MAE={fit['MAE_kW']:.1f} kW")
# Figure-level title deliberately not drawn; see note above.
# plt.suptitle("GHI to PV conversion: deterministic curve and its residual scatter "
#              "(non-forecastable; biases all policies identically)", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "r1_ghi_pv_map.png"); plt.close()

# sensitivity: the R^2 above is depressed by daytime records with PV == 0
# (inverter offline / curtailment), visible as the flat band at pv = 0 and the diagonal
# arm in the residual panel. Report the fit with and without them; DO NOT overwrite
# r1_ghi_pv_fit.json (hard rule 2) - this is a separate artifact.
on = mm["pv"].values > 1.0
ss = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
y_on, yh_on = mm["pv"].values[on], f(mm["ghi"].values)[on]
sens = dict(note="Goodness of fit of f(G) excluding daytime records with site PV output == 0 "
                 "(inverter offline / curtailment). Reported alongside, never instead of, "
                 "r1_ghi_pv_fit.json, which uses every daytime record.",
            n_points_all=int(len(mm)), n_points_pv_zero=int((~on).sum()),
            pct_daytime_pv_zero=round(float(100 * (~on).mean()), 2),
            R2_all=round(float(ss(mm["pv"].values, f(mm["ghi"].values))), 4),
            R2_excl_zero_output=round(float(ss(y_on, yh_on)), 4),
            RMSE_kW_excl_zero_output=round(float(np.sqrt(np.mean((y_on - yh_on) ** 2))), 2),
            MAE_kW_excl_zero_output=round(float(np.mean(np.abs(y_on - yh_on))), 2))
json.dump(sens, open(MET / "r1_ghi_pv_fit_sensitivity.json", "w"), indent=2)
print("GHI->PV fit sensitivity:", json.dumps(sens, indent=1))

# ---------------------------------------------------------------- summary
def headline(site, hm, proto):
    s = R[(R.site == site) & (R.horizon_min == hm) & (R.protocol == proto)]
    s = s[s.policy.isin(["icp", "mondrian", "cqr", "mondrian_cqr"])]
    if not len(s): return None
    return dict(best_mean=str(s.loc[s.mean_daily.idxmin(), "policy"]),
                best_cvar=str(s.loc[s.cvar95_daily.idxmin(), "policy"]),
                mcqr_vc_mean=float(s[s.policy == "mondrian_cqr"].value_captured_mean.iloc[0]),
                mcqr_vc_cvar=float(s[s.policy == "mondrian_cqr"].value_captured_cvar.iloc[0]),
                icp_vc_mean=float(s[s.policy == "icp"].value_captured_mean.iloc[0]),
                icp_vc_cvar=float(s[s.policy == "icp"].value_captured_cvar.iloc[0]))

summary = dict(
    generated_by="03_code/r1/r1_dispatch.py + r1_j5_aggregate.py",
    theta_grid=list(np.round(np.arange(0.75, 0.9501, 0.025), 4)),
    protocols=LABEL,
    rankings={f"{s}|{hm}|{p}": headline(s, hm, p)
              for s in ["yulara", "asp"] for hm in [5, 15, 30, 60]
              for p in ["R0_test_argmin_fine", "A", "B_critical_fractile"]},
    theta_selection=TL.to_dict("records"),
    n_bootstrap=10000,
    soc_feasible=bool((R.soc_min >= 0.0989).all() and (R.soc_max <= 1.001).all()),
)
json.dump(summary, open(MET / "r1_j5_summary.json", "w"), indent=2, default=str)
print("wrote tables:", sorted(os.path.basename(p) for p in glob.glob(str(TAB / "r1_*"))))
print("wrote figures:", sorted(os.path.basename(p) for p in glob.glob(str(FIG / "r1_*"))))
print("SoC feasible:", summary["soc_feasible"])
