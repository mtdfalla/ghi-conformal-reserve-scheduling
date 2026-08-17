"""Final pass, dispatch half — paired day-block bootstrap on VALUE-CAPTURED DIFFERENCES.

S1 produced marginal confidence intervals for VC(mean) and VC(CVaR) per policy
(`r1_j5_cvar_ci.csv`). A ranking claim ("Mondrian-CQR captures more of the gap than
ICP") needs the interval on the DIFFERENCE, which is tighter and is the correct
test: two overlapping marginal CIs do not imply an insignificant difference, and
two disjoint ones are a conservative proxy. This script resamples whole days
(paired: the same day draw is applied to every policy and to the deterministic
anchor) and reports the CI of VC_A - VC_B directly.

VC = 1 - cost / cost_deterministic, exactly as in `r1_dispatch.py:row`.
Each policy is evaluated at its Protocol-A theta, i.e. the selection that excludes the test year.

Usage (from 03_code):  python3 r1/r1_j5_vc_stats.py [B]
Writes 04_results/tables/r1_j5_vc_diff_ci.csv
"""
import sys, itertools; sys.path.insert(0, "utils")
import numpy as np, pandas as pd
import config as CFG

B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED = 42
CVAR_ALPHA = 0.95
POLS = ["icp", "mondrian", "cqr", "mondrian_cqr"]

z = np.load(CFG.TAB / "r1_j5_daily_costs.npz", allow_pickle=True)
BAF = pd.read_csv(CFG.TAB / "r1_j5_before_after.csv")
A = BAF[BAF.protocol == "A"]


def cvar_rows(X, alpha=CVAR_ALPHA):
    k = max(1, int(np.ceil((1 - alpha) * X.shape[1])))
    part = np.partition(X, X.shape[1] - k, axis=1)[:, X.shape[1] - k:]
    return part.mean(axis=1)


rows = []
for (site, hm), g in A.groupby(["site", "horizon_min"]):
    tag = f"{site}_h{hm // 5}"
    det = z[f"{tag}|deterministic|na"]
    nd = len(det)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nd, size=(B, nd))
    detB = det[idx]
    det_mean_b = detB.mean(axis=1); det_cvar_b = cvar_rows(detB)

    vcm = {}; vcc = {}; theta = {}
    for _, r in g.iterrows():
        pol = r["policy"]
        if pol not in POLS:
            continue
        th = float(r["theta"])
        key = f"{tag}|{pol}|{th:g}"
        if key not in z.files:
            print("missing", key); continue
        dc = z[key][idx]
        vcm[pol] = 1 - dc.mean(axis=1) / det_mean_b
        vcc[pol] = 1 - cvar_rows(dc) / det_cvar_b
        theta[pol] = th

    for a, b in itertools.combinations([p for p in POLS if p in vcm], 2):
        for name, dd in (("VC_mean", vcm[a] - vcm[b]), ("VC_cvar", vcc[a] - vcc[b])):
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append(dict(site=site, horizon_min=hm, metric=name,
                             policy_a=a, policy_b=b,
                             theta_a=theta[a], theta_b=theta[b],
                             diff_point=round(float(dd.mean()), 4),
                             ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                             p_two_sided=round(float(2 * min((dd <= 0).mean(), (dd >= 0).mean())), 5),
                             significant_95="yes" if (lo > 0) == (hi > 0) else "no",
                             n_days=nd, B=B))

out = pd.DataFrame(rows).sort_values(["site", "metric", "horizon_min", "policy_a", "policy_b"])
out.to_csv(CFG.TAB / "r1_j5_vc_diff_ci.csv", index=False)
print(out[(out.site == "yulara") & (out.policy_a == "icp") & (out.policy_b == "mondrian_cqr")]
      .to_string(index=False))
print(f"\nwrote r1_j5_vc_diff_ci.csv ({len(out)} rows)")
