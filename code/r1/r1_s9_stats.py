"""Final pass - four statistical analyses that run entirely from committed result tables.

WHY THIS FILE EXISTS
--------------------
Four questions are answered by re-analysing files already in results/tables/, with no
model refit and no dispatch re-run:

  multiplicity      96 decision-layer p-values and 608 interval-layer rows would
                    otherwise be reported unadjusted. Holm is applied within
                    PRE-DECLARED families.
  selection regret  what the swapped-half reserve-level choice actually costs on the
                    test year. This is a stronger and more honest statement than exact
                    equality of the two selections.
  block bootstrap   the day bootstrap resamples days independently. A stationary
                    (geometric) block bootstrap over multi-day blocks is added as a
                    sensitivity, because adjacent weather days need not be exchangeable.
  repeated measures the per-day ANOVA treats repeated measurements as independent.
                    Refit with cluster-robust standard errors by day, and with day as a
                    random effect.

Every output takes the r1_ prefix and refuses to overwrite an existing file.

Usage (from the repository root):  python3 code/r1/r1_s9_stats.py [--force]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import statsmodels.api as sm             # noqa: E402
import statsmodels.formula.api as smf    # noqa: E402

import config as CFG                     # noqa: E402

R1 = "r1_"
SEED = 42
B_BOOT = 10000


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (n - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ---------------------------------------------------- multiplicity
def multiplicity(force):
    out = []

    # Family 1 (decision layer): all 96 value-captured pairwise comparisons.
    d = pd.read_csv(CFG.TAB / "r1_j5_vc_diff_ci.csv")
    d = d.assign(family="decision_vc_all96", p_holm=holm(d.p_two_sided.values))
    out.append(d[["family", "site", "horizon_min", "metric", "policy_a", "policy_b",
                  "diff_point", "p_two_sided", "p_holm"]])

    # Family 2 (decision layer, PRE-DECLARED PRIMARY): marginal ICP against the fused
    # regime-aware policy, both metrics, four horizons, primary site. This is the paper's
    # headline contrast and the one a reader is asked to act on.
    prim = d[(d.site == "yulara")
             & (d[["policy_a", "policy_b"]].apply(lambda r: set(r) == {"icp", "mondrian_cqr"}, axis=1))]
    prim = prim.assign(family="decision_vc_primary", p_holm=holm(prim.p_two_sided.values))
    out.append(prim[["family", "site", "horizon_min", "metric", "policy_a", "policy_b",
                     "diff_point", "p_two_sided", "p_holm"]])

    # Family 3 (interval layer): CRPS differences at the primary site, pooled scope,
    # day-length HAC Diebold-Mariano - the conservative variant the manuscript quotes.
    s = pd.read_csv(CFG.TAB / "r1_j2_stats.csv")
    s = s[(s.kind == "CRPS_diff") & (s.site == "yulara") & (s.scope == "all") & s.dm_p_hac_day.notna()]
    s = s.assign(family="interval_crps_yulara_all", p_holm=holm(s.dm_p_hac_day.values))
    out.append(s.rename(columns={"value": "diff_point", "dm_p_hac_day": "p_two_sided"})[
        ["family", "site", "horizon_min", "method_a", "method_b", "diff_point",
         "p_two_sided", "p_holm"]].assign(metric="CRPS"))

    res = pd.concat(out, ignore_index=True)
    res.to_csv(guarded(CFG.TAB / f"{R1}s9_multiplicity.csv", force), index=False)
    for fam, g in res.groupby("family"):
        raw = int((g.p_two_sided < 0.05).sum())
        adj = int((g.p_holm < 0.05).sum())
        print(f"  [multiplicity] {fam:26} n={len(g):3}  significant raw={raw:3}  after Holm={adj:3}")
    return res


# ------------------------------------------------ selection regret
def selection_regret(force):
    fr = pd.read_csv(CFG.TAB / "r1_j5_frontier.csv")
    fr = fr[fr.set == "test2024"]
    th = pd.read_csv(CFG.TAB / "r1_j5_theta_selection.csv")
    key = {(r.site, int(r.horizon_min), r.policy, round(float(r.theta), 4)): r
           for r in fr.itertuples()}
    rows = []
    for r in th.itertuples():
        a, s = round(float(r.theta_protocolA), 4), round(float(r.theta_protocolA_swapped), 4)
        ka = key.get((r.site, int(r.horizon_min), r.policy, a))
        ks = key.get((r.site, int(r.horizon_min), r.policy, s))
        if ka is None or ks is None:
            continue
        rows.append(dict(
            site=r.site, horizon_min=int(r.horizon_min), policy=r.policy,
            theta_A=a, theta_swapped=s, theta_shift=round(s - a, 4),
            mean_at_A=ka.mean_daily, mean_at_swapped=ks.mean_daily,
            regret_mean=round(ks.mean_daily - ka.mean_daily, 4),
            regret_mean_pct=round(100 * (ks.mean_daily - ka.mean_daily) / ka.mean_daily, 3),
            cvar_at_A=ka.cvar95_daily, cvar_at_swapped=ks.cvar95_daily,
            regret_cvar=round(ks.cvar95_daily - ka.cvar95_daily, 4),
            regret_cvar_pct=round(100 * (ks.cvar95_daily - ka.cvar95_daily) / ka.cvar95_daily, 3)))
    d = pd.DataFrame(rows)
    d.to_csv(guarded(CFG.TAB / f"{R1}s9_selection_regret.csv", force), index=False)
    print("  [selection regret] |regret| on the test year, over all 8 site x horizon cells per policy:")
    for pol, g in d.groupby("policy"):
        print(f"          {pol:13} mean|mean-regret|={g.regret_mean_pct.abs().mean():6.2f}%  "
              f"max={g.regret_mean_pct.abs().max():6.2f}%   "
              f"mean|CVaR-regret|={g.regret_cvar_pct.abs().mean():6.2f}%  "
              f"max={g.regret_cvar_pct.abs().max():6.2f}%")
    return d


# -------------------------------------------------- block bootstrap
def block_bootstrap(force):
    """Stationary (geometric-length) block bootstrap over consecutive operating days.

    The committed paired day-block bootstrap resamples days i.i.d., which preserves
    within-day dependence but treats days as exchangeable. Multi-day weather systems
    break that. Expected block lengths of 2, 3, 5 and 7 days are reported beside it.
    Resampling stays PAIRED: every policy and the deterministic anchor see the same
    day sequence, exactly as in r1_dispatch.boot_ci.
    """
    z = np.load(CFG.TAB / "r1_j5_daily_costs.npz", allow_pickle=True)
    keys = list(z.keys())
    rows = []
    rng = np.random.default_rng(SEED)

    def cvar(X, alpha=0.95):
        k = max(1, int(np.ceil((1 - alpha) * X.shape[1])))
        part = np.partition(X, X.shape[1] - k, axis=1)[:, X.shape[1] - k:]
        return part.mean(axis=1)

    def idx_stationary(n, exp_len, B, rng):
        """B x n index matrix; blocks start uniformly, lengths ~ Geometric(1/exp_len)."""
        p = 1.0 / exp_len
        out = np.empty((B, n), dtype=np.int64)
        for b in range(B):
            pos, fill = 0, []
            while pos < n:
                start = rng.integers(0, n)
                L = max(1, int(rng.geometric(p)))
                fill.append((np.arange(start, start + L) % n))
                pos += L
            out[b] = np.concatenate(fill)[:n]
        return out

    # keys are "<site>_h<steps>|<policy>|<theta>"; the anchors carry theta "na".
    # The reported policy is the one at the Protocol-A theta, so the sensitivity is
    # computed at exactly the operating point the paper reports.
    th = pd.read_csv(CFG.TAB / "r1_j5_theta_selection.csv")
    thA = {(r.site, int(r.horizon_min), r.policy): round(float(r.theta_protocolA), 4)
           for r in th.itertuples()}
    groups = sorted({k.split("|")[0] for k in keys})
    for grp in groups:
        site, hs = grp.rsplit("_h", 1)
        hm = int(hs) * 5
        dkey = f"{grp}|deterministic|na"
        if dkey not in keys:
            continue
        det = z[dkey].astype(float)
        n = det.size
        for pname in ("icp", "mondrian", "cqr", "mondrian_cqr"):
            t_ = thA.get((site, hm, pname))
            if t_ is None:
                continue
            key = next((k for k in keys
                        if k.startswith(f"{grp}|{pname}|")
                        and abs(float(k.split("|")[2]) - t_) < 1e-9), None)
            if key is None:
                continue
            m = z[key].astype(float)
            if m.size != n:
                continue
            for exp_len in (1, 2, 3, 5, 7):
                I = (rng.integers(0, n, size=(B_BOOT, n)) if exp_len == 1
                     else idx_stationary(n, exp_len, B_BOOT, rng))
                Xm, Xd = m[I], det[I]
                mm, md = Xm.mean(1), Xd.mean(1)
                cm, cd = cvar(Xm), cvar(Xd)
                vcm = 1 - mm / np.where(md > 0, md, np.nan)
                vcc = 1 - cm / np.where(cd > 0, cd, np.nan)
                rows.append(dict(
                    site=site, horizon_min=hm, policy=pname, theta=t_, n_days=n,
                    block="iid_days" if exp_len == 1 else f"stationary_exp{exp_len}d",
                    expected_block_days=exp_len, B=B_BOOT,
                    vc_mean_lo=round(float(np.nanpercentile(vcm, 2.5)), 4),
                    vc_mean_hi=round(float(np.nanpercentile(vcm, 97.5)), 4),
                    vc_cvar_lo=round(float(np.nanpercentile(vcc, 2.5)), 4),
                    vc_cvar_hi=round(float(np.nanpercentile(vcc, 97.5)), 4)))
    d = pd.DataFrame(rows)
    if d.empty:
        print("  [block bootstrap] no daily-cost vectors matched the expected key layout; keys were:", keys[:8])
        return d
    d.to_csv(guarded(CFG.TAB / f"{R1}s9_blockboot.csv", force), index=False)
    w = d.assign(w_mean=d.vc_mean_hi - d.vc_mean_lo, w_cvar=d.vc_cvar_hi - d.vc_cvar_lo)
    print("  [block bootstrap] mean 95% interval WIDTH by block length (wider = more dependence admitted):")
    for b, g in w.groupby("expected_block_days"):
        print(f"          exp block {b}d: VC(mean) {g.w_mean.mean():.4f}   VC(CVaR) {g.w_cvar.mean():.4f}")
    return d


# ------------------------------------------------ repeated measures
def anova_clustered(force):
    """Repeated measures, treated as such.

    The committed ANOVA fits ordinary least squares to per-day RMSE with model, day class,
    their interaction and year as fixed effects. Every day contributes one row per model,
    so the rows are not independent: day class and year are DAY-level covariates, while
    model is a WITHIN-day repeated measure. Two instruments are reported here:

      (a) a linear mixed model with a random intercept for day, which is the textbook
          repeated-measures specification, plus the intraclass correlation it implies;
      (b) cluster-robust (by day) Wald tests on the same fixed-effect design.

    The point of (a) is the ICC: it says how much of the apparent sample size is real.
    """
    d = pd.read_csv(CFG.TAB / "r1_p2_cv_perday_errors.csv")
    rows = []
    for h, g in d.groupby("horizon_min"):
        g = g.copy()
        g["day"] = g.year.astype(str) + "_" + g.date.astype(str)
        f = "RMSE ~ C(model)*C(day_class) + C(year)"
        ols = smf.ols(f, data=g).fit()
        a_ols = sm.stats.anova_lm(ols, typ=2)
        mix = smf.mixedlm(f, data=g, groups=g["day"]).fit(reml=True, method="lbfgs")
        var_day = float(np.asarray(mix.cov_re).ravel()[0])
        icc = var_day / (var_day + float(mix.scale))
        slices = mix.model.data.design_info.term_name_slices
        clu = ols.get_robustcov_results(cov_type="cluster", groups=g["day"].astype("category").cat.codes)
        for term, label in [("C(model)", "model"), ("C(day_class)", "day_class"),
                            ("C(model):C(day_class)", "model:day_class"), ("C(year)", "year")]:
            if term not in a_ols.index or term not in slices:
                continue
            sl = slices[term]
            cols = list(range(sl.start, sl.stop))

            def wald(res, cols):
                R = np.zeros((len(cols), len(res.params)))
                for r_, c_ in enumerate(cols):
                    R[r_, c_] = 1.0
                w = res.wald_test(R, use_f=False, scalar=True)
                return (float(np.asarray(w.statistic).ravel()[0]),
                        float(np.asarray(w.pvalue).ravel()[0]))

            chi_m, p_m = wald(mix, cols)
            chi_c, p_c = wald(clu, cols)
            rows.append(dict(
                horizon_min=int(h), term=label, level=("within-day" if label in ("model", "model:day_class") else "between-day"),
                F_ols=round(float(a_ols.loc[term, "F"]), 4), p_ols=float(a_ols.loc[term, "PR(>F)"]),
                chi2_mixed_day_RE=round(chi_m, 4), p_mixed_day_RE=p_m,
                chi2_cluster_by_day=round(chi_c, 4), p_cluster_by_day=p_c,
                n_obs=int(len(g)), n_days=int(g["day"].nunique()),
                var_day=round(var_day, 3), var_resid=round(float(mix.scale), 3), icc_day=round(icc, 4),
                still_significant_at_05=bool(p_m < 0.05)))
    r = pd.DataFrame(rows)
    r.to_csv(guarded(CFG.TAB / f"{R1}s9_anova_repeated.csv", force), index=False)
    print("  [repeated measures] repeated-measures refit. ICC(day) by horizon: " +
          ", ".join(f"h={int(x.horizon_min)}:{x.icc_day:.3f}" for _, x in r.drop_duplicates('horizon_min').iterrows()))
    for _, x in r.iterrows():
        print(f"          h={x.horizon_min:>2} {x.term:16} ({x.level:10}) p_ols={x.p_ols:.2e}  "
              f"p_mixed={x.p_mixed_day_RE:.2e}  {'significant' if x.still_significant_at_05 else 'NOT significant'}")
    return r


def main():
    force = "--force" in sys.argv
    print("R1 / S9 statistics from committed data (no refit, no dispatch re-run)")
    print("=" * 78)
    multiplicity(force)
    selection_regret(force)
    anova_clustered(force)
    block_bootstrap(force)
    print("=" * 78)
    print("DONE")


if __name__ == "__main__":
    main()


# ------------------------------- block bootstrap (paired)
def block_bootstrap_paired(force):
    """The paper's claim is about PAIRED differences, so test those, not marginal intervals.

    Section X-B says the ICP-minus-Mondrian-CQR differences are significant at every horizon
    on both expected cost and tail risk. Comparing two marginal intervals is a weaker and
    inappropriate test for a paired design, so the sensitivity is run on the paired
    difference itself, under the same stationary multi-day blocks.
    """
    z = np.load(CFG.TAB / "r1_j5_daily_costs.npz", allow_pickle=True)
    th = pd.read_csv(CFG.TAB / "r1_j5_theta_selection.csv")
    thA = {(r.site, int(r.horizon_min), r.policy): round(float(r.theta_protocolA), 4)
           for r in th.itertuples()}
    keys = list(z.keys())
    rng = np.random.default_rng(SEED)

    def cvar(X, a=0.95):
        k = max(1, int(np.ceil((1 - a) * X.shape[1])))
        return np.partition(X, X.shape[1] - k, axis=1)[:, X.shape[1] - k:].mean(1)

    def idx(n, L, B):
        if L == 1:
            return rng.integers(0, n, size=(B, n))
        p = 1.0 / L
        out = np.empty((B, n), dtype=np.int64)
        for b in range(B):
            pos, fill = 0, []
            while pos < n:
                s = rng.integers(0, n)
                l = max(1, int(rng.geometric(p)))
                fill.append(np.arange(s, s + l) % n)
                pos += l
            out[b] = np.concatenate(fill)[:n]
        return out

    def vec(site, h, pol):
        grp = f"{site}_h{h // 5}"
        t = thA[(site, h, pol)]
        k = next(k for k in keys if k.startswith(f"{grp}|{pol}|")
                 and abs(float(k.split("|")[2]) - t) < 1e-9)
        return z[k].astype(float)

    rows = []
    for site in ("yulara", "asp"):
        for h in (5, 15, 30, 60):
            det = z[f"{site}_h{h // 5}|deterministic|na"].astype(float)
            a, b = vec(site, h, "icp"), vec(site, h, "mondrian_cqr")
            n = det.size
            for L in (1, 2, 3, 5, 7):
                I = idx(n, L, B_BOOT)
                Xa, Xb, Xd = a[I], b[I], det[I]
                for metric, va, vb in (
                        ("VC_mean", 1 - Xa.mean(1) / Xd.mean(1), 1 - Xb.mean(1) / Xd.mean(1)),
                        ("VC_cvar", 1 - cvar(Xa) / cvar(Xd), 1 - cvar(Xb) / cvar(Xd))):
                    d = va - vb
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    rows.append(dict(site=site, horizon_min=h, metric=metric,
                                     contrast="icp_minus_mondrian_cqr",
                                     block="iid_days" if L == 1 else f"stationary_exp{L}d",
                                     expected_block_days=L, B=B_BOOT,
                                     diff_lo=round(float(lo), 4), diff_hi=round(float(hi), 4),
                                     excludes_zero=bool(lo > 0 or hi < 0)))
    r = pd.DataFrame(rows)
    r.to_csv(guarded(CFG.TAB / f"{R1}s9_blockboot_paired.csv", force), index=False)
    bad = r[~r.excludes_zero]
    print(f"  [block bootstrap] paired differences: {len(r)} cells, "
          f"{int(r.excludes_zero.sum())} exclude zero, {len(bad)} do not")
    if len(bad):
        print(bad[["site", "horizon_min", "metric", "block", "diff_lo", "diff_hi"]].to_string(index=False))
    return r


if __name__ == "__main__" and "--paired" in sys.argv:
    block_bootstrap_paired("--force" in sys.argv)
