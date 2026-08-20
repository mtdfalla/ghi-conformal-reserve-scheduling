"""Measure the two finite-sample quantile rules on the ACTUAL dispatch pools.

WHY THIS FILE EXISTS
--------------------
The interval layer defines its conformal quantile as an order statistic
(`conformal.py:conformal_q`, the ceil((n+1)p)-th smallest score); the dispatch
layer's counted-GHI policies (`r1_dispatch.py:counted_ghi`) use numpy.quantile
(type-7 linear interpolation) on their calibration score pools, as the article's
policy equation states ("the empirical (1-theta)-quantile of a calibration pool").
The two estimators differ at finite n, and the difference grows as pools shrink -
the Mondrian variants fall back to the marginal pool only below 31 members, so
pools in the low hundreds are permitted. This script MEASURES the difference on
the real pools rather than asserting it is small:

 (1) OFFSETS - for every site x horizon x policy x theta on the selection grid,
     the counted-GHI offset under both rules, its absolute difference, and the
     pool size, on the full-2023 evaluation pools (per-regime pools for the
     Mondrian variants). Before the lower clip at zero, the counted-GHI
     difference equals the offset difference exactly, at every timestamp.

 (2) SELECTION STABILITY - the Protocol-A theta selection (scores on 2023-H1,
     argmin of mean daily cost on 2023-H2) re-run IN MEMORY with numpy.quantile
     replaced by the order-statistic rule, for every battery scenario: does any
     selected theta change? The re-run reads only the /tmp caches; NO committed
     result file is read, regenerated or overwritten by the comparison. As a
     control, the numpy-quantile selection is asserted equal to the shipped
     `r1_j5_theta_selection.csv` before the comparison is trusted.

Usage (from the code directory, code/ or code/):
    python3 r1/r1_b07_quantile_rules.py [--force]
Requires the S0 caches (bash r1/r1_restore_env.sh, then r1_fit_extra_taus.py).
Writes  results/tables/r1_b07_quantile_rules.csv     (the offsets, rule by rule)
        results/tables/r1_b07_theta_stability.csv    (the selection comparison)
Refuses to overwrite existing outputs without --force (or R1_REBUILD=1).
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import config as CFG                     # noqa: E402
import r1_dispatch as D                  # noqa: E402


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith("r1_"):
        raise WriteGuard(f"REFUSING to write '{path.name}': the r1_ prefix is required.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def os_quantile(a: np.ndarray, q: float) -> float:
    """The order-statistic rule: the ceil((n+1)q)-th smallest, clipped to [1, n].
    This is `conformal.py:conformal_q` applied at level q."""
    a = np.sort(np.asarray(a, dtype=float))
    n = a.size
    k = min(max(int(np.ceil((n + 1) * q)), 1), n)
    return float(a[k - 1])


@contextmanager
def order_stat_rule():
    """Swap numpy.quantile for the order-statistic rule inside the with-block.
    Within r1_dispatch, np.quantile is called only by counted_ghi, always with a
    1-D pool and a scalar level, so the redirect is exact for the measurement."""
    orig = np.quantile
    np.quantile = lambda a, q, *args, **kw: os_quantile(a, float(q))
    try:
        yield
    finally:
        np.quantile = orig


def offset_rows(site: str, h: int, S, taus) -> list[dict]:
    """(1): both rules on the full-2023 pools, every policy x theta."""
    rows = []
    fit = S["FULL"]
    e = fit["y"] - fit["p"]
    regimes = list(np.unique(fit["g"]))
    for th in D.THETA_GRID:
        th = round(float(th), 4)
        tau = min(taus, key=lambda z: abs(z - (1 - th)))
        s_all = fit["q"][tau] - fit["y"]
        pools = [("icp", "all", e, 1 - th),
                 ("cqr", "all", s_all, th)]
        for g in regimes:
            eg = e[fit["g"] == g]
            sg = s_all[fit["g"] == g]
            pools.append(("mondrian", str(g), eg if len(eg) > 30 else e, 1 - th))
            pools.append(("mondrian_cqr", str(g), sg if len(sg) > 30 else s_all, th))
        for pol, scope, pool, level in pools:
            q_np = float(np.quantile(pool, level))
            q_os = os_quantile(pool, level)
            rows.append(dict(site=site, horizon_min=h * 5, policy=pol, scope=scope,
                             theta=th, n_pool=int(len(pool)),
                             offset_npquantile=round(q_np, 4),
                             offset_orderstat=round(q_os, 4),
                             abs_diff_wm2=round(abs(q_np - q_os), 4)))
    return rows


def main() -> None:
    force = "--force" in sys.argv or os.environ.get("R1_REBUILD") == "1"
    th_ship = pd.read_csv(CFG.TAB / "r1_j5_theta_selection.csv")
    shipped = {(r.site, int(r.horizon_min), r.policy): round(float(r.theta_protocolA), 4)
               for r in th_ship.itertuples()}

    off_rows, sel_rows = [], []
    control_bad = []
    for site in ("yulara", "asp"):
        for h in (1, 3, 6, 12):
            P = D.load_preds(site, h)
            taus = P["taus"]
            S = D.build_sets(P)
            M2 = D.day_matrix(S["H2"]["idx"])[1]
            off_rows += offset_rows(site, h, S, taus)

            for batt, (E, Pw) in D.BATT.items():
                c_o = D.C_O_DEF
                c_u = c_o * D.RATIO_DEF
                sel_np = D.sweep(S["H1"], S["H2"], M2, taus, D.THETA_GRID, E, Pw, c_u, c_o)
                with order_stat_rule():
                    sel_os = D.sweep(S["H1"], S["H2"], M2, taus, D.THETA_GRID, E, Pw, c_u, c_o)
                for pol in D.CP_POLICIES:
                    thA_np, obj_np = D.select_theta(sel_np, pol, D.THETA_GRID, 0.0)
                    thA_os, obj_os = D.select_theta(sel_os, pol, D.THETA_GRID, 0.0)
                    if batt == "default":
                        ship = shipped.get((site, h * 5, pol))
                        if ship is not None and abs(ship - thA_np) > 1e-9:
                            control_bad.append((site, h * 5, pol, ship, thA_np))
                    sel_rows.append(dict(site=site, horizon_min=h * 5, battery=batt,
                                         policy=pol,
                                         theta_npquantile=thA_np, theta_orderstat=thA_os,
                                         sel_obj_npquantile=round(obj_np, 3),
                                         sel_obj_orderstat=round(obj_os, 3),
                                         theta_changes=bool(abs(thA_np - thA_os) > 1e-9)))
            print(f"  {site} h={h*5}min done", flush=True)

    if control_bad:
        raise SystemExit("CONTROL FAILED - the in-memory numpy-quantile selection does not "
                         "reproduce the shipped Protocol-A thetas:\n  " +
                         "\n  ".join(map(str, control_bad)))
    print("control: the in-memory numpy-quantile selection reproduces every shipped "
          "Protocol-A theta (default battery), 32 of 32")

    off = pd.DataFrame(off_rows)
    sel = pd.DataFrame(sel_rows)
    off.to_csv(guarded(CFG.TAB / "r1_b07_quantile_rules.csv", force), index=False)
    sel.to_csv(guarded(CFG.TAB / "r1_b07_theta_stability.csv", force), index=False)

    n_flip = int(sel.theta_changes.sum())
    print(f"\nmax |offset difference| over all pools/levels: {off.abs_diff_wm2.max():.4f} W/m2")
    worst = off.loc[off.abs_diff_wm2.idxmax()]
    print(f"  at {worst.site} h={worst.horizon_min} {worst.policy}/{worst.scope} "
          f"theta={worst.theta} (n={worst.n_pool})")
    print(f"smallest pool used without fallback: "
          f"{int(off[off.scope != 'all'].n_pool.min())}")
    print(f"theta selections that change under the order-statistic rule: {n_flip} of {len(sel)}")
    if n_flip:
        print(sel[sel.theta_changes].to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
