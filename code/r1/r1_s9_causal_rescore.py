"""Final pass - what the imputation exposure is worth, measured rather than argued.

WHY THIS FILE EXISTS
--------------------
code/preprocessing/p2_clean.py fills short GHI gaps by time interpolation, which uses the
observation on the FAR side of the gap - a value that had not arrived at issue time. And
code/utils/datasets.py excludes an evaluated sample only when its TARGET was imputed, not
when an issue-time or lagged input was. So a fraction of every reported number is computed on
feature windows containing a value reconstructed from the future.

This is answered by measuring the exposure rather than by re-running the pipeline:

  * the contamination mask is computed exactly from the feature window the model actually
    reads - lags {0,1,2,3,4,5,6,9,12} plus the 6-step rolling window, i.e. offsets 0..6, 9, 12;
  * the CONFORMAL layer is refit on the clean calibration rows. That is not a model refit -
    split-conformal calibration is a quantile of residuals - so the point and quantile models
    are untouched and no cached prediction changes;
  * everything is then re-scored on the clean test rows and compared with the published run.

WHAT THIS DOES NOT DO, stated so the disclosure cannot overclaim: the GBM and the quantile
models are still FITTED on training years that contain interpolated inputs. That is feature
noise in fitting, not leakage of test information into a test score, and correcting it would
require a full re-fit, which is not done here. The article states this.

Usage (from the repository root):  python3 code/r1/r1_s9_causal_rescore.py [--force]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "utils"))
sys.path.insert(0, str(_ROOT / "conformal"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import config as CFG                     # noqa: E402
import conformal as CP                   # noqa: E402
import datasets as D                     # noqa: E402

R1 = "r1_"
COV = 0.90
TAUS_Q = [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]
SITES = {"yulara": "/tmp/base.parquet", "asp": "/tmp/base_asp.parquet"}
HS = [1, 3, 6, 12]
OFFSETS = sorted(set(D.LAGS) | set(range(0, D.ROLL)))     # the window the features read


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def contamination(base: pd.DataFrame) -> pd.Series:
    """True where any input the feature window reads was interpolated."""
    imp = base["ghi_imputed"].astype(bool)
    out = pd.Series(False, index=base.index)
    for L in OFFSETS:
        out |= imp.shift(L).fillna(False).astype(bool)
    return out


def score_block(y, pt, qc_lo, qc_hi, qt_lo, qt_hi, yc, pc, csc, cst, gc, gt, mask_c, mask_t):
    """Fit the conformal layer on the (possibly filtered) calibration rows and score the
    (possibly filtered) test rows. Returns per-regime PICP for the five static methods."""
    yc_, pc_, csc_, gc_ = yc[mask_c], pc[mask_c], csc[mask_c], gc[mask_c]
    qc_lo_, qc_hi_ = qc_lo[mask_c], qc_hi[mask_c]
    y_, pt_, cst_, gt_ = y[mask_t], pt[mask_t], cst[mask_t], gt[mask_t]
    qt_lo_, qt_hi_ = qt_lo[mask_t], qt_hi[mask_t]

    res = {}
    s = CP.icp_fit(yc_, pc_)
    res["icp"] = CP.icp_interval(pt_, s, COV)
    sn = CP.icpn_fit(yc_, pc_, csc_)
    res["icp_norm"] = CP.icpn_interval(pt_, cst_, sn, COV)
    sm = CP.mondrian_fit(yc_, pc_, gc_)
    res["mondrian"] = CP.mondrian_interval(pt_, gt_, sm, COV, s)
    sq = CP.cqr_fit(yc_, qc_lo_, qc_hi_)
    res["cqr"] = CP.cqr_interval(qt_lo_, qt_hi_, sq, COV)
    smq = {g: CP.cqr_fit(yc_[gc_ == g], qc_lo_[gc_ == g], qc_hi_[gc_ == g]) for g in np.unique(gc_)}
    qte = np.empty(len(y_))
    for g in np.unique(gt_):
        sc = smq.get(g, sq)
        qte[gt_ == g] = CP.conformal_q(sc if len(sc) >= 30 else sq, COV)
    res["mondrian_cqr"] = (np.clip(qt_lo_ - qte, 0, None), qt_hi_ + qte)

    rows = []
    scopes = [("all", np.ones(len(y_), bool)), ("clear", gt_ == "clear"),
              ("transitional", gt_ == "transitional"), ("cloudy", gt_ == "cloudy")]
    for m, (lo, hi) in res.items():
        for sc, msk in scopes:
            if msk.sum() < 30:
                continue
            p = CP.picp(y_[msk], lo[msk], hi[msk])
            rows.append(dict(method=m, scope=sc, n=int(msk.sum()), PICP=round(p, 4),
                             ACE=round(p - COV, 4),
                             PINAW=round(CP.pinaw(y_[msk], lo[msk], hi[msk]), 4)))
    return rows, y_, pt_


def main():
    force = "--force" in sys.argv
    print("exposure of the interpolated-input contamination, measured")
    print("=" * 92)
    out, expo = [], []
    for site, bp in SITES.items():
        base = pd.read_parquet(bp)
        cont = contamination(base)
        for h in HS:
            d = D.make_xy(base, h).sort_index()
            ca, te = d[d.year == 2023], d[d.year == 2024]
            c_ca = cont.reindex(ca.index).fillna(False).to_numpy(dtype=bool)
            c_te = cont.reindex(te.index).fillna(False).to_numpy(dtype=bool)
            PRE = f"/tmp/r1cache/{site}_h{h}"
            pc = np.load(f"{PRE}_point_c.npy"); pt = np.load(f"{PRE}_point_t.npy")
            qc = {t_: np.load(f"{PRE}_q{t_}_c.npy") for t_ in TAUS_Q}
            qt = {t_: np.load(f"{PRE}_q{t_}_t.npy") for t_ in TAUS_Q}
            yc, y = ca["y_ghi"].to_numpy(), te["y_ghi"].to_numpy()
            csc = np.maximum(ca["y_ghi_cs"].to_numpy(), 50.0)
            cst = np.maximum(te["y_ghi_cs"].to_numpy(), 50.0)
            gc, gt = ca["base_regime"].to_numpy().astype(str), te["base_regime"].to_numpy().astype(str)

            expo.append(dict(site=site, horizon_min=h * 5,
                             n_calib=len(ca), n_calib_contaminated=int(c_ca.sum()),
                             pct_calib=round(100 * c_ca.mean(), 2),
                             n_test=len(te), n_test_contaminated=int(c_te.sum()),
                             pct_test=round(100 * c_te.mean(), 2)))

            full_c = np.ones(len(ca), bool); full_t = np.ones(len(te), bool)
            for label, mc, mt in (("published", full_c, full_t), ("clean", ~c_ca, ~c_te)):
                rows, y_, pt_ = score_block(y, pt, qc[0.05], qc[0.95], qt[0.05], qt[0.95],
                                            yc, pc, csc, cst, gc, gt, mc, mt)
                rmse = float(np.sqrt(np.mean((y_ - pt_) ** 2)))
                mae = float(np.mean(np.abs(y_ - pt_)))
                for r in rows:
                    out.append(dict(site=site, horizon_min=h * 5, variant=label,
                                    point_RMSE=round(rmse, 4), point_MAE=round(mae, 4), **r))
            print(f"  {site:6} h={h*5:>2}min  calib {c_ca.mean()*100:5.2f}% / test {c_te.mean()*100:5.2f}% contaminated")

    ex = pd.DataFrame(expo)
    ex.to_csv(guarded(CFG.TAB / f"{R1}s9_causal_exposure.csv", force), index=False)
    d = pd.DataFrame(out)
    d.to_csv(guarded(CFG.TAB / f"{R1}s9_causal_rescore.csv", force), index=False)

    print("\n  HEADLINE COMPARISON — 90% nominal, 5 min, per regime (published vs clean subset)")
    print("  " + "-" * 88)
    piv = d[(d.horizon_min == 5)].pivot_table(index=["site", "method", "scope"],
                                              columns="variant", values="PICP")
    piv["delta"] = (piv["clean"] - piv["published"]).round(4)
    print(piv.round(4).to_string())
    print("\n  POINT FORECAST (GBM) — RMSE published vs clean")
    p2 = d.drop_duplicates(["site", "horizon_min", "variant"]).pivot_table(
        index=["site", "horizon_min"], columns="variant", values="point_RMSE")
    p2["delta"] = (p2["clean"] - p2["published"]).round(3)
    p2["pct"] = (100 * (p2["clean"] - p2["published"]) / p2["published"]).round(2)
    print(p2.round(3).to_string())
    print("=" * 92)
    print("DONE")


if __name__ == "__main__":
    main()
