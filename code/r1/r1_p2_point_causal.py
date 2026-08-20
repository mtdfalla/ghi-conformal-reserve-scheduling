"""Final pass - classical point benchmark RE-RUN UNDER CAUSAL REGIMES.

WHY THIS FILE EXISTS
--------------------
the first-pass point-forecast table is pre-causal (`p2_all_models_comparison.csv`, dated 2026-06-22
18:06) while Tables 3-5 are POST-. made the clear-sky scalar train-only,
which changed ghi_cs -> kt -> the features AND the target reconstruction, so every
Table 2 number is on a different footing from the rest of the paper.

The deep half is handled by `r1_deep_causal.py` (runs A and C). This file is
the classical half: persistence, smart persistence, linear AR (ridge) and GBM, all
re-run on the post- frame. It is also the ONLY source of the post-causal GBM
**MAE and R2**, which did not exist before this session - the S0 cache stores only
predictions, and only RMSE was ever recomputed from them.

THE GBM CONFIGURATION QUESTION (, recorded this session)
------------------------------------------------------------
the first pass used two different GBM configurations:
  * `run_phase2_baselines.py`  max_iter=400, lr=0.05, max_leaf_nodes=63  -> made
    Table 2's GBM row (88.59 / 113.75 / 125.23 / 138.21 W/m2)
  * `r1_fit_cache.py` / the conformal + dispatch layer
                              max_iter=150, lr=0.08, max_leaf_nodes=31
and the article (Experimental Setup and the Appendix) describes the
SECOND one - "150 boosting iterations, learning rate 0.08, up to 31 leaves". So the first pass's
Table 2 was produced by a model the paper does not describe.

This script uses the config the article states (150 / 0.08 / 31), which is also
the configuration behind every interval and dispatch number. That makes the
whole paper one model. The choice is recorded as / OPEN_QUESTIONS Q19.

It refits the GBM from scratch and then CROSS-CHECKS the refit against the S0 cache
at /tmp/r1cache (built by the committed `r1_fit_cache.py`); the two must agree to
float tolerance, which is what makes the number reproducible rather than asserted.

OUTPUTS (all r1_-prefixed; nothing is overwritten)
    04_results/tables/r1_p2_point_causal.csv           overall metrics, 4 models x 4 h
    04_results/tables/r1_p2_point_causal_by_regime.csv per-regime metrics
    04_results/tables/r1_p2_point_dm_classical.csv     DM among the classical models
    04_results/tables/r1_p2_point_pred_h{1,3,6,12}.parquet  per-observation test-2024
    04_results/metrics/r1_p2_point_causal.json         provenance + the first-pass diff table

Run from the code directory (03_code/ in the working tree, code/ in a release checkout):  python3 r1/r1_p2_point_causal.py
Requires the S0 restore to have run (`bash r1/r1_restore_env.sh`).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parents[1]   # 03_code/ in the working tree, code/ in a release checkout
sys.path.insert(0, str(CODE / "utils"))
sys.path.insert(0, str(CODE / "evaluation"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import config as CFG                     # noqa: E402
import datasets as D                     # noqa: E402
import metrics as M                      # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB   # noqa: E402
from sklearn.linear_model import Ridge                              # noqa: E402
from sklearn.pipeline import make_pipeline                          # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

HORIZONS = [1, 3, 6, 12]
R1_PREFIX = "r1_"

# The POST- inputs, pinned by the same hashes r1_deep_causal.py pins, so the
# classical and deep halves of Table 2 are provably built on the same data.
EXPECTED_SHA256 = {
    "02_data/regime_labels/yulara_regimes_5min.parquet":
        "13d9659d90dab8242049e8858523e46fd0dad708eb29ca32138e0d9070c57e6b",
    "02_data/cleaned/yulara_clean_5min.parquet":
        "5dfe4445c8ab12b0a5ea889dfe7921c6496d2e79fb37dad421feeccfdec7bf39",
    "02_data/cleaned/yulara_quality_flags.parquet":
        "4e9e40a96f99d1b9e2510d569d1ce7524830e97fc0171b9675782310cfdfc275",
}


def _data_path(rel: str) -> Path:
    """Resolve an authors'-tree data path ("02_data/...") in either layout.
    In a release checkout the same file lives under data/; config.py detects
    which layout is present, so route the lookup through it."""
    p = REPO / rel
    if p.exists():
        return p
    return CFG.DATA_CLEAN.parent / Path(rel).relative_to("02_data")

# the first pass's point-forecast table / p2_all_models_comparison.csv, for the diff table.
R0_TABLE2 = {   # model -> {h_min: (MAE, RMSE, R2)}
    "persistence":       {5: (45.4713, 98.4738, 0.9035), 15: (82.3072, 141.3260, 0.7994),
                          30: (118.6774, 171.7383, 0.7023), 60: (181.5562, 230.0487, 0.4721)},
    "smart_persistence": {5: (40.7985, 97.8294, 0.9048), 15: (62.3391, 133.8908, 0.8199),
                          30: (75.0590, 148.3545, 0.7779), 60: (90.1800, 162.3155, 0.7372)},
    "linear_ar":         {5: (44.5013, 91.8287, 0.9161), 15: (65.9885, 117.9597, 0.8602),
                          30: (78.4329, 129.9100, 0.8297), 60: (93.0329, 142.9333, 0.7962)},
    "gbm":               {5: (41.6346, 88.5871, 0.9219), 15: (60.4569, 113.7534, 0.8700),
                          30: (70.4620, 125.2280, 0.8417), 60: (82.6528, 138.2119, 0.8094)},
}


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1_PREFIX):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_provenance() -> dict:
    prov, bad = {}, []
    for rel, exp in EXPECTED_SHA256.items():
        p = _data_path(rel)
        got = sha256_of(p)
        prov[rel] = {"sha256": got, "expected_sha256": exp, "match": got == exp}
        if got != exp:
            bad.append(f"{rel}: expected {exp} got {got}")
    if bad:
        if CFG.LAYOUT == "release-checkout":
            # A reconstructed file can never byte-match the pinned originals: the
            # shipped parquets were written by an earlier pandas whose
            # time-interpolation arithmetic differs at the 1e-13 level in imputed
            # cells (measured; the imputation FLAGS are content-identical). The
            # pin stays fatal in the authors' tree, where it guards against
            # stale inputs.
            print("Input provenance: reconstructed data does not byte-match the "
                  "pinned originals (expected for a fresh reconstruction; values "
                  "agree to ~1e-13 in interpolated cells):")
            for _line in bad:
                print("  " + _line)
        else:
            raise SystemExit("Input provenance FAILED:\n  " + "\n  ".join(bad))
    print("Input provenance OK - the three post-causal inputs match the pinned hashes.")
    return prov


def gbm_model() -> HGB:
    """The configuration the ARTICLE states, and the one r1_fit_cache.py uses for
    every interval and dispatch number. See the docstring."""
    return HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.1, n_iter_no_change=10, random_state=CFG.SEED)


def main() -> None:
    force = "--force" in sys.argv or os.environ.get("R1_REBUILD") == "1"   # reproduce.sh sets R1_REBUILD=1
    t_start = time.time()
    print(f"repo   : {REPO}")
    print(f"python : {platform.python_version()}")
    import sklearn
    print(f"sklearn: {sklearn.__version__}   pandas: {pd.__version__}   numpy: {np.__version__}")
    prov = check_provenance()

    # Pre-flight EVERY output path before fitting anything.
    planned = [CFG.TAB / f"{R1_PREFIX}p2_point_causal.csv",
               CFG.TAB / f"{R1_PREFIX}p2_point_causal_by_regime.csv",
               CFG.TAB / f"{R1_PREFIX}p2_point_dm_classical.csv",
               CFG.MET / f"{R1_PREFIX}p2_point_causal.json"]
    planned += [CFG.TAB / f"{R1_PREFIX}p2_point_pred_h{h}.parquet" for h in HORIZONS]
    blocked = []
    for p in planned:
        try:
            guarded(p, force)
        except WriteGuard as e:
            blocked.append(str(e))
    if blocked:
        raise SystemExit("Pre-flight FAILED (nothing was fitted):\n\n" + "\n\n".join(blocked))
    print(f"outputs: {len(planned)} paths pre-checked, all clear.")

    base = pd.read_parquet("/tmp/base.parquet")
    print(f"base   : {len(base):,} rows  (S0 restore expects 920,451)")

    rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)   # noqa: E731
    rows, rows_reg, dm_rows, cache_checks = [], [], [], []

    for h in HORIZONS:
        t0 = time.time()
        d = D.make_xy(base, h).sort_index()
        tr = d[d.year <= 2022]
        ca = d[d.year == 2023]
        te = d[d.year == 2024]
        F = D.FEATURES
        y = te["y_ghi"].values
        cs = te["y_ghi_cs"].values
        reg = te["base_regime"].values

        pred = {
            "persistence": te["base_ghi"].values,
            "smart_persistence": rec(te["base_kt"].values, cs),
        }
        ar = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ar.fit(tr[F].values, tr["y_kt"].values)
        pred["linear_ar"] = rec(ar.predict(te[F].values), cs)

        g = gbm_model()
        g.fit(tr[F].values, tr["y_kt"].values)
        pred["gbm"] = rec(g.predict(te[F].values), cs)

        # --- cross-check the refit against the committed S0 cache -----------------
        cache_f = f"/tmp/r1cache/yulara_h{h}_point_t.npy"
        chk = {"horizon_min": h * 5, "cache_file": cache_f, "cache_present": os.path.exists(cache_f)}
        if chk["cache_present"]:
            cached = np.load(cache_f)
            chk["same_length"] = bool(len(cached) == len(pred["gbm"]))
            if chk["same_length"]:
                chk["max_abs_diff"] = float(np.max(np.abs(cached - pred["gbm"])))
                chk["allclose_1e-6"] = bool(np.allclose(cached, pred["gbm"], atol=1e-6))
                chk["cache_RMSE"] = float(np.sqrt(np.mean((y - cached) ** 2)))
        chk["refit_RMSE"] = float(np.sqrt(np.mean((y - pred["gbm"]) ** 2)))
        cache_checks.append(chk)
        print(f"h={h*5:>2} min : n_train {len(tr):,}  n_test {len(te):,}  "
              f"GBM refit RMSE {chk['refit_RMSE']:.2f}  cache RMSE "
              f"{chk.get('cache_RMSE', float('nan')):.2f}  "
              f"maxdiff {chk.get('max_abs_diff', float('nan')):.3e}", flush=True)

        ref = pred["smart_persistence"]
        for name, p in pred.items():
            m = M.all_metrics(y, p, ref=ref)
            m.update(model=name, horizon_min=h * 5, regime_basis="causal_D012")
            rows.append(m)
            for rg in ["clear", "transitional", "cloudy"]:
                mk = reg == rg
                if mk.sum() > 30:
                    mm = M.all_metrics(y[mk], p[mk], ref=ref[mk])
                    mm.update(model=name, horizon_min=h * 5, regime=rg, regime_basis="causal_D012")
                    rows_reg.append(mm)

        for a, b in [("gbm", "smart_persistence"), ("gbm", "linear_ar"),
                     ("smart_persistence", "persistence")]:
            s, pv = M.diebold_mariano(y, pred[a], pred[b], h=h)
            dm_rows.append(dict(horizon_min=h * 5, model_A=a, model_B=b, n=int(len(y)),
                                DM=round(float(s), 4), p_value=float(f"{pv:.3e}"),
                                A_better=bool(s < 0), hac_lag=h - 1))

        pd.DataFrame({"y_ghi": y, "ghi_cs": cs, "regime": reg,
                      **{k: pred[k] for k in pred}},
                     index=te.index).to_parquet(
            guarded(CFG.TAB / f"{R1_PREFIX}p2_point_pred_h{h}.parquet", force))
        print(f"          done in {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)[["model", "horizon_min", "n", "MAE", "RMSE", "nRMSE", "R2",
                              "skill_vs_ref", "regime_basis"]].round(4)
    res.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_point_causal.csv", force), index=False)

    resr = pd.DataFrame(rows_reg)[["model", "horizon_min", "regime", "n", "MAE", "RMSE", "R2",
                                   "skill_vs_ref", "regime_basis"]].round(4)
    resr.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_point_causal_by_regime.csv", force), index=False)

    dmt = pd.DataFrame(dm_rows)
    dmt.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_point_dm_classical.csv", force), index=False)

    # --- the first-pass-vs-final-pass difference table ('s evidence column) -------------------
    diffs = []
    for _, r in res.iterrows():
        r0 = R0_TABLE2.get(r["model"], {}).get(int(r["horizon_min"]))
        if r0:
            diffs.append(dict(model=r["model"], horizon_min=int(r["horizon_min"]),
                              R0_MAE=r0[0], R1_MAE=float(r["MAE"]), d_MAE=round(float(r["MAE"]) - r0[0], 4),
                              R0_RMSE=r0[1], R1_RMSE=float(r["RMSE"]), d_RMSE=round(float(r["RMSE"]) - r0[1], 4),
                              R0_R2=r0[2], R1_R2=float(r["R2"]), d_R2=round(float(r["R2"]) - r0[2], 4)))

    meta = dict(
        run_id="(a) classical point benchmark under causal regimes",
        finished_utc=pd.Timestamp.utcnow().isoformat(),
        elapsed_s=round(time.time() - t_start, 1),
        python=platform.python_version(), sklearn=sklearn.__version__,
        pandas=pd.__version__, numpy=np.__version__,
        gbm_config=dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                        random_state=int(CFG.SEED),
                        note="the configuration the article states; the first-pass table used "
                             "max_iter=400 / lr=0.05 / 63 leaves"),
        inputs=prov, s0_cache_crosscheck=cache_checks,
        r0_vs_r1=diffs,
        note="Deep rows for Table 2 come from run C (r1_p2_deep_causal_full.csv), "
             "not from this file. Paired DM vs the deep models is r1_p2_deep_vs_gbm.py.",
    )
    json.dump(meta, open(guarded(CFG.MET / f"{R1_PREFIX}p2_point_causal.json", force), "w"),
              indent=2, default=str)

    print("\n=== OVERALL (test 2024, post-causal), GHI W/m^2 ===")
    print(res.to_string(index=False))
    print("\n=== first pass -> final pass difference ===")
    print(pd.DataFrame(diffs).to_string(index=False))
    print("\n=== DM (classical) ===")
    print(dmt.to_string(index=False))
    print(f"\ndone in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
