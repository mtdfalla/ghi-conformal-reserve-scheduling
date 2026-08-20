"""Final pass - assemble Table 2 and the point-forecast figures from the r1_ files.

the first-pass point-forecast table mixed a pre-causal classical benchmark with a pre-causal deep benchmark.
This script builds the replacement from three post-causal sources and nothing else:

    04_results/tables/r1_p2_point_causal.csv      persistence / smart / linear AR / GBM
    04_results/tables/r1_p2_deep_causal_full.csv  run C  <- Table 2's deep rows
    04_results/tables/r1_p2_deep_causal.csv       run A  <- reported as the
                                                                  smaller-budget point

RUN C IS THE SOURCE FOR THE TABLE. Run A is more flattering to our argument
and is therefore NOT the row we print; it appears only as the second budget, so a
reader can see that the conclusion is not an artifact of the training budget.

It also restates the EXTERNAL-SITE point metrics from the r1_ cache rather than from
the Phase-6 j3_point_metrics.csv, because requires every number the article reports
to come from an r1_ file: the two differ slightly (DKASC 5 min: 84.14 vs 83.88 W/m2)
for exactly the library-version reasons documents.

OUTPUTS
    04_results/tables/r1_p2_table2.csv         the article table, long form
    04_results/tables/r1_j3_point_causal.csv   external-site point metrics, r1_ sourced
    04_results/figures/r1_p2_all_models_rmse.png
    04_results/figures/r1_p2_skill_vs_horizon.png

Run from the code directory (03_code/ in the working tree, code/ in a release checkout):  python3 r1/r1_p2_table2_build.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parents[1]   # 03_code/ in the working tree, code/ in a release checkout
sys.path.insert(0, str(CODE / "utils"))
sys.path.insert(0, str(CODE / "evaluation"))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import config as CFG                    # noqa: E402
import datasets as D                    # noqa: E402
import metrics as M                     # noqa: E402

HORIZONS = [1, 3, 6, 12]
R1_PREFIX = "r1_"
PRETTY = {"persistence": "Naive persistence", "smart_persistence": "Smart persistence",
          "linear_ar": "Linear AR", "gru": "GRU", "gru_tcn": "GRU-TCN", "gbm": "GBM"}
ORDER = ["persistence", "smart_persistence", "linear_ar", "gru", "gru_tcn", "gbm"]


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1_PREFIX):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    force = "--force" in sys.argv or os.environ.get("R1_REBUILD") == "1"   # reproduce.sh sets R1_REBUILD=1
    cls = pd.read_csv(CFG.TAB / f"{R1_PREFIX}p2_point_causal.csv")
    runC = pd.read_csv(CFG.TAB / f"{R1_PREFIX}p2_deep_causal_full.csv")
    runA = pd.read_csv(CFG.TAB / f"{R1_PREFIX}p2_deep_causal.csv")

    # skill vs smart persistence must be computed against the SAME reference the
    # classical rows use; the deep rows have a different valid set, so the reference
    # RMSE is taken per horizon from the classical table and the caveat is recorded.
    ref = cls[cls.model == "smart_persistence"].set_index("horizon_min")["RMSE"].to_dict()

    rows = []
    for _, r in cls.iterrows():
        rows.append(dict(model=r["model"], model_pretty=PRETTY[r["model"]],
                         budget="-", horizon_min=int(r["horizon_min"]), n=int(r["n"]),
                         MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"],
                         skill_vs_smart_persistence=r["skill_vs_ref"],
                         in_table2=True, source=f"{R1_PREFIX}p2_point_causal.csv"))
    for df, budget, in_tab, src in [
            (runC, "30 epochs / all windows (run C,)", True, f"{R1_PREFIX}p2_deep_causal_full.csv"),
            (runA, "15 epochs / 150,000 windows (run A)", False, f"{R1_PREFIX}p2_deep_causal.csv")]:
        for _, r in df.iterrows():
            h = int(r["horizon_min"])
            rows.append(dict(model=r["model"], model_pretty=PRETTY[r["model"]],
                             budget=budget, horizon_min=h, n=int(r["n"]),
                             MAE=round(float(r["MAE"]), 4), RMSE=round(float(r["RMSE"]), 4),
                             R2=round(float(r["R2"]), 4),
                             skill_vs_smart_persistence=round(1 - float(r["RMSE"]) / ref[h], 4),
                             in_table2=in_tab, source=src))

    t2 = pd.DataFrame(rows)
    t2["_o"] = t2["model"].map({m: i for i, m in enumerate(ORDER)})
    t2 = t2.sort_values(["horizon_min", "_o", "budget"]).drop(columns="_o")
    t2.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_table2.csv", force), index=False)

    # ---- external site: point metrics from the r1_ cache, not from Phase 6 --------
    base_asp = pd.read_parquet("/tmp/base_asp.parquet")
    rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)   # noqa: E731
    asp_rows = []
    for h in HORIZONS:
        d = D.make_xy(base_asp, h).sort_index()
        te = d[d.year == 2024]
        y = te["y_ghi"].values
        cs = te["y_ghi_cs"].values
        reg = te["base_regime"].values
        p = np.load(f"/tmp/r1cache/asp_h{h}_point_t.npy")
        assert len(p) == len(y), f"asp h={h}: cache/frame length mismatch"
        sp = rec(te["base_kt"].values, cs)
        m = M.all_metrics(y, p, ref=sp)
        m.update(model="gbm", horizon_min=h * 5, scope="all", site="asp",
                 regime_basis="causal_D012")
        asp_rows.append(m)
        for rg in ["clear", "transitional", "cloudy"]:
            mk = reg == rg
            if mk.sum() > 30:
                mm = M.all_metrics(y[mk], p[mk], ref=sp[mk])
                mm.update(model="gbm", horizon_min=h * 5, scope=rg, site="asp",
                          regime_basis="causal_D012")
                asp_rows.append(mm)
    asp = pd.DataFrame(asp_rows)[["site", "model", "horizon_min", "scope", "n", "MAE",
                                  "RMSE", "nRMSE", "R2", "skill_vs_ref",
                                  "regime_basis"]].round(4)
    asp.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}j3_point_causal.csv", force), index=False)

    # ---- figures -----------------------------------------------------------------
    tab = t2[t2.in_table2]
    hs = [5, 15, 30, 60]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in ORDER:
        s = tab[tab.model == m].set_index("horizon_min").reindex(hs)
        ax.plot(hs, s["RMSE"], marker="o", lw=2 if m == "gbm" else 1.4,
                label=PRETTY[m], zorder=3 if m == "gbm" else 2)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Test-2024 RMSE (W/m$^2$)")
    # Figure-level title deliberately not drawn: the caption in the article carries the
    # description, and a title above a caption is redundant. Per-panel titles are kept.
    # ax.set_title("Point-forecast RMSE by horizon (post-causal,)")
    ax.set_xticks(hs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(guarded(CFG.FIG / f"{R1_PREFIX}p2_all_models_rmse.png", force), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in ORDER:
        if m == "smart_persistence":
            continue
        s = tab[tab.model == m].set_index("horizon_min").reindex(hs)
        ax.plot(hs, 100 * s["skill_vs_smart_persistence"], marker="o",
                lw=2 if m == "gbm" else 1.4, label=PRETTY[m])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Skill vs smart persistence (%)")
    # Figure-level title deliberately not drawn; see note above.
    # ax.set_title("Forecast skill by horizon (post-causal,)")
    ax.set_xticks(hs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(guarded(CFG.FIG / f"{R1_PREFIX}p2_skill_vs_horizon.png", force), dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print("=== Table 2 (in_table2 rows) ===")
    print(tab[["model_pretty", "horizon_min", "n", "MAE", "RMSE", "R2",
               "skill_vs_smart_persistence"]].to_string(index=False))
    print("\n=== run A (second budget, not in Table 2) ===")
    print(t2[(~t2.in_table2)][["model_pretty", "horizon_min", "RMSE"]].to_string(index=False))
    print("\n=== external site (r1_ sourced) ===")
    print(asp[asp.scope == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
