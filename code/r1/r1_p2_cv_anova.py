"""Final pass - expanding-window cross-validation and ANOVA, RE-RUN POST-CAUSAL.

WHY THIS FILE EXISTS
--------------------
The first-pass point-forecast section carries two further pre-causal results alongside
Table 2: the expanding-window cross-validation across six held-out years, and the
ANOVA on per-day RMSE with model / day-class / year factors. Both are computed from
`datasets.make_xy`, so both moved when made the clear-sky scalar train-only.
Leaving them at their first-pass values while Table 2 is rebuilt would leave the same
inconsistency exists to close, one paragraph further down.

Configuration note: like `r1_p2_point_causal.py`, this uses the GBM the ARTICLE
describes (150 iterations, learning rate 0.08, 31 leaves). the first-pass CV script used
200 / 0.07 / 31 - a third configuration.

OUTPUTS
    04_results/tables/r1_p2_cv_perday_errors.csv   tidy per-(year, day, model, h)
    04_results/tables/r1_p2_cv_summary_by_year.csv mean per-day RMSE + GBM skill
    04_results/tables/r1_p2_anova.csv              F and p per term per horizon
    04_results/metrics/r1_p2_cv_anova.json         headline ranges for the text

Run from 03_code:  python3 r1/r1_p2_cv_anova.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "03_code"
sys.path.insert(0, str(CODE / "utils"))
sys.path.insert(0, str(CODE / "evaluation"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import config as CFG                     # noqa: E402
import datasets as D                     # noqa: E402
import metrics as M                      # noqa: E402
import statsmodels.api as sm             # noqa: E402
import statsmodels.formula.api as smf    # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB   # noqa: E402

TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
HORIZONS = [1, 3, 6, 12]
R1_PREFIX = "r1_"


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
    force = "--force" in sys.argv
    t0 = time.time()
    planned = [CFG.TAB / f"{R1_PREFIX}p2_cv_perday_errors.csv",
               CFG.TAB / f"{R1_PREFIX}p2_cv_summary_by_year.csv",
               CFG.TAB / f"{R1_PREFIX}p2_anova.csv",
               CFG.MET / f"{R1_PREFIX}p2_cv_anova.json"]
    blocked = []
    for p in planned:
        try:
            guarded(p, force)
        except WriteGuard as e:
            blocked.append(str(e))
    if blocked:
        raise SystemExit("Pre-flight FAILED (nothing was fitted):\n\n" + "\n\n".join(blocked))

    base = pd.read_parquet("/tmp/base.parquet")
    reg = pd.read_parquet(CFG.DATA_REGIME / "yulara_regimes_5min.parquet")
    dclass = reg["day_class"]
    rec = lambda kt, cs: np.clip(np.clip(kt, 0, 1.5) * cs, 0, None)   # noqa: E731

    recs = []
    for h in HORIZONS:
        d = D.make_xy(base, h)
        d = d.assign(date=d.index.date, dclass=dclass.reindex(d.index).values)
        for ty in TEST_YEARS:
            tr = d[d["year"] < ty]
            te = d[d["year"] == ty]
            if len(tr) < 5000 or len(te) < 1000:
                continue
            g = HGB(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                    random_state=CFG.SEED)
            g.fit(tr[D.FEATURES].values, tr["y_kt"].values)
            cs = te["y_ghi_cs"].values
            tmp = pd.DataFrame({
                "date": te["date"].values, "dclass": te["dclass"].values,
                "y": te["y_ghi"].values,
                "persistence": te["base_ghi"].values,
                "smart_persistence": rec(te["base_kt"].values, cs),
                "gbm": rec(g.predict(te[D.FEATURES].values), cs)})
            for (dt, dcl), grp in tmp.groupby(["date", "dclass"]):
                if len(grp) < 10 or dcl is None or (isinstance(dcl, float) and np.isnan(dcl)):
                    continue
                for nm in ["persistence", "smart_persistence", "gbm"]:
                    e = grp[nm].values - grp["y"].values
                    recs.append(dict(year=ty, date=str(dt), day_class=str(dcl), model=nm,
                                     horizon_min=h * 5, n=len(grp),
                                     RMSE=float(np.sqrt(np.mean(e ** 2))),
                                     MAE=float(np.mean(np.abs(e)))))
            print(f"h={h*5:>2} test={ty}: train {len(tr):,} test {len(te):,} "
                  f"GBM RMSE {M.rmse(te['y_ghi'].values, rec(g.predict(te[D.FEATURES].values), cs)):.2f} "
                  f"@{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_cv_perday_errors.csv", force), index=False)

    piv = (df.pivot_table(index=["horizon_min", "year"], columns="model", values="RMSE",
                          aggfunc="mean").reset_index())
    piv["gbm_skill_vs_smart_pct"] = (100 * (1 - piv["gbm"] / piv["smart_persistence"])).round(2)
    piv = piv.round(2)
    piv.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_cv_summary_by_year.csv", force), index=False)

    arows = []
    for h in HORIZONS:
        sub = df[df.horizon_min == h * 5].copy()
        model = smf.ols("RMSE ~ C(model) + C(day_class) + C(model):C(day_class) + C(year)",
                        data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        for term, label in [("C(model)", "model"), ("C(day_class)", "day_class"),
                            ("C(model):C(day_class)", "model:day_class"), ("C(year)", "year")]:
            arows.append(dict(horizon_min=h * 5, term=label,
                              F=round(float(aov.loc[term, "F"]), 2),
                              p_value=float(f"{aov.loc[term, 'PR(>F)']:.3g}"),
                              df=int(aov.loc[term, "df"]), n=int(len(sub))))
    anova = pd.DataFrame(arows)
    anova.to_csv(guarded(CFG.TAB / f"{R1_PREFIX}p2_anova.csv", force), index=False)

    dayF = anova[anova.term == "day_class"]["F"]
    dayP = anova[anova.term == "day_class"]["p_value"]
    skill = piv["gbm_skill_vs_smart_pct"]
    meta = dict(
        run_id="(d) expanding-window CV + ANOVA, post-causal",
        finished_utc=pd.Timestamp.utcnow().isoformat(), elapsed_s=round(time.time() - t0, 1),
        gbm_config=dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                        note="the configuration the article states"),
        test_years=TEST_YEARS, n_perday_records=int(len(df)),
        day_class_F_range=[float(dayF.min()), float(dayF.max())],
        day_class_p_max=float(dayP.max()),
        gbm_skill_pct_range=[float(skill.min()), float(skill.max())],
        gbm_skill_positive_in_every_cell=bool((skill > 0).all()),
        model_F_range=[float(anova[anova.term == 'model']['F'].min()),
                       float(anova[anova.term == 'model']['F'].max())],
    )
    json.dump(meta, open(guarded(CFG.MET / f"{R1_PREFIX}p2_cv_anova.json", force), "w"),
              indent=2, default=str)

    pd.set_option("display.width", 220)
    print("\n=== ANOVA ===")
    print(anova.to_string(index=False))
    print("\n=== CV by year ===")
    print(piv.to_string(index=False))
    print("\n" + json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
