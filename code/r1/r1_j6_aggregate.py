"""Final pass - regenerate the robustness layer in the current stack, under r1_ names.

WHY THIS FILE EXISTS
--------------------
The calibration-size and feature-ablation tables, and the drift paragraph and its figure,
were first produced in the earlier pass, before the clear-sky scalar was made train-only.
They are regenerated here so that every table and figure the article reports comes from
one stack.

`_j6_drift.py` and `_j6_ablations.py` write only to /tmp, so they are safe to run as they
stand. `_j6_aggregate.py` is NOT safe: it writes `j6_*.csv` straight over the Phase-6 files,
which this project forbids. This wrapper does the same aggregation and writes `r1_j6_*`
instead, leaving every first-pass file byte-for-byte intact, and prints the earlier values
beside the regenerated ones so the change is visible rather than silent.

Run the underlying jobs first (from 03_code):
    for y in 2019 2020 2021 2022 2023 2024; do python3 _j6_drift.py $y; done
    python3 _j6_ablations.py calib
    for s in full no_roll lags_only minimal; do python3 _j6_ablations.py feat $s; done

Then, from the repository root:  python3 code/r1/r1_j6_aggregate.py [--force]
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

import config as CFG                     # noqa: E402

R1 = "r1_"
OUT = "/tmp/j6out"
HUES = {"icp": "#888888", "mondrian": "#11bb77", "aci": "#2277cc"}
PRETTY = {"icp": "ICP", "mondrian": "Mondrian", "aci": "ACI"}


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compare(label, old: pd.DataFrame, new: pd.DataFrame, keys, cols):
    """Print the published value beside the regenerated one, so nothing changes silently."""
    print(f"\n  --- {label}: Phase-6 (published) against the current stack ---")
    m = old.merge(new, on=keys, suffixes=("_published", "_r1"))
    for c in cols:
        a, b = f"{c}_published", f"{c}_r1"
        if a not in m or b not in m:
            continue
        m[f"d_{c}"] = (m[b] - m[a]).round(4)
    show = keys + [x for c in cols for x in (f"{c}_published", f"{c}_r1", f"d_{c}") if x in m]
    print(m[show].to_string(index=False))
    return m


def main():
    force = "--force" in sys.argv

    # ---------------- drift ----------------
    rows = []
    for f in sorted(glob.glob(f"{OUT}/drift_*.json")):
        j = json.load(open(f))
        if j.get("skipped"):
            continue
        rows += j["rows"]
    drift = pd.DataFrame(rows)
    drift.to_csv(guarded(CFG.TAB / f"{R1}j6_drift_coverage.csv", force), index=False)
    d_all = drift[drift.scope == "all"].groupby("method")["PICP"].agg(["mean", "std", "min", "max"]).round(4)
    d_tr = drift[drift.scope == "transitional"].groupby("method")["PICP"].agg(["mean", "std", "min", "max"]).round(4)
    d_all.to_csv(guarded(CFG.TAB / f"{R1}j6_drift_summary_all.csv", force))
    d_tr.to_csv(guarded(CFG.TAB / f"{R1}j6_drift_summary_transitional.csv", force))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for k, scope in enumerate(["all", "transitional"]):
        sub = drift[drift.scope == scope]
        for m_ in ["icp", "mondrian", "aci"]:
            s = sub[sub.method == m_].sort_values("year")
            ax[k].plot(s["year"], s["PICP"], "-o", color=HUES[m_], label=PRETTY[m_])
        ax[k].axhline(0.90, ls="--", c="k", lw=1)
        ax[k].set_title(scope)
        ax[k].set_xlabel("test year")
        ax[k].set_ylim(0.55, 1.02)
        ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("PICP (5-min, 90% target)")
    ax[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(guarded(CFG.FIG / f"{R1}j6_drift_picp_by_year.png", force), dpi=200)
    plt.close()

    print("== DRIFT summary, current stack ==")
    print("  all scope:\n", d_all.to_string())
    print("  transitional:\n", d_tr.to_string())
    for name, new in (("j6_drift_summary_all.csv", d_all), ("j6_drift_summary_transitional.csv", d_tr)):
        p = CFG.TAB / name
        if p.exists():
            old = pd.read_csv(p, index_col=0)
            j = old.join(new, lsuffix="_published", rsuffix="_r1")
            for c in ("mean", "std", "min", "max"):
                j[f"d_{c}"] = (j[f"{c}_r1"] - j[f"{c}_published"]).round(4)
            print(f"\n  --- {name}: published against current ---")
            print(j[[x for c in ("mean", "std", "min", "max")
                     for x in (f"{c}_published", f"{c}_r1", f"d_{c}")]].to_string())

    # ---------------- calibration-set size ----------------
    calib = pd.DataFrame(json.load(open(f"{OUT}/calib.json")))
    calib.to_csv(guarded(CFG.TAB / f"{R1}j6_calib_size.csv", force), index=False)
    if (CFG.TAB / "j6_calib_size.csv").exists():
        compare("Table 10 (calibration-set size)", pd.read_csv(CFG.TAB / "j6_calib_size.csv"),
                calib, ["calib_months", "method"], ["calib_n", "PICP", "PINAW"])

    # ---------------- feature ablation ----------------
    feats = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/feat_*.json"))])
    feats = feats.sort_values("n_features", ascending=False)
    feats.to_csv(guarded(CFG.TAB / f"{R1}j6_feature_ablation.csv", force), index=False)
    if (CFG.TAB / "j6_feature_ablation.csv").exists():
        compare("Table 11 (feature ablation)", pd.read_csv(CFG.TAB / "j6_feature_ablation.csv"),
                feats, ["set"], ["n_features", "RMSE", "PICP_mondrian", "PINAW_mondrian"])

    full = feats[feats.set == "full"].RMSE.iloc[0]
    feats2 = feats.assign(delta_vs_full_pct=(100 * (feats.RMSE / full - 1)).round(2))
    print("\n  Table 11 as it will be printed (delta recomputed against the current full-set RMSE):")
    print(feats2[["set", "n_features", "RMSE", "delta_vs_full_pct", "PICP_mondrian", "time_s"]].to_string(index=False))

    json.dump(dict(drift_all=d_all.to_dict(), drift_transitional=d_tr.to_dict(),
                   feature_ablation=feats2.to_dict("records")),
              open(guarded(CFG.MET / f"{R1}j6_summary.json", force), "w"), indent=2, default=str)
    print("\nWrote r1_j6_* tables, figure and summary. No Phase-6 file was touched.")


if __name__ == "__main__":
    main()
