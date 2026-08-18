"""Final pass - one denominator for the weather-regime split, and Figure 1 rebuilt on it.

WHY THIS FILE EXISTS
--------------------
Three different denominators were in play:

  * code/preprocessing/p3_clearsky_regimes.py writes p1_regime_distribution.csv over ALL
    daytime rows, so its percentages include a `night` bucket of 21,882 daytime rows - rows
    that are daytime but whose clear-sky index is missing, and which the regime script leaves
    at its initialisation label. Its split is 52.49 / 34.99 / 7.48 (+5.04 `night`).
  * code/preprocessing/p4_eda.py builds Figure 1 over CLASSIFIED rows only, giving
    55.28 / 36.85 / 7.88.
  * An earlier draft printed 52 / 36 / 7, taking 52 and 7 from the first denominator and
    36 from neither, so the three summed to 95 per cent with no explanation.

This script computes both denominators explicitly, names the unclassified rows for what they
are, and rebuilds Figure 1 from the same final table the article quotes. No model is fitted
and no dispatch is simulated, so no other reported number can move.

Usage (from the repository root):  python3 code/r1/r1_s9_regimes.py [--force]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from decimal import Decimal, ROUND_HALF_UP   # noqa: E402

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import config as CFG                     # noqa: E402

R1 = "r1_"
ORDER = ["clear", "transitional", "cloudy"]
COLORS = {"clear": "#f4a300", "transitional": "#7fb0d0", "cloudy": "#5a6b7b"}


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def split_for(reg: pd.DataFrame, label: str):
    day = reg[reg["is_day"] == True]                                   # noqa: E712
    n_day = len(day)
    unclass = int((~day["regime"].isin(ORDER)).sum())
    kt_missing = int(day["kt"].isna().sum())
    classified = day[day["regime"].isin(ORDER)]
    rows = []
    for r in ORDER:
        c = int((classified["regime"] == r).sum())
        rows.append(dict(site=label, regime=r, count=c,
                         pct_of_all_daytime=round(100 * c / n_day, 2),
                         pct_of_classified=round(100 * c / len(classified), 2)))
    rows.append(dict(site=label, regime="unclassified", count=unclass,
                     pct_of_all_daytime=round(100 * unclass / n_day, 2),
                     pct_of_classified=np.nan))
    d = pd.DataFrame(rows)
    print(f"  [{label}] daytime rows {n_day:,}; unclassified {unclass:,} "
          f"({100*unclass/n_day:.2f}% — all of them rows with a missing clear-sky index: "
          f"{'yes' if unclass == kt_missing else 'NO, ' + str(kt_missing) + ' have missing kt'})")
    print(d.to_string(index=False))
    return d, classified, n_day, unclass


def main():
    force = "--force" in sys.argv
    print("regime split on one stated denominator")
    print("=" * 78)

    y = pd.read_parquet(CFG.DATA_REGIME / "yulara_regimes_5min.parquet")
    dy, cls_y, n_day_y, unclass_y = split_for(y, "yulara")

    frames = [dy]
    asp_path = CFG.DATA_DKASC / "regime_labels" / "asp_regimes_5min.parquet"
    if asp_path.exists():
        a = pd.read_parquet(asp_path)
        da, _, _, _ = split_for(a, "dkasc")
        frames.append(da)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(guarded(CFG.TAB / f"{R1}s9_regime_distribution.csv", force), index=False)

    # ---- Figure 1, rebuilt on the SAME table, with the denominator named ----
    cls_y = cls_y.copy()
    cls_y["month"] = cls_y.index.month
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    vc = (cls_y["regime"].value_counts(normalize=True).reindex(ORDER) * 100)
    axes[0].bar(ORDER, vc.values, color=[COLORS[o] for o in ORDER])
    for i, v in enumerate(vc.values):
        # half-up to one decimal, so the bar label, the prose and the table cannot disagree
        # by a binary-rounding artefact (36.85 formats as "36.8" under %.1f).
        lab = Decimal(str(round(float(v), 2))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        axes[0].text(i, v + 0.5, f"{lab}%", ha="center", fontsize=12)
    axes[0].set_ylabel("% of classified daytime steps", fontsize=12)
    axes[0].tick_params(labelsize=12)
    mon = (cls_y.groupby(["month", "regime"]).size().unstack()
           .reindex(columns=ORDER).fillna(0))
    mon = mon.div(mon.sum(1), axis=0) * 100
    bottom = np.zeros(len(mon))
    for o in ORDER:
        axes[1].bar(mon.index, mon[o].values, bottom=bottom, label=o, color=COLORS[o])
        bottom += mon[o].values
    axes[1].set_xlabel("month", fontsize=12)
    axes[1].set_ylabel("% of classified daytime", fontsize=12)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(fontsize=11)
    plt.tight_layout()
    figp = guarded(CFG.FIG / f"{R1}s9_regime_distribution.png", force)
    plt.savefig(figp, dpi=200)
    plt.close()
    print(f"\n  figure -> {figp}")
    print("  NOTE: the figure uses the CLASSIFIED denominator and now says so on both axes;")
    print("        the manuscript must state the same denominator and give the unclassified count.")
    print("=" * 78)
    print("DONE")


if __name__ == "__main__":
    main()
