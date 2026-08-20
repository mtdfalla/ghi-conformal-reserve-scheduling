"""Final pass - make Table 7 a true mirror of Table 8.

WHY THIS FILE EXISTS
--------------------
Table 7 is captioned "External-site (DKASC) mirror of Table 8" but is not one: it carries
ICP, Mondrian and Mondrian-CQR only - CQR is absent - and it has no confidence intervals
and no absolute CVaR column, all of which Table 8 has. The external site should be
presented exactly as the main site is.

Nothing needs re-running. Every missing value is already committed:
  r1_j5_protocols.csv   theta, mean_daily, cvar95_daily, value_captured_{mean,cvar}
  r1_j5_cvar_ci.csv     the paired day-block bootstrap intervals

This script assembles the mirror in exactly Table 8's column order and prints it as LaTeX
rows, so the article table is a transcription rather than a re-derivation.

Usage (from the repository root):  python3 code/r1/r1_s9_dkasc_mirror.py [--force]
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import pandas as pd                      # noqa: E402

import config as CFG                     # noqa: E402

R1 = "r1_"
PRETTY = {"icp": "ICP", "mondrian": "Mondrian", "cqr": "CQR", "mondrian_cqr": "Mondrian-CQR"}
ORDER = ["icp", "mondrian", "cqr", "mondrian_cqr"]


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ci(s):
    lo, hi = [float(x) for x in str(s).strip("[]").split(",")]
    return f"[{lo:.3f}, {hi:.3f}]"


def main():
    force = "--force" in sys.argv or os.environ.get("R1_REBUILD") == "1"   # reproduce.sh sets R1_REBUILD=1
    pr = pd.read_csv(CFG.TAB / "r1_j5_protocols.csv")
    ic = pd.read_csv(CFG.TAB / "r1_j5_cvar_ci.csv")
    site = "asp"

    rows = []
    for h in (5, 15, 30, 60):
        for pol in ORDER:
            p = pr[(pr.site == site) & (pr.horizon_min == h) & (pr.protocol == "A") & (pr.policy == pol)]
            c = ic[(ic.site == site) & (ic.horizon_min == h) & (ic.protocol == "A") & (ic.policy == pol)]
            if p.empty:
                print(f"  MISSING protocol row: {site} h={h} {pol}")
                continue
            p = p.iloc[0]
            rows.append(dict(
                site=site, horizon_min=h, policy=pol, theta=round(float(p.theta), 3),
                mean_daily=round(float(p.mean_daily), 2), cvar95_daily=round(float(p.cvar95_daily), 2),
                vc_mean=round(float(p.value_captured_mean), 3),
                vc_cvar=round(float(p.value_captured_cvar), 3),
                vc_mean_ci=(ci(c.vc_mean_ci.iloc[0]) if not c.empty else ""),
                vc_cvar_ci=(ci(c.vc_cvar_ci.iloc[0]) if not c.empty else "")))
    d = pd.DataFrame(rows)
    d.to_csv(guarded(CFG.TAB / f"{R1}s9_dkasc_mirror.csv", force), index=False)

    # anchors, for the caption - stated the same way Table 8 states them
    anc = pr[(pr.site == site) & (pr.protocol == "A") & (pr.policy.isin(["deterministic", "oracle"]))]
    print("\n  Anchors for the caption (DKASC, Protocol A):")
    for _, a in anc.sort_values(["policy", "horizon_min"]).iterrows():
        print(f"    {a.policy:14} h={int(a.horizon_min):>2}  mean ${a.mean_daily:>8.2f}   CVaR ${a.cvar95_daily:>9.2f}")
    n_days = None
    try:
        import numpy as np
        z = np.load(CFG.TAB / "r1_j5_daily_costs.npz", allow_pickle=True)
        n_days = int(z["asp_h1|deterministic|na"].size)
    except Exception:
        pass
    print(f"    operating days at DKASC: {n_days}")

    print("\n  LaTeX rows, in Table 8's column order "
          "(Horizon & Method & theta & mean $ & CVaR $ & VC(mean) [CI] & VC(CVaR) [CI]):\n")
    last = None
    for _, r in d.iterrows():
        hcell = f"{r.horizon_min} min" if r.horizon_min != last else ""
        last = r.horizon_min
        print(f"    {hcell:7} & {PRETTY[r.policy]:13} & ${r.theta:.3f}$ & ${r.mean_daily:.2f}$ & "
              f"${r.cvar95_daily:.2f}$ & ${r.vc_mean:.3f}$ {r.vc_mean_ci} & "
              f"${r.vc_cvar:.3f}$ {r.vc_cvar_ci} \\\\")
    print("\n  CQR against Mondrian-CQR at DKASC, which the current Table 7 cannot show:")
    for h in (5, 15, 30, 60):
        a = d[(d.horizon_min == h) & (d.policy == "cqr")]
        b = d[(d.horizon_min == h) & (d.policy == "mondrian_cqr")]
        if a.empty or b.empty:
            continue
        print(f"    h={h:>2}  CQR VC(mean) {a.vc_mean.iloc[0]:.3f} vs M-CQR {b.vc_mean.iloc[0]:.3f}  "
              f"(diff {b.vc_mean.iloc[0]-a.vc_mean.iloc[0]:+.3f})")


if __name__ == "__main__":
    main()
