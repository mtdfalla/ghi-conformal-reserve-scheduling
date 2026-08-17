"""Final pass - regenerate the two calibration figures from the r1_ tables.

WHY THIS FILE EXISTS
--------------------
The point-forecast table and the external-site table carry `r1_` numbers while
Figs. `picp` and `external` were still the Phase-6 PNGs. The differences are small
(the S1 re-fit moves PICP by <= 0.002,) and would not be visible, but a figure
built from one dataset sitting beside a table built from another is precisely the
inconsistency was about, one float down. Two further reasons to rebuild:

  * the ACI rows must show the **delayed** feedback variant, and the Phase-6
    figure predates that fix entirely;
  * the external-site figure is the visual half of, whose whole complaint was
    that the two sites were not computed the same way.

Everything is read from `04_results/tables/r1_j2_interval_metrics.csv`, the same file
Table 3 is traced to in NUMBER_TRACE.md section 3, so figure and table cannot drift.

OUTPUTS (r1_-prefixed, nothing overwritten)
    04_results/figures/r1_j2_picp_by_regime_5min.png
    04_results/figures/r1_j3_crosssite_calibration.png

Run from 03_code:  python3 r1/r1_j2_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "03_code" / "utils"))

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import config as CFG                     # noqa: E402

R1_PREFIX = "r1_"
NOMINAL = 0.9
H = 5
SCOPES = ["clear", "transitional", "cloudy"]
METHODS = [("icp", "ICP (marginal)", "static"),
           ("icp_norm", "ICP-norm", "static"),
           ("cqr", "CQR", "static"),
           ("mondrian", "Mondrian", "static"),
           ("mondrian_cqr", "Mondrian-CQR", "static"),
           ("aci", "ACI (delayed)", "delayed"),
           ("aci_regime", "ACI-regime (delayed)", "delayed")]


class WriteGuard(Exception):
    pass


def guarded(path: Path, force: bool = False) -> Path:
    if not path.name.startswith(R1_PREFIX):
        raise WriteGuard(f"REFUSING to write '{path.name}': hard rule 2 requires the r1_ prefix.")
    if path.exists() and not force:
        raise WriteGuard(f"REFUSING to overwrite existing file:\n  {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pick(iv, site, method, variant, scope):
    r = iv[(iv.site == site) & (iv.method == method) & (iv.variant == variant)
           & (iv.horizon_min == H) & (iv.nominal == NOMINAL) & (iv.scope == scope)]
    assert len(r) == 1, f"{site}/{method}/{variant}/{scope}: {len(r)} rows"
    return float(r["PICP"].iloc[0])


def main() -> None:
    force = "--force" in sys.argv
    iv = pd.read_csv(CFG.TAB / f"{R1_PREFIX}j2_interval_metrics.csv")

    # ---- Fig. picp: per-regime coverage at Yulara -----------------------------
    x = np.arange(len(SCOPES))
    w = 0.115
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (m, lab, var) in enumerate(METHODS):
        vals = [pick(iv, "yulara", m, var, s) for s in SCOPES]
        ax.bar(x + (i - 3) * w, vals, w, label=lab)
    ax.axhline(NOMINAL, color="k", ls="--", lw=1.2, zorder=5)
    ax.text(2.42, NOMINAL + 0.004, "nominal 0.90", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SCOPES])
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("PICP at 90% nominal")
    # Figure-level title deliberately not drawn: the caption in the article carries the
    # description, and a title above a caption is redundant. Per-panel titles are kept.
    # ax.set_title("Yulara, 5 min: coverage by weather regime\n"
    #              "marginal CP fails in the transitional regime; regime-aware variants restore it",
    #              fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(guarded(CFG.FIG / f"{R1_PREFIX}j2_picp_by_regime_5min.png", force), dpi=200)
    plt.close(fig)

    # ---- Fig. external: the same panel for both sites -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (site, title) in zip(axes, [("yulara", "Yulara (primary)"),
                                        ("asp", "DKASC Alice Springs (external)")]):
        for i, (m, lab, var) in enumerate(METHODS):
            vals = [pick(iv, site, m, var, s) for s in SCOPES]
            ax.bar(x + (i - 3) * w, vals, w, label=lab if site == "yulara" else None)
        ax.axhline(NOMINAL, color="k", ls="--", lw=1.2, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in SCOPES])
        ax.set_ylim(0.60, 1.02)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("PICP at 90% nominal")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower left")
    # Figure-level title deliberately not drawn; see note above.
    # fig.suptitle("Per-regime calibration transfers: same pipeline, same code path, two arid sites",
    #              fontsize=11)
    fig.tight_layout()
    fig.savefig(guarded(CFG.FIG / f"{R1_PREFIX}j3_crosssite_calibration.png", force), dpi=200)
    plt.close(fig)

    # ---- print what went in, so the caption can be checked against it ---------
    for site in ["yulara", "asp"]:
        print(f"--- {site} ---")
        for m, lab, var in METHODS:
            print(f"  {lab:22s} " + "  ".join(f"{s[:5]}={pick(iv, site, m, var, s):.4f}"
                                              for s in SCOPES))


if __name__ == "__main__":
    main()
