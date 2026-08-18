# Calibrated, Adaptive, and Risk-Aware Probabilistic Ultra-Short-Term Solar Forecasting for Reserve Scheduling in Hybrid PV Systems

Reproducible code and results for the accompanying article. The study shows that for
ultra-short-term (5–60 min) global horizontal irradiance (GHI) forecasting in hybrid PV
operation, the operational lever is **calibrated, weather-regime-aware uncertainty** and
its decision value — not point-forecast architecture. Marginal conformal prediction is
near-calibrated on average but severely under-covers the **transitional** (broken-cloud)
regime; **regime-conditional (Mondrian)**, **Mondrian-CQR**, and **online adaptive (ACI)**
conformal prediction restore calibration, and the gains translate into lower expected cost
**and** lower tail risk (CVaR) in a battery-buffered reserve-scheduling model. Findings
reproduce on an independent arid-climate site (DKASC, Alice Springs) and across a nine-year
record.

**Hardware.** A standard multi-core CPU is sufficient for the entire pipeline; no GPU and no
cluster are required. The two compact deep baselines are the only components that benefit
from a GPU, and they are optional — their outputs are included.

---

## How the analysis is organised

The analysis was run in two passes, and both are kept so that every reported number can be
traced to the code that produced it.

| | First pass | Final pass |
|---|---|---|
| Code | `code/` — phase drivers `_j2_*` … `_j6_*` | `code/r1/` |
| Tables, metrics, figures | `results/**` without a prefix | `results/**` with the **`r1_`** prefix |

The `r1_` prefix simply marks the second run; it carries no other meaning.

**The article reports the final pass.** Every conformal, dispatch and point-forecast number
in the article comes from an `r1_`-prefixed file produced by `code/r1/`. The one exception is
the two ablation tables in the robustness section, which come from the first-pass `j6_*`
files; the article says so where it reports them.

What the final pass changed, in one line each: it re-runs the point-forecast benchmark on a
strictly causal feature frame; it applies the *h*-step feedback delay that an online adaptive
method must obey at horizons beyond one step; it selects the reserve level on data that
exclude the test year; and it adds paired significance tests to every ranked claim. Each is
described in the module docstring of the script that implements it.

Nothing from the first pass was overwritten. Where a number moved between the two passes,
both versions are on disk.

**The two phase reports dated 2026-08-16 are generated, not hand-written.** Every number in
them is read out of the CSVs at generation time, and the generator will re-derive and check
all 390 of them against `results/tables/`:

```bash
python code/r1/r1_regen_phase_reports.py --verify
```

---

## Repository structure

```
code/            All pipeline code
  preprocessing/   cleaning, calibrated clear-sky index, weather regimes (Yulara + DKASC)
  utils/           config + dataset/feature construction
  evaluation/      metrics (RMSE, Diebold–Mariano, PICP/PINAW/Winkler, CRPS)
  conformal/       conformal.py (ICP/norm/Mondrian/CQR) + conformal_adaptive.py (ACI, Mondrian-CQR)
  models/          compact GRU / GRU-TCN (PyTorch; optional)
  dispatch/        dispatch_soc.py (battery-SoC reserve scheduling with CVaR) + ASSUMPTIONS.md
  r1/              final-pass analysis: reserve-level selection without the test year,
                   delayed-feedback ACI, significance testing, causal deep re-run
  run_phase2_baselines.py, _j2_*, _j3_*, _j4_*, _j5_*, _j6_*  experiment drivers
results/
  tables/          every reported number as CSV (j2_*, …, p2_*; r1_* for the final pass)
  figures/         every figure the pipeline produces, including those in the article (PNG, 200 dpi)
  metrics/         summary JSONs
  reports/         per-phase reports (Markdown) describing methods and results
data/              instructions to obtain the raw data (NOT redistributed here)
```

## Installation

Python 3.10+.

```bash
pip install -r requirements.lock.txt      # exact pins; what was actually run
# pip install -r requirements.txt        # the dependency list, without versions
```

[`MANIFEST.md`](MANIFEST.md) maps every table and figure in the article to the script
that generates it, the result file it is read from, and that file's SHA-256, so any
exhibit can be traced without running anything.

## Data

The raw data are third-party and **not redistributed** in this repository; see
[`data/README.md`](data/README.md) for download instructions and the exact filenames the
preprocessing scripts expect. Both sites come from the Desert Knowledge Australia Solar
Centre (DKASC):

- **Yulara** solar-battery-diesel hybrid (2016–2024, 5-min) — primary site.
- **DKASC Alice Springs** Class-A weather station (2020–2024, 5-min) — external validation.

`code/utils/config.py` detects whether it is running in this release layout (`data/`,
`results/`) or in the authors' working tree (`02_data/`, `04_results/`) and resolves every
path accordingly. Nothing needs configuring.

## Reproduce

After placing the raw data as described in `data/README.md`:

```bash
bash reproduce.sh          # the full pipeline, first pass then final pass
bash reproduce.sh r1       # the final pass only
```

The individual steps, and which are optional, are in [`REPRODUCE.md`](REPRODUCE.md). All
scripts are config-driven via `code/utils/config.py`; outputs are written to `results/`.

**Two caveats, stated rather than buried.**

1. The compact deep baselines need PyTorch and are the only part that is not CPU-cheap. They
   are optional, and their outputs are in `results/tables/r1_p2_deep_causal*.csv`.
2. The reserve-dispatch numbers are **not** bit-for-bit reproducible against the first-pass
   tables. Re-running the identical protocol preserves every ranking at every reserve level,
   but mean daily costs differ by 8.5 % on average (maximum 34 % on a small base). Two causes
   are identified: the gradient-boosting models refit under a different library version, and
   the test year resolves to 357 operating days in the committed data where an earlier frame
   yielded 356. No comparison depends on the absolute level, because every comparison is
   computed inside a single consistent run.
   `results/reports/2026-08-16-J5-Dispatch-SoC-CVaR-Report.md` documents it.

## Key results (test year 2024, 5-min, 90 % nominal)

- **Conditional calibration is the finding.** Marginal conformal prediction is
  near-calibrated on average but transitional-regime coverage is only **0.713** against a
  0.90 target. Regime-conditional and adaptive methods restore per-regime and temporal
  calibration, and **Mondrian-CQR is the only method tested whose transitional-regime
  coverage error is statistically indistinguishable from zero**.
- **Sharpness is a secondary, honestly-sized claim.** Mondrian-CQR's CRPS advantage over CQR
  is statistically significant but **0.17 %–0.26 %**. It is reported with its
  Diebold–Mariano *p*-value, not as a bare "lowest CRPS".
- **Decision value, with the reserve level selected without the test year.** Regime-aware
  uncertainty captures **96.0 %** of the achievable expected-cost saving and **90.4 %** of
  the tail-risk (CVaR) saving at 5 min, against **92.6 %** and **76.3 %** for marginal
  intervals. At 30 min the two tail-risk figures are **84.7 %** and **47.0 %** — the
  advantage of calibration is largest exactly in the tail.
- **Point accuracy.** A tuned gradient-boosting model leads the compact deep baselines by
  **1.8 %–3.2 %** on common support at every horizon, after the deep training budget was more
  than tripled.
- The calibration failure and its correction reproduce on the external site and across
  expanding-window held-out years.

## Citation

If you use this code or the reproduction pipeline, please cite the accompanying article; see
[`CITATION.cff`](CITATION.cff). The archived release has a DOI, and citing the concept DOI
(`10.5281/zenodo.20956895`) always resolves to the newest version. The raw data must be cited
per DKASC's terms (see [`data/README.md`](data/README.md)).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The third-party DKASC data are
subject to the provider's own terms and are not covered by this license.
