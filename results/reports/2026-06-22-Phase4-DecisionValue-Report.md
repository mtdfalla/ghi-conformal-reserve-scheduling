# Phase 4 — Decision Value of Forecast Uncertainty

**Date:** 2026-06-22  **Status:** Core complete (the IEEM differentiator).

Goal: show that calibrated, **regime-aware** prediction intervals translate into real
**operational/economic value** for a solar-battery-diesel hybrid — not just better
statistics. This is what moves the paper from "an ML forecasting study" into IEEM's
Decision-Analysis / OR scope.

## Decision problem (operating reserve, newsvendor)
At each 5-min step the operator schedules dispatchable supply (diesel + battery) to back
the PV it "counts on" (a forecast quantile PV_q). Shortfall (PV over-counted) costs
`c_u`/kWh (fast-start / lost load); over-provision costs `c_o`/kWh (wasted fuel /
curtailment). Optimal PV_q = forecast quantile at τ'=c_o/(c_u+c_o). **Cost depends only
on PV forecast-quantile error → independent of the unknown net load (Q4).**
GHI→PV via the deterministic Q5 mapping f (cap ≈1046 kW). Assumptions & sensitivity:
`03_code/dispatch/ASSUMPTIONS.md`.

## Policies
deterministic (point) · ICP (marginal interval) · **Mondrian (regime-conditional)** ·
perfect foresight (oracle).

## Headline results (test 2024)
**Value captured** = share of the deterministic→oracle cost gap that a policy closes.

| Horizon | cost ratio c_u/c_o | ICP | **Mondrian** |
|---|---|---|---|
| 5 min | 5 | 7.4% | **17.6%** |
| 5 min | 10 | 22.7% | **36.9%** |
| 5 min | 19 | 40.1% | **53.8%** |
| 30 min | 10 | 30.3% | **44.6%** |
| 30 min | 19 | 50.3% | **61.3%** |

At cost ratio 10, 5-min: reserve cost falls from **$479/day (deterministic) → $371 (ICP)
→ $302 (Mondrian)** — Mondrian cuts cost ~37% vs deterministic. The advantage grows
with cost asymmetry (when shortfalls are expensive, calibrated uncertainty matters more).

## Why regime-conditional wins (the mechanism)
Value captured **by weather regime** (5-min, ratio 10):

| Regime | ICP | Mondrian |
|---|---|---|
| clear | −152% | **−17%** |
| transitional | +29% | **+37%** |
| cloudy | +60% | **+61%** |

In **clear** conditions forecasts are accurate, so adding a reserve margin *wastes* money:
marginal ICP over-widens and is strongly counter-productive (−152%), while **Mondrian
stays tight (−17%, near break-even)**. In **cloudy/transitional** conditions both add
value. Net: regime-aware intervals put width where risk is and remove it where it isn't —
exactly the calibration story of Phase 3, now in dollars.

## Takeaways for the paper
1. **Calibrated, regime-conditional uncertainty has measurable operational value** —
   ~37–45% of the achievable reserve-cost saving at realistic cost asymmetry, vs
   ~23–30% for marginal CP. This is the decision-analysis contribution.
2. The value is **regime-driven and cost-asymmetry-driven**, consistent with Phases 2–3.
3. Result is **load-independent** and **CPU-cheap**, with documented assumptions +
   sensitivity (cost ratio, battery buffer) — robust to unknown site specifics.

## Outputs
Table: `04_results/tables/p4_costofuncertainty.csv`. Figures: `p4_value_vs_costratio.png`,
`p4_cost_by_policy.png`, `p4_value_by_regime.png`. Code: `03_code/run_phase4.py`,
`03_code/dispatch/ASSUMPTIONS.md`. Mapping: `04_results/metrics/p4_ghi_pv_mapping.json`.

## Scope / extensions (journal, Phase 6)
Full battery SoC dynamics + economic dispatch with CVaR risk; CQR-based reserve;
multi-horizon; external site.
