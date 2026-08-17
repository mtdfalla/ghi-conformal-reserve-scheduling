# Phase 4 — Operating model & assumptions (Q1 defaults)

Stylised solar-battery-diesel hybrid; **real-time operating-reserve** decision under
PV forecast uncertainty (newsvendor). Documented defaults; results reported with a
**cost-ratio sweep** and **battery sensitivity**, so conclusions are robust to the
exact (unknown) parameters.

## Decision problem (per 5-min step, h-ahead)
The operator must schedule dispatchable supply (diesel + battery) to back the PV it
"counts on". It counts on a PV quantile PV_q from the forecast.
- Actual PV = f(GHI_actual);   PV_q = f(GHI_forecast_quantile)   [f = Q5 mapping]
- shortfall (PV over-counted): cost c_u · max(PV_q − PV_act − P_b, 0)
- over-provision (PV under-counted): cost c_o · max(PV_act − PV_q − P_b, 0)
- battery acts as a free fast-balancing band of ±P_b kW (energy-sufficient at 5-min).

Newsvendor-optimal PV_q = PV quantile at τ' = c_o/(c_u+c_o); the calibrated forecast
quantile supplies it. **Cost depends only on PV forecast-quantile error → independent
of the (unknown) net load (Q4).**

## Policies compared
- deterministic (point forecast; ignores uncertainty)
- interval-aware ICP (marginal) / Mondrian (regime-conditional)
- perfect foresight (oracle; lower bound)

## Default parameters (with sensitivity)
| Param | Default | Sensitivity range | Basis |
|---|---|---|---|
| Diesel fuel / over-provision cost c_o | 0.30 $/kWh | 0.25–0.40 | remote diesel genset |
| Shortfall cost c_u (VOLL / fast-start) | 3.0 $/kWh | ratio c_u/c_o ∈ {3,5,10,19} | value-of-lost-load literature |
| Battery buffer power P_b | 0 (core), 300 kW (sensitivity) | 0–500 kW | ~0.3C on ~1 MW PV |
| PV capacity (from mapping f) | ~1046 kW | — | binned-median GHI→PV on 2023 |
| Horizons | 5 & 30 min | — | real-time reserve |

## Metrics
- Total penalty cost ($) over daytime test 2024; per operating-day.
- **Value captured %** = (cost_deterministic − cost_method)/(cost_deterministic − cost_oracle).
- Cost-of-uncertainty = cost_method − cost_oracle.
- Stratified by weather regime.

## Scope note
Conference: operating-reserve cost-of-uncertainty (clean, load-independent). Full
battery SoC dynamics + economic dispatch with CVaR, the model used in this work.
Real PV scatter beyond f(GHI) is non-forecastable operational noise (excluded).
