# 03_code

All code, organized by pipeline stage. Keep functions importable and CLI-runnable; fix seeds.

- `preprocessing/` — load, clean, regime labels, supervised-window construction, splits.
- `models/` — point forecasters: persistence, smart persistence, ARIMA, gradient boosting, compact GRU/GRU-TCN.
- `conformal/` — conformal prediction wrappers (ICP, CQR, ACI, Mondrian).
- `dispatch/` — irradiance→power mapping + reserve/dispatch simulation (decision layer).
- `evaluation/` — point/probabilistic metrics, Diebold-Mariano, ANOVA, plots.
- `utils/` — shared helpers (paths, config, clear-sky, IO).

Convention: each script writes results to `../04_results/` and logs params used.
See `../00_admin/ENVIRONMENT.md` for dependencies and how to run.
