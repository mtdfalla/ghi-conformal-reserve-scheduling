#!/usr/bin/env bash
# Reproduce all results. Run from the repo root after installing requirements and
# placing the raw data per data/README.md. Outputs are written under results/.
#
# Phases 1-7 are the FIRST analysis pass (unprefixed result files).
# Phase 8 is the FINAL pass (r1_-prefixed result files) and is what backs every
# number the article reports. Phase 8 depends on phases 1-3 having run.
#
#   bash reproduce.sh            # everything
#   bash reproduce.sh r1         # the final pass only (phases 1-3 must have run)
set -e
cd "$(dirname "$0")/code"

STAGE="${1:-all}"

if [ "$STAGE" != "r1" ]; then

echo "[1/8] preprocessing (clean + clear-sky + regimes)"
python preprocessing/p2_clean.py
python preprocessing/p3_clearsky_regimes.py
python preprocessing/dkasc_prepare.py

echo "[2/8] point benchmark"
python run_phase2_baselines.py   # deep models optional: see models/HOW_TO_RUN_DEEP_MODEL.md

echo "[3/8] multi-horizon conformal (J2)"
for h in 1 3 6 12; do python _j2_fit.py "$h"; python _j2_one_horizon.py "$h"; done
python _j2_aggregate.py

echo "[4/8] external site (J3)"
for h in 1 3 6 12; do python _j3_one_horizon.py "$h"; done
python _j3_aggregate.py

echo "[5/8] multivariate feature study (J4)"
for h in 1 3 6 12; do python _j4_one_horizon.py "$h"; done
python _j4_aggregate.py

echo "[6/8] reserve scheduling + CVaR (J5)"
for h in 1 6; do python _j5_fit.py "$h"; python dispatch/dispatch_soc.py "$h"; done
python _j5_aggregate.py

echo "[7/8] robustness, drift, ablations (J6)"
for y in 2019 2020 2021 2022 2023 2024; do python _j6_drift.py "$y"; done
python _j6_ablations.py calib
for s in full no_roll lags_only minimal; do python _j6_ablations.py feat "$s"; done
python _j6_aggregate.py

fi

echo "[8/8] final pass — reserve-level selection without the test year, delayed-feedback ACI, significance"

# 8a. caches both sites need: feature frames, GHI->PV map, GBM point + quantile predictions
bash r1/r1_restore_env.sh
for site in yulara asp; do for h in 1 3 6 12; do python r1/r1_fit_extra_taus.py "$site" "$h"; done; done

# 8b. point forecasting under the causal regimes, and against the deep baselines
#     r1_deep_causal.py needs PyTorch and is the only optional step; its outputs are shipped.
python r1/r1_p2_point_causal.py
python r1/r1_p2_cv_anova.py
python r1/r1_p2_deep_vs_gbm.py
python r1/r1_p2_table2_build.py

# 8c. conformal with h-step delayed ACI feedback, crossing, bound capping
for site in yulara asp; do for h in 1 3 6 12; do python r1/r1_j2_delayed.py "$site" "$h"; done; done
python r1/r1_j2_stats.py
python r1/r1_j2_aggregate.py
python r1/r1_j2_figures.py

# 8d. reserve dispatch with the reserve level selected WITHOUT the test year
for site in yulara asp; do for h in 1 3 6 12; do python r1/r1_dispatch.py "$site" "$h" all; done; done
python r1/r1_j5_aggregate.py
python r1/r1_j5_vc_stats.py

# 8e. generate the J2 and J5 phase reports from the tables, then verify every number against them
python r1/r1_regen_phase_reports.py
python r1/r1_regen_phase_reports.py --verify

echo "Done. See../results/ for tables and figures."
