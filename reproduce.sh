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

# Rebuild mode: the final-pass (r1_) scripts refuse to overwrite their own shipped
# outputs unless forced, which is the right default for interactive use but would
# stop a full reproduction at the first regenerated file. Reproduction is exactly
# the case where overwriting those outputs is intended, so it is enabled here.
# Unset this to keep the overwrite guards active.
export R1_REBUILD=1

STAGE="${1:-all}"

if [ "$STAGE" != "r1" ]; then

echo "[1/8] preprocessing (clean + clear-sky + regimes)"
# --legacy-gapfill reproduces the shipped article artifacts bit-for-bit: it applies
# the pre-guard interpolation (first 6 steps of a gap of any length are filled).
# The guarded cleaner (no flag) is the documented default for NEW work; the two
# modes must never be mixed within one run. See data/README.md.
python preprocessing/p2_clean.py --legacy-gapfill
python preprocessing/p3_clearsky_regimes.py
python preprocessing/dkasc_prepare.py

echo "[2/8] point benchmark"
python run_phase2_baselines.py   # deep models optional: see models/HOW_TO_RUN_DEEP_MODEL.md

# Build the shared Yulara feature-frame cache that the per-horizon scripts of
# phases 3-7 read. It was previously built only by hand (or by phase 8a), so a
# clean run died at phase 3. Idempotent: skipped when already present.
python - <<'PYEOF'
import os, sys
sys.path.insert(0, "utils")
if not os.path.exists("/tmp/base.parquet"):
    import datasets as D
    D.build_base().to_parquet("/tmp/base.parquet")
    print("built /tmp/base.parquet")
PYEOF

echo "[3/8] multi-horizon conformal (J2)"
for h in 1 3 6 12; do python _j2_fit.py "$h"; python _j2_one_horizon.py "$h"; done
python _j2_aggregate.py

echo "[4/8] external site (J3)"
for h in 1 3 6 12; do python _j3_one_horizon.py "$h"; done
python _j3_aggregate.py

echo "[5/8] multivariate feature study (J4)"
for h in 1 3 6 12; do python _j4_one_horizon.py "$h"; done
python _j4_aggregate.py

# Build the GHI->PV plant map the dispatch layer reads at import. It was
# previously built only by phase 8a, so the first-pass dispatch of phase 6
# could not run in a clean layout. Deterministic from the cleaned 2023 records;
# safe to re-run.
python r1/r1_build_ghi_pv_map.py > /dev/null
echo "built /tmp/ghi_pv_map.npz"

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

# 8f. the analyses that run entirely from the tables written above: one denominator for the
#     regime split (and Figure 1), multiplicity control, selection regret, the multi-day block
#     bootstrap, the repeated-measures refit, the external-site mirror table, the robustness
#     layer under r1_ names, and the measurement of the interpolated-input exposure.
python r1/r1_s9_regimes.py
python r1/r1_s9_stats.py
python r1/r1_s9_dkasc_mirror.py
python r1/r1_j6_aggregate.py
python r1/r1_s9_causal_rescore.py

# 8g. check the cleaner's run-length guard, and the count of long-run interpolated cells it
#     withholds, against the shipped data.
python r1/r1_s10_verify_runlength_guard.py

echo "Done. See../results/ for tables and figures."
