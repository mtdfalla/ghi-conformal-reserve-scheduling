#!/bin/bash
# One-command environment restore. /tmp caches are lost between
# sessions. This rebuilds everything S1-S3 need in ~8 minutes on 2 CPUs.
#   bash 03_code/r1/r1_restore_env.sh
set -e
# Install the exact pinned environment. This MUST NOT be allowed to fail silently:
# a partial install produces numbers that look fine and are not reproducible.
LOCK="$(cd "$(dirname "$0")/../.." && pwd)/requirements.lock.txt"
if [ -f "$LOCK" ]; then
  pip install -r "$LOCK" --break-system-packages -q
else
  echo "requirements.lock.txt not found at $LOCK" >&2
  exit 1
fi
cd "$(dirname "$0")/.."                      # -> the code directory (03_code, or code/ in a release checkout)
python3 - <<'PY'
import sys; sys.path.insert(0,'utils')
import pandas as pd, numpy as np, datasets as D, config as C, os
if not os.path.exists('/tmp/base.parquet'):
    D.build_base().to_parquet('/tmp/base.parquet'); print('built /tmp/base.parquet')
if not os.path.exists('/tmp/base_asp.parquet'):
    # C.DATA_DKASC resolves to 02_data/DKASC in the authors' tree and data/DKASC in a
    # public release checkout; see the layout note in utils/config.py.
    reg = pd.read_parquet(C.DATA_DKASC/"regime_labels"/"asp_regimes_5min.parquet")
    df = pd.DataFrame(index=reg.index)
    for c in ["ghi","ghi_cs","kt","zenith","is_day","regime","ghi_imputed"]: df[c]=reg[c]
    df["year"]=df.index.year
    hod=df.index.hour+df.index.minute/60
    df["hod_sin"]=np.sin(2*np.pi*hod/24); df["hod_cos"]=np.cos(2*np.pi*hod/24)
    doy=df.index.dayofyear
    df["doy_sin"]=np.sin(2*np.pi*doy/365); df["doy_cos"]=np.cos(2*np.pi*doy/365)
    df["cosz"]=np.cos(np.radians(df["zenith"].clip(0,90)))
    for L in D.LAGS: df[f"kt_l{L}"]=df["kt"].shift(L)
    df["kt_rmean"]=df["kt"].rolling(D.ROLL,min_periods=D.ROLL).mean()
    df["kt_rstd"]=df["kt"].rolling(D.ROLL,min_periods=D.ROLL).std()
    df.to_parquet('/tmp/base_asp.parquet'); print('built /tmp/base_asp.parquet')
PY
python3 r1/r1_build_ghi_pv_map.py > /dev/null && echo "built /tmp/ghi_pv_map.npz"
for site in yulara asp; do for h in 1 3 6 12; do
  for a in 1 2 3 4 5 6; do
    python3 r1/r1_fit_cache.py $site $h >> /tmp/r1_s0.log 2>&1
    grep -q "ALL_DONE $site h=$((h*5))min" /tmp/r1_s0.log && break
  done
done; done
echo "S0_COMPLETE" | tee -a /tmp/r1_s0.log
grep ALL_DONE /tmp/r1_s0.log
