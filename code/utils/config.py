"""Shared configuration for the GHI Forecasting project.

Single source of truth for paths, site parameters, column mapping, and cleaning
constants. Import in any script with:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
    import config as C

TWO DIRECTORY LAYOUTS ARE SUPPORTED, and the difference is detected, never configured:

  authors' working tree      public release checkout
  -------------------------  ----------------------------
  02_data/Original Dataset   data/raw
  02_data/cleaned            data/cleaned
  02_data/regime_labels      data/regime_labels
  02_data/interim            data/interim
  02_data/DKASC              data/DKASC
  04_results/{tables,...}    results/{tables,...}
  03_code/                   code/

The working-tree layout is preferred whenever `02_data` or `04_results` is present, so
behaviour in the authors' tree is byte-for-byte what it always was. The release layout is
used only in a checkout of the published repository, where those directories do not exist
and the shipped results live under `results/`. Without this, a third party running
`reproduce.sh` from a clean checkout would silently create an empty `04_results/` beside
the `results/` directory holding the published outputs, and every input path would miss.
"""
from pathlib import Path

# ---- Paths (resolved relative to repo root, layout detected) ----
BASE = Path(__file__).resolve().parents[2]          # repo root

_WORKING = (BASE / "02_data").is_dir() or (BASE / "04_results").is_dir()
if _WORKING:
    LAYOUT       = "working-tree"
    _DATA        = BASE / "02_data"
    DATA_RAW     = _DATA / "Original Dataset"
    RESULTS      = BASE / "04_results"
    CODE         = BASE / "03_code"
else:
    LAYOUT       = "release-checkout"
    _DATA        = BASE / "data"
    DATA_RAW     = _DATA / "raw"
    RESULTS      = BASE / "results"
    CODE         = BASE / "code"

DATA_CLEAN   = _DATA / "cleaned"
DATA_REGIME  = _DATA / "regime_labels"
DATA_INTERIM = _DATA / "interim"
DATA_DKASC   = _DATA / "DKASC"
FIG          = RESULTS / "figures"
TAB          = RESULTS / "tables"
MET          = RESULTS / "metrics"
REPORTS      = RESULTS / "reports" if LAYOUT == "release-checkout" else RESULTS

for _p in (DATA_CLEAN, DATA_REGIME, DATA_INTERIM, FIG, TAB, MET):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Site (Yulara / Ayers Rock Resort, NT, Australia) ----
SITE = dict(
    name="Yulara",
    latitude=-25.2406,
    longitude=130.9889,
    altitude=492,          # metres (approx)
    tz="Australia/Darwin", # ACST, UTC+9:30, no daylight saving
)

# ---- Column mapping: clean_name -> raw_name ----
# Pyranometer_1 is the PRIMARY GHI: clean & consistent across ALL years.
# "Global_Horizontal_Radiation" is faulty 2016-2022 (near-zero daytime) and only
# reliable 2023-2024 -> kept as a secondary cross-check ("ghi_alt").
COLS = {
    "ghi":      "3052_Environment_DG_Weather_Station_Pyranometer_1",
    "ghi_alt":  "3052_Environment_DG_Weather_Station_Global_Horizontal_Radiation",
    "temp_air": "3052_Environment_DG_Weather_Station_Weather_Temperature_Celsius",
    "wind_spd": "3052_Environment_DG_Weather_Station_Wind_Speed",
    "wind_max": "3052_Environment_DG_Weather_Station_Max_Wind_Speed",
    "wind_dir": "3052_Environment_DG_Weather_Station_Wind_Direction",
    "pressure": "3052_Environment_DG_Weather_Station_Air_Pressure",
    "rain_day": "3052_Environment_DG_Weather_Station_Weather_Daily_Rainfall",
    "temp_mod1":"3052_Environment_DG_Weather_Station_Temperature_Probe_1",
    "temp_mod2":"3052_Environment_DG_Weather_Station_Temperature_Probe_2",
    "pv_total": "3050_Total_Site_PV_Generation_Active_Power",
}
TS = "timestamp"

# ---- Cleaning constants ----
FREQ              = "5min"
SENTINEL_ABS      = 9e4
GHI_PHYS_MAX      = 1500.0
GHI_NIGHT_CLIP    = 0.0
SHORT_GAP_STEPS   = 6          # interpolate gaps up to 6 steps (30 min)

# Plausibility bounds (clean_name -> (lo, hi))
BOUNDS = {
    "ghi":      (-50, 1500),
    "ghi_alt":  (-50, 1500),
    "temp_air": (-15, 60),
    "wind_spd": (0, 60),
    "wind_max": (0, 90),
    "wind_dir": (0, 360),
    "pressure": (850, 1050),
    "rain_day": (0, 500),
    "temp_mod1":(-20, 90),
    "temp_mod2":(-20, 90),
    "pv_total": (-5, 2000),
}

SEED = 42
