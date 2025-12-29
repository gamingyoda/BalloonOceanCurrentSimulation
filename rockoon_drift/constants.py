# -*- coding: utf-8 -*-
"""
Default CMEMS dataset IDs and variable names used by the drift simulator.

Notes:
  - Dataset IDs can change. Always allow overriding via CLI.
  - Variables should match the chosen dataset variable names.
"""

# Currents (physical analysis/forecast, 1/12°, 1-hourly)
CURR_DATASET_DEFAULT = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
CURR_UVAR_DEFAULT = "uo"
CURR_VVAR_DEFAULT = "vo"

# Wind (global L4 NRT, 0.125°, 1-hourly)
WIND_DATASET_DEFAULT = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
WIND_UVAR_DEFAULT = "eastward_wind"
WIND_VVAR_DEFAULT = "northward_wind"

# Waves (Stokes drift, 0.083°, 3-hourly; allow override to PT1H-i etc.)
WAVES_DATASET_DEFAULT = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
WAVES_UVAR_DEFAULT = "VSDX"
WAVES_VVAR_DEFAULT = "VSDY"


# =========================
# ECMWF Open Data (IFS) wind
# =========================
# Uses ECMWF Open Data IFS deterministic forecasts to obtain 10m wind.
# Downloaded as GRIB and converted to NetCDF for OpenDrift.
ECMWF_MODEL_DEFAULT = "ifs"
ECMWF_RESOL_DEFAULT = "0p25"   # 0.25 deg (global)
ECMWF_PARAM_U10 = "10u"
ECMWF_PARAM_V10 = "10v"
# Typical deterministic horizon is up to 240h (10 days). Adjust if your endpoint differs.
ECMWF_MAX_STEP_HOURS_DEFAULT = 240
