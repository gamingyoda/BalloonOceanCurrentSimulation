# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.readers import reader_constant

@dataclass(frozen=True)
class OceanDriftConfig:
    lon: float
    lat: float
    start_utc: datetime  # tz-naive UTC
    duration: timedelta
    dt_seconds: int
    dt_out_seconds: int
    windage_percents: list[float]
    n_per_member: int
    include_wind: bool = True
    include_waves: bool = True
    const_wind_u: float = 0.0
    const_wind_v: float = 0.0
    coastline_action: str = "stranding"

# Backwards compatibility for older callers
DriftRunConfig = OceanDriftConfig

def build_model(
    currents_nc: str | Path,
    wind_nc: str | Path | None,
    waves_nc: str | Path | None,
    const_wind_u: float,
    const_wind_v: float,
    dt_out_minutes: int,
    include_wind: bool = True,
    include_waves: bool = True,
    coastline_action: str = "stranding",
    loglevel: int = 20,
) -> OceanDrift:
    o = OceanDrift(loglevel=loglevel)
    readers = [Reader(str(currents_nc))]

    # Wind: prefer NetCDF, else constant; if disabled, use zero wind
    if include_wind:
        if wind_nc and Path(wind_nc).is_file():
            try:
                readers.append(Reader(str(wind_nc)))
            except Exception as e:
                print(f"[WARN] wind Reader failed: {e} -> fallback to constant wind")
                readers.append(reader_constant.Reader({"x_wind": const_wind_u, "y_wind": const_wind_v}))
        else:
            readers.append(reader_constant.Reader({"x_wind": const_wind_u, "y_wind": const_wind_v}))
    else:
        readers.append(reader_constant.Reader({"x_wind": 0.0, "y_wind": 0.0}))

    # Waves/Stokes: optional
    if include_waves and waves_nc and Path(waves_nc).is_file():
        try:
            readers.append(Reader(str(waves_nc)))
        except Exception as e:
            print(f"[WARN] waves Reader failed: {e} -> continue without Stokes drift")

    o.add_reader(readers)
    o.set_config("general:coastline_action", coastline_action)
    o.set_config("general:time_step_output_minutes", float(dt_out_minutes))
    return o

def run_oceandrift(
    cfg: OceanDriftConfig,
    currents_nc: str | Path,
    wind_nc: str | Path | None,
    waves_nc: str | Path | None,
    out_nc: str | Path,
    loglevel: int = 20,
) -> Path:
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    o = build_model(
        currents_nc=currents_nc,
        wind_nc=wind_nc,
        waves_nc=waves_nc,
        const_wind_u=cfg.const_wind_u,
        const_wind_v=cfg.const_wind_v,
        dt_out_minutes=int(cfg.dt_out_seconds / 60),
        include_wind=cfg.include_wind,
        include_waves=cfg.include_waves,
        coastline_action=cfg.coastline_action,
        loglevel=loglevel,
    )

    # Seed ensemble (windage members); disable windage when include_wind is False
    for pct in cfg.windage_percents:
        wind_factor = (pct / 100.0) if cfg.include_wind else 0.0
        o.seed_elements(
            cfg.lon,
            cfg.lat,
            time=cfg.start_utc,
            number=cfg.n_per_member,
            wind_drift_factor=wind_factor,
            z=0,
        )

    o.run(
        duration=cfg.duration,
        time_step=cfg.dt_seconds,
        time_step_output=cfg.dt_out_seconds,
        outfile=str(out_nc),
    )
    return out_nc

# Backwards compatibility
run_drift = run_oceandrift
