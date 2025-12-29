# -*- coding: utf-8 -*-
"""ECMWF IFS wind (Open Data) downloader + NetCDF converter.

Why this exists:
  - CMEMS currents/waves products can provide *forecasts*.
  - Many CMEMS wind products are NRT/analysis only, not future forecasts.
  - For "where will it drift" you often want wind forecasts too.

Implementation notes:
  - Uses `ecmwf-opendata` to download IFS deterministic GRIB messages for 10m winds.
  - Converts GRIB -> NetCDF using xarray+cfgrib so OpenDrift can read it via Reader_netCDF_CF_generic.
  - Spatial subsetting is done *after* download (ECMWF Open Data does not guarantee bbox-subset at download).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import os
from typing import Iterable

import pandas as pd
import xarray as xr
import requests

from .geo import BBox
from .netcdf_utils import nc_time_bounds, window_within_bounds

@dataclass(frozen=True)
class ECMWFWindSpec:
    source: str = "aws"        # 'aws' is usually the most reliable mirror for Open Data
    model: str = "ifs"
    resol: str = "0p25"
    max_step_hours: int = 240

def _hash_key(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]

def _choose_cycle_latest_covering(end_utc: datetime, spec: ECMWFWindSpec) -> datetime:
    """Pick a forecast cycle (init time) that covers `end_utc`.

    Strategy:
      - Ask the client for the latest available cycle for a long step (e.g., 240h).
      - Use that cycle as long as end_utc is within horizon.
    """
    try:
        from ecmwf.opendata import Client  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "ECMWF wind requested but `ecmwf-opendata` is not installed.\n"
            "Install: pip install ecmwf-opendata"
        ) from e

    client = Client(source=spec.source)
    # Depending on ecmwf-opendata version, latest() may return a datetime or an object with `.datetime`.
    latest = client.latest(type="fc", step=spec.max_step_hours, param="10u")
    cycle_dt = latest.datetime if hasattr(latest, "datetime") else latest
    cycle = cycle_dt.replace(tzinfo=None)

    if end_utc > (cycle + timedelta(hours=spec.max_step_hours)):
        raise RuntimeError(
            "Requested end time is beyond the available ECMWF Open Data forecast horizon.\n"
            f"cycle={cycle} horizon={spec.max_step_hours}h end_utc={end_utc}"
        )
    return cycle

def _steps_for_window(cycle: datetime, start_utc: datetime, end_utc: datetime) -> list[int]:
    """Return forecast steps (hours) that cover [start_utc, end_utc].

    IFS open data (0.25°, oper) provides 3-hourly steps up to ~144h, then 6-hourly
    beyond that. Request only the available step grid to avoid 404s.
    """
    lead_start_h = max(0, int(((start_utc - cycle).total_seconds()) // 3600))
    lead_end_h = max(0, int(((end_utc - cycle).total_seconds() + 3599) // 3600))  # ceil to hour

    steps: list[int] = []

    def add_range(a: int, b: int, step: int):
        if b < a:
            return
        steps.extend(range(a, b + 1, step))

    # 0–144h: 3-hourly
    start3 = (lead_start_h // 3) * 3
    end3 = min(((lead_end_h + 2) // 3) * 3, 144)
    add_range(start3, end3, 3)

    # >144h: 6-hourly from 150h onward (align to multiple of 6 starting at 150)
    if lead_end_h > 144:
        start6 = max(150, ((max(lead_start_h, 150) + 5) // 6) * 6)
        end6 = ((lead_end_h + 5) // 6) * 6
        add_range(start6, end6, 6)

    # Ensure we include a guard step just before start if available
    guard = max(0, start3 - 3)
    if guard not in steps:
        steps.append(guard)

    return sorted(set(steps))

def _open_grib_u10v10(grib_path: Path) -> xr.Dataset:
    """Open GRIB and return dataset with u10/v10 at 10m height.

    Requires `cfgrib` + `eccodes`.
    """
    try:
        import cfgrib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "To convert ECMWF GRIB -> NetCDF you need `cfgrib` and `eccodes`.\n"
            "Install: pip install cfgrib\n"
            "Ubuntu (recommended): sudo apt install -y libeccodes0 eccodes\n"
        ) from e

    # filter_by_keys selects the 10m above ground fields
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
        },
    )
    return ds

def _to_time_series(ds: xr.Dataset) -> xr.Dataset:
    """Convert (time, step, lat, lon) to (time, lat, lon) with time=valid_time."""
    # ds typically contains dims: time (init) length 1, step length N, latitude, longitude.
    if "step" not in ds.dims:
        return ds

    def _init_time() -> pd.Timestamp:
        # cfgrib may expose init time as scalar coord, not a dim
        for key in ["time", "forecast_reference_time", "dataDate"]:
            if key in ds.coords:
                val = ds.coords[key]
                try:
                    return pd.to_datetime(val.values.ravel()[0])
                except Exception:
                    continue
        return pd.Timestamp(0)

    if "valid_time" in ds.variables:
        vt = ds["valid_time"]
        if "time" in vt.dims:
            vt = vt.isel(time=0)
    else:
        init = _init_time()
        vt = xr.DataArray(
            [init + pd.to_timedelta(int(s / 1e9), unit="s") for s in ds["step"].values],
            dims=("step",),
        )

    # If there is no time dim, don't isel; just drop helper coords if present
    drop_keys: list[str] = []
    for v in ["time", "valid_time"]:
        if v in ds.variables or v in ds.coords:
            drop_keys.append(v)

    ds2 = ds
    if "time" in ds.dims:
        ds2 = ds.isel(time=0)
    ds2 = ds2.drop_vars([v for v in drop_keys if v in ds2.variables], errors="ignore")
    ds2 = ds2.drop_vars([v for v in drop_keys if v in ds2.coords], errors="ignore")

    ds2 = ds2.assign_coords(time=("step", pd.to_datetime(vt.values)))
    ds2 = ds2.swap_dims({"step": "time"}).drop_vars("step")
    return ds2

def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude to [-180, 180) and sort."""
    if "longitude" not in ds.coords:
        return ds
    lon = ds["longitude"].values
    lon2 = ((lon + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(longitude=lon2).sortby("longitude")
    return ds

def _subset_bbox(ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
    # longitude may be ascending; latitude might be descending
    if "longitude" in ds.coords:
        ds = ds.sel(longitude=slice(bbox.min_lon, bbox.max_lon))
    if "latitude" in ds.coords:
        lat = ds["latitude"].values
        if len(lat) >= 2 and lat[0] > lat[-1]:
            ds = ds.sel(latitude=slice(bbox.max_lat, bbox.min_lat))
        else:
            ds = ds.sel(latitude=slice(bbox.min_lat, bbox.max_lat))
    return ds

def download_ecmwf_wind_netcdf(
    bbox: BBox,
    start_utc: datetime,
    end_utc: datetime,
    out_dir: str | Path,
    spec: ECMWFWindSpec = ECMWFWindSpec(),
    force_download: bool = False,
) -> Path:
    """Download ECMWF Open Data IFS wind (10u/10v) and output NetCDF.

    Returns the path to the produced NetCDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = _hash_key(
        f"{bbox.min_lon:.3f},{bbox.max_lon:.3f},{bbox.min_lat:.3f},{bbox.max_lat:.3f}",
        start_utc.isoformat(),
        end_utc.isoformat(),
        spec.source,
        spec.model,
        spec.resol,
    )
    out_grib = out_dir / f"ecmwf_wind_{key}.grib2"
    out_nc = out_dir / f"ecmwf_wind_{key}.nc"

    # If cache exists and covers the window, reuse
    if out_nc.exists() and out_nc.stat().st_size > 0 and not force_download:
        try:
            b = nc_time_bounds(out_nc)
            if window_within_bounds(start_utc, end_utc, b):
                print(f"[ECMWF] cache covers request: {out_nc} [{b.tmin}..{b.tmax}]")
                return out_nc
            print(f"[ECMWF] cache stale: {out_nc} [{b.tmin}..{b.tmax}] -> refresh")
        except Exception as e:
            print(f"[WARN] ECMWF cache read failed: {e} -> refresh")

    # Choose cycle and steps
    cycle = _choose_cycle_latest_covering(end_utc, spec)
    steps = _steps_for_window(cycle, start_utc, end_utc)

    # Download GRIB
    try:
        from ecmwf.opendata import Client  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "ECMWF wind requested but `ecmwf-opendata` is not installed.\n"
            "Install: pip install ecmwf-opendata"
        ) from e

    sources = []
    for s in [spec.source, "aws", "ecmwf", "azure", "google"]:
        if s not in sources:
            sources.append(s)

    date_str = cycle.strftime("%Y%m%d")
    time_str = cycle.strftime("%H")

    last_err: Exception | None = None
    used_source = None
    for src in sources:
        client = Client(source=src)
        print(
            f"[ECMWF] retrieving IFS wind (source={src}): cycle={cycle} "
            f"steps={steps[0]}..{steps[-1]}h (n={len(steps)})"
        )
        try:
            client.retrieve(
                type="fc",
                date=date_str,
                time=time_str,
                step=steps,
                param=["10u", "10v"],
                target=str(out_grib),
            )
            used_source = src
            break
        except requests.exceptions.HTTPError as e:
            last_err = e
            print(f"[WARN] ECMWF source={src} failed: {e}")
            continue
        except Exception as e:
            last_err = e
            print(f"[WARN] ECMWF source={src} failed: {e}")
            continue

    if used_source is None:
        raise RuntimeError(f"ECMWF Open Data retrieval failed across sources {sources}: {last_err}")

    if not out_grib.exists() or out_grib.stat().st_size == 0:
        raise RuntimeError(f"ECMWF GRIB download failed: {out_grib}")

    # GRIB -> NetCDF (subset to bbox)
    ds = _open_grib_u10v10(out_grib)
    ds = _to_time_series(ds)
    ds = _normalize_lon(ds)
    ds = _subset_bbox(ds, bbox)

    # Rename to CF-ish names OpenDrift reads well
    # cfgrib often names variables as "u10" and "v10"
    rename = {}
    for cand_u, cand_v in [("u10", "v10"), ("10u", "10v")]:
        if cand_u in ds.data_vars:
            rename[cand_u] = "eastward_wind"
        if cand_v in ds.data_vars:
            rename[cand_v] = "northward_wind"
    ds = ds.rename(rename)

    if "eastward_wind" not in ds.data_vars or "northward_wind" not in ds.data_vars:
        raise RuntimeError(
            "ECMWF wind NetCDF conversion failed: expected u10/v10 variables not found. "
            f"vars={list(ds.data_vars)}"
        )

    ds["eastward_wind"].attrs.update({"standard_name": "eastward_wind", "units": "m s-1"})
    ds["northward_wind"].attrs.update({"standard_name": "northward_wind", "units": "m s-1"})
    ds.attrs.update(
        {
            "title": "ECMWF Open Data IFS 10m wind (subset for rockoon drift)",
            "source": f"ECMWF Open Data (source={used_source})",
            "cycle_utc": cycle.isoformat(),
        }
    )

    # Ensure time is sorted
    ds = ds.sortby("time")

    ds.to_netcdf(out_nc)
    print(f"[OK] ECMWF wind NetCDF: {out_nc}")
    return out_nc
