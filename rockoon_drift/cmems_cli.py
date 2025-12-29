# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .geo import BBox, bbox_around
from .netcdf_utils import nc_time_bounds, nc_has_vars, window_within_bounds

_RE_BOUNDS = re.compile(r"dataset coordinates \[([0-9\-:\+T ]+),\s*([0-9\-:\+T ]+)\]")

def _parse_dataset_bounds_from_error(errtext: str) -> tuple[Optional[datetime], Optional[datetime]]:
    if not errtext:
        return None, None
    m = _RE_BOUNDS.search(errtext)
    if not m:
        return None, None
    import pandas as pd
    s = pd.to_datetime(m.group(1)).to_pydatetime().replace(tzinfo=None)
    e = pd.to_datetime(m.group(2)).to_pydatetime().replace(tzinfo=None)
    return s, e

@dataclass(frozen=True)
class SubsetRequest:
    dataset_id: str
    variables: tuple[str, ...]
    bbox: BBox
    start_utc: datetime
    end_utc: datetime
    depth_min: float | None = None
    depth_max: float | None = None

def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def _cache_key(req: SubsetRequest) -> str:
    note = {
        "dataset": req.dataset_id,
        "vars": list(req.variables),
        "bbox": [req.bbox.min_lon, req.bbox.max_lon, req.bbox.min_lat, req.bbox.max_lat],
        "start": req.start_utc.isoformat(),
        "end": req.end_utc.isoformat(),
        "dmin": req.depth_min,
        "dmax": req.depth_max,
    }
    h = hashlib.sha256(json.dumps(note, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return h

def ensure_cli_available() -> None:
    if shutil.which("copernicusmarine") is None:
        raise RuntimeError(
            "copernicusmarine CLI が見つかりません。\n"
            "  pip install copernicus-marine-client\n"
            "を venv 内で実行し、'copernicusmarine --help' が通る状態にしてください。"
        )

def subset_to_netcdf(
    req: SubsetRequest,
    cache_dir: str | Path,
    force_download: bool = False,
    max_retries: int = 3,
    allow_time_shift: bool = True,
    time_tolerance_minutes: float = 180.0,
    verbose: bool = True,
) -> Path:
    """
    Robust wrapper around `copernicusmarine subset`:

    - Deterministic cache filename (dataset+params hash)
    - Validate: file exists, has variables, and covers [start,end]
    - Retry on common time-range errors by trimming/shifting within dataset bounds
    - Atomic write via temp file then rename
    """
    ensure_cli_available()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(req)
    name = f"{_safe_slug(req.dataset_id)}_{key}.nc"
    out_nc = cache_dir / name
    tmp_nc = cache_dir / (name + ".part")

    # Cache hit?
    if out_nc.exists() and out_nc.stat().st_size > 0 and not force_download:
        try:
            b = nc_time_bounds(out_nc)
            ok_time = window_within_bounds(req.start_utc, req.end_utc, b)
            ok_vars = nc_has_vars(out_nc, req.variables)
            if ok_time and ok_vars:
                if verbose:
                    print(f"[CMEMS] cache hit: {out_nc} [{b.tmin}..{b.tmax}]")
                return out_nc
            if verbose:
                print(f"[CMEMS] cache invalid -> refresh: {out_nc}")
        except Exception as e:
            if verbose:
                print(f"[CMEMS] cache read failed -> refresh: {e}")

    start, end = req.start_utc, req.end_utc
    dur = end - start

    def _run_subset(s: datetime, e: datetime) -> None:
        if tmp_nc.exists():
            tmp_nc.unlink()
        cmd = ["copernicusmarine", "subset", "--dataset-id", req.dataset_id]
        for v in req.variables:
            cmd += ["--variable", v]
        cmd += [
            "--start-datetime", s.isoformat(),
            "--end-datetime", e.isoformat(),
            "--minimum-longitude", f"{req.bbox.min_lon:.6f}",
            "--maximum-longitude", f"{req.bbox.max_lon:.6f}",
            "--minimum-latitude",  f"{req.bbox.min_lat:.6f}",
            "--maximum-latitude",  f"{req.bbox.max_lat:.6f}",
            "--output-directory", str(cache_dir),
            "--output-filename", tmp_nc.name,
            "--force-download",
        ]
        if req.depth_min is not None and req.depth_max is not None:
            cmd += ["--minimum-depth", str(req.depth_min), "--maximum-depth", str(req.depth_max)]

        if verbose:
            print("[CMEMS] subset:", " ".join(cmd))

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _resolve_downloaded_path() -> Path | None:
        """Handle copernicusmarine adding .nc to filenames (or numbered suffixes)."""
        if tmp_nc.exists():
            return tmp_nc
        candidates = [p for p in tmp_nc.parent.glob(tmp_nc.name + "*") if p.is_file()]
        if not candidates:
            return None
        # pick the most recently modified candidate to avoid stale files
        return max(candidates, key=lambda p: p.stat().st_mtime)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            _run_subset(start, end)
            actual_nc = _resolve_downloaded_path()
            if actual_nc is None or actual_nc.stat().st_size == 0:
                raise RuntimeError("subset output missing or empty")

            if not nc_has_vars(actual_nc, req.variables):
                raise RuntimeError(f"downloaded NetCDF missing variables: {req.variables}")

            b = nc_time_bounds(actual_nc)
            tol = timedelta(minutes=time_tolerance_minutes)
            if not window_within_bounds(start, end, b):
                if start >= (b.tmin - tol) and end <= (b.tmax + tol):
                    if verbose:
                        print(
                            f"[CMEMS] accept within ±{time_tolerance_minutes} min: "
                            f"requested [{start}..{end}] got [{b.tmin}..{b.tmax}]"
                        )
                else:
                    raise RuntimeError(
                        f"downloaded NetCDF does not cover window: want [{start}..{end}] got [{b.tmin}..{b.tmax}]"
                    )

            # normalize filename to deterministic cache name
            actual_nc.replace(out_nc)
            if verbose:
                print(f"[CMEMS] OK: {out_nc} [{b.tmin}..{b.tmax}]")
            return out_nc

        except subprocess.CalledProcessError as e:
            last_err = e
            errtxt = (e.stderr or "") + "\n" + (e.stdout or "")
            ds_s, ds_e = _parse_dataset_bounds_from_error(errtxt)

            if verbose:
                print(f"[CMEMS] attempt {attempt} failed (returncode={e.returncode})")
                if ds_s and ds_e:
                    print(f"        dataset time bounds: [{ds_s}..{ds_e}]")

            if allow_time_shift and ds_s and ds_e:
                adj_s = max(start, ds_s)
                adj_e = min(end, ds_e)

                if adj_e > adj_s:
                    if verbose:
                        print(f"[CMEMS] retry within bounds: [{adj_s}..{adj_e}]")
                    start, end = adj_s, adj_e
                    continue

                if start > ds_e:
                    end = ds_e
                    start = max(ds_s, ds_e - dur)
                    if verbose:
                        print(f"[CMEMS] shift back: [{start}..{end}]")
                    continue

                if end < ds_s:
                    start = ds_s
                    end = min(ds_e, ds_s + dur)
                    if verbose:
                        print(f"[CMEMS] shift forward: [{start}..{end}]")
                    continue

            raise RuntimeError("copernicusmarine subset failed:\n" + errtxt) from e

        except Exception as e:
            last_err = e
            if verbose:
                print(f"[CMEMS] attempt {attempt} failed: {e}")
            if tmp_nc.exists():
                tmp_nc.unlink()
            if attempt == max_retries:
                raise
            continue

    raise RuntimeError(f"subset failed after retries: {last_err}")

def make_request(
    dataset_id: str,
    lon: float,
    lat: float,
    start_utc: datetime,
    end_utc: datetime,
    radius_km: float,
    variables: tuple[str, ...],
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> SubsetRequest:
    bb = bbox_around(lon, lat, radius_km)
    return SubsetRequest(
        dataset_id=dataset_id,
        variables=variables,
        bbox=bb,
        start_utc=start_utc,
        end_utc=end_utc,
        depth_min=depth_min,
        depth_max=depth_max,
    )


# Convenience wrapper used by cli.py (new interface)
def cmems_subset(
    out_nc: str | Path,
    dataset_id: str,
    bbox: BBox,
    start_utc: datetime,
    end_utc: datetime,
    var_u: str,
    var_v: str,
    zmin: float | None,
    zmax: float | None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # Reuse existing out_nc if valid and not forced
    if out_nc.exists() and out_nc.stat().st_size > 0 and not force_download:
        try:
            b = nc_time_bounds(out_nc)
            if window_within_bounds(start_utc, end_utc, b) and nc_has_vars(out_nc, (var_u, var_v)):
                if verbose:
                    print(f"[CMEMS] reuse existing file: {out_nc} [{b.tmin}..{b.tmax}]")
                return out_nc
        except Exception:
            pass

    req = SubsetRequest(
        dataset_id=dataset_id,
        variables=(var_u, var_v),
        bbox=bbox,
        start_utc=start_utc,
        end_utc=end_utc,
        depth_min=zmin,
        depth_max=zmax,
    )

    downloaded = subset_to_netcdf(
        req,
        cache_dir=out_nc.parent,
        force_download=force_download,
        allow_time_shift=True,
        verbose=verbose,
    )

    if downloaded != out_nc:
        shutil.copy2(downloaded, out_nc)

    return out_nc
