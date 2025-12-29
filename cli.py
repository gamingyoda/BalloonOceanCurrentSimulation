#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rockoon sea drift simulator: CMEMS -> OpenDrift (OceanDrift)

1) Download CMEMS subsets (currents + optional wind + optional waves/Stokes)
2) Validate NetCDF time coverage and required variables
3) Run OpenDrift ensemble with windage
4) Export NetCDF + CSV + (optional) pretty PNG
"""

from __future__ import annotations
import argparse
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rockoon_drift.constants import (
    CURR_DATASET_DEFAULT, CURR_UVAR_DEFAULT, CURR_VVAR_DEFAULT,
    WIND_DATASET_DEFAULT, WIND_UVAR_DEFAULT, WIND_VVAR_DEFAULT,
    WAVES_DATASET_DEFAULT, WAVES_UVAR_DEFAULT, WAVES_VVAR_DEFAULT,
)
from rockoon_drift.timeutil import parse_time_utc, Window, window_with_margin, intersect_windows, adjust_requested_to_available
from rockoon_drift.netcdf_utils import nc_time_bounds
from rockoon_drift.cmems_cli import make_request, subset_to_netcdf
from rockoon_drift.ecmwf_cli import download_ecmwf_wind_netcdf
from rockoon_drift.opendrift_runner import DriftRunConfig, run_drift
from rockoon_drift.outputs import export_csvs, export_pretty_map
from rockoon_drift.manifest import RunManifest, write_manifest
from rockoon_drift.geo import bbox_around

def _parse_windage_list(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CMEMS → OpenDrift drift simulation after sea landing")
    ap.add_argument("--lon", type=float, required=False, help="start longitude (deg)")
    ap.add_argument("--lat", type=float, required=False, help="start latitude (deg)")
    ap.add_argument("--time-utc", type=str, required=False, help="e.g. 2025-09-07T10:30:00 or 'now'")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--dt", type=int, default=600)
    ap.add_argument("--dt-out", type=int, default=600)
    ap.add_argument("--radius-km", type=float, default=120.0)

    ap.add_argument("--curr-id", type=str, default=CURR_DATASET_DEFAULT)
    ap.add_argument("--curr-u", type=str, default=CURR_UVAR_DEFAULT)
    ap.add_argument("--curr-v", type=str, default=CURR_VVAR_DEFAULT)

    ap.add_argument(
        "--wind-source",
        type=str,
        choices=["auto", "cmems", "ecmwf"],
        default="auto",
        help="wind data source (auto=try CMEMS then fallback to ECMWF)",
    )
    ap.add_argument("--wind-id", type=str, default=WIND_DATASET_DEFAULT)
    ap.add_argument("--wind-u", type=str, default=WIND_UVAR_DEFAULT)
    ap.add_argument("--wind-v", type=str, default=WIND_VVAR_DEFAULT)
    ap.add_argument("--const-wind-only", action="store_true", help="skip wind download; use constant wind only")

    ap.add_argument("--waves-id", type=str, default=WAVES_DATASET_DEFAULT)
    ap.add_argument("--waves-u", type=str, default=WAVES_UVAR_DEFAULT)
    ap.add_argument("--waves-v", type=str, default=WAVES_VVAR_DEFAULT)
    ap.add_argument("--const-waves-only", action="store_true", help="skip waves download; use constant Stokes drift only")

    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=2.0)

    ap.add_argument("--windage", type=str, default="0.5,1,2,3,4")
    ap.add_argument("--n-per", type=int, default=200)

    ap.add_argument("--const-wind-u", type=float, default=6.0)
    ap.add_argument("--const-wind-v", type=float, default=0.0)
    ap.add_argument("--const-waves-u", type=float, default=0.0, help="eastward Stokes drift (m/s) when using const waves")
    ap.add_argument("--const-waves-v", type=float, default=0.0, help="northward Stokes drift (m/s) when using const waves")

    ap.add_argument("--cache-dir", type=str, default="./copernicus-cache")
    ap.add_argument("--outdir", type=str, default="./outputs")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--pretty-png", action="store_true")
    ap.add_argument("--no-waves", action="store_true")
    ap.add_argument("--no-wind", action="store_true")

    ap.add_argument("--web", action="store_true", help="start simple web UI instead of running CLI once")
    ap.add_argument("--web-host", type=str, default="127.0.0.1", help="web UI listen host")
    ap.add_argument("--web-port", type=int, default=8000, help="web UI listen port")
    ap.add_argument("--open-browser", action="store_true", help="open browser to the web UI on start")
    return ap


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    if args.lon is None or args.lat is None or args.time_utc is None:
        raise ValueError("lon, lat, time-utc are required for simulation")

    t0 = parse_time_utc(args.time_utc)
    requested = Window(t0, t0 + timedelta(hours=args.hours))
    dl_window = window_with_margin(center=t0, duration_hours=args.hours, margin_hours=6.0)
    bbox = bbox_around(args.lon, args.lat, args.radius_km)

    cache_dir = Path(args.cache_dir)
    outdir = Path(args.outdir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    curr_req = make_request(
        dataset_id=args.curr_id,
        lon=args.lon,
        lat=args.lat,
        start_utc=dl_window.start,
        end_utc=dl_window.end,
        radius_km=args.radius_km,
        variables=(args.curr_u, args.curr_v),
        depth_min=args.zmin,
        depth_max=args.zmax,
    )
    currents_nc = subset_to_netcdf(curr_req, cache_dir=cache_dir, force_download=args.force_download)

    wind_nc = None
    wind_source_used = None
    if args.const_wind_only:
        print("[INFO] wind download skipped (const wind only mode)")
    elif not args.no_wind:
        def _download_wind_cmems() -> Path:
            wind_req = make_request(
                dataset_id=args.wind_id,
                lon=args.lon,
                lat=args.lat,
                start_utc=dl_window.start,
                end_utc=dl_window.end,
                radius_km=args.radius_km,
                variables=(args.wind_u, args.wind_v),
            )
            return subset_to_netcdf(wind_req, cache_dir=cache_dir, force_download=args.force_download)

        def _download_wind_ecmwf() -> Path:
            return download_ecmwf_wind_netcdf(
                bbox=bbox,
                start_utc=dl_window.start,
                end_utc=dl_window.end,
                out_dir=cache_dir,
                force_download=args.force_download,
            )

        if args.wind_source == "ecmwf":
            wind_nc = _download_wind_ecmwf()
            wind_source_used = "ecmwf"
        elif args.wind_source == "cmems":
            try:
                wind_nc = _download_wind_cmems()
                wind_source_used = "cmems"
            except Exception as e:
                raise RuntimeError(
                    "wind download failed and fallback to constant wind is disabled. "
                    "Use --const-wind-only or --no-wind if you want to proceed without wind data."
                ) from e
        else:
            try:
                wind_nc = _download_wind_cmems()
                wind_source_used = "cmems"
            except Exception as e_cmems:
                print("[INFO] CMEMS wind failed; falling back to ECMWF (auto mode)")
                try:
                    wind_nc = _download_wind_ecmwf()
                    wind_source_used = "ecmwf"
                except Exception as e_ecmwf:
                    raise RuntimeError(
                        "wind download failed in auto mode (CMEMS then ECMWF). "
                        "Use --const-wind-only or --no-wind if you want to proceed without wind data."
                    ) from e_ecmwf

    waves_nc = None
    if args.const_waves_only:
        print("[INFO] waves download skipped (const waves only mode)")
    elif not args.no_waves:
        waves_req = make_request(
            dataset_id=args.waves_id,
            lon=args.lon,
            lat=args.lat,
            start_utc=dl_window.start,
            end_utc=dl_window.end,
            radius_km=args.radius_km,
            variables=(args.waves_u, args.waves_v),
        )
        try:
            waves_nc = subset_to_netcdf(waves_req, cache_dir=cache_dir, force_download=args.force_download)
        except Exception as e:
            raise RuntimeError(
                "waves download failed (so Stokes drift cannot be used). "
                "If you want to run without waves, pass --no-waves. "
                f"Root cause: {e}"
            ) from e

    def _build_bounds(curr_nc, wind_path, waves_path):
        b_list = []
        meta = {
            "currents": {"path": str(curr_nc), "time": None},
            "wind": {"path": str(wind_path) if wind_path else None, "time": None},
            "waves": {"path": str(waves_path) if waves_path else None, "time": None},
        }

        c_b = nc_time_bounds(curr_nc); b_list.append(("currents", c_b))
        meta["currents"]["time"] = {"tmin": str(c_b.tmin), "tmax": str(c_b.tmax)}
        if wind_path:
            w_b = nc_time_bounds(wind_path); b_list.append(("wind", w_b))
            meta["wind"]["time"] = {"tmin": str(w_b.tmin), "tmax": str(w_b.tmax)}
        if waves_path:
            s_b = nc_time_bounds(waves_path); b_list.append(("waves", s_b))
            meta["waves"]["time"] = {"tmin": str(s_b.tmin), "tmax": str(s_b.tmax)}
        return b_list, meta

    def _common(active_bounds):
        return intersect_windows([Window(b.tmin, b.tmax) for _, b in active_bounds]) if active_bounds else None

    bounds, ds_meta = _build_bounds(currents_nc, wind_nc, waves_nc)
    common = _common(bounds)

    if common is None and args.wind_source == "auto" and wind_source_used == "cmems" and (not args.no_wind) and (not args.const_wind_only):
        print("[INFO] No common time window; retry wind with ECMWF (auto fallback)")
        wind_nc = download_ecmwf_wind_netcdf(
            bbox=bbox,
            start_utc=dl_window.start,
            end_utc=dl_window.end,
            out_dir=cache_dir,
            force_download=args.force_download,
        )
        wind_source_used = "ecmwf"
        bounds, ds_meta = _build_bounds(currents_nc, wind_nc, waves_nc)
        common = _common(bounds)

    if common is None:
        detail = " | ".join([f"{name}=[{b.tmin}..{b.tmax}]" for name, b in bounds])
        raise RuntimeError(
            "No common time intersection among datasets and fallback is disabled. "
            "Adjust your requested time/window or disable a data source with --no-wind/--no-waves/--const-wind-only.\n"
            f"Bounds: {detail}"
        )

    sim_window = adjust_requested_to_available(requested, common)
    if sim_window != requested:
        print("[INFO] requested window adjusted to available dataset range")
        print(f"       requested=[{requested.start}..{requested.end}]")
        print(f"       available=[{common.start}..{common.end}]")
        print(f"       adjusted =[ {sim_window.start}..{sim_window.end} ]")

    include_wind = not args.no_wind
    include_waves = (not args.no_waves) and (not args.const_waves_only) and (waves_nc is not None)

    if args.const_waves_only:
        print("[INFO] using constant Stokes drift (waves disabled in readers)")

    cfg = DriftRunConfig(
        lon=args.lon,
        lat=args.lat,
        start_utc=sim_window.start,
        duration=sim_window.duration,
        dt_seconds=args.dt,
        dt_out_seconds=args.dt_out,
        windage_percents=_parse_windage_list(args.windage),
        n_per_member=args.n_per,
        include_wind=include_wind,
        include_waves=include_waves,
        const_wind_u=args.const_wind_u,
        const_wind_v=args.const_wind_v,
    )

    out_nc = outdir / "balloon_drift.nc"
    result_nc = run_drift(cfg, currents_nc=currents_nc, wind_nc=wind_nc, waves_nc=waves_nc, out_nc=out_nc)

    csv_mean = outdir / "balloon_positions_mean.csv"
    csv_all = outdir / "balloon_positions_all.csv"
    export_csvs(result_nc, csv_mean, csv_all)
    print(f"[OK] CSV mean: {csv_mean}")
    print(f"[OK] CSV all : {csv_all}")

    if args.pretty_png:
        pretty = outdir / "balloon_drift_pretty.png"
        export_pretty_map(result_nc, currents_nc, pretty)

    manifest = RunManifest(
        created_utc=datetime.utcnow().isoformat(timespec="seconds"),
        params={
            "lon": args.lon,
            "lat": args.lat,
            "time_utc": args.time_utc,
            "requested_hours": args.hours,
            "sim_start_utc": sim_window.start.isoformat(),
            "sim_end_utc": sim_window.end.isoformat(),
            "dt": args.dt,
            "dt_out": args.dt_out,
            "windage": cfg.windage_percents,
            "n_per": cfg.n_per_member,
            "radius_km": args.radius_km,
            "wind_source_requested": args.wind_source,
            "wind_source_used": wind_source_used,
        },
        datasets=ds_meta,
        outputs={"netcdf": str(result_nc), "csv_mean": str(csv_mean), "csv_all": str(csv_all)},
    )
    manifest_path = outdir / "run_manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[OK] manifest: {manifest_path}")
    print("[DONE]")

    return {
        "netcdf": str(result_nc),
        "csv_mean": str(csv_mean),
        "csv_all": str(csv_all),
        "manifest": str(manifest_path),
        "sim_window": {
            "start": sim_window.start.isoformat(),
            "end": sim_window.end.isoformat(),
            "duration_hours": sim_window.duration.total_seconds() / 3600.0,
        },
        "datasets": ds_meta,
        "wind_source_used": wind_source_used,
    }


def start_web(args: argparse.Namespace) -> None:
    from webapp import create_app

    app = create_app(args)
    url = f"http://{args.web_host}:{args.web_port}"
    print(f"[WEB] starting UI at {url}")
    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            print("[WARN] could not open browser; start manually")
    app.run(host=args.web_host, port=args.web_port, debug=False)


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.web:
        start_web(args)
        return

    # デフォルトはWeb UIを起動する。CLIとして使う場合のみlon/lat/timeを指定する。
    if args.lon is None or args.lat is None or args.time_utc is None:
        print("[INFO] 引数が指定されていないためWeb UIを起動します。CLIで実行する場合は --lon --lat --time-utc を指定してください。")
        start_web(args)
        return

    result = run_simulation(args)
    print(f"[RESULT] NetCDF : {result['netcdf']}")
    print(f"[RESULT] CSV mean: {result['csv_mean']}")
    print(f"[RESULT] CSV all : {result['csv_all']}")


if __name__ == "__main__":
    main()
