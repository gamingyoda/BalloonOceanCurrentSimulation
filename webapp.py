"""Lightweight Flask UI for running the drift simulator and visualizing results on a map."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file

from cli import build_arg_parser, run_simulation
from rockoon_drift.outputs import nc_to_geojson


UI_DIR = Path(__file__).resolve().parent / "web_ui"
INDEX_HTML_PATH = UI_DIR / "index.html"


def _fresh_args(base_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(base_args))


def _apply_payload(args: argparse.Namespace, payload: dict[str, Any]) -> argparse.Namespace:
    def _to_bool(val: Any) -> bool:
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "on"}
        return bool(val)

    float_fields = ["lon", "lat", "hours", "radius_km", "const_wind_u", "const_wind_v", "const_waves_u", "const_waves_v"]
    int_fields = ["dt", "dt_out", "n_per"]
    bool_fields = ["pretty_png", "force_download", "no_wind", "no_waves", "const_wind_only", "const_waves_only"]
    str_fields = [
        "time_utc",
        "windage",
        "wind_source",
        "curr_id",
        "curr_u",
        "curr_v",
        "wind_id",
        "wind_u",
        "wind_v",
        "waves_id",
        "waves_u",
        "waves_v",
        "cache_dir",
        "outdir",
    ]

    for field in float_fields:
        if field in payload and payload[field] is not None:
            setattr(args, field, float(payload[field]))
    for field in int_fields:
        if field in payload and payload[field] is not None:
            setattr(args, field, int(payload[field]))
    for field in bool_fields:
        if field in payload and payload[field] is not None:
            setattr(args, field, _to_bool(payload[field]))
    for field in str_fields:
        if field in payload and payload[field] is not None:
            setattr(args, field, str(payload[field]))

    return args


def create_app(base_args: argparse.Namespace | None = None) -> Flask:
    defaults = _fresh_args(base_args) if base_args else build_arg_parser().parse_args([])
    app = Flask(__name__)

    @app.get("/")
    def index():
        if INDEX_HTML_PATH.exists():
            return send_file(INDEX_HTML_PATH, mimetype="text/html")
        return "index.html not found", 500

    @app.post("/api/simulate")
    def api_simulate():
        payload = request.get_json(force=True, silent=True) or {}
        args = _apply_payload(_fresh_args(defaults), payload)
        try:
            result = run_simulation(args)
            geojson = nc_to_geojson(Path(result["netcdf"]))
            return jsonify({"status": "ok", "geojson": geojson, "result": result, "wind_source_used": result.get("wind_source_used")})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 400

    return app


__all__ = ["create_app"]
