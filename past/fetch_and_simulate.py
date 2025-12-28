#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# source ~/venvs/opendrift/bin/activate

"""
fetch_and_simulate.py

落下点（lon/lat/UTC）を引数で受け取り、
Copernicus Marine (CMEMS) から
  - 海流（uo/vo）
  - 風（eastward_wind/northward_wind）
  - 波のストークスドリフト（VSDX/VSDY）
を自動ダウンロードして OpenDrift (OceanDrift) に投入。
風受け率アンサンブルで漂流を実行し、NetCDF/PNG/CSVを出力します。

前提:
  - Python 3.10+（WSL Ubuntu 推奨）
  - venv 有効化後に必要パッケージを pip install 済み
  - copernicusmarine login 済み（初回のみ）

インストールの目安:
  sudo apt update && sudo apt install -y \
    libproj-dev proj-data proj-bin libgeos-dev libnetcdf-dev libudunits2-dev \
    gdal-bin libgdal-dev
  pip install -U pip setuptools wheel
  pip install "opendrift==1.14.3" xarray netcdf4 pandas matplotlib cmocean cartopy coloredlogs geojson copernicus-marine-client
  # roaring-landmask エラー時: pip install roaring-landmask
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
import math
import xarray as xr
import pandas as pd

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.readers import reader_constant

import re
import numpy as np

def nc_time_bounds(path: str):
    """NetCDF の time 次元の最小/最大（tz-naive UTC）を返す。"""
    ds = xr.open_dataset(path)
    if "time" not in ds:
        raise ValueError(f"{path}: time 次元が見つかりません")
    t = pd.to_datetime(ds["time"].values)
    return t.min().to_pydatetime().replace(tzinfo=None), t.max().to_pydatetime().replace(tzinfo=None)

def window_within_bounds(start: datetime, end: datetime, bmin: datetime, bmax: datetime) -> bool:
    return (start >= bmin) and (end <= bmax)

def parse_dataset_bounds_from_error(errtext: str):
    """
    copernicusmarine subset のエラー文
    'exceed the dataset coordinates [YYYY-.., YYYY-..]'
    をパースして (start,end) を返す。
    """
    if not errtext:
        return None, None
    m = re.search(r"dataset coordinates \[([0-9\-:\+T ]+),\s*([0-9\-:\+T ]+)\]", errtext)
    if not m:
        return None, None
    s = pd.to_datetime(m.group(1)).to_pydatetime().replace(tzinfo=None)
    e = pd.to_datetime(m.group(2)).to_pydatetime().replace(tzinfo=None)
    return s, e


# ======== 既定の CMEMS データセット ID / 変数名 ========
# 海流（GLOBAL_ANALYSISFORECAST_PHY_001_024 の 1/12°, 1-hourly サブセット）
CURR_DATASET_DEFAULT = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
CURR_UVAR = "uo"
CURR_VVAR = "vo"

# 風（WIND_GLO_PHY_L4_NRT_012_004 の 0.125°, 1-hourly サブセット）
WIND_DATASET_DEFAULT = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
WIND_UVAR = "eastward_wind"
WIND_VVAR = "northward_wind"

# 波（GLOBAL_ANALYSISFORECAST_WAV_001_027 の 0.083°, 3-hourly サブセット）
# 3-hourly (PT3H-i) を既定とし、必要なら PT1H-i を --waves-id で指定
WAVES_DATASET_DEFAULT = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
WAVES_STOKES_U = "VSDX"  # CF: sea_surface_wave_stokes_drift_x_velocity
WAVES_STOKES_V = "VSDY"  # CF: sea_surface_wave_stokes_drift_y_velocity


# ========= ユーティリティ =========
def deg_buffer_from_km(lat_deg: float, radius_km: float):
    """km の半径を緯度経度度数へ近似変換。経度は緯度によって縮む。"""
    lat_per_km = 1.0 / 111.32
    lon_per_km = 1.0 / (111.32 * max(0.1, math.cos(math.radians(lat_deg))))
    return radius_km * lon_per_km, radius_km * lat_per_km


def cmems_subset(
    out_nc: str,
    dataset_id: str,
    lon: float,
    lat: float,
    start_utc: datetime,
    end_utc: datetime,
    radius_km: float,
    var_u: str,
    var_v: str,
    zmin: float | None = None,
    zmax: float | None = None,
    force_download: bool = False,
):
    """
    仕様:
      1) 既存ファイルが要求時間をカバー → それを使う
      2) カバーしない/壊れてる → 最新を取り直し (--force-download)
      3) それでも 'dataset coordinates を超える' エラー → データセットが提供する上限に
         自動で [start,end] を切り詰めて再試行（共通範囲が無ければ例外）
    """
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)

    # 1) 既存キャッシュが要求時間をカバーするか？
    if os.path.isfile(out_nc) and os.path.getsize(out_nc) > 0 and not force_download:
        try:
            bmin, bmax = nc_time_bounds(out_nc)
            if window_within_bounds(start_utc, end_utc, bmin, bmax):
                print(f"[CMEMS] cache covers request: {out_nc} [{bmin}..{bmax}]")
                return out_nc
            else:
                print(f"[CMEMS] cache stale: {out_nc} [{bmin}..{bmax}] -> refresh")
                force_download = True
        except Exception as e:
            print(f"[WARN] cache read failed: {e} -> refresh")
            force_download = True

    # 2) 取得を実行
    def _try_subset(t0: datetime, t1: datetime):
        dlon, dlat = deg_buffer_from_km(lat, radius_km)
        x0, x1 = lon - dlon, lon + dlon
        y0, y1 = lat - dlat, lat + dlat
        cmd = [
            "copernicusmarine", "subset",
            "--dataset-id", dataset_id,
            "--variable", var_u, "--variable", var_v,
            "--start-datetime", t0.replace(second=0, microsecond=0).isoformat(),
            "--end-datetime",   t1.replace(second=0, microsecond=0).isoformat(),
            "--minimum-longitude", f"{x0:.6f}", "--maximum-longitude", f"{x1:.6f}",
            "--minimum-latitude",  f"{y0:.6f}", "--maximum-latitude",  f"{y1:.6f}",
            "--output-directory", os.path.dirname(out_nc),
            "--output-filename",  os.path.basename(out_nc),
        ]
        if force_download:
            cmd.append("--force-download")
        if zmin is not None and zmax is not None:
            cmd += ["--minimum-depth", str(zmin), "--maximum-depth", str(zmax)]
        print("[CMEMS] subset:", " ".join(cmd))
        # 重要: 出力を捕捉してエラーメッセージを解析する
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    try:
        _try_subset(start_utc, end_utc)
    except subprocess.CalledProcessError as e:
        # 既存: データセット範囲の抽出
        errtxt = (e.stderr or "") + "\n" + (e.stdout or "")
        ds_s, ds_e = parse_dataset_bounds_from_error(errtxt)

        if ds_s and ds_e:
            req_s, req_e = start_utc, end_utc
            req_dur = req_e - req_s

            # まずは通常のトリミングで重なりを試す
            adj_s = max(req_s, ds_s)
            adj_e = min(req_e, ds_e)

            if adj_e > adj_s:
                print(f"[CMEMS] retry within dataset range: {adj_s} .. {adj_e}")
                force_download = True
                _try_subset(adj_s, adj_e)

            else:
                # 依然オーバーラップ無し → 自動シフト
                # ケースA: 要求窓が将来側（データ最終時刻より後）
                if req_s > ds_e:
                    adj_e = ds_e
                    adj_s = max(ds_s, ds_e - req_dur)
                    print(f"[CMEMS] requested > dataset end -> shift back: {adj_s} .. {adj_e}")
                    force_download = True
                    _try_subset(adj_s, adj_e)

                # ケースB: 要求窓が過去側（データ開始時刻より前）
                elif req_e < ds_s:
                    adj_s = ds_s
                    adj_e = min(ds_e, ds_s + req_dur)
                    print(f"[CMEMS] requested < dataset start -> shift forward: {adj_s} .. {adj_e}")
                    force_download = True
                    _try_subset(adj_s, adj_e)

                else:
                    # ありえないが保険
                    raise RuntimeError(
                        f"No overlap with dataset time range even after shift: "
                        f"requested [{req_s}..{req_e}] vs dataset [{ds_s}..{ds_e}]"
                    )
        else:
            # 他のエラーはそのまま投げる
            raise


    if not os.path.isfile(out_nc):
        raise RuntimeError("subset 出力が見つかりませんでした。")

    return out_nc



def build_model(
    curr_nc_path: str,
    wind_nc_path: str | None,
    waves_nc_path: str | None,
    const_wind_u: float,
    const_wind_v: float,
    dt_out_minutes: int,
):
    """
    OceanDrift モデルを構築し、海流・風・波（Stokes）を add_reader。
    風は NetCDF 優先、なければ定数へフォールバック。
    """
    o = OceanDrift(loglevel=20)
    readers = []

    # Currents（必須）
    readers.append(Reader(curr_nc_path))

    # Wind（任意：ファイル優先、失敗時は定数風）
    if wind_nc_path and os.path.isfile(wind_nc_path):
        try:
            readers.append(Reader(wind_nc_path))
        except Exception as e:
            print(f"[WARN] wind Reader 失敗: {e} → 定数風にフォールバック")
            readers.append(reader_constant.Reader({"x_wind": const_wind_u, "y_wind": const_wind_v}))
    else:
        readers.append(reader_constant.Reader({"x_wind": const_wind_u, "y_wind": const_wind_v}))

    # Waves / Stokes drift（任意：あれば追加）
    if waves_nc_path and os.path.isfile(waves_nc_path):
        try:
            readers.append(Reader(waves_nc_path))
        except Exception as e:
            print(f"[WARN] waves Reader 失敗: {e} → Stokes drift なしで継続")

    o.add_reader(readers)
    o.set_config("general:coastline_action", "stranding")
    o.set_config("general:time_step_output_minutes", float(dt_out_minutes))
    return o


def export_png(o: OceanDrift, png_path: str):
    """
    図の出力。まずは海岸線背景で試み、失敗時は背景なしで再試行。
    """
    try:
        o.plot(filename=png_path, background=["coastline"], buffer=1.5, fast=True)
        print(f"[OK] PNG: {png_path}")
    except Exception as e:
        print(f"[WARN] 背景 coastlines 失敗: {e} → 背景なしで再試行")
        o.plot(filename=png_path, background=None, buffer=1.5, fast=True)
        print(f"[OK] PNG (no background): {png_path}")


def export_csvs(nc_path: str, csv_mean: str, csv_all: str):
    """
    NetCDF → 10分ごとの平均位置CSV / 全粒子CSV を出力。
    """
    ds = xr.open_dataset(nc_path)
    times = pd.to_datetime(ds["time"].values)

    lon_mean = ds["lon"].mean(dim="trajectory").values
    lat_mean = ds["lat"].mean(dim="trajectory").values
    pd.DataFrame({"time_utc": times, "lon": lon_mean, "lat": lat_mean}).to_csv(csv_mean, index=False)
    print(f"[OK] CSV mean: {csv_mean}")

    lon_df = ds["lon"].to_pandas()
    lat_df = ds["lat"].to_pandas()
    df_all = (
        lon_df.stack().rename("lon").to_frame()
        .join(lat_df.stack().rename("lat")).reset_index()
        .rename(columns={"level_0": "time_utc", "level_1": "particle_id"})
    )
    df_all.to_csv(csv_all, index=False)
    print(f"[OK] CSV all: {csv_all}")

def export_pretty_map(nc_path: str, curr_nc_path: str | None, png_path: str,
                      quiver_currents: bool = True, quiver_stride: int = 6,
                      dpi: int = 220):
    """
    読みやすいPNGを出力:
      - 背景: 陸域/海岸線/国境 + 緯度経度グリッド(ラベル)
      - 粒子: 全軌跡(半透明) + 平均軌跡(太線)
      - マーカー: 開始(★) と 最終時刻(×)
      - ベクトル: その時刻の海流ベクトル(間引き)
    """
    ds = xr.open_dataset(nc_path)

    # 粒子位置（time, trajectory）
    lon = ds["lon"].values
    lat = ds["lat"].values
    times = pd.to_datetime(ds["time"].values)

    # 描画範囲（余白つき）
    lon_min = float(np.nanmin(lon)); lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat)); lat_max = float(np.nanmax(lat))
    pad_lon = max(0.2, 0.1*(lon_max - lon_min + 1e-6))
    pad_lat = max(0.2, 0.1*(lat_max - lat_min + 1e-6))
    extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)

    # 背景
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f1f1")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 全軌跡（薄め）
    for tr in range(lon.shape[1]):
        ax.plot(lon[:, tr], lat[:, tr], transform=proj, linewidth=0.6, alpha=0.25)

    # 平均軌跡（太め）
    lon_mean = np.nanmean(lon, axis=1)
    lat_mean = np.nanmean(lat, axis=1)
    ax.plot(lon_mean, lat_mean, transform=proj, linewidth=2.2)

    # 開始/終了マーカー（平均位置ベース）
    ax.plot(lon_mean[0],  lat_mean[0],  marker="*", markersize=10, transform=proj)
    ax.plot(lon_mean[-1], lat_mean[-1], marker="x", markersize=8, transform=proj)

    # 凡例テキスト
    ax.text(0.01, 0.99, f"Start: {times[0].strftime('%Y-%m-%d %H:%MZ')}\n"
                        f"End  : {times[-1].strftime('%Y-%m-%d %H:%MZ')}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"), fontsize=9)

    # 海流ベクトル（任意）
    if quiver_currents and curr_nc_path and os.path.isfile(curr_nc_path):
        try:
            dc = xr.open_dataset(curr_nc_path)
            # 粒子の中央時刻に最も近いスライス
            mid_t = times[len(times)//2]
            it = int(np.argmin(np.abs(pd.to_datetime(dc["time"].values) - mid_t)))
            # 代表時刻 it の表層 (z最上位or2D) を取得
            # よくある変数名: uo, vo or x/y_sea_water_velocity
            for cand_u, cand_v in [("uo","vo"),
                                   ("x_sea_water_velocity","y_sea_water_velocity"),
                                   ("eastward_sea_water_velocity","northward_sea_water_velocity")]:
                if (cand_u in dc.variables) and (cand_v in dc.variables):
                    U = dc[cand_u].isel(time=it)
                    V = dc[cand_v].isel(time=it)
                    break
            # 次元整形（zがあれば表層を取る）
            if "depth" in U.dims:
                U = U.isel(depth=0); V = V.isel(depth=0)
            # 座標名の正規化
            xname = [d for d in U.dims if d.lower().startswith(("lon","x"))][0]
            yname = [d for d in U.dims if d.lower().startswith(("lat","y"))][0]
            lons = U[xname].values
            lats = U[yname].values
            uu = U.values; vv = V.values
            # 間引き
            uu = uu[::quiver_stride, ::quiver_stride]
            vv = vv[::quiver_stride, ::quiver_stride]
            qlons = lons[::quiver_stride]
            qlats = lats[::quiver_stride]
            Q = ax.quiver(qlons, qlats, uu, vv, transform=proj, scale=10, width=0.002)
            ax.quiverkey(Q, X=0.83, Y=1.02, U=0.5, label="0.5 m/s", labelpos="E", coordinates='axes')
        except Exception as e:
            print(f"[WARN] currents quiver skipped: {e}")

    plt.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] pretty PNG: {png_path}")


# ========= メイン =========
def main():
    ap = argparse.ArgumentParser(
        description="CMEMS (海流+風+波Stokes) 自動取得 → OpenDrift OceanDrift（気球の海上漂流）"
    )
    ap.add_argument("--lon", type=float, required=True, help="落下経度(東経+, 例 135.0)")
    ap.add_argument("--lat", type=float, required=True, help="落下緯度(北緯+, 例 33.5)")
    ap.add_argument("--time-utc", type=str, required=True, help="落下時刻(UTC, 例 2025-09-07T10:30:00 または 'now')")
    ap.add_argument("--hours", type=int, default=6, help="シミュレーション時間[h] (既定:6)")
    ap.add_argument("--dt", type=int, default=600, help="計算刻み[秒] (既定:600)")
    ap.add_argument("--dt-out", type=int, default=600, help="出力刻み[秒] (既定:600)")
    ap.add_argument("--radius-km", type=float, default=120.0, help="CMEMS 取得バッファ半径[km] (既定:120)")
    ap.add_argument("--pretty-png", action="store_true",help="見やすい地図レイアウトのPNGも出力する")


    # データセット切替用
    ap.add_argument("--curr-id", type=str, default=CURR_DATASET_DEFAULT, help="海流 dataset-id")
    ap.add_argument("--wind-id", type=str, default=WIND_DATASET_DEFAULT, help="風 dataset-id")
    ap.add_argument("--waves-id", type=str, default=WAVES_DATASET_DEFAULT, help="波(Stokes) dataset-id")

    # 表層海流の深さレンジ（0–2 m を既定）
    ap.add_argument("--zmin", type=float, default=0.0, help="海流の最小深さ[m] (既定:0)")
    ap.add_argument("--zmax", type=float, default=2.0, help="海流の最大深さ[m] (既定:2)")

    # 風受け率アンサンブル
    ap.add_argument("--windage", type=str, default="0.5,1,2,3,4", help="風受け率%%のカンマ区切り (例 '0.5,1,2,3,4')")
    ap.add_argument("--n-per", type=int, default=200, help="各メンバーの粒子数 (既定:200)")

    # 風のフォールバック（定数風）
    ap.add_argument("--const-wind-u", type=float, default=6.0, help="定数風u[m/s]（風データ無い時）")
    ap.add_argument("--const-wind-v", type=float, default=0.0, help="定数風v[m/s]（風データ無い時）")

    # 既存ファイルを直接指定したい場合（任意）
    ap.add_argument("--curr-nc", type=str, default="", help="既存の海流 NetCDF（任意）")
    ap.add_argument("--wind-nc", type=str, default="", help="既存の風 NetCDF（任意）")
    ap.add_argument("--waves-nc", type=str, default="", help="既存の波(Stokes) NetCDF（任意）")

    ap.add_argument("--outdir", type=str, default="./outputs", help="出力ディレクトリ (既定: ./outputs)")
    ap.add_argument("--force-download", action="store_true", help="既存ファイルがあっても再ダウンロードする")

    args = ap.parse_args()

    # UTC tz-naive に統一（'now' 対応）
    if args.time_utc.lower() == "now":
        t0 = datetime.utcnow().replace(second=0, microsecond=0)
    else:
        t0 = datetime.fromisoformat(args.time_utc).replace(second=0, microsecond=0)

    duration = timedelta(hours=args.hours)
    # 解析の安全マージンとして前後 +6h
    t_start = t0 - timedelta(hours=6)
    t_end = t0 + duration + timedelta(hours=6)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("./copernicus-data", exist_ok=True)

    # --- 海流 ---
    if args.curr_nc:
        curr_nc = args.curr_nc
        print(f"[INFO] currents (user): {curr_nc}")
    else:
        curr_nc = os.path.join("./copernicus-data", "cmems_currents_auto.nc")
        cmems_subset(
            out_nc=curr_nc,
            dataset_id=args.curr_id,
            lon=args.lon,
            lat=args.lat,
            start_utc=t_start,
            end_utc=t_end,
            radius_km=args.radius_km,
            var_u=CURR_UVAR,
            var_v=CURR_VVAR,
            zmin=args.zmin,
            zmax=args.zmax,
            force_download=args.force_download,
        )

    # --- 風 ---
    if args.wind_nc:
        wind_nc = args.wind_nc
        print(f"[INFO] wind (user): {wind_nc}")
    else:
        wind_nc = os.path.join("./copernicus-data", "cmems_wind_auto.nc")
        cmems_subset(
            out_nc=wind_nc,
            dataset_id=args.wind_id,
            lon=args.lon,
            lat=args.lat,
            start_utc=t_start,
            end_utc=t_end,
            radius_km=args.radius_km,
            var_u=WIND_UVAR,
            var_v=WIND_VVAR,
            force_download=args.force_download,
        )

    # --- 波（Stokes drift）---
    if args.waves_nc:
        waves_nc = args.waves_nc
        print(f"[INFO] waves (user): {waves_nc}")
    else:
        waves_nc = os.path.join("./copernicus-data", "cmems_waves_stokes_auto.nc")
        cmems_subset(
            out_nc=waves_nc,
            dataset_id=args.waves_id,
            lon=args.lon,
            lat=args.lat,
            start_utc=t_start,
            end_utc=t_end,
            radius_km=args.radius_km,
            var_u=WAVES_STOKES_U,
            var_v=WAVES_STOKES_V,
            force_download=args.force_download,
        )

    # --- 3データの time 範囲を取得（データセット側の純粋な共通範囲を先に計算）---
    cmin, cmax = nc_time_bounds(curr_nc)
    wmin, wmax = nc_time_bounds(wind_nc)
    smin, smax = nc_time_bounds(waves_nc)

    ds_common_start = max(cmin, wmin, smin)
    ds_common_end   = min(cmax, wmax, smax)

    if ds_common_end <= ds_common_start:
        raise RuntimeError(
            "海流・風・波のNetCDF間で共通の時間範囲がありません。\n"
            f"currents:[{cmin}..{cmax}]  wind:[{wmin}..{wmax}]  waves:[{smin}..{smax}]"
        )

    # 希望窓（t0..t0+duration）が共通範囲に入らない場合は自動スライド
    requested_start = t0
    requested_end   = t0 + duration
    ds_span = ds_common_end - ds_common_start

    if requested_end <= ds_common_start or requested_start >= ds_common_end:
        # 完全に外れている：共通範囲の中で「できるだけ新しい窓」を選ぶ
        if duration <= ds_span:
            # 希望の長さを満たせる → 後ろ詰め（最新側優先）
            new_start = ds_common_end - duration
            new_end   = ds_common_end
        else:
            # 共通範囲が短い → 取りうる最大長に短縮
            new_start = ds_common_start
            new_end   = ds_common_end
        print("[INFO] requested window is outside dataset range -> auto slide")
        print(f"       requested=[{requested_start}..{requested_end}]")
        print(f"       dataset  =[{ds_common_start}..{ds_common_end}]")
        print(f"       adjusted =[{new_start}..{new_end}]")
        t0 = new_start
        duration = new_end - new_start
    else:
        # 一部でも重なっている：共通範囲内に切り詰め
        new_start = max(requested_start, ds_common_start)
        new_end   = min(requested_end, ds_common_end)
        if (new_end - new_start) < duration:
            print("[INFO] requested window partially exceeds dataset -> trim to dataset")
            print(f"       requested=[{requested_start}..{requested_end}]")
            print(f"       dataset  =[{ds_common_start}..{ds_common_end}]")
            print(f"       adjusted =[{new_start}..{new_end}]")
        t0 = new_start
        duration = new_end - new_start

    # 念のため、ゼロ長や負値を防止
    if duration.total_seconds() <= 0:
        raise RuntimeError(
            "有効なシミュレーション時間が確保できませんでした。"
            f" dataset common=[{ds_common_start}..{ds_common_end}]"
        )

    # --- モデル構築 & シード ---
    o = build_model(
        curr_nc_path=curr_nc,
        wind_nc_path=wind_nc,
        waves_nc_path=waves_nc,
        const_wind_u=args.const_wind_u,
        const_wind_v=args.const_wind_v,
        dt_out_minutes=int(args.dt_out / 60),
    )

    for pct_str in args.windage.split(","):
        pct = float(pct_str.strip())
        o.seed_elements(
            args.lon,
            args.lat,
            time=t0,
            number=args.n_per,
            wind_drift_factor=pct / 100.0,
            z=0,
        )

    # --- 実行 & 出力 ---
    out_nc = os.path.join(args.outdir, "balloon_drift.nc")
    o.run(duration=duration, time_step=args.dt, time_step_output=args.dt_out, outfile=out_nc)
    print(f"[OK] NetCDF: {out_nc}")

    # 既存の簡易PNG
    png_path = os.path.join(args.outdir, "balloon_drift.png")
    export_png(o, png_path)

    # 見やすい版（任意）
    if args.pretty_png:
        pretty_path = os.path.join(args.outdir, "balloon_drift_pretty.png")
        export_pretty_map(out_nc, curr_nc, pretty_path, quiver_currents=True, quiver_stride=6, dpi=220)


    csv_mean = os.path.join(args.outdir, "balloon_positions_mean_10min.csv")
    csv_all = os.path.join(args.outdir, "balloon_positions_all_10min.csv")
    export_csvs(out_nc, csv_mean, csv_all)

    print("[DONE] すべて完了しました。")


if __name__ == "__main__":
    main()
