# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

import pandas as pd
import xarray as xr

def export_csvs(nc_path: str | Path, csv_mean: str | Path, csv_all: str | Path) -> None:
    ds = xr.open_dataset(nc_path)
    times = pd.to_datetime(ds["time"].values)

    lon_mean = ds["lon"].mean(dim="trajectory").values
    lat_mean = ds["lat"].mean(dim="trajectory").values
    pd.DataFrame({"time_utc": times, "lon": lon_mean, "lat": lat_mean}).to_csv(csv_mean, index=False)

    lon_df = ds["lon"].to_pandas()
    lat_df = ds["lat"].to_pandas()
    df_all = (
        lon_df.stack().rename("lon").to_frame()
        .join(lat_df.stack().rename("lat")).reset_index()
        .rename(columns={"level_0": "time_utc", "level_1": "particle_id"})
    )
    df_all.to_csv(csv_all, index=False)

def export_pretty_map(
    nc_path: str | Path,
    currents_nc: str | Path | None,
    png_path: str | Path,
    quiver_currents: bool = True,
    quiver_stride: int = 6,
    dpi: int = 220,
) -> None:
    """Nice-looking plot (requires matplotlib + cartopy)."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        print(f"[WARN] pretty map skipped (missing matplotlib/cartopy): {e}")
        return

    ds = xr.open_dataset(nc_path)
    lon = ds["lon"].values
    lat = ds["lat"].values
    times = pd.to_datetime(ds["time"].values)

    lon_min = float(np.nanmin(lon)); lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat)); lat_max = float(np.nanmax(lat))
    pad_lon = max(0.2, 0.1*(lon_max - lon_min + 1e-6))
    pad_lat = max(0.2, 0.1*(lat_max - lat_min + 1e-6))
    extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f1f1")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    for tr in range(lon.shape[1]):
        ax.plot(lon[:, tr], lat[:, tr], transform=proj, linewidth=0.6, alpha=0.25)

    lon_mean = np.nanmean(lon, axis=1)
    lat_mean = np.nanmean(lat, axis=1)
    ax.plot(lon_mean, lat_mean, transform=proj, linewidth=2.2)

    ax.plot(lon_mean[0],  lat_mean[0],  marker="*", markersize=10, transform=proj)
    ax.plot(lon_mean[-1], lat_mean[-1], marker="x", markersize=8, transform=proj)

    ax.text(0.01, 0.99, f"Start: {times[0].strftime('%Y-%m-%d %H:%MZ')}\n"
                        f"End  : {times[-1].strftime('%Y-%m-%d %H:%MZ')}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"), fontsize=9)

    if quiver_currents and currents_nc and Path(currents_nc).is_file():
        try:
            dc = xr.open_dataset(currents_nc)
            mid_t = times[len(times)//2]
            it = int(np.argmin(np.abs(pd.to_datetime(dc["time"].values) - mid_t)))

            for cand_u, cand_v in [
                ("uo","vo"),
                ("x_sea_water_velocity","y_sea_water_velocity"),
                ("eastward_sea_water_velocity","northward_sea_water_velocity"),
            ]:
                if (cand_u in dc.variables) and (cand_v in dc.variables):
                    U = dc[cand_u].isel(time=it)
                    V = dc[cand_v].isel(time=it)
                    break
            else:
                raise KeyError("No known current variables found in currents NetCDF")

            if "depth" in U.dims:
                U = U.isel(depth=0); V = V.isel(depth=0)

            xname = [d for d in U.dims if d.lower().startswith(("lon","x"))][0]
            yname = [d for d in U.dims if d.lower().startswith(("lat","y"))][0]
            lons = U[xname].values
            lats = U[yname].values
            uu = U.values; vv = V.values

            uu = uu[::quiver_stride, ::quiver_stride]
            vv = vv[::quiver_stride, ::quiver_stride]
            qlons = lons[::quiver_stride]
            qlats = lats[::quiver_stride]
            Q = ax.quiver(qlons, qlats, uu, vv, transform=proj, scale=10, width=0.002)
            ax.quiverkey(Q, X=0.83, Y=1.02, U=0.5, label="0.5 m/s", labelpos="E", coordinates="axes")
        except Exception as e:
            print(f"[WARN] currents quiver skipped: {e}")

    plt.tight_layout()
    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] pretty PNG: {png_path}")


def export_simple_png(nc_path: str | Path, png_path: str | Path, dpi: int = 180) -> None:
    """Lightweight PNG without cartopy (just lon-lat tracks)."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] simple png skipped (missing matplotlib): {e}")
        return

    ds = xr.open_dataset(nc_path)
    lon = ds["lon"].values  # (time, trajectory)
    lat = ds["lat"].values
    times = pd.to_datetime(ds["time"].values)

    # mean track
    lon_m = np.nanmean(lon, axis=1)
    lat_m = np.nanmean(lat, axis=1)

    plt.figure()
    # plot a subset of trajectories for speed
    ntraj = lon.shape[1]
    step = max(1, ntraj // 200)
    for j in range(0, ntraj, step):
        plt.plot(lon[:, j], lat[:, j], linewidth=0.5, alpha=0.25)

    plt.plot(lon_m, lat_m, linewidth=2.0, label="mean")
    plt.scatter([lon[0, 0]], [lat[0, 0]], marker="o", s=20, label="start")
    plt.scatter([lon[-1, 0]], [lat[-1, 0]], marker="x", s=30, label="end")
    plt.xlabel("lon [deg]")
    plt.ylabel("lat [deg]")
    plt.title(f"Drift tracks (UTC {times[0]} to {times[-1]})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=dpi)
    plt.close()
