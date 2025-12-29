# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class BBox:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

def deg_buffer_from_km(lat_deg: float, radius_km: float) -> tuple[float, float]:
    """Convert km radius to lon/lat degree buffers (rough)."""
    lat_per_km = 1.0 / 111.32
    lon_per_km = 1.0 / (111.32 * max(0.1, math.cos(math.radians(lat_deg))))
    return radius_km * lon_per_km, radius_km * lat_per_km

def bbox_around(lon: float, lat: float, radius_km: float) -> BBox:
    dlon, dlat = deg_buffer_from_km(lat, radius_km)
    return BBox(lon - dlon, lon + dlon, lat - dlat, lat + dlat)
