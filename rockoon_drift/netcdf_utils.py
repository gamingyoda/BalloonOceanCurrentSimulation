# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

@dataclass(frozen=True)
class NcBounds:
    tmin: datetime
    tmax: datetime

def nc_time_bounds(path: str | Path) -> NcBounds:
    ds = xr.open_dataset(path)
    if "time" not in ds:
        raise ValueError(f"{path}: 'time' coordinate not found")
    t = pd.to_datetime(ds["time"].values)
    return NcBounds(
        t.min().to_pydatetime().replace(tzinfo=None),
        t.max().to_pydatetime().replace(tzinfo=None),
    )

def nc_has_vars(path: str | Path, vars_needed: Iterable[str]) -> bool:
    ds = xr.open_dataset(path)
    for v in vars_needed:
        if v not in ds.variables:
            return False
    return True

def window_within_bounds(start: datetime, end: datetime, bounds: NcBounds) -> bool:
    return (start >= bounds.tmin) and (end <= bounds.tmax)
