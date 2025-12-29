# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta

ISO_HINT = "YYYY-MM-DDTHH:MM:SS (UTC, tz-naive) or 'now'"

def parse_time_utc(s: str) -> datetime:
    """Parse UTC time (tz-naive). Accepts 'now' or ISO string."""
    if s.lower() == "now":
        return datetime.utcnow().replace(second=0, microsecond=0)
    # allow trailing 'Z'
    s2 = s.rstrip("Z")
    t = datetime.fromisoformat(s2)
    return t.replace(tzinfo=None, second=0, microsecond=0)

@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

def window_with_margin(center: datetime, duration_hours: float, margin_hours: float) -> Window:
    duration = timedelta(hours=duration_hours)
    start = center - timedelta(hours=margin_hours)
    end = center + duration + timedelta(hours=margin_hours)
    return Window(start, end)

def intersect_windows(windows: list[Window]) -> Window | None:
    if not windows:
        return None
    s = max(w.start for w in windows)
    e = min(w.end for w in windows)
    if e <= s:
        return None
    return Window(s, e)

def adjust_requested_to_available(requested: Window, available: Window) -> Window:
    """
    Ensure requested lies within available.

    - If requested is fully outside: slide it to the newest possible window.
    - If partially outside: trim to fit.
    """
    req_s, req_e = requested.start, requested.end
    av_s, av_e = available.start, available.end

    if req_e <= av_s or req_s >= av_e:
        dur = requested.duration
        span = available.duration
        if dur <= span:
            new_end = av_e
            new_start = av_e - dur
        else:
            new_start, new_end = av_s, av_e
        return Window(new_start, new_end)

    return Window(max(req_s, av_s), min(req_e, av_e))
