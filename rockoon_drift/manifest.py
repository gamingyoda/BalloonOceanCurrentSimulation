# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass
class RunManifest:
    created_utc: str
    params: dict[str, Any]
    datasets: dict[str, Any]
    outputs: dict[str, str]

def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
