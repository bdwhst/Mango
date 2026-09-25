"""Atomic JSON files (state.json, manifest.json, best.json) and phase seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_seed(a: int, b: int) -> int:
    """Deterministic 63-bit seed for a phase (not required to match the C++ deriveSeed)."""
    x = (a * 0x9E3779B97F4A7C15 + b * 0xC2B2AE3D27D4EB4F + 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    x = (x * 0x7FB5D329728EA185) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    return x & 0x7FFFFFFFFFFFFFFF
