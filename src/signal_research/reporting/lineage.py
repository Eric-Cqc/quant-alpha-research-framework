"""Stable JSON lineage payloads."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def environment_metadata() -> dict[str, object]:
    """Return the interpreter and runtime dependency versions for lineage."""
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")}
    return {"python_version": platform.python_version(), "dependency_versions": dependencies}


def results_hash(frames: dict[str, object], lineage_core: dict[str, Any]) -> str:
    chunks: list[bytes] = []
    for name in sorted(frames):
        frame = frames[name]
        chunks.append(name.encode("utf-8") + b"\n" + frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8"))
    chunks.append(canonical_json_bytes(lineage_core))
    return sha256_bytes(b"\n".join(chunks))
