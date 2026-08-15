"""Canonical signatures and exact duplicate reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from signal_research.signal_generation.schema import SignalSpec


@dataclass(frozen=True)
class DedupRecord:
    name: str
    signature: str
    kept: bool
    duplicate_of: str | None = None


def canonical_signature(spec: SignalSpec) -> str:
    payload = {"family": spec.family, "parameters": dict(sorted(spec.parameters.items()))}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def remove_exact_duplicates(specs: list[SignalSpec]) -> tuple[list[SignalSpec], list[DedupRecord]]:
    seen: dict[str, SignalSpec] = {}
    kept: list[SignalSpec] = []
    report: list[DedupRecord] = []
    for spec in specs:
        signature = canonical_signature(spec)
        previous = seen.get(signature)
        if previous is None:
            seen[signature] = spec
            kept.append(spec)
            report.append(DedupRecord(spec.name, signature, True))
        else:
            report.append(DedupRecord(spec.name, signature, False, previous.name))
    return kept, report
