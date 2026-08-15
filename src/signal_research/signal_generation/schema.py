"""Immutable description of a candidate signal."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    required_fields: tuple[str, ...]
    parameters: Mapping[str, Any]
    horizon_days: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "required_fields": list(self.required_fields),
            "parameters": dict(self.parameters),
            "horizon_days": self.horizon_days,
            "rationale": self.rationale,
        }
