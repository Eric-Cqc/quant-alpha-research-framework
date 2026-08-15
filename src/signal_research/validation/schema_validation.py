"""Typed validation for signal specifications."""

from __future__ import annotations

import re
from typing import Iterable

from signal_research.signal_generation.schema import SignalSpec


class SpecValidationError(ValueError):
    """Base class for invalid candidate specifications."""


class MissingFieldError(SpecValidationError):
    pass


class ParameterValidationError(SpecValidationError):
    pass


class HorizonValidationError(SpecValidationError):
    pass


def validate_signal_spec(spec: SignalSpec, available_fields: Iterable[str] = ("open", "high", "low", "close", "volume")) -> SignalSpec:
    fields = set(available_fields)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.name):
        raise SpecValidationError(f"invalid name: {spec.name!r}")
    if not spec.family:
        raise SpecValidationError("family is required")
    missing = sorted(set(spec.required_fields) - fields)
    if missing:
        raise MissingFieldError(f"undeclared required fields: {missing}")
    if spec.horizon_days < 1 or spec.horizon_days > 252:
        raise HorizonValidationError("horizon_days must be between 1 and 252")
    if not spec.parameters:
        raise ParameterValidationError("parameters cannot be empty")
    for key, value in spec.parameters.items():
        if not isinstance(key, str):
            raise ParameterValidationError("parameter names must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParameterValidationError(f"parameter {key} must be numeric")
        if value <= 0:
            raise ParameterValidationError(f"parameter {key} must be positive")
    if spec.family == "momentum_12_1" and not {"skip", "lookback"}.issubset(spec.parameters):
        raise ParameterValidationError("momentum requires skip and lookback")
    if spec.family == "vol_scaled_mean_reversion" and not {"w", "v"}.issubset(spec.parameters):
        raise ParameterValidationError("scaled reversal requires w and v")
    if spec.family == "volume_price_trend" and not {"w", "volume_window"}.issubset(spec.parameters):
        raise ParameterValidationError("volume trend requires w and volume_window")
    if spec.family in {"short_term_reversal", "trend_quality"} and "w" not in spec.parameters:
        raise ParameterValidationError(f"{spec.family} requires w")
    return spec
