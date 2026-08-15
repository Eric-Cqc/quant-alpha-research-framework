"""Candidate grid for the demonstration funnel."""

from __future__ import annotations

from .schema import SignalSpec


FIELDS = {
    "momentum_12_1": ("close",),
    "short_term_reversal": ("close",),
    "vol_scaled_mean_reversion": ("close",),
    "volume_price_trend": ("open", "high", "low", "close", "volume"),
    "trend_quality": ("close",),
}


def _spec(name: str, family: str, parameters: dict, rationale: str) -> SignalSpec:
    return SignalSpec(name, family, FIELDS[family], parameters, 1, rationale)


def build_candidate_grid() -> list[SignalSpec]:
    candidates: list[SignalSpec] = []
    for lookback in [63, 126, 252, 378, 504]:
        for skip in [5, 21, 42] if lookback < 504 else [5, 21]:
            candidates.append(_spec(f"momentum_{lookback}_{skip}", "momentum_12_1", {"skip": skip, "lookback": lookback}, "Test medium-horizon persistence."))
    for w in [2, 5, 10, 21, 42, 63]:
        candidates.append(_spec(f"reversal_{w}", "short_term_reversal", {"w": w}, "Test short-horizon reversal."))
    for w in [5, 10, 21, 42, 63, 84]:
        for v in [21, 42]:
            candidates.append(_spec(f"scaled_reversal_{w}_{v}", "vol_scaled_mean_reversion", {"w": w, "v": v}, "Scale reversal by recent dispersion."))
    for w in [5, 10, 20, 40, 80]:
        for volume_window in [10, 20]:
            candidates.append(_spec(f"volume_trend_{w}_{volume_window}", "volume_price_trend", {"w": w, "volume_window": volume_window}, "Combine signed candle movement and relative activity."))
    for w in [10, 21, 42, 63, 126, 189, 252, 315]:
        candidates.append(_spec(f"trend_quality_{w}", "trend_quality", {"w": w}, "Prefer stable directional log-price paths."))

    # These entries make schema validation and duplicate removal visible in the report.
    candidates.append(SignalSpec("invalid_missing_field", "short_term_reversal", ("close", "not_a_field"), {"w": 5}, 1, "Intentional validation fixture."))
    candidates.append(SignalSpec("invalid_horizon", "short_term_reversal", ("close",), {"w": 5}, 0, "Intentional validation fixture."))
    candidates.append(SignalSpec("invalid_parameter", "short_term_reversal", ("close",), {"w": -3}, 1, "Intentional validation fixture."))
    candidates.append(_spec("reversal_5_duplicate", "short_term_reversal", {"w": 5}, "Intentional exact duplicate fixture."))
    return candidates
