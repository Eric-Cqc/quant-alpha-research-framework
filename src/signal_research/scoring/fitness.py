"""Configurable but fixed-weight composite fitness."""

from __future__ import annotations

import math


def composite_fitness(metrics: dict[str, float], weights: dict[str, float], turnover_floor: float = 1e-3) -> float:
    turnover = max(float(metrics.get("turnover", 0.0)), turnover_floor)
    ann_return = float(metrics.get("annualized_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    drawdown = float(metrics.get("max_drawdown", 0.0))
    return float(
        weights.get("sharpe", 1.0) * sharpe
        + weights.get("return", 0.0) * math.copysign(math.sqrt(abs(ann_return) / turnover), ann_return)
        - weights.get("turnover", 0.0) * turnover
        - weights.get("drawdown", 0.0) * drawdown
    )
