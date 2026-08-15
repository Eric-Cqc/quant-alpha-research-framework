"""Transparent, configurable candidate risk gates."""

from __future__ import annotations

import math

import pandas as pd


def apply_gates(metrics: pd.DataFrame, limits: dict[str, float]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    reasons: dict[str, list[str]] = {}
    passed: list[bool] = []
    min_active_observations = limits.get("min_active_observations", limits.get("min_observations"))
    if min_active_observations is None:
        raise KeyError("min_active_observations")
    for row in metrics.itertuples(index=False):
        failures: list[str] = []
        if row.sharpe < limits["min_sharpe"]:
            failures.append("minimum Sharpe")
        if row.turnover > limits["max_turnover"]:
            failures.append("turnover limit")
        if row.max_drawdown > limits["max_drawdown"]:
            failures.append("drawdown limit")
        if row.max_abs_weight > limits["max_abs_weight"] + 1e-12:
            failures.append("weight cap")
        active_observations = getattr(row, "active_observations", row.observations)
        if active_observations < min_active_observations:
            failures.append("minimum active observations")
        pnl_std = getattr(row, "pnl_std", float("nan"))
        if not math.isfinite(pnl_std):
            annualized_volatility = getattr(row, "annualized_volatility", float("nan"))
            pnl_std = annualized_volatility / math.sqrt(252.0) if math.isfinite(annualized_volatility) else float("nan")
        if row.turnover <= 1e-12 and (not math.isfinite(pnl_std) or abs(pnl_std) <= 1e-12):
            failures.append("inactive or degenerate")
        reasons[row.name] = failures
        passed.append(not failures)
    result = metrics.copy()
    result["gate_pass"] = passed
    result["gate_reasons"] = ["; ".join(reasons[row.name]) for row in metrics.itertuples(index=False)]
    return result, reasons
