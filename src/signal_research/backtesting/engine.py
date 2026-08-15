"""Signal-to-return backtest engine with explicit next-day convention."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .transactions import linear_cost, one_way_turnover
from .weights import cross_sectional_weights


@dataclass
class BacktestResult:
    daily_pnl: pd.Series
    turnover: pd.Series
    weights: pd.Series
    signal: pd.Series
    forward_returns: pd.Series


def backtest_signal(
    signal: pd.Series,
    forward_returns: pd.Series,
    cost_bps: float = 5.0,
    quantile: float = 0.2,
    max_abs_weight: float = 0.05,
) -> BacktestResult:
    if not signal.index.equals(forward_returns.index):
        raise ValueError("signal and forward returns must share the exact index")
    weights = cross_sectional_weights(signal, quantile=quantile, max_abs_weight=max_abs_weight)
    turnover = one_way_turnover(weights)
    gross = (weights * forward_returns.fillna(0.0)).groupby(level="date").sum()
    daily_pnl = gross - linear_cost(turnover, cost_bps)
    daily_pnl.name = "daily_pnl"
    return BacktestResult(daily_pnl, turnover, weights, signal, forward_returns)
