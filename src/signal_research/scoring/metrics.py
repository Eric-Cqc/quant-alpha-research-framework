"""Performance, concentration, and information metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(pnl: pd.Series) -> float:
    wealth = (1.0 + pnl.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(-drawdown.min()) if len(drawdown) else float("nan")


def metrics_for_backtest(result) -> dict[str, float]:
    pnl = result.daily_pnl.dropna()
    weights = result.weights.unstack("ticker").fillna(0.0)
    active_dates = weights.abs().sum(axis=1).loc[lambda values: values > 1e-12].index
    observations = int(pnl.reindex(active_dates).dropna().size)
    mean = float(pnl.mean()) if len(pnl) else float("nan")
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan")
    annualized_return = mean * 252.0
    annualized_volatility = std * np.sqrt(252.0) if np.isfinite(std) else float("nan")
    sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0
    signal = result.signal
    forward = result.forward_returns
    ics: list[float] = []
    for _, values in signal.groupby(level="date", sort=True):
        returns = forward.loc[values.index]
        valid = values.notna() & returns.notna()
        if int(valid.sum()) >= 3:
            ics.append(float(values[valid].rank().corr(returns[valid].rank())))
    ic_series = pd.Series(ics, dtype=float).dropna()
    mean_ic = float(ic_series.mean()) if len(ic_series) else 0.0
    ic_ir = float(mean_ic / ic_series.std(ddof=1) * np.sqrt(252.0)) if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else 0.0
    max_weight = float(weights.abs().max().max()) if not weights.empty else 0.0
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(pnl),
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "turnover": float(result.turnover.mean()) if len(result.turnover) else 0.0,
        "max_abs_weight": max_weight,
        "observations": observations,
        "active_observations": observations,
        "pnl_std": std,
    }


def metrics_from_pnl(pnl: pd.Series) -> dict[str, float]:
    pnl = pnl.dropna()
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan")
    mean = float(pnl.mean()) if len(pnl) else float("nan")
    return {
        "annualized_return": mean * 252.0,
        "annualized_volatility": std * np.sqrt(252.0) if np.isfinite(std) else float("nan"),
        "sharpe": mean / std * np.sqrt(252.0) if std > 0 else 0.0,
        "max_drawdown": _max_drawdown(pnl),
        "observations": int(len(pnl)),
    }
