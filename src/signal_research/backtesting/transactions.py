"""Turnover and linear transaction cost helpers."""

from __future__ import annotations

import pandas as pd


def one_way_turnover(weights: pd.Series) -> pd.Series:
    wide = weights.unstack("ticker").fillna(0.0).sort_index()
    changes = wide.diff().abs().sum(axis=1) / 2.0
    if len(changes):
        changes.iloc[0] = wide.iloc[0].abs().sum() / 2.0
    return changes.rename("turnover")


def linear_cost(turnover: pd.Series, cost_bps: float) -> pd.Series:
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    return turnover * cost_bps / 10_000.0
