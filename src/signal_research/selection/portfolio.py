"""Fitness-ranked selection with family caps and equal-volatility combination."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_portfolio(metrics: pd.DataFrame, top_k: int, family_caps: dict[str, int]) -> list[str]:
    ranked = metrics.sort_values(["fitness", "name"], ascending=[False, True], kind="mergesort")
    counts: dict[str, int] = {}
    selected: list[str] = []
    for row in ranked.itertuples(index=False):
        cap = family_caps.get(row.family, top_k)
        if counts.get(row.family, 0) >= cap:
            continue
        selected.append(row.name)
        counts[row.family] = counts.get(row.family, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def equal_volatility_combine(returns: pd.DataFrame, names: list[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float, name="combined_pnl")
    vol = returns[names].std(ddof=1).replace(0.0, np.nan)
    inverse = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inverse.sum() == 0:
        inverse[:] = 1.0
    weights = inverse / inverse.sum()
    return returns[names].mul(weights, axis=1).sum(axis=1).rename("combined_pnl")
