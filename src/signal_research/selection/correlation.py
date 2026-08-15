"""Greedy deterministic return-correlation filtering."""

from __future__ import annotations

import pandas as pd


def greedy_correlation_filter(
    returns: pd.DataFrame,
    scores: dict[str, float],
    threshold: float = 0.8,
) -> tuple[list[str], dict[str, str]]:
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for name in ordered:
        if not accepted:
            accepted.append(name)
            continue
        correlations = returns[[name] + accepted].corr()[name].drop(name).abs()
        if bool((correlations > threshold).any()):
            rejected[name] = "pairwise return correlation exceeds threshold"
        else:
            accepted.append(name)
    return accepted, rejected
