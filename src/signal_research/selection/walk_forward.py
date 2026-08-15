"""Leakage-checked rolling train/validation/test splits."""

from __future__ import annotations

import pandas as pd

from signal_research.scoring.metrics import metrics_from_pnl


def rolling_splits(dates: pd.Index, train_days: int, validation_days: int, test_days: int, step_days: int) -> list[tuple[pd.Index, pd.Index, pd.Index]]:
    unique = pd.Index(dates).sort_values().unique()
    splits = []
    start = 0
    while start + train_days + validation_days + test_days <= len(unique):
        train = unique[start : start + train_days]
        validation = unique[start + train_days : start + train_days + validation_days]
        test = unique[start + train_days + validation_days : start + train_days + validation_days + test_days]
        assert train[-1] < validation[0] < test[0]
        splits.append((train, validation, test))
        start += step_days
    return splits


def run_walk_forward(
    returns: pd.DataFrame,
    names: list[str],
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    top_k: int,
) -> pd.DataFrame:
    dates = returns.index
    rows: list[dict[str, object]] = []
    for split_id, (train, validation, test) in enumerate(rolling_splits(dates, train_days, validation_days, test_days, step_days)):
        train_scores = {name: metrics_from_pnl(returns.loc[train, name])["sharpe"] for name in names}
        ranked = sorted(names, key=lambda name: (-train_scores[name], name))[:top_k]
        for name in ranked:
            train_end = train[-1]
            validation_start = validation[0]
            test_start = test[0]
            assert train_end < validation_start < test_start
            rows.append({
                "split": split_id,
                "name": name,
                "train_sharpe": train_scores[name],
                "validation_sharpe": metrics_from_pnl(returns.loc[validation, name])["sharpe"],
                "test_sharpe": metrics_from_pnl(returns.loc[test, name])["sharpe"],
                "train_end": str(train_end),
                "validation_start": str(validation_start),
                "test_start": str(test_start),
            })
    return pd.DataFrame(rows)
