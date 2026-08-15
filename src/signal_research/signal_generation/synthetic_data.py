"""Deterministic synthetic OHLCV-like panel generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_panel(
    seed: int = 20260815,
    n_tickers: int = 100,
    n_days: int = 750,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    """Create a deterministic, market-like panel with no external data."""
    if n_tickers < 4 or n_days < 30:
        raise ValueError("n_tickers must be at least 4 and n_days at least 30")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    tickers = np.array([f"SYN{i:03d}" for i in range(n_tickers)])

    market = np.zeros(n_days)
    market_shock = rng.normal(0.0, 0.0060, n_days)
    for i in range(1, n_days):
        market[i] = 0.18 * market[i - 1] + market_shock[i]

    groups = rng.integers(0, 8, size=n_tickers)
    group_factors = np.zeros((n_days, 8))
    group_shocks = rng.normal(0.0, 0.0038, (n_days, 8))
    for t in range(1, n_days):
        group_factors[t] = 0.12 * group_factors[t - 1] + group_shocks[t]

    beta_market = rng.normal(0.45, 0.12, n_tickers)
    beta_group = rng.normal(0.30, 0.14, n_tickers)
    drift = rng.normal(0.0, 0.00012, n_tickers)
    idio_vol = rng.uniform(0.018, 0.042, n_tickers)
    liquidity = rng.normal(0.0, 0.45, n_tickers)
    idio_shocks = rng.normal(0.0, idio_vol[None, :], (n_days, n_tickers))
    idio_returns = np.zeros_like(idio_shocks)
    idio_returns[0] = idio_shocks[0]
    for t in range(1, n_days):
        idio_returns[t] = -0.015 * idio_returns[t - 1] + idio_shocks[t]
    returns = (
        drift[None, :]
        + market[:, None] * beta_market[None, :]
        + group_factors[:, groups] * beta_group[None, :]
        + idio_returns
    )
    returns = np.clip(returns, -0.25, 0.25)
    close = 50.0 * np.exp(np.cumsum(returns, axis=0))
    overnight = rng.normal(0.0, 0.0045, (n_days, n_tickers))
    previous_close = np.vstack([close[0][None, :], close[:-1]])
    open_price = previous_close * np.exp(overnight)
    intraday_move = np.abs(returns - overnight)
    envelope = np.maximum(0.002, 0.65 * intraday_move + rng.uniform(0.001, 0.008, (n_days, n_tickers)))
    high = np.maximum(open_price, close) * (1.0 + envelope)
    low = np.minimum(open_price, close) * (1.0 - envelope)
    volume = np.exp(
        13.0
        + liquidity[None, :]
        + 2.0 * np.abs(returns)
        + rng.normal(0.0, 0.30, (n_days, n_tickers))
    )

    arrays = [np.repeat(dates.to_numpy(), n_tickers), np.tile(tickers, n_days)]
    index = pd.MultiIndex.from_arrays(arrays, names=["date", "ticker"])
    values = np.column_stack([open_price.ravel(), high.ravel(), low.ravel(), close.ravel(), volume.ravel()])
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close", "volume"])
