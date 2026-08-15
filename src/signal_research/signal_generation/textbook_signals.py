"""Five public, textbook signal families implemented with pandas idioms."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel[field].unstack("ticker").sort_index()


def momentum_12_1(panel: pd.DataFrame, skip: int = 21, lookback: int = 252) -> pd.Series:
    """Log price change from `lookback` days ago to `skip` days ago."""
    close = _wide(panel, "close")
    result = np.log(close.shift(skip) / close.shift(skip + lookback))
    return result.stack().reindex(panel.index).rename("signal")


def short_term_reversal(panel: pd.DataFrame, w: int = 5) -> pd.Series:
    """Negative recent log return."""
    close = _wide(panel, "close")
    result = -np.log(close / close.shift(w))
    return result.stack().reindex(panel.index).rename("signal")


def vol_scaled_mean_reversion(panel: pd.DataFrame, w: int = 10, v: int = 21) -> pd.Series:
    """Recent reversal divided by trailing daily-return volatility."""
    close = _wide(panel, "close")
    log_return = np.log(close / close.shift(1))
    result = -np.log(close / close.shift(w)) / (log_return.rolling(v, min_periods=v).std() + EPS)
    return result.stack().reindex(panel.index).rename("signal")


def volume_price_trend(panel: pd.DataFrame, w: int = 20, volume_window: int = 20) -> pd.Series:
    """Rolling signed candle position weighted by relative volume."""
    close = _wide(panel, "close")
    open_price = _wide(panel, "open")
    high = _wide(panel, "high")
    low = _wide(panel, "low")
    volume = _wide(panel, "volume")
    candle = (close - open_price) / (high - low + EPS)
    relative_volume = np.log1p(volume / volume.rolling(volume_window, min_periods=volume_window).median())
    result = (candle * relative_volume).rolling(w, min_periods=w).mean()
    return result.stack().reindex(panel.index).rename("signal")


def _trend_quality_array(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan
    x = np.arange(values.size, dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denominator = float(np.sum(x_centered * x_centered))
    slope = float(np.sum(x_centered * y_centered) / denominator)
    fitted = values.mean() + slope * x_centered
    residual = values - fitted
    total = float(np.sum(y_centered * y_centered))
    if total <= EPS:
        return 0.0
    r_squared = max(0.0, 1.0 - float(np.sum(residual * residual)) / total)
    residual_vol = float(np.sqrt(np.sum(residual * residual) / max(1, values.size - 2)))
    return slope * r_squared / (residual_vol + EPS)


def trend_quality(panel: pd.DataFrame, w: int = 42) -> pd.Series:
    """Rolling log-price slope weighted by fit quality and residual scale."""
    close = _wide(panel, "close")
    result = np.log(close).rolling(w, min_periods=w).apply(_trend_quality_array, raw=True)
    return result.stack().reindex(panel.index).rename("signal")


SIGNAL_FUNCTIONS = {
    "momentum_12_1": momentum_12_1,
    "short_term_reversal": short_term_reversal,
    "vol_scaled_mean_reversion": vol_scaled_mean_reversion,
    "volume_price_trend": volume_price_trend,
    "trend_quality": trend_quality,
}
