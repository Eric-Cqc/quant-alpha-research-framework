import numpy as np
import pandas as pd

from signal_research.backtesting.weights import cross_sectional_weights


def _signal(n_tickers: int) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-02"]), [f"SYN{i:03d}" for i in range(n_tickers)]],
        names=["date", "ticker"],
    )
    return pd.Series(np.arange(n_tickers, dtype=float), index=index)


def test_weight_cap_and_side_gross_are_enforced_when_feasible():
    weights = cross_sectional_weights(_signal(100), quantile=0.2, max_abs_weight=0.05)
    long = weights[weights > 0]
    short = weights[weights < 0]
    assert float(weights.abs().max()) <= 0.05 + 1e-12
    assert np.isclose(long.sum(), 0.5)
    assert np.isclose(short.sum(), -0.5)
    assert np.isclose(weights.sum(), 0.0)


def test_infeasible_side_cap_is_deterministic_and_stays_neutral():
    signal = _signal(4)
    first = cross_sectional_weights(signal, quantile=0.2, max_abs_weight=0.05)
    second = cross_sectional_weights(signal, quantile=0.2, max_abs_weight=0.05)
    pd.testing.assert_series_equal(first, second)
    assert float(first.abs().max()) <= 0.05 + 1e-12
    assert np.isclose(first[first > 0].sum(), 0.05)
    assert np.isclose(first[first < 0].sum(), -0.05)
    assert np.isclose(first.sum(), 0.0)
