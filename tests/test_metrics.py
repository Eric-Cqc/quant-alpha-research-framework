import numpy as np
import pandas as pd

from signal_research.backtesting.engine import BacktestResult
from signal_research.risk.gates import apply_gates
from signal_research.scoring.metrics import metrics_for_backtest


def test_metrics_hand_fixture():
    dates = pd.bdate_range("2020-01-01", periods=3)
    tickers = ["SYN000", "SYN001"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    pnl = pd.Series([0.01, -0.005, 0.02], index=dates)
    signal = pd.Series([1.0, -1.0] * 3, index=index)
    forward = pd.Series([0.0] * len(index), index=index)
    weights = signal * 0.0
    turnover = pd.Series([0.1, 0.2, 0.3], index=dates)
    result = BacktestResult(pnl, turnover, weights, signal, forward)
    metrics = metrics_for_backtest(result)
    expected_std = np.std([0.01, -0.005, 0.02], ddof=1)
    assert metrics["annualized_return"] == 0.025 * 252 / 3
    assert np.isclose(metrics["sharpe"], np.mean([0.01, -0.005, 0.02]) / expected_std * np.sqrt(252))
    assert np.isclose(metrics["max_drawdown"], 1 - (1.01 * 0.995) / 1.01)
    assert np.isclose(metrics["turnover"], 0.2)
    assert metrics["observations"] == 0


def test_observations_count_active_weight_dates():
    dates = pd.bdate_range("2020-01-01", periods=3)
    tickers = ["SYN000", "SYN001"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    pnl = pd.Series([0.0, 0.0, 0.01], index=dates)
    signal = pd.Series([1.0, -1.0] * 3, index=index)
    forward = pd.Series([0.0] * len(index), index=index)
    weights = pd.Series([0.0, 0.0, 0.0, 0.0, 0.5, -0.5], index=index)
    turnover = pd.Series([0.0, 0.0, 0.5], index=dates)
    result = BacktestResult(pnl, turnover, weights, signal, forward)
    assert metrics_for_backtest(result)["observations"] == 1


def test_gates_reject_inactive_degenerate_candidate_and_use_active_limit():
    metrics = pd.DataFrame(
        [{"name": "inactive", "sharpe": 1.0, "turnover": 0.0, "max_drawdown": 0.0, "max_abs_weight": 0.0, "observations": 5, "pnl_std": 0.0}]
    )
    gated, reasons = apply_gates(
        metrics,
        {"min_sharpe": 0.0, "max_turnover": 1.0, "max_drawdown": 1.0, "max_abs_weight": 0.1, "min_active_observations": 10},
    )
    assert not bool(gated.loc[0, "gate_pass"])
    assert reasons["inactive"] == ["minimum active observations", "inactive or degenerate"]
