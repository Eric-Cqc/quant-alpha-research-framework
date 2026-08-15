from pathlib import Path

import pytest

from signal_research.config import ConfigError, load_config


def test_invalid_config_rejects_bad_quantile(tmp_path: Path):
    text = """
[general]
seed = 1
n_tickers = 10
n_days = 200
start = "2020-01-02"
[backtest]
quantile = 0.7
cost_bps = 5
max_abs_weight = 0.05
[selection]
correlation_threshold = 0.8
top_k = 3
[gates]
min_sharpe = 0
max_turnover = 1
max_drawdown = 0.5
max_abs_weight = 0.05
min_observations = 10
[walk_forward]
train_days = 100
validation_days = 40
test_days = 40
step_days = 40
[fitness]
sharpe = 1
"""
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)
