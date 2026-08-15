import hashlib

import pandas as pd

from signal_research.signal_generation.synthetic_data import generate_synthetic_panel
from signal_research.signal_generation.textbook_signals import momentum_12_1


def test_same_seed_has_identical_panel_and_signal():
    first = generate_synthetic_panel(seed=7, n_tickers=8, n_days=80)
    second = generate_synthetic_panel(seed=7, n_tickers=8, n_days=80)
    assert hashlib.sha256(first.to_csv().encode()).hexdigest() == hashlib.sha256(second.to_csv().encode()).hexdigest()
    pd.testing.assert_series_equal(momentum_12_1(first, skip=5, lookback=20), momentum_12_1(second, skip=5, lookback=20))
