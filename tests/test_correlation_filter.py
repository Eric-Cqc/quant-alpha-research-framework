import pandas as pd

from signal_research.selection.correlation import greedy_correlation_filter


def test_correlation_filter_is_score_ordered_and_deterministic():
    returns = pd.DataFrame({"b": [1.0, 2.0, 3.0], "a": [1.0, 2.0, 3.0], "c": [3.0, 1.0, 2.0]})
    selected, rejected = greedy_correlation_filter(returns, {"a": 1.0, "b": 1.0, "c": 0.5}, 0.9)
    assert selected == ["a", "c"]
    assert "b" in rejected
