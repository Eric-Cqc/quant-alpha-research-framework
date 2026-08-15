import numpy as np
import pandas as pd
import pytest

from signal_research.signal_generation.synthetic_data import generate_synthetic_panel
from signal_research.signal_generation.registry import FIELDS
from signal_research.signal_generation.textbook_signals import SIGNAL_FUNCTIONS, _wide, short_term_reversal
from signal_research.validation.static_checks import RuntimeSignalError, StaticCheckError, assert_truncation_invariant, check_signal_function, truncation_cutoffs


def unsafe_file(panel):
    return open("secret.txt").read()


def unsafe_socket(panel):
    return socket.socket().connect(("127.0.0.1", 80))


def unsafe_concat(panel):
    field = "cl" + "ose"
    return panel[field]


def unsafe_fstring(panel):
    return panel[f'{"cl"}ose']


def unsafe_field(panel):
    return panel["volume"]


def unsafe_wide_field(panel):
    close = _wide(panel, "close")
    volume = _wide(panel, "volume")
    return (close * volume).stack()


def unsafe_wide_field_via_variable(panel):
    field = "volume"
    return _wide(panel, field).stack()


def unsafe_wide_field_via_rebound_variable(panel):
    field = "close"
    field = "vol" + "ume"
    return _wide(panel, field).stack()


def unsafe_wide_field_via_default_parameter(panel, field="volume"):
    return _wide(panel, field).stack()


def unsafe_wide_field_via_plain_parameter(panel, field):
    return _wide(panel, field).stack()


def unsafe_whole_frame_mean(panel):
    return panel.mean(1).rename("signal")


def unsafe_whole_frame_shift(panel):
    return panel.shift(1).mean(1).rename("signal")


def unsafe_panel_alias(panel):
    frame = panel
    return frame.mean(1).rename("signal")


def unsafe_positional_only(panel, /):
    return panel.mean(1).rename("signal")


def unsafe_varargs(*args):
    return args[0].mean(1).rename("signal")


def unsafe_shadowed_wide(panel):
    def _wide(frame, field):
        return frame.mean(1)

    return _wide(panel, "close").rename("signal")


def test_static_check_rejects_import():
    def unsafe(panel):
        import os
        return panel["close"]

    with pytest.raises(StaticCheckError):
        check_signal_function(unsafe, {"close"})


def test_static_check_rejects_undeclared_field():
    def unsafe(panel):
        return panel["not_declared"]

    with pytest.raises(StaticCheckError):
        check_signal_function(unsafe, {"close"})


@pytest.mark.parametrize(
    "function",
    [
        unsafe_file,
        unsafe_socket,
        unsafe_concat,
        unsafe_fstring,
        unsafe_field,
        unsafe_wide_field,
        unsafe_wide_field_via_variable,
        unsafe_wide_field_via_rebound_variable,
        unsafe_wide_field_via_default_parameter,
        unsafe_wide_field_via_plain_parameter,
        unsafe_whole_frame_mean,
        unsafe_whole_frame_shift,
        unsafe_panel_alias,
        unsafe_positional_only,
        unsafe_varargs,
        unsafe_shadowed_wide,
    ],
)
def test_static_check_rejects_unsafe_or_dynamic_access(function):
    with pytest.raises(StaticCheckError):
        check_signal_function(function, {"close"})


def test_static_check_allows_the_five_textbook_families():
    for family, function in SIGNAL_FUNCTIONS.items():
        check_signal_function(function, set(FIELDS[family]))


def test_lookahead_truncation_invariance():
    panel = generate_synthetic_panel(n_tickers=6, n_days=45)

    def future_using(panel):
        return panel["close"].groupby(level="ticker").shift(-1)

    with pytest.raises(RuntimeSignalError):
        assert_truncation_invariant(future_using, panel, panel.index.get_level_values("date").unique()[20])
    assert_truncation_invariant(short_term_reversal, panel, panel.index.get_level_values("date").unique()[20])


def test_lookahead_check_uses_candidate_parameters():
    panel = generate_synthetic_panel(n_tickers=6, n_days=45)
    cutoff = panel.index.get_level_values("date").unique()[20]

    def default_safe_param_unsafe(panel, w=1):
        close = _wide(panel, "close")
        return close.shift(1 - w).stack().reindex(panel.index).rename("signal")

    assert_truncation_invariant(default_safe_param_unsafe, panel, cutoff)
    with pytest.raises(RuntimeSignalError):
        assert_truncation_invariant(default_safe_param_unsafe, panel, cutoff, {"w": 5})


def test_multiple_cutoffs_catch_single_cutoff_masking():
    panel = generate_synthetic_panel(n_tickers=6, n_days=60)
    cutoffs = truncation_cutoffs(panel)
    assert len(cutoffs) >= 4

    masked_cutoff = cutoffs[2]

    def masks_one_cutoff(panel):
        close = _wide(panel, "close")
        ahead = close.shift(-1)
        ahead.loc[ahead.index == masked_cutoff] = np.nan
        return ahead.stack().reindex(panel.index).rename("signal")

    caught = 0
    for cutoff in cutoffs:
        try:
            assert_truncation_invariant(masks_one_cutoff, panel, cutoff)
        except RuntimeSignalError:
            caught += 1
    # The masked cutoff slips through alone; every other cutoff catches it.
    assert caught == len(cutoffs) - 1
