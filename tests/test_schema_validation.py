import pytest

from signal_research.signal_generation.schema import SignalSpec
from signal_research.validation.schema_validation import HorizonValidationError, MissingFieldError, validate_signal_spec


def test_valid_spec_and_typed_failures():
    spec = SignalSpec("demo", "short_term_reversal", ("close",), {"w": 5}, 1, "test")
    assert validate_signal_spec(spec) == spec
    with pytest.raises(MissingFieldError):
        validate_signal_spec(SignalSpec("bad", "short_term_reversal", ("secret_field",), {"w": 5}, 1, "test"))
    with pytest.raises(HorizonValidationError):
        validate_signal_spec(SignalSpec("bad", "short_term_reversal", ("close",), {"w": 5}, 0, "test"))
