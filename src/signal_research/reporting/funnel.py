"""Funnel counts and rejection reasons."""

from __future__ import annotations

import pandas as pd


def funnel_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["stage", "count", "rejection_reason"])
