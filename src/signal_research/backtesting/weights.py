"""Cross-sectional long-short portfolio weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _waterfill(names: pd.Index, side_gross: float, cap: float) -> pd.Series:
    """Allocate a side gross target while redistributing any clipped mass."""
    allocation = pd.Series(0.0, index=names, dtype=float)
    if len(names) == 0:
        return allocation

    capacity = len(names) * cap
    if capacity < side_gross - 1e-15:
        # The requested gross is infeasible.  Saturate every selected name at
        # the cap; this is deterministic and makes the constraint honest.
        allocation.loc[names] = cap
        return allocation

    remaining = side_gross
    open_names = list(names)
    while open_names and remaining > 0:
        share = remaining / len(open_names)
        clipped = [name for name in open_names if share > cap]
        if not clipped:
            allocation.loc[open_names] = share
            break
        allocation.loc[clipped] = cap
        remaining -= len(clipped) * cap
        open_names = [name for name in open_names if name not in clipped]
    return allocation


def cross_sectional_weights(signal: pd.Series, quantile: float = 0.2, max_abs_weight: float = 0.05) -> pd.Series:
    """Build capped, dollar-neutral cross-sectional long-short weights.

    The normal convention is unit gross exposure: the long side sums to +0.5
    and the short side to -0.5.  Each side is allocated with iterative
    redistribute-and-clip (water-filling), so the final weights respect
    ``max_abs_weight``.  If a side has too few names for its target gross
    (``n_side * max_abs_weight < 0.5``), the target is infeasible; that side
    is deterministically equal-weighted at the cap and its gross is reduced.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between zero and one half")
    if max_abs_weight <= 0:
        raise ValueError("max_abs_weight must be positive")
    result = pd.Series(0.0, index=signal.index, name="weight")
    for _, cross_section in signal.groupby(level="date", sort=True):
        finite = cross_section.dropna()
        if len(finite) < 4:
            continue
        n = max(1, int(np.floor(len(finite) * quantile)))
        ordered = finite.sort_values(kind="mergesort")
        short_names = ordered.index[:n]
        long_names = ordered.index[-n:]
        raw = pd.Series(0.0, index=finite.index, dtype=float)
        raw.loc[long_names] = _waterfill(long_names, 0.5, max_abs_weight)
        raw.loc[short_names] = -_waterfill(short_names, 0.5, max_abs_weight)
        result.loc[raw.index] = raw
    return result
