"""TOML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    n_tickers: int
    n_days: int
    start: str
    quantile: float
    cost_bps: float
    max_abs_weight: float
    correlation_threshold: float
    top_k: int
    family_caps: dict[str, int]
    gate_limits: dict[str, float]
    walk_forward: dict[str, int]
    fitness_weights: dict[str, float]
    turnover_floor: float


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing [{section}].{key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    general = raw.get("general", {})
    backtest = raw.get("backtest", {})
    selection = raw.get("selection", {})
    gates = raw.get("gates", {})
    walk = raw.get("walk_forward", {})
    fitness = raw.get("fitness", {})
    quantile = float(_required(backtest, "quantile", "backtest"))
    cost_bps = float(_required(backtest, "cost_bps", "backtest"))
    n_days = int(_required(general, "n_days", "general"))
    windows = {key: int(_required(walk, key, "walk_forward")) for key in ("train_days", "validation_days", "test_days", "step_days")}
    if not 0 < quantile < 0.5:
        raise ConfigError("backtest.quantile must be between 0 and 0.5")
    if cost_bps < 0:
        raise ConfigError("backtest.cost_bps cannot be negative")
    if int(_required(general, "n_tickers", "general")) < 4 or n_days < 30:
        raise ConfigError("general dimensions are too small")
    if any(value <= 0 for value in windows.values()) or windows["step_days"] > windows["test_days"] + windows["validation_days"]:
        raise ConfigError("walk-forward windows must be positive and step must be practical")
    if windows["train_days"] + windows["validation_days"] + windows["test_days"] > n_days:
        raise ConfigError("walk-forward windows do not fit the configured panel")
    correlation_threshold = float(_required(selection, "correlation_threshold", "selection"))
    if not 0 <= correlation_threshold <= 1:
        raise ConfigError("selection.correlation_threshold must be in [0, 1]")
    top_k = int(_required(selection, "top_k", "selection"))
    if top_k < 1:
        raise ConfigError("selection.top_k must be positive")
    limits = {key: float(value) for key, value in gates.items()}
    required_limits = {"min_sharpe", "max_turnover", "max_drawdown", "max_abs_weight"}
    if not required_limits.issubset(limits) or not ({"min_observations", "min_active_observations"} & limits.keys()):
        raise ConfigError("all gate limits are required")
    return ExperimentConfig(
        seed=int(_required(general, "seed", "general")),
        n_tickers=int(_required(general, "n_tickers", "general")),
        n_days=n_days,
        start=str(_required(general, "start", "general")),
        quantile=quantile,
        cost_bps=cost_bps,
        max_abs_weight=float(_required(backtest, "max_abs_weight", "backtest")),
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        family_caps={str(k): int(v) for k, v in selection.get("family_caps", {}).items()},
        gate_limits=limits,
        walk_forward=windows,
        fitness_weights={str(k): float(v) for k, v in fitness.items() if k != "turnover_floor"},
        turnover_floor=float(fitness.get("turnover_floor", 1e-3)),
    )
