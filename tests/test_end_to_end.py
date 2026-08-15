import json
from pathlib import Path

from signal_research.config import ExperimentConfig
from signal_research.reporting.report import run_experiment


def _small_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=12,
        n_tickers=12,
        n_days=220,
        start="2020-01-02",
        quantile=0.2,
        cost_bps=2.0,
        max_abs_weight=0.08,
        correlation_threshold=0.95,
        top_k=3,
        family_caps={},
        gate_limits={"min_sharpe": -1.0, "max_turnover": 2.0, "max_drawdown": 1.0, "max_abs_weight": 0.081, "min_observations": 50},
        walk_forward={"train_days": 100, "validation_days": 40, "test_days": 40, "step_days": 40},
        fitness_weights={"sharpe": 1.0, "return": 0.1, "turnover": 0.1, "drawdown": 0.1},
        turnover_floor=0.001,
    )


def test_end_to_end_is_deterministic_and_funnel_consistent(tmp_path: Path):
    first = run_experiment(_small_config(), tmp_path / "first")
    second = run_experiment(_small_config(), tmp_path / "second")
    assert first["lineage"]["results_hash"] == second["lineage"]["results_hash"]
    funnel = first["funnel"].set_index("stage")["count"]
    assert funnel["candidates"] >= funnel["valid"] >= funnel["deduped"] >= funnel["gates_passed"] >= funnel["selected"]
    selected = first["selected"]
    assert not selected.empty
    assert (selected["turnover"] > 0).all()
    assert (selected["pnl_std"] > 0).all()
    assert (selected["observations"] >= _small_config().gate_limits["min_observations"]).all()
    assert json.loads((tmp_path / "first" / "lineage.json").read_text())["results_hash"] == first["lineage"]["results_hash"]
