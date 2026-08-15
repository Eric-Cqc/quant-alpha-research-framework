"""End-to-end offline experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from signal_research import __version__
from signal_research.backtesting.engine import backtest_signal
from signal_research.config import ExperimentConfig
from signal_research.reporting.funnel import funnel_frame
from signal_research.reporting.lineage import canonical_json_bytes, environment_metadata, results_hash, sha256_bytes
from signal_research.reporting.plots import plot_correlation_heatmap, plot_equity_curves, plot_funnel, plot_sharpe_distribution
from signal_research.risk.gates import apply_gates
from signal_research.scoring.fitness import composite_fitness
from signal_research.scoring.metrics import metrics_for_backtest
from signal_research.selection.correlation import greedy_correlation_filter
from signal_research.selection.portfolio import select_portfolio
from signal_research.selection.walk_forward import run_walk_forward
from signal_research.signal_generation.registry import build_candidate_grid
from signal_research.signal_generation.synthetic_data import generate_synthetic_panel
from signal_research.signal_generation.textbook_signals import SIGNAL_FUNCTIONS
from signal_research.validation.dedup import remove_exact_duplicates
from signal_research.validation.schema_validation import SpecValidationError, validate_signal_spec
from signal_research.validation.static_checks import StaticCheckError, assert_truncation_invariant, check_signal_function, truncation_cutoffs, validate_runtime_signal


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def run_experiment(config: ExperimentConfig, out_dir: str | Path) -> dict[str, object]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = generate_synthetic_panel(config.seed, config.n_tickers, config.n_days, config.start)
    specs = build_candidate_grid()
    valid_specs = []
    invalid_reasons: list[str] = []
    for spec in specs:
        try:
            validate_signal_spec(spec)
            function = SIGNAL_FUNCTIONS[spec.family]
            check_signal_function(function, set(spec.required_fields))
            for cutoff in truncation_cutoffs(panel):
                assert_truncation_invariant(function, panel, cutoff, dict(spec.parameters))
            valid_specs.append(spec)
        except (SpecValidationError, StaticCheckError, KeyError, ValueError) as exc:
            invalid_reasons.append(f"{spec.name}: {exc}")
    deduped_specs, dedup_records = remove_exact_duplicates(valid_specs)
    close = panel["close"].unstack("ticker").sort_index()
    forward_wide = close.pct_change().shift(-1)
    forward_returns = forward_wide.stack().reindex(panel.index).rename("forward_return")
    backtests = {}
    rows: list[dict[str, object]] = []
    for spec in deduped_specs:
        function = SIGNAL_FUNCTIONS[spec.family]
        signal = function(panel, **dict(spec.parameters))
        validate_runtime_signal(signal, panel.index)
        result = backtest_signal(signal, forward_returns, config.cost_bps, config.quantile, config.max_abs_weight)
        metrics = metrics_for_backtest(result)
        metrics["fitness"] = composite_fitness(metrics, config.fitness_weights, config.turnover_floor)
        rows.append({"spec_id": spec.name, "name": spec.name, "family": spec.family, **metrics})
        backtests[spec.name] = result
    metrics_frame = pd.DataFrame(rows).sort_values(["fitness", "name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    gated, gate_reasons = apply_gates(metrics_frame, config.gate_limits)
    passed_names = gated.loc[gated["gate_pass"], "name"].tolist()
    return_frame = pd.DataFrame({name: backtests[name].daily_pnl for name in passed_names})
    return_frame.index.name = "date"
    scores = {row.name: float(row.fitness) for row in gated.loc[gated["gate_pass"]].itertuples(index=False)}
    correlation_passed, correlation_rejections = greedy_correlation_filter(return_frame, scores, config.correlation_threshold) if passed_names else ([], {})
    eligible_metrics = gated[gated["name"].isin(correlation_passed)].copy()
    selected_names = select_portfolio(eligible_metrics, config.top_k, config.family_caps)
    selected_frame = gated[gated["name"].isin(selected_names)].copy()
    selected_frame["portfolio_weight"] = 1.0 / len(selected_names) if selected_names else 0.0
    selected_frame = selected_frame.sort_values(["fitness", "name"], ascending=[False, True], kind="mergesort")
    corr_matrix = return_frame[correlation_passed].corr() if correlation_passed else pd.DataFrame()
    walk = run_walk_forward(return_frame[correlation_passed], correlation_passed, **config.walk_forward, top_k=config.top_k) if correlation_passed else pd.DataFrame()
    funnel = funnel_frame([
        {"stage": "candidates", "count": len(specs), "rejection_reason": ""},
        {"stage": "valid", "count": len(valid_specs), "rejection_reason": f"{len(invalid_reasons)} rejected by schema/static checks"},
        {"stage": "deduped", "count": len(deduped_specs), "rejection_reason": f"{len(valid_specs) - len(deduped_specs)} exact duplicates removed"},
        {"stage": "gates_passed", "count": len(passed_names), "rejection_reason": f"{len(deduped_specs) - len(passed_names)} failed risk gates"},
        {"stage": "correlation_passed", "count": len(correlation_passed), "rejection_reason": f"{len(correlation_rejections)} correlated candidates removed"},
        {"stage": "selected", "count": len(selected_names), "rejection_reason": f"{len(correlation_passed) - len(selected_names)} left outside top-k or family caps"},
    ])
    _write_frame(funnel, output / "funnel.csv")
    _write_frame(gated, output / "signal_metrics.csv")
    _write_frame(selected_frame, output / "selected_signals.csv")
    corr_to_write = corr_matrix.reset_index().rename(columns={"index": "name"}) if not corr_matrix.empty else pd.DataFrame(columns=["name"])
    _write_frame(corr_to_write, output / "correlation_matrix.csv")
    _write_frame(walk, output / "walk_forward.csv")

    lineage_core = {
        "config_hash": sha256_bytes(canonical_json_bytes(config.__dict__)),
        "data_hash": sha256_bytes(panel.to_csv(float_format="%.12g", lineterminator="\n").encode("utf-8")),
        "stages": {"candidates": len(specs), "valid": len(valid_specs), "deduped": len(deduped_specs), "gates_passed": len(passed_names), "correlation_passed": len(correlation_passed), "selected": len(selected_names)},
        "selected_ids": selected_names,
        "package_version": __version__,
        **environment_metadata(),
    }
    frames = {"funnel": funnel, "signal_metrics": gated, "selected_signals": selected_frame, "correlation_matrix": corr_to_write, "walk_forward": walk}
    result_digest = results_hash(frames, lineage_core)
    lineage = {**lineage_core, "results_hash": result_digest}
    (output / "lineage.json").write_bytes(canonical_json_bytes(lineage) + b"\n")
    plot_sharpe_distribution(gated, output / "sharpe_distribution.png")
    plot_correlation_heatmap(corr_matrix, output / "correlation_heatmap.png")
    plot_funnel(funnel, output / "funnel.png")
    plot_equity_curves({name: backtests[name].daily_pnl for name in selected_names}, output / "equity_curves.png")
    return {"funnel": funnel, "metrics": gated, "selected": selected_frame, "walk_forward": walk, "lineage": lineage, "backtests": backtests, "invalid_reasons": invalid_reasons}
