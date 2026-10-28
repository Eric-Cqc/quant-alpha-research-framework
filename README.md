# Signal Research Framework
This repo is a research framework. Not a product install path — see profile Selected for current work.


## What this is

This is a sanitized reconstructed portfolio version of an offline signal
research workflow. It demonstrates candidate generation, point-in-time checks,
portfolio simulation, risk gates, selection, walk-forward evaluation, and
reproducible reporting on synthetic data. The included metrics are research
demonstration metrics, not platform scores.

## Why I built it

The interesting engineering problem is not producing one impressive backtest.
It is building a workflow that can say why a candidate entered or left the
research funnel, prevent accidental look-ahead, control concentration and
turnover, and reproduce the same result later.

## Research workflow

Synthetic panel → textbook candidate grid → schema/static/runtime checks →
duplicate removal → next-day backtest → metrics and fitness → risk gates →
return-correlation filter → family-capped portfolio → walk-forward report.

## Architecture

The `src/signal_research/` package is split into signal generation, validation,
backtesting, scoring, selection, risk, and reporting. `configs/demo.toml` holds
all experiment thresholds. `examples/run_demo.py` is the end-to-end entry
point. The tests cover each major contract and determinism.

## AI-human collaboration

The human defines hypotheses, metrics, risk thresholds, and economic-plausibility
judgments. AI generates candidate transformations, automates experiments,
summarizes results, and detects duplicates. The system tracks provenance,
evaluates, filters, and ranks. AI is a research multiplier, not a black-box
strategy generator.

## Signal lifecycle

Each immutable `SignalSpec` names a public textbook family, its required panel
columns, numeric parameters, horizon, and rationale. Functions return a Series
aligned to the full `(date, ticker)` panel. The engine applies date `t` weights
to date `t+1` returns and subtracts linear costs from one-way turnover.

## Validation & risk gates

AST checks reject imports, file/network access, unsafe evaluation, and
undeclared fields. Runtime checks verify exact index alignment and truncation
invariance at multiple cutoffs spread through the sample. These checks are
designed to catch accidental look-ahead and unsafe patterns in candidate code,
not to sandbox a deliberately adversarial author — the funnel assumes
good-faith research code and makes mistakes loud, not malice impossible.
Risk gates then enforce minimum active observations, Sharpe, turnover,
drawdown, and maximum absolute weight. A greedy correlation filter removes
redundant return streams before family caps and top-k selection.

## Example experiment

The checked-in example is generated with 100 synthetic tickers over 750 business
days and a five-basis-point linear cost. The exact candidate counts, selected
signals, metrics, and `results_hash` are in `outputs/example_results/`; they are
refreshed by the commands below and are not live performance claims. In the
verified run, the funnel was 54 candidates → 51 valid → 50 deduped → 13 passed
risk gates → 6 correlation-filter survivors → 3 selected, with the selected
signals landing between 1.24299 and 1.43186 annualized Sharpe after costs. The
resulting hash was
`2fb543d1a0faf4be42829acb0c947f770412ffb31b8dd2db6e20d9faa4fe9a94`.

## Reproducibility

### Environment setup

From a clean checkout:

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

```bash
./.venv/bin/pytest
./.venv/bin/python examples/run_demo.py --config configs/demo.toml --out outputs/example_results
```

The run is offline and deterministic for a fixed configuration. `lineage.json`
contains configuration, synthetic-data, stage, selection, and results hashes.

## What is intentionally not included

There are no proprietary data fields, real competition signals or expressions,
credentials, personal identifiers, external-service calls, platform scores, or
claims of live profitability. The repository is a public educational
reconstruction using synthetic data and textbook signal families.

## WorldQuant-IQC disclosure

This is an independent sanitized reconstruction inspired by my private research
workflow for the WorldQuant BRAIN / IQC competition; no real competition
signals, platform data, or platform scores are included; not affiliated with
or endorsed by WorldQuant. The metrics here are not platform scores.

## Tech stack

Python, numpy, pandas, matplotlib, setuptools, TOML, and pytest. No network
service or database is required.

## Lessons learned

The most valuable controls are boring and explicit: a next-day return
convention, a truncation test, canonical duplicate signatures, deterministic
tie-breaking, risk gates, and lineage hashes. They make the research process
easier to review and harder to accidentally overstate.
