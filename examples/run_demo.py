"""Run the complete offline demonstration."""

from __future__ import annotations

import argparse

from signal_research.config import load_config
from signal_research.reporting.report import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline synthetic signal research demo")
    parser.add_argument("--config", default="configs/demo.toml")
    parser.add_argument("--out", default="outputs/example_results")
    args = parser.parse_args()
    result = run_experiment(load_config(args.config), args.out)
    print("Research funnel")
    print(result["funnel"].to_string(index=False))
    print("\nFinal selection")
    selected = result["selected"]
    columns = ["name", "family", "fitness", "sharpe", "annualized_return", "max_drawdown"]
    print(selected[columns].to_string(index=False) if not selected.empty else "No candidates selected")
    print(f"\nresults_hash: {result['lineage']['results_hash']}")


if __name__ == "__main__":
    main()
