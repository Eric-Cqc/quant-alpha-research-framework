"""Small, labeled, colorblind-safe experiment plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def plot_sharpe_distribution(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(metrics["sharpe"].dropna(), bins=14, color=PALETTE[0], edgecolor="white")
    ax.set(title="Candidate Sharpe distribution", xlabel="Annualized Sharpe", ylabel="Candidates")
    _save(fig, path)


def plot_correlation_heatmap(correlation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if correlation.empty:
        ax.text(0.5, 0.5, "No candidates passed correlation filtering", ha="center", va="center")
        ax.set_axis_off()
    else:
        image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(correlation.index)), correlation.index, fontsize=7)
        ax.set_title("Daily return correlation")
        fig.colorbar(image, ax=ax, label="Correlation")
    _save(fig, path)


def plot_funnel(funnel: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(funnel["stage"], funnel["count"], color=PALETTE[: len(funnel)])
    ax.set(title="Research funnel", ylabel="Candidate count")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)


def plot_equity_curves(results: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, pnl) in enumerate(sorted(results.items())):
        ax.plot((1.0 + pnl.fillna(0.0)).cumprod(), label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.3)
    ax.set(title="Selected signal equity curves", xlabel="Date", ylabel="Growth of 1 unit")
    if results:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)
