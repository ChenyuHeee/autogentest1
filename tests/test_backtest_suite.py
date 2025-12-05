"""Tests covering the parameter sweep utilities built around the backtester."""

from __future__ import annotations

import pandas as pd

from autogentest1.services.backtest_suite import run_parameter_sweep


def _mock_history(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.DataFrame({"Close": values}, index=dates)


def test_expand_grid_and_ranking() -> None:
    history = _mock_history([100, 99, 101, 103, 102, 104, 105, 107])
    grid = {"short_window": [2, 3], "long_window": [4, 5]}

    sweep = run_parameter_sweep(
        history,
        strategy="sma_crossover",
        parameter_grid=grid,
        base_params={"symbol": "TEST"},
        initial_capital=50_000.0,
        evaluation_metric="total_return",
    )

    summary = sweep["summary"]
    assert summary["total_runs"] == 4
    assert summary["successful_runs"] == 4
    assert summary["failed_runs"] == 0

    best = sweep["best"]
    assert best is not None
    assert best["rank"] == 1
    best_params = best["params"]
    assert best_params["short_window"] < best_params["long_window"]

    ranked_returns = [run["metrics"]["total_return"] for run in sweep["runs"] if "rank" in run]
    assert ranked_returns == sorted(ranked_returns, reverse=True)


def test_parameter_sweep_handles_invalid_combo_and_fallback_metric() -> None:
    history = _mock_history([100, 100, 100, 100, 100])
    grid = {"short_window": [2, 4], "long_window": [4]}

    sweep = run_parameter_sweep(
        history,
        strategy="sma_crossover",
        parameter_grid=grid,
        base_params={"symbol": "FLAT"},
        evaluation_metric="sharpe_ratio",
    )

    runs = sweep["runs"]
    errors = [run for run in runs if "error" in run]
    assert errors, "Expected at least one invalid configuration"

    best = sweep["best"]
    assert best is not None
    assert best["score_metric"] == "total_return"
    assert best["score"] == 0.0

    valid_runs = [run for run in runs if "error" not in run]
    assert len(valid_runs) == 1
    assert valid_runs[0]["metrics"]["total_return"] == 0.0
