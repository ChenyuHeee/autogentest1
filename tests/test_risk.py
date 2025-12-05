"""Unit tests for risk snapshot generation."""

from __future__ import annotations

import pandas as pd

from autogentest1.services.risk import (
    CorrelationTarget,
    RiskLimits,
    build_risk_snapshot,
)


def test_build_risk_snapshot_empty_history() -> None:
    history = pd.DataFrame({"Close": []})
    limits = RiskLimits(max_position_oz=5000, stress_var_millions=3.0, daily_drawdown_pct=3.0)

    snapshot = build_risk_snapshot(
        "XAUUSD",
        history,
        limits=limits,
        current_position_oz=1000,
        pnl_today_millions=-0.2,
    )

    assert snapshot["limit_position_oz"] == 5000
    assert snapshot["drawdown_alert"] is False
    assert snapshot["historical_var_99"] is None
    assert snapshot["scenario_outcomes"] == []
    assert snapshot["cross_asset_correlations"] == []
    assert snapshot["portfolio_var_millions"] is None
    assert snapshot["risk_alerts"] == []
    assert snapshot["base_limits"]["max_position_oz"] == 5000
    assert snapshot["adjusted_limits"]["max_position_oz"] == 5000
    assert snapshot["news_adjustment"] is None


def test_build_risk_snapshot_with_metrics() -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    data = {
        "Close": [
            1900.0,
            1910.0,
            1920.0,
            1905.0,
            1895.0,
            1915.0,
            1930.0,
        ]
    }
    history = pd.DataFrame(data, index=dates)
    limits = RiskLimits(max_position_oz=4000, stress_var_millions=2.5, daily_drawdown_pct=3.0)

    benchmark_series = {
        "DX-Y.NYB": pd.Series(
            [100.0, 101.0, 102.0, 101.5, 101.0, 102.5, 103.0], index=dates
        )
    }

    snapshot = build_risk_snapshot(
        "XAUUSD",
        history,
        limits=limits,
        current_position_oz=1200,
        pnl_today_millions=0.1,
        benchmark_series=benchmark_series,
        correlation_targets=[CorrelationTarget(symbol="DX-Y.NYB", label="DXY", window=3)],
        correlation_window=3,
    )

    assert snapshot["historical_var_99"] is not None
    assert snapshot["scenario_outcomes"]
    labels = {entry["label"] for entry in snapshot["scenario_outcomes"]}
    assert {"minus_2pct", "plus_2pct"}.issubset(labels)
    assert all("projected_pnl_millions" in entry for entry in snapshot["scenario_outcomes"])
    assert snapshot["portfolio_var_millions"] is not None
    assert snapshot["cross_asset_correlations"]
    correlation_symbols = {item["symbol"] for item in snapshot["cross_asset_correlations"]}
    assert "DX-Y.NYB" in correlation_symbols


def test_build_risk_snapshot_applies_news_coupling() -> None:
    history = pd.DataFrame(
        {
            "Close": [
                1900.0,
                1910.0,
                1925.0,
                1895.0,
                1880.0,
                1870.0,
                1860.0,
            ]
        }
    )
    limits = RiskLimits(max_position_oz=4000, stress_var_millions=2.5, daily_drawdown_pct=2.0)
    bearish_news = {
        "score": -0.3,
        "confidence": 0.9,
        "classification": "bearish",
        "score_trend": -0.05,
    }

    snapshot = build_risk_snapshot(
        "XAUUSD",
        history,
        limits=limits,
        current_position_oz=1000,
        pnl_today_millions=-0.4,
        news_snapshot=bearish_news,
    )

    assert snapshot["news_adjustment"] is not None
    assert snapshot["adjusted_limits"]["max_position_oz"] < limits.max_position_oz
    assert snapshot["adjusted_limits"]["stress_var_millions"] < limits.stress_var_millions


def test_build_risk_snapshot_alert_flags() -> None:
    history = pd.DataFrame({"Close": [1900.0, 1700.0, 1500.0, 1400.0, 1300.0, 1250.0, 1200.0]})
    limits = RiskLimits(max_position_oz=1000, stress_var_millions=0.5, daily_drawdown_pct=1.0)

    snapshot = build_risk_snapshot(
        "XAUUSD",
        history,
        limits=limits,
        current_position_oz=5000,
        pnl_today_millions=-2.0,
        benchmark_series={},
    )

    alerts = set(snapshot["risk_alerts"])
    assert "position_limit_exceeded" in alerts
    assert "drawdown_limit_breached" in alerts
    assert "var_limit_exceeded" in alerts or "var_limit_warning" in alerts


def test_build_risk_snapshot_includes_data_quality() -> None:
    dates = pd.date_range("2024-06-01", periods=5, freq="D", tz="UTC")
    history = pd.DataFrame({
        "Close": [1900.0, 1910.0, 1920.0, 1915.0, 1925.0],
        "Volume": [1000, 1050, 980, 990, 1010],
    }, index=dates)
    history.attrs["provider_key"] = "yfinance"
    history.attrs["provider_label"] = "Yahoo Finance"
    history.attrs["data_age_minutes"] = 25.0
    history.attrs["data_last_timestamp"] = dates[-1].isoformat()
    history.attrs["history_rows"] = len(history)

    limits = RiskLimits(max_position_oz=4000, stress_var_millions=2.5, daily_drawdown_pct=3.0)

    snapshot = build_risk_snapshot(
        "XAUUSD",
        history,
        limits=limits,
        current_position_oz=1000,
        pnl_today_millions=0.2,
        max_data_age_minutes=30,
        data_provider="yfinance",
        data_mode="hybrid",
    )

    dq = snapshot["data_quality"]
    assert dq["provider"] == "yfinance"
    assert dq["provider_label"] == "Yahoo Finance"
    assert dq["max_age_minutes"] == 30
    assert dq["fresh"] is True
    assert dq["history_rows"] == len(history)
