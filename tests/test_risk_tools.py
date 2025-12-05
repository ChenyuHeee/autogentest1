"""Unit tests for ToolsProxy-facing risk helpers."""

from __future__ import annotations

import pandas as pd

from autogentest1.services.risk import CorrelationTarget
from autogentest1.tools import risk_tools


def test_compute_risk_profile_respects_overrides(monkeypatch) -> None:
    history = pd.DataFrame({"Close": [1900.0, 1910.0, 1925.0]})

    class DummySettings:
        max_position_oz = 2000
        stress_var_millions = 3.0
        daily_drawdown_pct = 2.5
        default_position_oz = 750.0
        pnl_today_millions = 0.15
        risk_news_coupling_enabled = False
        news_api_key = None
        alpha_vantage_api_key = None
        market_data_max_age_minutes = 30
        data_provider = "yfinance"
        data_mode = "hybrid"

    captured: dict = {}

    def fake_get_settings():
        return DummySettings()

    def fake_fetch(symbol: str, days: int):
        captured["fetch_args"] = (symbol, days)
        return history

    def fake_build_snapshot(
        symbol: str,
        history: pd.DataFrame,
        *,
        limits,
        current_position_oz: float,
        pnl_today_millions: float,
        correlation_targets,
        news_snapshot,
        apply_news_adjustment: bool,
        **_: object,
    ):
        captured["snapshot_args"] = {
            "symbol": symbol,
            "history": history,
            "limits": limits,
            "position": current_position_oz,
            "pnl": pnl_today_millions,
            "targets": correlation_targets,
            "news_snapshot": news_snapshot,
            "apply_news_adjustment": apply_news_adjustment,
        }
        return {"status": "ok"}

    target_override = [CorrelationTarget(symbol="TEST", label="Test", window=5)]

    monkeypatch.setattr(risk_tools, "get_settings", fake_get_settings)
    monkeypatch.setattr(risk_tools, "fetch_price_history", fake_fetch)
    monkeypatch.setattr(risk_tools, "build_risk_snapshot", fake_build_snapshot)

    result = risk_tools.compute_risk_profile(
        symbol="XAUUSD",
        days=60,
        position_oz=900.0,
        pnl_today_millions=0.05,
        correlation_targets=target_override,
        news_snapshot={"score": 0.0, "confidence": 0.0, "classification": "neutral"},
    )

    assert result == {"status": "ok"}
    assert captured["fetch_args"] == ("XAUUSD", 60)
    snapshot_args = captured["snapshot_args"]
    assert snapshot_args["position"] == 900.0
    assert snapshot_args["pnl"] == 0.05
    assert snapshot_args["limits"].max_position_oz == DummySettings.max_position_oz
    assert snapshot_args["targets"] == target_override
    assert snapshot_args["news_snapshot"] == {"score": 0.0, "confidence": 0.0, "classification": "neutral"}
    assert snapshot_args["apply_news_adjustment"] is True