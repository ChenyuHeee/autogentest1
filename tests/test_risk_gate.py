"""Tests for the hard risk gate enforcement layer."""

from __future__ import annotations

from typing import Any

from autogentest1.config.settings import Settings
from autogentest1.services.risk_gate import enforce_hard_limits


def _base_settings(**overrides: Any) -> Settings:
    return Settings(deepseek_api_key="test-key", **overrides)


def _build_response(include_stop: bool = True, *, position_oz: float = -1000.0) -> dict[str, object]:
    orders = [
        {
            "instrument": "XAUUSD",
            "side": "SELL",
            "size_oz": 1000,
            "type": "LIMIT",
            "entry": 4200,
        }
    ]
    if include_stop:
        orders.append(
            {
                "instrument": "XAUUSD",
                "side": "BUY",
                "size_oz": 1000,
                "type": "STOP",
                "entry": 4220,
            }
        )
    return {
        "details": {
            "trading_plan": {
                "base_plan": {
                    "position_oz": position_oz,
                }
            },
            "execution_checklist": {
                "orders": orders,
            },
            "risk_compliance_signoff": {
                "risk_metrics": {
                    "position_utilization": 0.4,
                    "stress_test_worst_loss_millions": -0.8,
                }
            },
        }
    }


def _base_context() -> dict[str, dict[str, Any]]:
    return {
        "risk_snapshot": {
            "position_utilization": 0.25,
            "pnl_today_millions": 0.2,
            "drawdown_threshold_millions": -0.3,
            "risk_alerts": [],
            "cross_asset_correlations": [],
        }
    }


def test_enforce_hard_limits_no_violation() -> None:
    settings = _base_settings()
    response = _build_response()
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is False
    assert report.violations == []


def test_enforce_hard_limits_drawdown_violation() -> None:
    settings = _base_settings()
    response = _build_response()
    context = _base_context()
    context["risk_snapshot"]["pnl_today_millions"] = -0.5
    context["risk_snapshot"]["drawdown_threshold_millions"] = -0.3

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "DAILY_DRAWDOWN" in codes


def test_enforce_hard_limits_requires_stop_loss() -> None:
    settings = _base_settings()
    response = _build_response(include_stop=False)
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    assert any(violation.code == "STOP_LOSS_MISSING" for violation in report.violations)


def test_enforce_hard_limits_stop_distance_violation() -> None:
    settings = _base_settings()
    response = {
        "details": {
            "trading_plan": {"base_plan": {"position_oz": -1000}},
            "execution_checklist": {
                "orders": [
                    {
                        "instrument": "XAUUSD",
                        "side": "SELL",
                        "size_oz": 1000,
                        "type": "LIMIT",
                        "entry": 4200,
                        "stop": 4400,
                    }
                ]
            },
            "risk_compliance_signoff": {
                "risk_metrics": {
                    "position_utilization": 0.4,
                    "stress_test_worst_loss_millions": -0.8,
                }
            },
        }
    }
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "STOP_DISTANCE_TOO_WIDE" in codes


def test_enforce_hard_limits_liquidity_violation() -> None:
    settings = _base_settings()
    response = {
        "details": {
            "trading_plan": {"base_plan": {"position_oz": -2000}},
            "execution_checklist": {
                "orders": [
                    {
                        "instrument": "XAUUSD",
                        "side": "SELL",
                        "size_oz": 2000,
                        "type": "LIMIT",
                        "entry": 4200,
                        "stop": 4250,
                    }
                ]
            },
            "risk_compliance_signoff": {
                "risk_metrics": {
                    "position_utilization": 0.6,
                    "stress_test_worst_loss_millions": -1.2,
                }
            },
        }
    }
    context = _base_context()
    context["risk_snapshot"]["liquidity_metrics"] = {
        "latest_volume": 500.0,
        "avg_volume": 600.0,
        "spread_bps": 45.0,
        "atr_based_slippage_bps": 40.0,
    }

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "LIQUIDITY_DEPTH_INSUFFICIENT" in codes
    assert "LIQUIDITY_SPREAD_EXCEEDED" in codes
    assert "LIQUIDITY_SLIPPAGE_EXCEEDED" in codes
