"""Tests for the hard risk gate enforcement layer."""

from __future__ import annotations

import json
from typing import Any, Dict

from autogentest1.config.settings import Settings
from autogentest1.services.risk_gate import enforce_hard_limits


def _base_settings(**overrides: Any) -> Settings:
    params: Dict[str, Any] = {"audit_log_enabled": False}
    params.update(overrides)
    return Settings(deepseek_api_key="test-key", **params)


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
            "data_quality": {
                "provider": "polygon",
                "provider_label": "Polygon",
                "age_minutes": 5.0,
                "max_age_minutes": 30.0,
                "fresh": True,
                "history_rows": 90,
            },
        },
        "portfolio_state": {
            "risk_controls": {
                "consecutive_losses": 0,
                "baseline_vol_annualized": None,
                "cooldown_until": None,
            }
        },
    }


def test_enforce_hard_limits_no_violation() -> None:
    settings = _base_settings()
    response = _build_response()
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is False
    assert report.violations == []


def test_enforce_hard_limits_payload_orders_detected() -> None:
    settings = _base_settings()
    response = {
        "details": {
            "payload": {
                "trading_plan": {
                    "base_plan": {
                        "position_oz": 800,
                        "entry": 4198,
                        "stop": 4165,
                        "targets": [4265],
                    }
                },
                "execution_checklist": {
                    "orders": [
                        {
                            "instrument": "XAUUSD",
                            "side": "BUY",
                            "size_oz": 800,
                            "type": "LIMIT",
                            "entry": 4198,
                            "stop": 4165,
                            "target": 4265,
                        }
                    ]
                },
            }
        }
    }
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is False
    assert report.evaluated_metrics["has_stop_protection"] is True


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
        "atr_based_slippage_bps": 45.0,
    }

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "LIQUIDITY_DEPTH_INSUFFICIENT" in codes
    assert "LIQUIDITY_SPREAD_EXCEEDED" in codes
    assert "LIQUIDITY_SLIPPAGE_EXCEEDED" in codes


def test_enforce_hard_limits_circuit_breaker_daily_loss() -> None:
    settings = _base_settings(
        circuit_breaker_enabled=True,
        circuit_breaker_daily_loss_limit_millions=1.0,
    )
    response = _build_response()
    context = _base_context()
    context["risk_snapshot"]["pnl_today_millions"] = -1.5

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "CIRCUIT_BREAKER_DAILY_LOSS" in codes


def test_enforce_hard_limits_flags_stale_data() -> None:
    settings = _base_settings()
    response = _build_response()
    context = _base_context()
    context["risk_snapshot"]["data_quality"].update({"age_minutes": 120.0, "max_age_minutes": 30.0})

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "MARKET_DATA_STALE" in codes


def test_enforce_hard_limits_relaxes_retail_data() -> None:
    settings = _base_settings()
    response = _build_response()
    context = _base_context()
    context["risk_snapshot"]["data_quality"].update(
        {
            "provider": "yfinance",
            "provider_label": "Yahoo Finance",
            "age_minutes": 10.0,
            "max_age_minutes": 30.0,
        }
    )
    context["risk_snapshot"]["liquidity_metrics"] = {
        "latest_volume": 30000.0,
        "avg_volume": 32000.0,
        "spread_bps": 45.0,
        "atr_based_slippage_bps": 32.0,
    }

    report = enforce_hard_limits(response, context=context, settings=settings)

    codes = {violation.code for violation in report.violations}
    assert "LIQUIDITY_SPREAD_EXCEEDED" not in codes
    assert "LIQUIDITY_SLIPPAGE_EXCEEDED" not in codes
    assert report.breached is False


def test_enforce_hard_limits_circuit_breaker_consecutive_losses() -> None:
    settings = _base_settings(circuit_breaker_enabled=True, circuit_breaker_max_consecutive_losses=3)
    response = _build_response()
    context = _base_context()
    context["portfolio_state"]["risk_controls"]["consecutive_losses"] = 4

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "CIRCUIT_BREAKER_CONSECUTIVE_LOSSES" in codes


def test_enforce_hard_limits_circuit_breaker_cooldown() -> None:
    settings = _base_settings(circuit_breaker_enabled=True)
    response = _build_response()
    context = _base_context()
    context["portfolio_state"]["risk_controls"]["cooldown_until"] = "2999-01-01T00:00:00+00:00"

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is True
    codes = {violation.code for violation in report.violations}
    assert "CIRCUIT_BREAKER_ACTIVE_COOLDOWN" in codes


def test_enforce_hard_limits_audit_logging(tmp_path) -> None:
    audit_dir = tmp_path / "audit"
    settings = _base_settings(audit_log_enabled=True, audit_log_directory=str(audit_dir))
    response = _build_response()
    context = _base_context()

    report = enforce_hard_limits(response, context=context, settings=settings)

    assert report.breached is False
    files = list(audit_dir.glob("audit-*.jsonl"))
    assert files
    raw = files[0].read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(raw[-1])
    assert event["event_type"] == "risk_gate.evaluation"
    assert event["payload"]["breached"] is False
