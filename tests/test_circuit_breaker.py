"""Tests for circuit breaker evaluation and audit logging."""

from __future__ import annotations

import json

from autogentest1.config.settings import Settings
from autogentest1.services.circuit_breaker import evaluate_circuit_breaker


def test_circuit_breaker_audit_trigger(tmp_path) -> None:
    audit_dir = tmp_path / "audit"
    settings = Settings(
        deepseek_api_key="test-key",
        circuit_breaker_enabled=True,
        audit_log_enabled=True,
        audit_log_directory=str(audit_dir),
    )
    risk_snapshot = {
        "pnl_today_millions": -3.0,
        "realized_vol_annualized": 0.2,
    }
    portfolio_state = {
        "risk_controls": {
            "consecutive_losses": 0,
            "baseline_vol_annualized": 0.2,
        }
    }

    evaluation = evaluate_circuit_breaker(
        settings=settings,
        risk_snapshot=risk_snapshot,
        portfolio_state=portfolio_state,
    )

    assert evaluation.triggered is True
    files = list(audit_dir.glob("audit-*.jsonl"))
    assert files
    raw = files[0].read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(raw[-1])
    assert event["event_type"] == "circuit_breaker.evaluation"
    assert event["payload"]["triggered"] is True
