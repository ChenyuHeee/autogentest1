"""Tests for portfolio state helpers."""

from __future__ import annotations

from pathlib import Path

from autogentest1.services import state


def _mock_state_path(tmp_path: Path) -> Path:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / "portfolio_state.json"


def test_load_state_returns_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(state, "_state_file_path", lambda: _mock_state_path(tmp_path))
    result = state.load_portfolio_state()
    assert result["positions"]["symbol"] == "XAUUSD"
    assert result["positions"]["net_oz"] == 0.0


def test_update_state_persists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(state, "_state_file_path", lambda: _mock_state_path(tmp_path))

    updated = state.update_portfolio_state({"positions": {"symbol": "XAUUSD", "net_oz": 100.0}})
    assert updated["positions"]["net_oz"] == 100.0
    saved = state.load_portfolio_state()
    assert saved["positions"]["net_oz"] == 100.0


def test_apply_state_patch_preserves_nested_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(state, "_state_file_path", lambda: _mock_state_path(tmp_path))

    base = state.load_portfolio_state()
    assert base["risk_controls"]["consecutive_losses"] == 0

    patch = {
        "risk_controls": {
            "consecutive_losses": 3,
            "cooldown_until": "2999-01-01T00:00:00+00:00",
        }
    }
    updated = state.apply_portfolio_state_patch(patch)

    assert updated["risk_controls"]["consecutive_losses"] == 3
    assert updated["risk_controls"]["cooldown_until"] == "2999-01-01T00:00:00+00:00"
    # Baseline keys should remain available after deep merge.
    assert "baseline_vol_annualized" in updated["risk_controls"]

    persisted = state.load_portfolio_state()
    assert persisted["risk_controls"]["consecutive_losses"] == 3
