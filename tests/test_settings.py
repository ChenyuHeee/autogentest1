"""Tests for configuration utilities."""

from __future__ import annotations

from autogentest1.config.dump_settings import dump_settings_dict
from autogentest1.config.settings import Settings


def test_local_model_agents_parsing(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_MODEL_AGENTS", "DataAgent, MacroAnalystAgent , ,QuantResearchAgent")
    settings = Settings(
        deepseek_api_key="test-key",
    )
    assert settings.local_model_agents == [
        "DataAgent",
        "MacroAnalystAgent",
        "QuantResearchAgent",
    ]


def test_local_model_defaults_loaded_from_env(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_MODEL_ENABLED", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_AUTO_ENABLE", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_AGENTS", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_BASE_URL", raising=False)
    settings = Settings(
        deepseek_api_key="test-key",
    )
    # Default value for local_model_enabled is False (disabled by default)
    assert settings.local_model_enabled is False
    # Default value for local_model_agents is empty list (no agents use local model by default)
    assert settings.local_model_agents == []


def test_code_execution_agents_default(monkeypatch) -> None:
    monkeypatch.delenv("CODE_EXECUTION_AGENTS", raising=False)
    settings = Settings(deepseek_api_key="test-key")
    assert settings.code_execution_enabled is True
    assert {"QuantResearchAgent", "TechAnalystAgent"}.issubset(set(settings.code_execution_agents))


def test_dump_settings_redacts_secrets(monkeypatch) -> None:
    settings = Settings(
        deepseek_api_key="sk-live-123456",
        alpha_vantage_api_key="alpha-XYZ",
        news_api_key=None,
    )
    payload = dump_settings_dict(settings=settings)
    assert payload["deepseek_api_key"] == "<redacted>"
    assert payload["alpha_vantage_api_key"] == "<redacted>"
    # Non-secret values should remain intact.
    assert payload["audit_log_enabled"] is True

    raw_payload = dump_settings_dict(settings=settings, include_secrets=True)
    assert raw_payload["deepseek_api_key"] == "sk-live-123456"
