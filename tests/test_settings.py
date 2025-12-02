"""Tests for configuration utilities."""

from __future__ import annotations

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
    settings = Settings(
        deepseek_api_key="test-key",
    )
    assert settings.local_model_enabled is True
    assert settings.local_model_agents == [
        "DataAgent",
        "MacroAnalystAgent",
        "FundamentalAnalystAgent",
        "QuantResearchAgent",
    ]


def test_code_execution_agents_default(monkeypatch) -> None:
    monkeypatch.delenv("CODE_EXECUTION_AGENTS", raising=False)
    settings = Settings(deepseek_api_key="test-key")
    assert settings.code_execution_enabled is True
    assert {"QuantResearchAgent", "TechAnalystAgent"}.issubset(set(settings.code_execution_agents))
