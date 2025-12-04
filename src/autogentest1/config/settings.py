"""Application settings loaded from environment variables."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _lenient_json_loads(value: str) -> Any:
    """Parse JSON values for settings while tolerating plain strings."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class Settings(BaseSettings):
    """Central configuration for the AutoGen agents and services."""

    deepseek_api_key: str = Field(...)
    deepseek_model: str = Field("deepseek-chat")
    deepseek_base_url: str = Field("https://api.deepseek.com/v1")
    data_provider: str = Field("yfinance")
    data_mode: str = Field("hybrid")
    default_symbol: str = Field("XAUUSD")
    default_days: int = Field(14)
    log_level: str = Field("INFO")
    max_position_oz: float = Field(5000.0)
    stress_var_millions: float = Field(3.0)
    daily_drawdown_pct: float = Field(3.0)
    default_position_oz: float = Field(0.0)
    pnl_today_millions: float = Field(0.0)
    compliance_allowed_instruments: List[str] = Field(
        default_factory=lambda: ["XAUUSD", "GC", "GC=F", "GLD"]
    )
    compliance_restricted_instruments: List[str] = Field(default_factory=list)
    compliance_allowed_counterparties: List[str] = Field(
        default_factory=lambda: ["CME", "ICE", "LBMA", "OTC_DESK"]
    )
    compliance_restricted_counterparties: List[str] = Field(default_factory=list)
    compliance_max_single_order_oz: float = Field(3000.0)
    compliance_require_stop_loss: bool = Field(True)
    compliance_require_take_profit: bool = Field(True)
    local_model_enabled: bool = Field(False)
    local_model_name: str | None = Field("qwen2.5-14b-instruct")
    local_model_base_url: str | None = Field("http://127.0.0.1:11434/v1")
    local_model_api_key: str | None = Field(None)
    local_model_agents: List[str] = Field(default_factory=list)
    code_execution_enabled: bool = Field(True)
    code_execution_agents: List[str] = Field(default_factory=lambda: [
        "QuantResearchAgent",
        "TechAnalystAgent",
    ])
    code_execution_timeout: int = Field(90)
    code_execution_workdir: str | None = Field(None)
    workflow_format_retry_limit: int = Field(2)
    workflow_max_rounds: int = Field(30)
    workflow_max_plan_retries: int = Field(3)
    market_data_max_age_minutes: int = Field(1440)
    human_override_timeout_seconds: int = Field(120)
    market_data_cache_minutes: int = Field(10)
    market_data_retry_total: int = Field(4)
    market_data_retry_backoff: float = Field(1.0)
    news_watcher_enabled: bool = Field(False)
    news_watcher_poll_seconds: int = Field(300)
    news_watcher_keywords: List[str] = Field(default_factory=lambda: ["war", "fed", "cpi", "rate", "hike"])
    news_watcher_vol_threshold: float = Field(0.4)
    alpha_vantage_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices("alpha_vantage_api_key", "alphavantage_api_key", "ALPHAVANTAGE_API_KEY"),
    )
    news_api_key: str | None = Field(None)
    tanshu_api_key: str | None = Field(None)
    tanshu_endpoint: str = Field("gjgold2")
    tanshu_symbol_code: str | None = Field(None)
    tanshu_symbol_map: Dict[str, str] = Field(default_factory=dict)
    twelve_data_api_key: str | None = Field(None)
    twelve_data_base_url: str = Field("https://api.twelvedata.com")
    twelve_data_symbol: str | None = Field(None)
    twelve_data_symbol_map: Dict[str, str] = Field(default_factory=dict)
    polygon_api_key: str | None = Field(None)
    polygon_base_url: str = Field("https://api.polygon.io")
    polygon_symbol_map: Dict[str, str] = Field(default_factory=dict)
    rag_index_root: str = Field("data/rag-index")
    rag_namespace: str = Field("gold-playbooks")
    rag_chunk_size: int = Field(200)
    rag_chunk_overlap: int = Field(32)
    rag_similarity_threshold: float = Field(0.12)
    rag_auto_ingest: bool = Field(True)
    rag_corpus_paths: List[str] = Field(default_factory=lambda: ["data/rag"])
    risk_news_coupling_enabled: bool = Field(True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def settings_json_loads(cls, value: str) -> Any:
        """Decode JSON values from env while tolerating plain comma-separated lists."""

        return _lenient_json_loads(value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Disable strict JSON decoding for environment-backed sources."""

        for source in (env_settings, dotenv_settings):
            if hasattr(source, "config"):
                source.config["enable_decoding"] = False

        return init_settings, env_settings, dotenv_settings, file_secret_settings

    @field_validator(
        "local_model_agents",
        "code_execution_agents",
        "news_watcher_keywords",
        "rag_corpus_paths",
        "compliance_allowed_instruments",
        "compliance_restricted_instruments",
        "compliance_allowed_counterparties",
        "compliance_restricted_counterparties",
        mode="before",
    )
    @classmethod
    def _parse_local_model_agents(cls, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @field_validator("tanshu_symbol_map", "twelve_data_symbol_map", "polygon_symbol_map", mode="before")
    @classmethod
    def _parse_symbol_map(cls, value: object) -> Dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {
                str(key).upper(): str(val)
                for key, val in value.items()
                if val is not None
            }
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return {}
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return {
                        str(key).upper(): str(val)
                        for key, val in parsed.items()
                        if val is not None
                    }
            except json.JSONDecodeError:
                mapping: Dict[str, str] = {}
                parts = [segment for segment in cleaned.split(",") if segment.strip()]
                for part in parts:
                    if ":" not in part:
                        continue
                    key, val = part.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key and val:
                        mapping[key.upper()] = val
                return mapping
        return {}

    @model_validator(mode="before")
    @classmethod
    def _coerce_numeric_bounds(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        if "workflow_format_retry_limit" in coerced:
            coerced["workflow_format_retry_limit"] = max(1, int(coerced["workflow_format_retry_limit"]))
        if "workflow_max_plan_retries" in coerced:
            coerced["workflow_max_plan_retries"] = max(1, int(coerced["workflow_max_plan_retries"]))
        if "workflow_max_rounds" in coerced:
            coerced["workflow_max_rounds"] = max(10, int(coerced["workflow_max_rounds"]))
        if "news_watcher_poll_seconds" in coerced:
            coerced["news_watcher_poll_seconds"] = max(30, int(coerced["news_watcher_poll_seconds"]))
        if "market_data_max_age_minutes" in coerced:
            coerced["market_data_max_age_minutes"] = max(1, int(coerced["market_data_max_age_minutes"]))
        if "code_execution_timeout" in coerced:
            coerced["code_execution_timeout"] = max(10, int(coerced["code_execution_timeout"]))
        if "human_override_timeout_seconds" in coerced:
            coerced["human_override_timeout_seconds"] = max(30, int(coerced["human_override_timeout_seconds"]))
        if "news_watcher_vol_threshold" in coerced:
            coerced["news_watcher_vol_threshold"] = max(0.05, float(coerced["news_watcher_vol_threshold"]))
        if "market_data_cache_minutes" in coerced:
            coerced["market_data_cache_minutes"] = max(1, int(coerced["market_data_cache_minutes"]))
        if "market_data_retry_total" in coerced:
            coerced["market_data_retry_total"] = max(0, int(coerced["market_data_retry_total"]))
        if "market_data_retry_backoff" in coerced:
            coerced["market_data_retry_backoff"] = max(0.0, float(coerced["market_data_retry_backoff"]))
        if "data_mode" in coerced:
            coerced["data_mode"] = str(coerced["data_mode"]).lower()
        if "rag_chunk_size" in coerced:
            coerced["rag_chunk_size"] = max(32, int(coerced["rag_chunk_size"]))
        if "rag_chunk_overlap" in coerced:
            overlap = max(0, int(coerced["rag_chunk_overlap"]))
            chunk = max(1, int(coerced.get("rag_chunk_size", 200)))
            coerced["rag_chunk_overlap"] = min(overlap, chunk - 1 if chunk > 1 else 0)
        if "rag_similarity_threshold" in coerced:
            coerced["rag_similarity_threshold"] = max(0.0, min(1.0, float(coerced["rag_similarity_threshold"])))
        return coerced

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_env_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "alphavantage_api_key" in data and "alpha_vantage_api_key" not in data:
            data = dict(data)
            data["alpha_vantage_api_key"] = data.pop("alphavantage_api_key")
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and cache the instance for reuse."""

    return Settings()  # type: ignore[call-arg]
