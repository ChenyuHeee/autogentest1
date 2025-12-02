"""Common agent construction helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from ..config.settings import Settings, get_settings


def _should_enable_code_execution(settings: Settings, agent_name: str) -> bool:
    return settings.code_execution_enabled and agent_name in set(settings.code_execution_agents)


def _build_code_execution_config(settings: Settings, agent_name: str) -> Dict[str, Any] | None:
    if not _should_enable_code_execution(settings, agent_name):
        return None

    work_dir = settings.code_execution_workdir or str(Path.cwd())
    executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    return {"executor": executor}


def build_llm_config(settings: Settings, *, agent_name: str) -> Dict[str, Any]:
    """Return the AutoGen llm_config dictionary with optional local overrides."""

    config_list: List[Dict[str, Any]] = []
    local_agents = set(settings.local_model_agents)

    if (
        settings.local_model_enabled
        and settings.local_model_name
        and agent_name in local_agents
    ):
        local_config: Dict[str, Any] = {
            "model": settings.local_model_name,
        }
        if settings.local_model_base_url:
            local_config["base_url"] = settings.local_model_base_url
        if settings.local_model_api_key:
            local_config["api_key"] = settings.local_model_api_key
        config_list.append(local_config)

    config_list.append(
        {
            "model": settings.deepseek_model,
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
        }
    )

    return {"config_list": config_list}


def create_llm_agent(name: str, system_prompt: str, settings: Settings | None = None) -> AssistantAgent:
    """Create an AutoGen AssistantAgent with optional local execution support."""

    effective_settings = settings or get_settings()
    code_execution_config = _build_code_execution_config(effective_settings, name)

    if effective_settings.deepseek_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = effective_settings.deepseek_api_key
    if effective_settings.deepseek_base_url and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = effective_settings.deepseek_base_url

    return AssistantAgent(
        name=name,
        system_message=system_prompt,
        llm_config=build_llm_config(effective_settings, agent_name=name),
        code_execution_config=code_execution_config or False,
    )


def create_user_proxy(name: str, code_execution_config: Dict[str, Any] | None = None) -> UserProxyAgent:
    """Create a UserProxyAgent for running Python tools on behalf of LLM agents."""

    return UserProxyAgent(
        name=name,
        system_message="You are an orchestration proxy that executes Python helpers when requested.",
        code_execution_config=code_execution_config or {"use_docker": False},
    )
