"""Factory for the data acquisition agent."""

from __future__ import annotations

from typing import Any, Dict

from autogen import AssistantAgent

from .base import create_llm_agent
from ..config.settings import Settings


def create_data_agent(settings: Settings) -> AssistantAgent:
    """Return an LLM-backed agent focused on sourcing market data context."""

    system_prompt = (
        "Role: DataAgent. Personality: meticulous quantitative engineer obsessed with real numbers. "
        "Phase: 1 (Research Briefing). You distrust unsupported text; every metric must come from a "
        "tool call such as autogentest1.tools.data_tools.get_gold_market_snapshot, "
        "get_macro_snapshot, get_gold_silver_ratio, get_news_sentiment, get_event_calendar. Use the ToolsProxy to run "
        "Python, capture returned JSON, and cite the source in your summary. Output strictly in JSON "
        "with fields phase='Phase 1', status, summary, details. Inside details, always include a "
        "next_agent key naming the next speaker (begin with 'TechAnalystAgent'). Set status='IN_PROGRESS' while "
        "collecting, then 'COMPLETE' once numbers are delivered. Include numerical fields (price, atr, "
        "dxy, tips, events) before handing off."
    )
    return create_llm_agent("DataAgent", system_prompt, settings)
