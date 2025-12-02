"""Factory for the technical analysis agent."""

from __future__ import annotations

from autogen import AssistantAgent

from .base import create_llm_agent
from ..config.settings import Settings


def create_tech_analyst_agent(settings: Settings) -> AssistantAgent:
    """Create an agent that interprets technical indicators and price action."""

    system_prompt = (
        "Role: TechAnalystAgent. Personality: battle-hardened price-action hunter. Bias: price "
        "discounts everything; macro talk is noise. Phase: 1 (Research Briefing) following DataAgent's "
        "figures. You can execute Python code to interrogate OHLCV data—build quick calculations for "
        "RSI, SMA crosses, ATR-based stops, and visualize ranges when needed. Always run at least one "
        "code cell before issuing conclusions and reference the results succinctly. Provide entry, stop, "
        "target zones and timing windows. If macro narrative disagrees but chart screams reversal, defend "
        "the technical case firmly. Output JSON with phase='Phase 1', status, summary, details (include "
        "key levels array, code_snippets). Conclude with status='COMPLETE' and cue FundamentalAnalystAgent."
    )
    return create_llm_agent("TechAnalystAgent", system_prompt, settings)
