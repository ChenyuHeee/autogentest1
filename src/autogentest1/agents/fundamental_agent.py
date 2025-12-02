"""Factory for the fundamental analysis agent."""

from __future__ import annotations

from autogen import AssistantAgent

from .base import create_llm_agent
from ..config.settings import Settings


def create_fundamental_analyst_agent(settings: Settings) -> AssistantAgent:
    """Create an agent focusing on supply, demand, and flow dynamics."""

    system_prompt = (
        "Role: FundamentalAnalystAgent. Personality: forensic supply/demand detective focused on "
        "physical flows. Bias: central bank buying, COMEX inventories, gold/silver ratio, and seasonal "
        "jewellery demand outweigh short-term price noise. Phase: 1 (Research Briefing). Use data from "
        "DataAgent (and request gold/silver ratio, news sentiment, or macro tools if missing) to judge whether "
        "physical flows confirm or refute the trade idea. Highlight structural shifts, inventory stress, ETF "
        "flows, and show how the latest news sentiment reinforces or contradicts positioning. Produce JSON with "
        "phase='Phase 1', status, summary, details (include supply_demand array) and set details.next_agent='QuantResearchAgent'. "
        "After assigning next_agent, mark status='COMPLETE'."
    )
    return create_llm_agent("FundamentalAnalystAgent", system_prompt, settings)
