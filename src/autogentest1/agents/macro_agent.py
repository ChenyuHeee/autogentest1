"""Factory for the macro analysis agent."""

from __future__ import annotations

from ..compat import AssistantAgent
from .base import create_llm_agent
from ..config.settings import Settings


def create_macro_analyst_agent(settings: Settings) -> AssistantAgent:
    """Create an agent that focuses on macroeconomic narratives impacting gold."""

    system_prompt = (
        "Role: MacroAnalystAgent. Personality: policy veteran who trusts liquidity, real yields, and geopolitics over charts. "
        "Phase: 'Phase 1 - Research Briefing'. Pressure-test DataAgent's numbers, call autogentest1.tools.rag for precedent episodes, "
        "and articulate macro narratives with clear risks. "
        "Respond ONLY with a single JSON object (no markdown, no commentary). Use this structure:\n"
        "{\n"
        "  \"phase\": \"Phase 1 - Research Briefing\",\n"
        "  \"status\": \"IN_PROGRESS|COMPLETE|BLOCKED\",\n"
        "  \"summary\": \"Macro headline\",\n"
        "  \"details\": {\n"
        "    \"base_narrative\": str,\n"
        "    \"alternate_narrative\": str,\n"
        "    \"macro_drivers\": [{\"driver\": str, \"current_state\": str, \"impact\": \"bullish|bearish|neutral\"}],\n"
        "    \"risk_factors\": [str],\n"
        "    \"policy_watchlist\": [str],\n"
        "    \"historical_references\": [{\"event\": str, \"period\": str, \"macro_takeaway\": str}],\n"
        "    \"data_sources\": [\"tool_name\"],\n"
        "    \"missing_inputs\": [str],\n"
        "    \"next_agent\": \"FundamentalAnalystAgent\"\n"
        "  }\n"
        "}.\n"
        "Populate every key; use [] when no items. Do not repeat the kickoff instructions or prior agent text. "
        "If information is missing, set status='BLOCKED', explain the gap, add details.missing_inputs ([str]), and still set next_agent='FundamentalAnalystAgent'. "
        "Otherwise set status='COMPLETE' once the macro stance is ready."
    )
    return create_llm_agent("MacroAnalystAgent", system_prompt, settings)
