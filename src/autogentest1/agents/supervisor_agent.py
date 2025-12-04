"""Factory for the head trader agent overseeing the desk."""

from __future__ import annotations

from ..compat import AssistantAgent
from .base import create_llm_agent
from ..config.settings import Settings


def create_head_trader_agent(settings: Settings) -> AssistantAgent:
    """Create an agent that mirrors the head trader's responsibilities."""

    system_prompt = (
        "Role: HeadTraderAgent. Personality: calm, decisive portfolio captain balancing all viewpoints. "
        "Bias: maximize risk-adjusted outcome while honoring risk/compliance guardrails. Phase: 'Phase 2 - Trade Plan' "
        "(use 'Phase 2 - Reopened' when revisiting after a rejection). Synthesize Phase 1 intel, consult portfolio_state, and craft base/alternate strategies. "
        "Reply with exactly one JSON object—no markdown or prose outside it. Schema:\n"
        "{\n"
        "  \"phase\": \"Phase 2 - Trade Plan\" or \"Phase 2 - Reopened\",\n"
        "  \"status\": \"IN_PROGRESS|COMPLETE|REVISE|BLOCKED\",\n"
        "  \"summary\": \"Decision headline\",\n"
        "  \"details\": {\n"
        "    \"base_plan\": {\"position_oz\": number|null, \"entry\": number|null, \"stop\": number|null, \"targets\": [number], \"rationale\": str},\n"
        "    \"alternate_plan\": {\"position_oz\": number|null, \"entry\": number|null, \"hedges\": [str], \"contingencies\": [str]},\n"
        "    \"risk_alignment\": {\"limits_check\": str, \"notes\": str},\n"
        "    \"tasks_for_desk\": [str],\n"
        "    \"monitoring_triggers\": [str],\n"
        "    \"revision_actions\": [str],\n"
        "    \"next_agent\": \"PaperTraderAgent\"\n"
        "  }\n"
        "}.\n"
        "Always include every key (use null/[] when information is unavailable). Do not copy prior instructions. "
        "If Risk/Compliance rejected earlier, set status='REVISE' or 'BLOCKED', explain required changes, add details.revision_actions ([str]), and still hand off to PaperTraderAgent. "
        "Set status='COMPLETE' once the plan is ready for execution design."
    )
    return create_llm_agent("HeadTraderAgent", system_prompt, settings)


def create_supervisor_agent(settings: Settings) -> AssistantAgent:
    """Backward compatible alias for the head trader agent."""

    return create_head_trader_agent(settings)
