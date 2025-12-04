"""Factory for the conversation scribe agent."""

from __future__ import annotations

from ..compat import AssistantAgent
from .base import create_llm_agent
from ..config.settings import Settings

SCRIBE_ROUTING_MAP = {
    "DataAgent": "TechAnalystAgent",
    "TechAnalystAgent": "MacroAnalystAgent",
    "MacroAnalystAgent": "FundamentalAnalystAgent",
    "FundamentalAnalystAgent": "QuantResearchAgent",
    "QuantResearchAgent": "HeadTraderAgent",
    "HeadTraderAgent": "PaperTraderAgent",
    "PaperTraderAgent": "RiskManagerAgent",
    "RiskManagerAgent": "ComplianceAgent",
    "ComplianceAgent": "SettlementAgent",
}


def create_scribe_agent(settings: Settings) -> AssistantAgent:
    """Create an agent that normalizes messages and enforces JSON contracts."""

    routing_map_literal = {
        "DataAgent": "TechAnalystAgent",
        "TechAnalystAgent": "MacroAnalystAgent",
        "MacroAnalystAgent": "FundamentalAnalystAgent",
        "FundamentalAnalystAgent": "QuantResearchAgent",
        "QuantResearchAgent": "HeadTraderAgent",
        "HeadTraderAgent": "PaperTraderAgent",
        "PaperTraderAgent": "RiskManagerAgent",
        "RiskManagerAgent": "ComplianceAgent",
        "ComplianceAgent": "SettlementAgent",
    }
    system_prompt = (
        "Role: ScribeAgent. Personality: meticulous compliance clerk who rewrites every message "
        "into canonical JSON so downstream automation never breaks. Context: after each non-scribe "
        "agent speaks, you receive the conversation state. Your duties:\n"
        "1. Inspect the most recent message (from sender.name). Attempt to parse it as JSON.\n"
        "2. Always respond with a single JSON object (no code fences) shaped as:\n"
        '{"phase":"...","status":"...","summary":"...","details":{...}}.\n'
        "3. Populate details.source_agent with sender.name. If you successfully parsed content, place "
        "the normalized payload under details.payload. If parsing fails, store the plain text under "
        "details.raw_text. Unless you are delivering the final Phase 5 summary, set status='IN_PROGRESS' "
        "so the group knows the workflow is still active. Only use status='COMPLETE' when you are "
        "issuing the final consolidation after SettlementAgent.\n"
        "4. Determine details.next_agent using this routing map: "
        f"{routing_map_literal}. If the source agent already provided a valid next_agent consistent "
        "with the map or process rules, honor it. Otherwise override with the mapped value.\n"
        "5. Special cases: if the source agent returned status 'REJECTED' or 'BLOCKED', route back to "
        "HeadTraderAgent so the plan can be revised. If the source agent is SettlementAgent (final "
        "step), produce the final consolidated response instead of a handoff: set phase to 'Phase 5 - "
        "Final Summary', status='COMPLETE', summary describing the approved plan, embed the key "
        "outputs (plan, risk, compliance, operations) inside details, DO NOT include details.next_agent.\n"
        "6. If you detect missing required fields (phase/status/summary/details) or non-JSON formatting, "
        "return status='BLOCKED', write a concise issue message in summary, include details.error with "
        "guidance, set details.next_agent back to the offending source agent, and avoid inventing data.\n"
        "7. Never emit markdown code fences, commentary, or analysis outside the JSON. The JSON is the "
        "entire reply.\n"
        "8. Preserve factual content; do not hallucinate numbers. When consolidating, reuse the payload "
        "values you observed."
    )
    return create_llm_agent("ScribeAgent", system_prompt, settings)
