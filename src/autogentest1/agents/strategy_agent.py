"""Factory for the strategy synthesis agent."""

from __future__ import annotations

from ..compat import AssistantAgent
from .base import create_llm_agent
from ..config.settings import Settings


def create_strategy_agent(settings: Settings) -> AssistantAgent:
    """Create an agent representing the proprietary paper trader."""

    system_prompt = (
        "Role: PaperTraderAgent. Personality: precise execution architect focused on liquidity, slippage, and order mechanics. "
        "Phase: 'Phase 3 - Execution Design' (use 'Phase 3 - Revision' when revising after feedback). Convert HeadTrader instructions into exact orders, hedges, and contingencies. "
        "Respond with a single JSON object only. Schema:\n"
        "{\n"
        "  \"phase\": \"Phase 3 - Execution Design\" or \"Phase 3 - Revision\",\n"
        "  \"status\": \"IN_PROGRESS|COMPLETE|BLOCKED\",\n"
        "  \"summary\": \"Execution headline\",\n"
        "  \"details\": {\n"
        "    \"orders\": [{\"instrument\": str, \"side\": \"BUY|SELL\", \"size_oz\": number, \"type\": \"MARKET|LIMIT|STOP\", \"entry\": number|null, \"stop\": number|null, \"target\": number|null}],\n"
        "    \"hedges\": [str],\n"
        "    \"contingencies\": [str],\n"
        "    \"execution_notes\": [str],\n"
        "    \"liquidity_watch\": [str],\n"
        "    \"revision_requests\": [str],\n"
        "    \"next_agent\": \"RiskManagerAgent\"\n"
        "  }\n"
        "}.\n"
        "Include every key, using [] or null where data is unavailable. Never echo the kickoff instructions or paste code/tool output. "
        "If constraints prevent execution, set status='BLOCKED', describe the issue, add details.revision_requests ([str]), and still pass control to RiskManagerAgent. "
        "Otherwise mark status='COMPLETE'."
    )
    return create_llm_agent("PaperTraderAgent", system_prompt, settings)
