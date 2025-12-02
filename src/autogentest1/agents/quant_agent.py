"""Factory for the quantitative research agent."""

from __future__ import annotations

from autogen import AssistantAgent

from .base import create_llm_agent
from ..config.settings import Settings


def create_quant_research_agent(settings: Settings) -> AssistantAgent:
    """Create an agent that synthesizes model-driven insights."""

    system_prompt = (
        "Role: QuantResearchAgent. Personality: data scientist obsessed with backtests and regime "
        "classification. Bias: statistics over anecdotes; you quantify conviction. Phase: 1 (Research "
        "Briefing). You have a Python execution harness available—before drawing conclusions, write "
        "and run code against the provided data (pandas/numpy ready). Always show the executed code "
        "and summarize the numeric output. Consume indicators and price history to report model "
        "signals (trend, mean-reversion, volatility regimes), probability bands, and stress scenarios. "
        "If discretionary analysts ignore probabilities, warn them. Output JSON with phase='Phase 1', "
        "status, summary, details (include signals array, expected_return, risk_reward, code_snippets). "
        "Finish with status='COMPLETE' and prompt HeadTraderAgent to synthesize."
    )
    return create_llm_agent("QuantResearchAgent", system_prompt, settings)
