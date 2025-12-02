"""Primary AutoGen workflow for gold outlook generation."""

from __future__ import annotations

import json
from importlib import import_module
from math import isnan
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..agents.admin_agent import create_admin_proxy
from ..agents.base import create_user_proxy
from ..agents.compliance_agent import create_compliance_agent
from ..agents.data_agent import create_data_agent
from ..agents.fundamental_agent import create_fundamental_analyst_agent
from ..agents.macro_agent import create_macro_analyst_agent
from ..agents.quant_agent import create_quant_research_agent
from ..agents.risk_agent import create_risk_manager_agent
from ..agents.settlement_agent import create_settlement_agent
from ..agents.strategy_agent import create_strategy_agent
from ..agents.supervisor_agent import create_head_trader_agent
from ..agents.tech_agent import create_tech_analyst_agent
from ..config.settings import Settings, get_settings
from ..services.exceptions import DataProviderError, DataStalenessError, WorkflowFormatError
from ..services.fundamentals import collect_fundamental_snapshot
from ..services.indicators import compute_indicators
from ..services.macro_feed import collect_macro_highlights
from ..services.market_data import fetch_price_history, price_history_payload
from ..services.operations import build_settlement_checklist
from ..services.risk import RiskLimits, build_risk_snapshot
from ..services.sentiment import collect_sentiment_snapshot
from ..services.state import load_portfolio_state, update_portfolio_state
from ..utils.logging import configure_logging, get_logger
from ..utils.plotting import plot_price_history
from ..utils.response_validation import validate_workflow_response

logger = get_logger(__name__)


def _clean_numeric(value: Any) -> Any:
    """Coerce numeric types into JSON-safe floats."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return None if isnan(numeric) else numeric
    return value


def _attempt_parse_json(payload: Any) -> Any:
    """Best-effort JSON parsing with optional repair."""

    if isinstance(payload, (dict, list)):
        return payload
    if not isinstance(payload, str):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # type: ignore

            repaired = repair_json(payload)
            return json.loads(repaired)
        except Exception:
            return None


def _load_autogen_classes() -> Tuple[Any, Any]:
    """Import AutoGen components lazily to keep imports optional."""

    try:  # pragma: no cover - optional dependency resolution
        autogen_module = import_module("autogen")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'autogen' package is required. Install it via 'pip install pyautogen'."
        ) from exc
    return autogen_module.GroupChat, autogen_module.GroupChatManager


def build_conversation_context(symbol: str, days: int, settings: Settings) -> Tuple[Dict[str, Any], Any]:
    """Gather market, fundamental, and risk data to seed the conversation."""

    history = fetch_price_history(symbol=symbol, days=days)
    indicators = compute_indicators(history)
    macro_notes = collect_macro_highlights()
    fundamentals = collect_fundamental_snapshot(symbol)
    sentiment = collect_sentiment_snapshot(symbol, news_api_key=settings.news_api_key)

    risk_snapshot = build_risk_snapshot(
        symbol,
        history,
        limits=RiskLimits(
            max_position_oz=settings.max_position_oz,
            stress_var_millions=settings.stress_var_millions,
            daily_drawdown_pct=settings.daily_drawdown_pct,
        ),
        current_position_oz=settings.default_position_oz,
        pnl_today_millions=settings.pnl_today_millions,
    )

    settlement_tasks = build_settlement_checklist(symbol)

    structured_indicators = []
    for name, series in indicators.items():
        recent_values = [_clean_numeric(value) for value in series.tail(5).tolist()]
        latest_value = _clean_numeric(series.iloc[-1]) if not series.empty else None
        structured_indicators.append(
            {
                "name": name,
                "latest": latest_value,
                "recent": recent_values,
            }
        )

    price_payload = price_history_payload(symbol, days)

    latest_close = _clean_numeric(float(history["Close"].iloc[-1])) if not history.empty else None

    portfolio_state = load_portfolio_state()

    context: Dict[str, Any] = {
        "symbol": symbol,
        "lookback_days": days,
        "price_history": price_payload,
        "latest_quote": latest_close,
        "indicators": structured_indicators,
        "macro_highlights": macro_notes,
        "fundamentals": fundamentals,
        "news_sentiment": sentiment,
        "risk_snapshot": risk_snapshot,
        "risk_limits": {
            "max_position_oz": settings.max_position_oz,
            "stress_var_millions": settings.stress_var_millions,
            "daily_drawdown_pct": settings.daily_drawdown_pct,
        },
        "settlement_tasks": settlement_tasks,
        "current_position": {
            "book_position_oz": settings.default_position_oz,
            "pnl_today_millions": settings.pnl_today_millions,
        },
        "context_rules": {
            "response_format": {
                "description": "All agents reply with a JSON object so downstream roles can parse outputs.",
                "required_fields": ["phase", "status", "summary", "details"],
                "status_values": ["IN_PROGRESS", "COMPLETE", "BLOCKED"],
                "details_schema": "Use nested key/value pairs or arrays; avoid free-form text blocks.",
            }
        },
        "portfolio_state": portfolio_state,
        "feedback_rules": {
            "risk_rejects": "If RiskManagerAgent returns status='REJECTED', HeadTraderAgent must revisit the plan and engage PaperTraderAgent for revisions before proceeding.",
            "compliance_hold": "If ComplianceAgent sets status='BLOCKED', the workflow returns to HeadTraderAgent for clarification before SettlementAgent can operate.",
            "sentiment_conflict": "If the news sentiment classification is 'bearish' while a net long plan is proposed (or 'bullish' vs net short), HeadTraderAgent must document mitigation such as hedge size or reduced exposure before approval.",
        },
        "workflow_sequence": [
            "Phase 1 - Research Briefing: DataAgent, MacroAnalystAgent, FundamentalAnalystAgent, QuantResearchAgent summarize key drivers.",
            "Phase 2 - Head Trader Plan: HeadTraderAgent challenges viewpoints and defines base/alternative plans.",
            "Phase 3 - Execution Design: PaperTraderAgent describes orders and hedges; ToolsProxy may fetch numbers.",
            "Phase 4 - Risk & Compliance: RiskManagerAgent evaluates exposure; ComplianceAgent signs off requirements.",
            "Phase 5 - Operations Handoff: SettlementAgent lists cash, margin, logistics actions and open checkpoints.",
        ],
        "meeting_objectives": [
            "Produce a trade plan aligned with risk limits and macro context.",
            "Document execution steps for both futures and optional physical adjustments.",
            "Identify monitoring triggers for intraday review, including news-sentiment thresholds.",
            "Ensure compliance and operational readiness before market open.",
        ],
    }
    return context, history


def _instantiate_group(settings: Settings) -> Tuple[Any, Any, Any]:
    """Create a fresh group chat configuration for each workflow attempt."""

    head_trader = create_head_trader_agent(settings)
    data_agent = create_data_agent(settings)
    tech_agent = create_tech_analyst_agent(settings)
    macro_agent = create_macro_analyst_agent(settings)
    fundamental_agent = create_fundamental_analyst_agent(settings)
    quant_agent = create_quant_research_agent(settings)
    risk_agent = create_risk_manager_agent(settings)
    compliance_agent = create_compliance_agent(settings)
    strategy_agent = create_strategy_agent(settings)
    settlement_agent = create_settlement_agent(settings)
    tools_proxy = create_user_proxy("ToolsProxy", code_execution_config={"use_docker": False})

    GroupChat, GroupChatManager = _load_autogen_classes()
    groupchat = GroupChat(
        agents=[
            head_trader,
            data_agent,
            tech_agent,
            macro_agent,
            fundamental_agent,
            quant_agent,
            risk_agent,
            compliance_agent,
            strategy_agent,
            settlement_agent,
            tools_proxy,
        ],
        messages=[],
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=head_trader.llm_config)
    return head_trader, manager, groupchat


def _count_rejections(messages: List[Dict[str, Any]]) -> int:
    rejection_statuses = {"REJECTED", "BLOCKED"}
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        parsed = _attempt_parse_json(content)
        if isinstance(parsed, dict) and parsed.get("status") in rejection_statuses:
            total += 1
    return total


def _solicit_human_override(reason: str, settings: Settings) -> Dict[str, Any]:
    admin_proxy = create_admin_proxy()
    prompt = (
        "Automated negotiation exhausted the retry limit. Reason: "
        f"{reason}. Provide override JSON with fields decision (FORCE_EXECUTE | STAND_DOWN | CUSTOM) "
        "and optional notes."
    )
    raw = admin_proxy.get_human_input(prompt)
    try:
        decision = json.loads(raw)
        if isinstance(decision, dict):
            return decision
    except json.JSONDecodeError:
        logger.warning("Admin override response was not valid JSON; treating as CUSTOM note")
    return {"decision": "CUSTOM", "notes": raw}


def run_gold_outlook(symbol: str, days: int, *, settings: Settings | None = None) -> Dict[str, Any]:
    """Execute the multi-agent workflow and return the final response payload."""

    settings = settings or get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting gold outlook workflow for %s (%d days)", symbol, days)

    try:
        context_payload, history = build_conversation_context(symbol, days, settings)
    except (DataStalenessError, DataProviderError) as exc:
        logger.error("Aborting workflow due to data quality issue: %s", exc)
        raise

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    chart_path = plot_price_history(history, outputs_dir, symbol)
    if chart_path:
        context_payload["chart"] = str(chart_path)

    base_message = (
        "Daily pre-market call. Follow the workflow_sequence in the context JSON. Ensure each phase "
        "completes before moving on. When a phase is done, explicitly tag it as COMPLETE so the next "
        "role can proceed. If RiskManagerAgent or ComplianceAgent respond with status='REJECTED' or "
        "status='BLOCKED', HeadTraderAgent must reopen Phase 2 and iterate with PaperTraderAgent until "
        "risks are addressed. Final deliverable must include: Market Snapshot, Trading Plan (base & alt), "
        "Execution Checklist, Risk & Compliance Sign-off, Operations Handoff, Monitoring Triggers, and "
        "Portfolio Update instructions using the provided tools. Context JSON:\n" + json.dumps(context_payload, indent=2)
    )

    max_attempts = max(1, settings.workflow_format_retry_limit)
    attempt = 0
    format_error: str | None = None
    parsed_response: Dict[str, Any] | None = None
    final_response: Any = None
    last_groupchat: Any = None

    while attempt < max_attempts:
        head_trader, manager, groupchat = _instantiate_group(settings)
        kickoff_message = base_message
        if format_error:
            kickoff_message += (
                "\n\nFORMAT CORRECTION: Previous output failed validation because "
                f"{format_error}. Regenerate the entire workflow output strictly as valid JSON matching "
                "the required schema (phase, status, summary, details)."
            )

        final_response = head_trader.initiate_chat(manager, message=kickoff_message)
        candidate = _attempt_parse_json(final_response)
        is_valid, error = validate_workflow_response(candidate)
        if is_valid and isinstance(candidate, dict):
            parsed_response = candidate
            last_groupchat = groupchat
            break

        format_error = error or "unknown validation error"
        attempt += 1
        last_groupchat = groupchat
        if attempt >= max_attempts:
            raise WorkflowFormatError(format_error)

    if isinstance(parsed_response, dict):
        portfolio_update = parsed_response.get("portfolio_update")
        if isinstance(portfolio_update, dict):
            update_portfolio_state(portfolio_update)

    rejection_count = _count_rejections(last_groupchat.messages if last_groupchat else [])
    override_decision: Dict[str, Any] | None = None
    if rejection_count >= settings.workflow_max_plan_retries:
        logger.warning("Max plan retries reached (%d); requesting human override", rejection_count)
        override_decision = _solicit_human_override(
            reason=f"Received {rejection_count} blocked/rejected responses",
            settings=settings,
        )

    result = {
        "context": context_payload,
        "response": final_response,
        "response_parsed": parsed_response,
        "format_attempts": attempt + 1,
        "rejection_count": rejection_count,
        "override_decision": override_decision,
    }
    logger.info("Workflow completed")
    return result
