"""Risk-focused helper functions for ToolsProxy calls."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from ..config.settings import get_settings
from ..services.market_data import fetch_price_history
from ..services.risk import (
    DEFAULT_CORRELATION_TARGETS,
    CorrelationTarget,
    RiskLimits,
    build_risk_snapshot,
)

logger = logging.getLogger(__name__)


def compute_risk_profile(
    *,
    symbol: str = "XAUUSD",
    days: int = 90,
    position_oz: Optional[float] = None,
    pnl_today_millions: Optional[float] = None,
    correlation_targets: Optional[Sequence[CorrelationTarget]] = None,
    news_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured risk snapshot for the specified book."""

    settings = get_settings()
    history = fetch_price_history(symbol, days=days)

    limits = RiskLimits(
        max_position_oz=settings.max_position_oz,
        stress_var_millions=settings.stress_var_millions,
        daily_drawdown_pct=settings.daily_drawdown_pct,
    )

    snapshot = news_snapshot
    apply_adjustment = settings.risk_news_coupling_enabled
    if snapshot is None and settings.risk_news_coupling_enabled:
        try:
            from ..services.sentiment import collect_sentiment_snapshot

            snapshot = collect_sentiment_snapshot(
                symbol,
                news_api_key=settings.news_api_key,
                alpha_vantage_api_key=settings.alpha_vantage_api_key,
            )
        except Exception as exc:  # pragma: no cover - network path / optional dependency
            logger.warning("新闻情绪快照获取失败，将使用静态限额：%s", exc)
            snapshot = None
            apply_adjustment = False
    elif snapshot is not None:
        apply_adjustment = True

    return build_risk_snapshot(
        symbol,
        history,
        limits=limits,
        current_position_oz=position_oz if position_oz is not None else settings.default_position_oz,
        pnl_today_millions=pnl_today_millions if pnl_today_millions is not None else settings.pnl_today_millions,
        correlation_targets=correlation_targets or DEFAULT_CORRELATION_TARGETS,
        news_snapshot=snapshot,
        apply_news_adjustment=apply_adjustment,
        max_data_age_minutes=settings.market_data_max_age_minutes,
        data_provider=settings.data_provider,
        data_mode=settings.data_mode,
    )
