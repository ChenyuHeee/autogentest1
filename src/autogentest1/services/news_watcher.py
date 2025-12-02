"""Asynchronous watcher that escalates breaking news events to risk management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from ..agents.base import create_user_proxy
from ..agents.risk_agent import create_risk_manager_agent
from ..config.settings import Settings, get_settings
from ..services.sentiment import collect_sentiment_snapshot

logger = logging.getLogger(__name__)


@dataclass
class WatchEvent:
    """Structured representation of a high-priority news trigger."""

    title: str
    summary: str
    published: str
    score: float
    source: str

    def render(self) -> str:
        timestamp = self.published or datetime.now(timezone.utc).isoformat()
        return (
            f"[{timestamp}] {self.source}: {self.title}\n"
            f"Summary: {self.summary}\n"
            f"Headline score: {self.score:.3f}"
        )


class NewsWatcher:
    """Poll sentiment feeds and escalate when critical headlines emerge."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.keywords = {keyword.lower(): keyword for keyword in self.settings.news_watcher_keywords}
        self.poll_interval = self.settings.news_watcher_poll_seconds
        self.threshold = float(self.settings.news_watcher_vol_threshold)

    async def run(self, *, symbol: str | None = None) -> None:
        """Continuously poll for breaking events."""

        if not self.settings.news_watcher_enabled:
            logger.info("News watcher disabled via settings; exiting")
            return

        ticker = symbol or self.settings.default_symbol
        logger.info(
            "Starting NewsWatcher for %s (interval=%ss, keywords=%s)",
            ticker,
            self.poll_interval,
            ", ".join(self.keywords.values()),
        )
        while True:
            try:
                await self._poll_once(ticker)
            except Exception as exc:  # pragma: no cover - resilience
                logger.exception("News watcher poll failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self, symbol: str) -> None:
        snapshot = collect_sentiment_snapshot(symbol)
        critical = self._detect_events(snapshot.get("headlines", []))
        if not critical:
            return
        logger.warning("News watcher identified %d critical headlines", len(critical))
        await self._trigger_emergency_review(symbol, critical)

    def _detect_events(self, headlines: Iterable[Dict[str, Any]]) -> List[WatchEvent]:
        events: List[WatchEvent] = []
        for item in headlines:
            title = str(item.get("title", ""))
            summary = str(item.get("summary", ""))
            combined = f"{title} {summary}".lower()
            score = float(item.get("score", 0.0))
            if self.keywords and not any(keyword in combined for keyword in self.keywords):
                continue
            if abs(score) < self.threshold:
                continue
            events.append(
                WatchEvent(
                    title=title,
                    summary=summary,
                    published=str(item.get("published", "")),
                    score=score,
                    source=str(item.get("source", "channel")),
                )
            )
        return events

    async def _trigger_emergency_review(self, symbol: str, events: List[WatchEvent]) -> None:
        risk_agent = create_risk_manager_agent(self.settings)
        proxy = create_user_proxy(
            "NewsWatcherProxy",
            code_execution_config={"use_docker": False},
        )

        bulletin = "\n\n".join(event.render() for event in events)
        message = (
            "Emergency news escalation triggered. Provide an immediate risk assessment in JSON with the fields: "
            "phase, status, summary, details (include hedging_actions, exposure_guidance, monitoring_triggers).\n"
            f"Symbol: {symbol}. Headlines:\n{bulletin}\n"
            "Respond with status='COMPLETE' when guidance is actionable."
        )

        logger.info("Dispatching emergency review to RiskManagerAgent")
        try:
            risk_agent.initiate_chat(proxy, message=message)
        except Exception as exc:  # pragma: no cover - best effort escalation
            logger.exception("Emergency review failed to execute: %s", exc)


async def run_default_watcher() -> None:
    """Convenience entry point for CLI usage."""

    watcher = NewsWatcher()
    await watcher.run()
