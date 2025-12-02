"""Adapter wrapping Alpha Vantage news sentiment endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta
from importlib import import_module
from typing import Any, Dict, Optional

from .base import NewsDataAdapter


class AlphaVantageNewsAdapter(NewsDataAdapter):
    """Retrieve sentiment data from Alpha Vantage's NEWS_SENTIMENT endpoint."""

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key

    def fetch_sentiment(self, symbol: str, *, limit: int = 20) -> Dict[str, Any]:
        if not self._api_key:
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "error": "alpha_vantage_api_key_not_configured",
            }

        requests = import_module("requests")  # type: ignore[assignment]
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self._api_key,
            "limit": limit,
        }
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        payload.setdefault("generated_at", datetime.utcnow().isoformat())
        payload.setdefault("symbol", symbol)
        payload["fetched_articles"] = len(payload.get("feed", []))
        if payload.get("feed"):
            latest_item = max(payload["feed"], key=lambda item: item.get("time_published", ""), default=None)
            if latest_item and latest_item.get("time_published"):
                try:
                    ts = datetime.strptime(latest_item["time_published"], "%Y%m%dT%H%M%S")
                    payload["latest_article_ts"] = ts.isoformat()
                except ValueError:
                    payload["latest_article_ts"] = latest_item["time_published"]
        else:
            payload.setdefault("latest_article_ts", None)
        expiry = datetime.utcnow() + timedelta(minutes=5)
        payload["expires_at"] = expiry.isoformat()
        return payload
