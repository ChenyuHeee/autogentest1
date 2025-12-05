"""Sentiment scoring utilities for news-based signals."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .news_ingest import NewsArticle, collect_news_articles

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_HISTORY_FILE = _OUTPUT_DIR / "sentiment_history.json"

_POSITIVE_TOKENS: Dict[str, float] = {
    "rally": 1.2,
    "support": 0.8,
    "demand": 0.7,
    "buy": 1.0,
    "surge": 1.1,
    "record": 0.9,
    "safe": 1.0,
    "haven": 1.1,
    "bull": 1.0,
    "bid": 0.8,
    "strong": 0.8,
    "gain": 0.8,
    "high": 0.6,
    "cut": 0.5,  # Rate cut is usually bullish for gold
    "easing": 0.7,
}

_NEGATIVE_TOKENS: Dict[str, float] = {
    "sell": 1.0,
    "slump": 1.2,
    "recession": 1.1, # Recession might be bullish (safe haven) or bearish (cash dash), context matters. Keeping as neg for now.
    "rate": 0.2,      # "Rate" itself is neutral, but often associated with hikes
    "hike": 1.2,
    "hawkish": 1.0,
    "risk": 0.5,      # Geopolitical risk is bullish, market risk is bearish. Ambiguous.
    "fear": 0.5,
    "outflow": 1.1,
    "pressure": 0.9,
    "dip": 0.8,
    "drop": 0.9,
    "weak": 0.8,
    "fall": 0.8,
}

_STOPWORDS: Sequence[str] = (
    "the",
    "and",
    "for",
    "with",
    "from",
    "gold",
    "price",
    "prices",
    "market",
    "metal",
    "metals",
    "will",
    "this",
    "that",
    "into",
    "after",
    "amid",
    "fed",
    "federal",
    "reserve",
    "usd",
    "dollar",
    "says",
    "saying",
    "over",
    "amid",
)

_WORD_PATTERN = re.compile(r"[A-Za-z']+")


def _token_scores(text: str) -> float:
    tokens = _WORD_PATTERN.findall(text.lower())
    if not tokens:
        return 0.0
    positive = sum(_POSITIVE_TOKENS.get(token, 0.0) for token in tokens)
    negative = sum(_NEGATIVE_TOKENS.get(token, 0.0) for token in tokens)
    net = positive - negative
    return net / max(len(tokens), 1)


def _extract_topics(articles: Iterable[NewsArticle], top_n: int = 6) -> List[str]:
    tokens: Counter[str] = Counter()
    stopwords = set(_STOPWORDS)
    for article in articles:
        text = f"{article.title} {article.summary}"
        for token in _WORD_PATTERN.findall(text.lower()):
            if len(token) <= 2 or token in stopwords:
                continue
            tokens[token] += 1
    return [term for term, _ in tokens.most_common(top_n)]


def _load_history(symbol: str) -> List[Dict[str, Any]]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        with _HISTORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            entries = data.get(symbol)
            if isinstance(entries, list):
                return entries
    except Exception:
        logger.debug("Failed to load sentiment history for symbol %s", symbol)
    return []


def _persist_history(symbol: str, timestamp: str, score: float) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if _HISTORY_FILE.exists():
        try:
            with _HISTORY_FILE.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)  # type: ignore[assignment]
        except Exception:
            payload = {}
    history = payload.get(symbol, []) if isinstance(payload.get(symbol), list) else []
    history.append({"timestamp": timestamp, "score": score})
    payload[symbol] = history[-60:]
    try:
        with _HISTORY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Unable to persist sentiment history: %s", exc)


def _compute_trend(symbol: str, current_score: float, generated_at: str) -> float:
    history = _load_history(symbol)
    previous_score = history[-1]["score"] if history else None
    _persist_history(symbol, generated_at, current_score)
    if previous_score is None:
        return 0.0
    return round(current_score - float(previous_score), 3)


def collect_sentiment_snapshot(
    symbol: str = "XAUUSD",
    *,
    news_api_key: str | None = None,
    alpha_vantage_api_key: str | None = None,
    simulation_date: str | None = None,
) -> Dict[str, Any]:
    """Return a sentiment snapshot derived from weighted news headlines.

    Args:
        symbol: Ticker symbol.
        news_api_key: Optional API key.
        alpha_vantage_api_key: Optional API key.
        simulation_date: If provided (YYYY-MM-DD), fetch historical news for backtesting.
    """

    articles = collect_news_articles(
        symbol,
        news_api_key=news_api_key,
        alpha_vantage_api_key=alpha_vantage_api_key,
        simulation_date=simulation_date,
    )

    if not articles:
        generated_at = simulation_date or datetime.now(timezone.utc).isoformat()
        if not simulation_date:
            _persist_history(symbol, generated_at, 0.0)
        return {
            "generated_at": generated_at,
            "symbol": symbol,
            "score": 0.0,
            "confidence": 0.0,
            "classification": "neutral",
            "score_trend": 0.0,
            "topics": [],
            "headlines": [],
            "positive_highlights": [],
            "negative_highlights": [],
        }

    enriched: List[Dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0
    weighted_abs = 0.0

    for article in articles:
        text = f"{article.title} {article.summary}".strip()
        score = _token_scores(text)
        total_weight += article.weight
        weighted_score += score * article.weight
        weighted_abs += abs(score) * article.weight
        enriched.append(
            {
                **article.to_dict(),
                "score": round(score, 3),
            }
        )

    sentiment = weighted_score / total_weight if total_weight else 0.0
    confidence = weighted_abs / total_weight if total_weight else 0.0

    if sentiment >= 0.05:
        classification = "bullish"
    elif sentiment <= -0.05:
        classification = "bearish"
    else:
        classification = "neutral"

    generated_at = simulation_date or datetime.now(timezone.utc).isoformat()
    trend = 0.0
    if not simulation_date:
        trend = _compute_trend(symbol, sentiment, generated_at)

    sorted_by_score = sorted(enriched, key=lambda item: item["score"], reverse=True)
    positive_highlights = sorted_by_score[:3]
    negative_highlights = list(reversed(sorted_by_score[-3:]))

    topics = _extract_topics(articles)

    return {
        "generated_at": generated_at,
        "symbol": symbol,
        "score": round(sentiment, 3),
        "confidence": round(confidence, 3),
        "classification": classification,
        "score_trend": trend,
        "topics": topics,
        "headlines": enriched,
        "positive_highlights": positive_highlights,
        "negative_highlights": negative_highlights,
    }


def export_sentiment_json(
    symbol: str = "XAUUSD",
    *,
    news_api_key: str | None = None,
    alpha_vantage_api_key: str | None = None,
) -> str:
    """Helper to export snapshot as JSON string."""

    snapshot = collect_sentiment_snapshot(
        symbol,
        news_api_key=news_api_key,
        alpha_vantage_api_key=alpha_vantage_api_key,
    )
    return json.dumps(snapshot, ensure_ascii=False)
