"""News ingestion utilities inspired by BettaFish multi-source gathering."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

try:  # pragma: no cover - optional dependency used when available
    import feedparser  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    feedparser = None  # type: ignore

try:  # pragma: no cover - requests shipped with many deps but optional
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_CACHE_FILE = _OUTPUT_DIR / "news_cache.json"
_ALPHA_CACHE_FILE = _OUTPUT_DIR / "alpha_news_cache.json"
_NEWS_ARCHIVE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rag" / "news_archive.json"
_ALPHA_CACHE_TTL = timedelta(hours=4)

_FALLBACK_HEADLINES: Sequence[Dict[str, Any]] = (
    {
        "source": "Bloomberg",
        "title": "Central banks extend gold buying spree as haven demand stays firm",
        "summary": "Official sector demand underpins bullion despite dollar strength.",
        "url": "https://www.bloomberg.com/",
        "weight": 1.4,
        "published": datetime.now(timezone.utc).isoformat(),
    },
    {
        "source": "Reuters",
        "title": "Gold slips as investors weigh Fed path while war risks linger",
        "summary": "Bullion eases but safe-haven bid remains with geopolitical tension.",
        "url": "https://www.reuters.com/",
        "weight": 1.2,
        "published": datetime.now(timezone.utc).isoformat(),
    },
    {
        "source": "Kitco",
        "title": "Physical premiums in Asia stay elevated on retail buying surge",
        "summary": "Asian buyers take advantage of price dips, keeping premiums strong.",
        "url": "https://www.kitco.com/",
        "weight": 1.1,
        "published": datetime.now(timezone.utc).isoformat(),
    },
)

_RSS_SOURCES: Sequence[Dict[str, Any]] = (
    {
        "id": "reuters-markets",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "weight": 1.3,
    },
    {
        "id": "marketwatch-gold",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
        "weight": 1.1,
    },
    {
        "id": "kitco-gold",
        "url": "https://www.kitco.com/rss/feed/",
        "weight": 1.0,
    },
)


def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _parse_alpha_timestamp(value: Optional[str]) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        # Alpha Vantage sometimes returns ISO format with timezone offset
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            return datetime.now(timezone.utc).isoformat()


def _load_alpha_cache(symbol: str) -> List[NewsArticle]:
    if not _ALPHA_CACHE_FILE.exists():
        return []
    try:
        with _ALPHA_CACHE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    entry = payload.get(symbol)
    if not isinstance(entry, dict):
        return []
    fetched_at = entry.get("fetched_at")
    try:
        fetched_dt = datetime.fromisoformat(str(fetched_at))
    except Exception:
        return []
    if datetime.now(timezone.utc) - fetched_dt > _ALPHA_CACHE_TTL:
        return []
    articles = entry.get("articles")
    if not isinstance(articles, list):
        return []
    result: List[NewsArticle] = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        try:
            result.append(NewsArticle(**item))
        except TypeError:
            continue
    return result


def _save_alpha_cache(symbol: str, articles: Iterable[NewsArticle]) -> None:
    _ensure_output_dir()
    try:
        with _ALPHA_CACHE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    payload[symbol] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "articles": [article.to_dict() for article in articles],
    }
    try:
        with _ALPHA_CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Unable to persist Alpha Vantage cache: %s", exc)


def _fetch_alpha_vantage_articles(symbol: str, api_key: Optional[str]) -> List[NewsArticle]:
    if not api_key or requests is None:
        return []
    cached = _load_alpha_cache(symbol)
    if cached:
        return cached
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "limit": "30",
        "apikey": api_key,
    }
    try:  # pragma: no cover - network path
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        logger.warning("Alpha Vantage NEWS_SENTIMENT request failed: %s", exc)
        return cached

    if not isinstance(payload, dict):
        return cached

    if "Note" in payload or "Information" in payload:
        logger.warning("Alpha Vantage API limit notice: %s", payload.get("Note") or payload.get("Information"))
        return cached

    feed = payload.get("feed")
    if not isinstance(feed, list):
        return cached

    articles: List[NewsArticle] = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        summary = (item.get("summary") or item.get("overall_sentiment_label") or "").strip()
        url = item.get("url") or item.get("source") or ""
        time_published = _parse_alpha_timestamp(item.get("time_published"))
        score = float(item.get("overall_sentiment_score") or 0.0)
        weight = max(0.6, 1.0 + abs(score))
        source = (item.get("source") or item.get("source_domain") or "AlphaVantage").strip()
        articles.append(
            NewsArticle(
                source=source,
                title=title,
                summary=summary,
                url=url,
                published=time_published,
                weight=weight,
            )
        )

    if articles:
        _save_alpha_cache(symbol, articles)
    return articles or cached


@dataclass
class NewsArticle:
    """Normalized news article for downstream processing."""

    source: str
    title: str
    summary: str
    url: str
    published: str
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "published": self.published,
            "weight": self.weight,
        }


def _fetch_newsapi_articles(news_api_key: Optional[str], query: str, page_size: int = 10) -> List[NewsArticle]:
    if not news_api_key or requests is None:
        return []
    try:  # pragma: no cover - network path
        params = {
            "apiKey": news_api_key,
            "q": query,
            "language": "en",
            "pageSize": page_size,
            "sortBy": "publishedAt",
        }
        response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles")
        normalized: List[NewsArticle] = []
        if isinstance(articles, list):
            for item in articles:
                title = (item.get("title") or "").strip()
                description = (item.get("description") or item.get("content") or "").strip()
                source_payload = item.get("source") or {}
                source_name = (source_payload.get("name") if isinstance(source_payload, dict) else source_payload) or "Unknown"
                published_at = item.get("publishedAt") or datetime.now(timezone.utc).isoformat()
                url = item.get("url") or ""
                if not title:
                    continue
                normalized.append(
                    NewsArticle(
                        source=str(source_name),
                        title=title,
                        summary=description,
                        url=url,
                        published=published_at,
                        weight=1.5,
                    )
                )
        return normalized
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to fetch NewsAPI headlines: %s", exc)
        return []


def _download_feed(url: str) -> bytes:
    try:  # pragma: no cover - network path
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "autogentest1/1.0"})
            response.raise_for_status()
            return response.content
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to download RSS feed %s: %s", url, exc)
        return b""


def _fetch_rss_articles() -> List[NewsArticle]:
    if feedparser is None:
        return []
    items: List[NewsArticle] = []
    for config in _RSS_SOURCES:
        try:  # pragma: no cover - network path
            payload = _download_feed(config["url"])
            if not payload:
                continue
            feed = feedparser.parse(payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse RSS feed %s: %s", config["id"], exc)
            continue
        entries = getattr(feed, "entries", []) or []
        for entry in entries[:10]:
            title = (getattr(entry, "title", "") or "").strip()
            if not title:
                continue
            summary = (getattr(entry, "summary", "") or getattr(entry, "description", "") or "").strip()
            link = getattr(entry, "link", "")
            published = getattr(entry, "published", None) or datetime.now(timezone.utc).isoformat()
            items.append(
                NewsArticle(
                    source=str(config["id"]),
                    title=title,
                    summary=summary,
                    url=link,
                    published=published,
                    weight=float(config.get("weight", 1.0)),
                )
            )
    return items


def _load_cache() -> List[Dict[str, Any]]:
    if not _CACHE_FILE.exists():
        return []
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                return payload
    except Exception:
        logger.debug("Failed to load news cache; regenerating")
    return []


def _save_cache(articles: Iterable[NewsArticle]) -> None:
    _ensure_output_dir()
    serialized = [article.to_dict() for article in articles]
    try:
        with _CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(serialized, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Unable to persist news cache: %s", exc)


def _load_historical_news(date_str: str) -> List[NewsArticle]:
    """Load news from the local archive for a specific date (YYYY-MM-DD)."""
    if not _NEWS_ARCHIVE_FILE.exists():
        return []
    try:
        with _NEWS_ARCHIVE_FILE.open("r", encoding="utf-8") as handle:
            archive = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load news archive: %s", exc)
        return []

    # Try exact match
    articles_data = archive.get(date_str)
    if not articles_data:
        # Fallback: try to find the closest previous date if exact match fails?
        # For now, strict matching.
        return []

    articles: List[NewsArticle] = []
    for item in articles_data:
        try:
            articles.append(NewsArticle(**item))
        except TypeError:
            continue
    return articles


def collect_news_articles(
    symbol: str,
    *,
    news_api_key: Optional[str] = None,
    alpha_vantage_api_key: Optional[str] = None,
    limit: int = 30,
    simulation_date: Optional[str] = None,
) -> List[NewsArticle]:
    """Fetch news articles for the given symbol from multiple sources with caching.

    Args:
        symbol: Ticker symbol (e.g. XAUUSD).
        news_api_key: API key for NewsAPI.
        alpha_vantage_api_key: API key for Alpha Vantage.
        limit: Max articles to return.
        simulation_date: If provided (YYYY-MM-DD), fetch from historical archive instead of live APIs.
    """

    if simulation_date:
        return _load_historical_news(simulation_date)

    candidate_articles: List[NewsArticle] = []

    rss_articles = _fetch_rss_articles()
    candidate_articles.extend(rss_articles)

    query = f"gold OR {symbol} OR precious metals"
    api_articles = _fetch_newsapi_articles(news_api_key, query=query, page_size=limit)
    candidate_articles.extend(api_articles)

    alpha_articles = _fetch_alpha_vantage_articles(symbol, alpha_vantage_api_key)
    candidate_articles.extend(alpha_articles)

    if not candidate_articles:
        cached = _load_cache()
        if cached:
            return [NewsArticle(**item) for item in cached if isinstance(item, dict)]
        return [NewsArticle(**item) for item in _FALLBACK_HEADLINES]

    deduped: Dict[str, NewsArticle] = {}
    for article in candidate_articles:
        key = article.title.lower()
        if key not in deduped:
            deduped[key] = article

    ordered_articles = sorted(deduped.values(), key=lambda item: item.published, reverse=True)[:limit]
    _save_cache(ordered_articles)
    return ordered_articles


__all__ = [
    "NewsArticle",
    "collect_news_articles",
]
