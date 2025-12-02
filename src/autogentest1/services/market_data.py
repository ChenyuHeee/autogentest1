"""Market data retrieval utilities."""

from __future__ import annotations

import math
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import requests_cache
except ImportError:  # pragma: no cover - dependency enforced via pyproject
    requests_cache = None

from ..config.settings import get_settings
from ..utils.logging import get_logger
from ..utils.serialization import df_to_records
from .data_providers import IBKRAdapter, MarketDataAdapter, YahooFinanceAdapter
from .exceptions import DataProviderError, DataStalenessError
from .indicators import compute_indicators

logger = get_logger(__name__)

_CACHE_SESSION: Optional[Session] = None
_CACHE_SETTINGS: Dict[str, Any] = {}


def _cache_path() -> str:
    cache_root = Path(tempfile.gettempdir()) / "autogentest1"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(cache_root / "market_data_cache")


def _cached_session(settings) -> Optional[Session]:
    if requests_cache is None:
        return None

    global _CACHE_SESSION, _CACHE_SETTINGS
    config_key = {
        "expire": settings.market_data_cache_minutes,
        "retry_total": settings.market_data_retry_total,
        "retry_backoff": settings.market_data_retry_backoff,
    }
    if _CACHE_SESSION is not None and _CACHE_SETTINGS == config_key:
        return _CACHE_SESSION

    expire_after = timedelta(minutes=settings.market_data_cache_minutes)
    session = requests_cache.CachedSession(
        cache_name=_cache_path(),
        backend="sqlite",
        expire_after=expire_after,
        ignored_parameters=["_ts"],
    )
    retry = Retry(
        total=settings.market_data_retry_total,
        backoff_factor=settings.market_data_retry_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _CACHE_SESSION = session
    _CACHE_SETTINGS = config_key
    return session


def _mock_price_history(symbol: str, days: int) -> pd.DataFrame:
    periods = max(days, 30)
    idx = pd.bdate_range(end=datetime.utcnow(), periods=periods)
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    base_price = 1850 + (seed % 200) * 0.5
    drift = rng.normal(0.05, 0.02)
    shocks = rng.normal(0, 1.8, size=len(idx))
    closes = base_price + np.cumsum(drift + shocks)
    highs = closes + np.abs(rng.normal(1.2, 0.6, size=len(idx)))
    lows = closes - np.abs(rng.normal(1.3, 0.5, size=len(idx)))
    opens = closes + rng.normal(0, 0.8, size=len(idx))
    volumes = np.full(len(idx), 100_000 + int(seed % 50_000))
    data = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Adj Close": closes,
            "Volume": volumes,
        },
        index=idx,
    )
    data.index.name = "Date"
    logger.warning("Using mock market data for %s (%d days)", symbol, days)
    return data.tail(days)


def _retry_fetch(
    adapter: MarketDataAdapter,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    session: Optional[Session],
    settings,
) -> pd.DataFrame:
    attempts = max(1, settings.market_data_retry_total + 1)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return adapter.fetch_price_history(symbol, start=start, end=end, session=session)
        except DataProviderError:
            raise
        except Exception as exc:  # pragma: no cover - exercised via live fetch in integration runs
            last_error = exc
            wait_seconds = max(0.0, settings.market_data_retry_backoff) * math.pow(2, attempt - 1)
            logger.warning(
                "Market data fetch attempt %d/%d failed for %s: %s", attempt, attempts, symbol, exc
            )
            if attempt < attempts and wait_seconds:
                time.sleep(min(wait_seconds, 30))

    raise DataProviderError(f"Failed to fetch market data for {symbol}: {last_error}") from last_error


def _select_adapter(provider_name: str) -> MarketDataAdapter:
    provider = provider_name.lower().strip()
    if provider in {"yfinance", "yahoo"}:
        return YahooFinanceAdapter()
    if provider in {"ibkr", "interactivebrokers"}:
        return IBKRAdapter()
    raise DataProviderError(f"Unsupported market data provider: {provider_name}")


def _ensure_freshness(history: pd.DataFrame, *, max_age_minutes: int) -> None:
    if history.empty:
        return
    last_index = history.index.max()
    if last_index is None:
        return
    if isinstance(last_index, pd.Timestamp):
        last_dt = last_index.to_pydatetime()
    else:
        last_dt = datetime.fromisoformat(str(last_index))
    age = datetime.utcnow() - last_dt
    if age > timedelta(minutes=max_age_minutes):
        raise DataStalenessError(
            f"Latest data point is {int(age.total_seconds() // 60)} minutes old (limit {max_age_minutes} min)."
        )


def fetch_price_history(symbol: str, days: int = 14) -> pd.DataFrame:
    """Fetch recent price history using the configured data adapter."""

    settings = get_settings()
    mode = (settings.data_mode or "live").lower()

    if mode not in {"live", "hybrid", "mock"}:
        logger.warning("Unknown data_mode '%s', defaulting to 'live'", mode)
        mode = "live"

    if mode == "mock":
        return _mock_price_history(symbol, days)

    end = datetime.utcnow()
    start = end - timedelta(days=days * 2)
    adapter = _select_adapter(settings.data_provider)
    session = _cached_session(settings)

    logger.info("Downloading price history for %s via %s", symbol, settings.data_provider)

    try:
        data = _retry_fetch(adapter, symbol, start=start, end=end, session=session, settings=settings)
    except DataProviderError as exc:
        if mode == "hybrid":
            logger.error("Market data provider failed, falling back to mock data: %s", exc)
            return _mock_price_history(symbol, days)
        raise

    if data.empty:
        logger.warning("No price data retrieved for %s", symbol)
        if mode == "hybrid":
            return _mock_price_history(symbol, days)
        return data

    data = data.tail(days)
    data.index = pd.to_datetime(data.index)
    _ensure_freshness(data, max_age_minutes=settings.market_data_max_age_minutes)
    return data


def latest_quote(symbol: str) -> Optional[float]:
    """Return the latest close price for quick status updates."""

    history = fetch_price_history(symbol, days=1)
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


def price_history_payload(symbol: str, days: int = 14) -> Dict[str, Any]:
    """Return a JSON-serializable payload describing price history."""

    history = fetch_price_history(symbol, days=days)
    return {
        "symbol": symbol,
        "lookback_days": days,
        "frequency": "daily",
        "records": df_to_records(history.tail(days), include_index=True),
    }


def market_snapshot(symbol: str, days: int = 30) -> Dict[str, Any]:
    """Return key market metrics such as latest price and volatility."""

    history = fetch_price_history(symbol, days=days)
    indicators = compute_indicators(history)
    latest_close = float(history["Close"].iloc[-1]) if not history.empty else None

    atr_series = indicators.get("atr_14")
    atr_latest = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else None

    return {
        "symbol": symbol,
        "latest_close": latest_close,
        "atr_14": atr_latest,
        "history_sample": df_to_records(history.tail(5), include_index=True),
    }
