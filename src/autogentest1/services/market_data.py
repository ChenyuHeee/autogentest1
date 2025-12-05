"""Market data retrieval utilities."""

from __future__ import annotations

import math
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from .data_providers import (
    AlphaVantageFXAdapter,
    CacheConfig,
    DataSourceAdapter,
    IBKRAdapter,
    PolygonAdapter,
    RetryConfig,
    TanshuGoldAdapter,
    TwelveDataAdapter,
    YahooFinanceAdapter,
)
from .data_router import DataSourceRouter
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
    now_utc = datetime.now(timezone.utc)
    idx = pd.bdate_range(end=now_utc, periods=periods, tz="UTC").tz_localize(None)
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
    logger.warning("使用模拟行情：%s（%d天）", symbol, days)
    return data.tail(days)


def _retry_fetch(
    fetcher: Callable[[], pd.DataFrame],
    *,
    symbol: str,
    provider_label: str,
    settings,
) -> pd.DataFrame:
    attempts = max(1, settings.market_data_retry_total + 1)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return fetcher()
        except DataProviderError:
            raise
        except Exception as exc:  # pragma: no cover - exercised via live fetch in integration runs
            last_error = exc
            wait_seconds = max(0.0, settings.market_data_retry_backoff) * math.pow(2, attempt - 1)
            logger.warning(
                "行情抓取失败（第%d/%d次，标的=%s，来源=%s）：%s",
                attempt,
                attempts,
                symbol,
                provider_label,
                exc,
            )
            if attempt < attempts and wait_seconds:
                time.sleep(min(wait_seconds, 30))

    raise DataProviderError(
        f"多次尝试后仍无法获取行情 {symbol}（来源 {provider_label}）：{last_error}"
    ) from last_error


def _ensure_freshness(history: pd.DataFrame, *, max_age_minutes: int) -> Optional[float]:
    """Validate market data freshness and annotate the dataframe."""

    if history.empty:
        history.attrs.setdefault("data_age_minutes", None)
        return None
    last_index = history.index.max()
    if last_index is None:
        history.attrs.setdefault("data_age_minutes", None)
        return None
    if isinstance(last_index, pd.Timestamp):
        last_dt = last_index.to_pydatetime()
    else:
        last_dt = datetime.fromisoformat(str(last_index))
    now_utc = datetime.now(timezone.utc)
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    else:
        last_dt = last_dt.astimezone(timezone.utc)
    age = now_utc - last_dt
    age_minutes = age.total_seconds() / 60 if age else 0.0

    history.attrs["data_last_timestamp"] = last_dt.isoformat()
    history.attrs["data_age_minutes"] = age_minutes

    if age > timedelta(minutes=max_age_minutes):
        raise DataStalenessError(
            f"最新行情已经超过{int(age.total_seconds() // 60)}分钟（上限{max_age_minutes}分钟）"
        )
    return age_minutes


def _normalized_provider(value: Optional[str]) -> str:
    return (value or "yfinance").lower().strip()


def _instantiate_adapter(provider_key: str, settings) -> Optional[Tuple[DataSourceAdapter, str]]:
    try:
        if provider_key in {"yfinance", "yahoo"}:
            return YahooFinanceAdapter(), "Yahoo Finance"
        if provider_key in {"tanshu", "tanshuapi", "tanshu_gold", "tanshu-gold"}:
            adapter = TanshuGoldAdapter(
                settings.tanshu_api_key,
                endpoint=settings.tanshu_endpoint,
                symbol_map=settings.tanshu_symbol_map,
                default_symbol_code=settings.tanshu_symbol_code,
            )
            return adapter, "探数黄金"
        if provider_key in {"twelvedata", "twelve_data", "12data", "twelve"}:
            adapter = TwelveDataAdapter(
                settings.twelve_data_api_key,
                base_url=settings.twelve_data_base_url,
                symbol_map=settings.twelve_data_symbol_map,
                default_symbol=settings.twelve_data_symbol,
            )
            return adapter, "Twelve Data"
        if provider_key in {"alpha_vantage", "alphavantage", "alpha-vantage", "alpha"}:
            adapter = AlphaVantageFXAdapter(settings.alpha_vantage_api_key)
            return adapter, "Alpha Vantage"
        if provider_key in {"ibkr", "interactivebrokers"}:
            return IBKRAdapter(), "IBKR"
        if provider_key in {"polygon", "polygon.io"}:
            adapter = PolygonAdapter(
                settings.polygon_api_key,
                base_url=settings.polygon_base_url,
                symbol_map=settings.polygon_symbol_map,
            )
            return adapter, "Polygon.io"
    except DataProviderError as exc:
        message = str(exc)
        if "未配置" in message or "not configured" in message.lower():
            logger.info("数据源初始化失败（%s）：%s", provider_key, message)
        else:
            logger.warning("数据源初始化失败（%s）：%s", provider_key, message)
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("初始化行情适配器异常（%s）：%s", provider_key, exc)
        return None
    return None


def _build_provider_chain(settings) -> List[Tuple[str, DataSourceAdapter, str]]:
    primary = _normalized_provider(settings.data_provider)
    chain: List[Tuple[str, DataSourceAdapter, str]] = []
    seen: set[str] = set()

    def add(provider: str) -> None:
        key = _normalized_provider(provider)
        if key in seen:
            return
        entry = _instantiate_adapter(key, settings)
        if entry is None:
            return
        chain.append((key, entry[0], entry[1]))
        seen.add(key)

    add(primary)

    fallback_matrix = {
        "yfinance": ["polygon", "twelvedata", "tanshu", "alpha_vantage"],
        "yahoo": ["polygon", "twelvedata", "tanshu", "alpha_vantage"],
        "tanshu": ["polygon", "twelvedata", "yfinance", "alpha_vantage"],
        "tanshuapi": ["polygon", "twelvedata", "yfinance", "alpha_vantage"],
        "tanshu_gold": ["polygon", "twelvedata", "yfinance", "alpha_vantage"],
        "tanshu-gold": ["polygon", "twelvedata", "yfinance", "alpha_vantage"],
        "twelvedata": ["polygon", "yfinance", "tanshu", "alpha_vantage"],
        "twelve_data": ["polygon", "yfinance", "tanshu", "alpha_vantage"],
        "12data": ["polygon", "yfinance", "tanshu", "alpha_vantage"],
        "twelve": ["polygon", "yfinance", "tanshu", "alpha_vantage"],
        "alpha_vantage": ["polygon", "twelvedata", "tanshu", "yfinance"],
        "alphavantage": ["polygon", "twelvedata", "tanshu", "yfinance"],
        "alpha-vantage": ["polygon", "twelvedata", "tanshu", "yfinance"],
        "alpha": ["polygon", "twelvedata", "tanshu", "yfinance"],
        "ibkr": ["polygon", "twelvedata", "tanshu", "yfinance", "alpha_vantage"],
        "interactivebrokers": ["polygon", "twelvedata", "tanshu", "yfinance", "alpha_vantage"],
        "polygon": ["twelvedata", "yfinance", "alpha_vantage", "tanshu"],
        "polygon.io": ["twelvedata", "yfinance", "alpha_vantage", "tanshu"],
    }

    fallback_candidates = fallback_matrix.get(primary, ["polygon", "twelvedata", "tanshu", "alpha_vantage", "yfinance"])

    # Ensure we consider common fallbacks even if not listed explicitly.
    fallback_candidates = list(fallback_candidates) + ["polygon", "twelvedata", "alpha_vantage", "tanshu", "yfinance"]

    for candidate in fallback_candidates:
        add(candidate)

    if not chain:
        raise DataProviderError(f"不支持的数据源：{settings.data_provider}")

    return chain


def _attempt_fetch_with_logging(
    router: DataSourceRouter,
    provider_key: str,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    session: Optional[Session],
    settings,
    provider_label: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    label = provider_label or "未知来源"
    logger.info("行情下载：%s（来源：%s）", symbol, label)

    def fetcher() -> pd.DataFrame:
        return router.fetch_ohlcv(
            symbol,
            start,
            end,
            timeframe="1d",
            prefer=provider_key,
            session=session,
        )

    try:
        data = _retry_fetch(fetcher, symbol=symbol, provider_label=label, settings=settings)
    except DataProviderError as exc:
        logger.warning("行情获取失败（%s）：%s", label, exc)
        return None, str(exc)

    if data.empty:
        logger.warning("行情无有效数据（%s）", label)
        return None, "无有效数据"

    return data, None


def fetch_price_history(symbol: str, days: int = 14) -> pd.DataFrame:
    """Fetch recent price history using the configured data adapter."""

    settings = get_settings()
    mode = (settings.data_mode or "live").lower()

    if mode not in {"live", "hybrid", "mock"}:
        logger.warning("未识别的数据模式 '%s'，已改用 live", mode)
        mode = "live"

    if mode == "mock":
        return _mock_price_history(symbol, days)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days * 2)
    session = _cached_session(settings)

    provider_chain = _build_provider_chain(settings)

    router = DataSourceRouter()
    if settings.market_data_cache_minutes > 0:
        router.configure_cache(
            CacheConfig(
                expire_after=timedelta(minutes=settings.market_data_cache_minutes),
                namespace="market",
            )
        )
    router.configure_retry(
        RetryConfig(
            total=max(0, settings.market_data_retry_total),
            backoff=max(0.0, settings.market_data_retry_backoff),
        )
    )

    for index, (provider_key, adapter, _) in enumerate(provider_chain):
        router.register(provider_key, adapter)
        if index == 0 and "live" not in router.adapters:
            router.adapters["live"] = router.adapters[provider_key]

    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    selected_provider_key: Optional[str] = None
    selected_provider_label: Optional[str] = None

    for index, (provider_key, _adapter, label) in enumerate(provider_chain):
        candidate, error = _attempt_fetch_with_logging(
            router,
            provider_key,
            symbol,
            start=start,
            end=end,
            session=session,
            settings=settings,
            provider_label=label,
        )
        if candidate is not None:
            data = candidate
            selected_provider_key = provider_key
            selected_provider_label = label
            break
        if index < len(provider_chain) - 1:
            next_label = provider_chain[index + 1][2]
            logger.info("切换备用行情源：%s", next_label)

    if data is None:
        if mode == "hybrid":
            logger.warning("所有行情源不可用，启用模拟数据：%s", symbol)
            return _mock_price_history(symbol, days)
        if error:
            logger.error("行情抓取失败：%s", error)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    attrs_snapshot = dict(getattr(data, "attrs", {}))
    data = data.tail(days)
    data.index = pd.to_datetime(data.index)
    data.attrs.update(attrs_snapshot)
    age_minutes = _ensure_freshness(data, max_age_minutes=settings.market_data_max_age_minutes)
    if selected_provider_key:
        data.attrs["provider_key"] = selected_provider_key
    if selected_provider_label:
        data.attrs["provider_label"] = selected_provider_label
    data.attrs["data_age_minutes"] = age_minutes
    data.attrs["history_rows"] = int(len(data))
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
