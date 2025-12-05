"""Risk monitoring utilities aligning with trading desk practices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isnan
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from ..utils.logging import get_logger
from .market_data import fetch_price_history
from .risk_math import ScenarioShock, apply_scenario, historical_var, rolling_correlation

if TYPE_CHECKING:  # pragma: no cover - type checker assistance only
    import pandas as pd

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Key risk guardrails supplied by configuration."""

    max_position_oz: float
    stress_var_millions: float
    daily_drawdown_pct: float


@dataclass(frozen=True)
class CorrelationTarget:
    """Configure cross-asset correlation diagnostics."""

    symbol: str
    label: str
    window: int = 20


DEFAULT_CORRELATION_TARGETS: Sequence[CorrelationTarget] = (
    CorrelationTarget(symbol="DX-Y.NYB", label="US Dollar Index (DXY)", window=20),
    CorrelationTarget(symbol="^GSPC", label="S&P 500 Index", window=20),
    CorrelationTarget(symbol="TLT", label="Long-Term Treasuries ETF", window=20),
)


DEFAULT_SCENARIO_SHOCKS: Sequence[ScenarioShock] = (
    ScenarioShock(label="minus_2pct", pct_change=-0.02),
    ScenarioShock(label="minus_1pct", pct_change=-0.01),
    ScenarioShock(label="plus_1pct", pct_change=0.01),
    ScenarioShock(label="plus_2pct", pct_change=0.02),
)


def _infer_market_session(timestamp: datetime) -> str:
    """Roughly bucket the current UTC hour into key trading sessions."""

    hour = timestamp.hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "newyork"
    return "off_hours"


def adjust_limits_with_news(
    limits: RiskLimits,
    news_snapshot: Mapping[str, Any],
    *,
    min_scale: float = 0.5,
    max_scale: float = 1.2,
) -> Tuple[RiskLimits, Dict[str, Any]]:
    """Scale risk limits based on headline-driven sentiment.

    The adjustment factors consider both the absolute news sentiment score and the
    confidence of that score. Higher confidence tightens limits, while a strongly
    bullish (bearish) bias can marginally expand (tighten) exposures within
    configured bounds.
    """

    score = float(news_snapshot.get("score", 0.0) or 0.0)
    confidence = float(news_snapshot.get("confidence", 0.0) or 0.0)
    classification = str(news_snapshot.get("classification", "neutral")).lower()
    trend = float(news_snapshot.get("score_trend", 0.0) or 0.0)

    confidence_clamped = max(0.0, min(2.0, confidence))
    score_clamped = max(-1.0, min(1.0, score))
    trend_clamped = max(-1.0, min(1.0, trend))

    directional_bias = 1.0 + (score_clamped + 0.3 * trend_clamped) * 0.25
    if classification == "bearish":
        directional_bias -= 0.1 * (abs(score_clamped) + 0.2 * confidence_clamped)
    elif classification == "bullish":
        directional_bias += 0.1 * max(0.0, score_clamped)

    tightening = 1.0 - 0.25 * confidence_clamped
    scale = max(min_scale, min(max_scale, directional_bias * tightening))

    adjusted = RiskLimits(
        max_position_oz=limits.max_position_oz * scale,
        stress_var_millions=limits.stress_var_millions * scale,
        daily_drawdown_pct=max(0.5, limits.daily_drawdown_pct * scale),
    )

    adjustment_meta = {
        "classification": classification,
        "score": score,
        "confidence": confidence,
        "trend": trend,
        "scale": scale,
    }

    return adjusted, adjustment_meta


def _fetch_benchmark_series(
    targets: Sequence[CorrelationTarget],
    *,
    lookback_days: int,
) -> Dict[str, "pd.Series"]:
    """Download benchmark closes for correlation diagnostics."""

    from importlib import import_module

    benchmark_series: Dict[str, "pd.Series"] = {}
    try:  # pragma: no cover - keep pandas optional
        pd = import_module("pandas")
    except ModuleNotFoundError:
        return benchmark_series

    for target in targets:
        try:
            history = fetch_price_history(target.symbol, days=lookback_days)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("基准行情下载失败：%s（%s）", target.symbol, exc)
            continue
        if history.empty:
            logger.debug("基准行情为空：%s", target.symbol)
            continue
        series = history.get("Close")
        if series is None or series.empty:
            logger.debug("基准缺少收盘价：%s", target.symbol)
            continue
        benchmark_series[target.symbol] = pd.Series(series.astype(float))
    return benchmark_series


def _compute_cross_asset_correlations(
    base_series: "pd.Series",
    benchmarks: Dict[str, "pd.Series"],
    *,
    targets: Sequence[CorrelationTarget],
) -> List[Dict[str, Any]]:
    """Calculate latest rolling correlation vs configured benchmarks."""

    from importlib import import_module
    import math

    diagnostics: List[Dict[str, Any]] = []
    if base_series.empty or not targets:
        return diagnostics

    try:  # pragma: no cover - keep pandas optional
        pd = import_module("pandas")
    except ModuleNotFoundError:
        return diagnostics

    base_series = pd.Series(base_series.astype(float)).dropna()
    if base_series.empty:
        return diagnostics

    for target in targets:
        peer = benchmarks.get(target.symbol)
        if peer is None:
            continue
        peer_series = pd.Series(peer.astype(float)).dropna()
        if peer_series.empty:
            continue

        aligned = pd.concat([base_series, peer_series], axis=1, join="inner").dropna()
        if aligned.empty:
            continue

        window = max(2, target.window)
        corr_series = rolling_correlation(aligned.iloc[:, 0], aligned.iloc[:, 1], window=window)
        latest = corr_series.dropna()
        value: Optional[float] = None
        if not latest.empty:
            value = float(latest.iloc[-1])
        else:
            tail = aligned.tail(window)
            if len(tail) == window:
                raw = float(tail.iloc[:, 0].corr(tail.iloc[:, 1]))
                value = None if math.isnan(raw) else raw

        if value is None:
            continue

        diagnostics.append(
            {
                "symbol": target.symbol,
                "label": target.label,
                "window": window,
                "value": value,
                "observations": int(len(aligned)),
            }
        )

    return diagnostics


def _compute_liquidity_metrics(history: Any, latest_price: Optional[float]) -> Dict[str, Any]:
    """Derive basic liquidity diagnostics from price history."""

    metrics: Dict[str, Any] = {}

    try:  # pragma: no cover - optional dependency resolution
        from importlib import import_module

        pd = import_module("pandas")
    except ModuleNotFoundError:
        return metrics

    if history is None or getattr(history, "empty", True):
        return metrics

    volume_series = None
    if "Volume" in history:
        try:
            volume_series = pd.Series(history["Volume"].astype(float)).dropna()
        except Exception:  # pragma: no cover - defensive
            volume_series = None

    if volume_series is not None and not volume_series.empty:
        latest_volume = float(volume_series.iloc[-1])
        avg_volume_lookback = min(len(volume_series), 20)
        avg_volume = float(volume_series.tail(avg_volume_lookback).mean()) if avg_volume_lookback else None
        metrics["latest_volume"] = latest_volume
        if avg_volume is not None and avg_volume > 0:
            metrics["avg_volume"] = avg_volume
            metrics["volume_ratio"] = latest_volume / avg_volume if avg_volume else None
        metrics["volume_observations"] = int(len(volume_series))

    atr_pct: Optional[float] = None
    if latest_price and "High" in history and "Low" in history and "Close" in history:
        highs = lows = closes = None
        try:
            highs = pd.Series(history["High"].astype(float))
            lows = pd.Series(history["Low"].astype(float))
            closes = pd.Series(history["Close"].astype(float))
            prev_close = closes.shift(1)
            true_range = pd.concat(
                [
                    highs - lows,
                    (highs - prev_close).abs(),
                    (lows - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            window = min(14, len(true_range))
            if window >= 3:
                atr = true_range.rolling(window=window, min_periods=3).mean().dropna()
                if not atr.empty:
                    atr_latest = float(atr.iloc[-1])
                    if atr_latest > 0:
                        atr_pct = (atr_latest / latest_price) * 100
                        metrics["atr_window"] = int(window)
                        metrics["atr_pct"] = atr_pct
                        metrics["atr_based_slippage_bps"] = atr_pct * 100
        except Exception:  # pragma: no cover - defensive guard
            highs = lows = None
            atr_pct = None

        if highs is not None and lows is not None:
            try:
                high = float(highs.iloc[-1])
                low = float(lows.iloc[-1])
                if latest_price and latest_price > 0 and high >= low:
                    spread_bps = ((high - low) / latest_price) * 10_000 / 2
                    metrics["spread_bps"] = float(max(0.0, spread_bps))
            except Exception:  # pragma: no cover - defensive
                pass

    if atr_pct is None and "spread_bps" in metrics:
        # Provide a proxy slippage estimate using spread when ATR missing.
        metrics["atr_based_slippage_bps"] = metrics["spread_bps"] * 1.5

    return {key: value for key, value in metrics.items() if value is not None}


def build_risk_snapshot(
    symbol: str,
    history: Any,
    *,
    limits: RiskLimits,
    current_position_oz: float,
    pnl_today_millions: float,
    benchmark_series: Optional[Dict[str, "pd.Series"]] = None,
    correlation_targets: Sequence[CorrelationTarget] = DEFAULT_CORRELATION_TARGETS,
    correlation_window: Optional[int] = None,
    correlation_windows: Optional[Sequence[int]] = None,
    scenario_shocks: Sequence[ScenarioShock] = DEFAULT_SCENARIO_SHOCKS,
    news_snapshot: Optional[Mapping[str, Any]] = None,
    apply_news_adjustment: bool = True,
    max_data_age_minutes: Optional[int] = None,
    data_provider: Optional[str] = None,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute realized and hypothetical risk metrics for the desk."""

    from importlib import import_module

    try:  # pragma: no cover - optional dependency resolution
        np = import_module("numpy")
        pd = import_module("pandas")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "The 'numpy' and 'pandas' packages are required for risk computations."
        ) from exc

    close_series = pd.Series(dtype=float)
    latest_price: Optional[float] = None
    portfolio_var_millions: Optional[float] = None
    drawdown_threshold: Optional[float] = None
    effective_limits = limits
    news_adjustment: Optional[Dict[str, Any]] = None

    if apply_news_adjustment and news_snapshot:
        try:
            effective_limits, news_adjustment = adjust_limits_with_news(limits, news_snapshot)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("新闻驱动风险参数调节失败：%s", exc)
            effective_limits = limits

    if correlation_windows is None:
        if correlation_window is not None:
            correlation_windows = (int(correlation_window),)
        else:
            correlation_windows = (20, 60, 120)

    provider_key = None
    provider_label = None
    data_age_minutes: Optional[float] = None
    last_timestamp: Optional[str] = None
    history_rows: Optional[int] = None

    if hasattr(history, "attrs"):
        provider_key = str(history.attrs.get("provider_key") or "").lower() or None
        provider_label = history.attrs.get("provider_label") or provider_key
        age_attr = history.attrs.get("data_age_minutes")
        if age_attr is not None:
            try:
                data_age_minutes = float(age_attr)
            except (TypeError, ValueError):
                data_age_minutes = None
        last_timestamp = history.attrs.get("data_last_timestamp")
        rows_attr = history.attrs.get("history_rows")
        if rows_attr is not None:
            try:
                history_rows = int(rows_attr)
            except (TypeError, ValueError):
                history_rows = None
    if data_provider and not provider_key:
        provider_key = str(data_provider).lower()
    if provider_label is None and provider_key is not None:
        provider_label = provider_key

    max_age_limit: Optional[float] = None
    if max_data_age_minutes is not None:
        try:
            max_age_limit = float(max_data_age_minutes)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            max_age_limit = None

    if history.empty:
        logger.warning("缺少行情数据，无法计算风险快照：%s", symbol)
        vol_annualized: Optional[float] = None
        drawdown_flag = False
        var_99: Optional[float] = None
        scenario_outcomes: List[Dict[str, Any]] = []
        cross_asset_correlations: List[Dict[str, Any]] = []
        liquidity_metrics: Dict[str, Any] = {}
    else:
        close_series = pd.Series(history["Close"].astype(float))
        latest_price = float(close_series.iloc[-1])
        returns = close_series.pct_change().dropna()
        vol_annualized = float(np.sqrt(252) * returns.std()) if not returns.empty else None
        drawdown_threshold = -effective_limits.daily_drawdown_pct / 100 * effective_limits.stress_var_millions
        drawdown_flag = pnl_today_millions <= drawdown_threshold
        var_value = historical_var(returns, confidence=0.99) if not returns.empty else float("nan")
        var_99 = None if np.isnan(var_value) else float(var_value)

        projections = apply_scenario(close_series, scenario_shocks)
        scenario_outcomes = []
        for label, value in projections:
            entry: Dict[str, Any] = {"label": label, "projected_price": value}
            if latest_price is not None:
                projected_pnl = (value - latest_price) * current_position_oz / 1_000_000
                entry["projected_pnl_millions"] = float(projected_pnl)
            scenario_outcomes.append(entry)

        if var_99 is not None and latest_price is not None:
            portfolio_var_millions = float(abs(var_99) * latest_price * current_position_oz / 1_000_000)

        if benchmark_series is None:
            max_window = max(correlation_windows) if correlation_windows else 20
            lookback = max(len(close_series) + max_window, max_window * 3, 60)
            benchmark_series = _fetch_benchmark_series(correlation_targets, lookback_days=lookback)

        cross_asset_correlations: List[Dict[str, Any]] = []
        for window in correlation_windows:
            diagnostics = _compute_cross_asset_correlations(
                close_series,
                benchmark_series,
                targets=[
                    CorrelationTarget(symbol=target.symbol, label=target.label, window=max(2, int(window)))
                    for target in correlation_targets
                ],
            )
            cross_asset_correlations.extend(diagnostics)

        liquidity_metrics = _compute_liquidity_metrics(history, latest_price)

    fresh: Optional[bool] = None
    if data_age_minutes is not None and max_age_limit is not None:
        fresh = data_age_minutes <= max_age_limit

    utilization = (
        current_position_oz / effective_limits.max_position_oz if effective_limits.max_position_oz else None
    )

    if vol_annualized is not None and isinstance(vol_annualized, float) and isnan(vol_annualized):
        vol_annualized = None

    var_limit_utilization: Optional[float] = None
    if portfolio_var_millions is not None and effective_limits.stress_var_millions:
        var_limit_utilization = portfolio_var_millions / effective_limits.stress_var_millions

    risk_alerts: List[str] = []
    if utilization is not None:
        if utilization > 1.0:
            risk_alerts.append("position_limit_exceeded")
        elif utilization >= 0.9:
            risk_alerts.append("position_limit_warning")

    if history.empty:
        drawdown_flag = False

    if drawdown_flag:
        risk_alerts.append("drawdown_limit_breached")

    if var_limit_utilization is not None:
        if var_limit_utilization > 1.0:
            risk_alerts.append("var_limit_exceeded")
        elif var_limit_utilization >= 0.8:
            risk_alerts.append("var_limit_warning")

    for outcome in scenario_outcomes:
        projected = outcome.get("projected_pnl_millions")
        if projected is not None and projected < -effective_limits.stress_var_millions:
            risk_alerts.append("scenario_loss_exceeds_limit")
            break

    snapshot_timestamp = datetime.now(timezone.utc)
    market_session = _infer_market_session(snapshot_timestamp)

    snapshot: Dict[str, Any] = {
        "symbol": symbol,
        "current_position_oz": current_position_oz,
        "limit_position_oz": effective_limits.max_position_oz,
        "position_utilization": utilization,
        "stress_var_limit_millions": effective_limits.stress_var_millions,
        "pnl_today_millions": pnl_today_millions,
        "drawdown_alert": drawdown_flag,
        "drawdown_threshold_millions": drawdown_threshold,
        "realized_vol_annualized": vol_annualized,
        "historical_var_99": var_99,
        "portfolio_var_millions": portfolio_var_millions,
        "var_limit_utilization": var_limit_utilization,
        "scenario_outcomes": scenario_outcomes,
        "cross_asset_correlations": cross_asset_correlations,
        "liquidity_metrics": liquidity_metrics,
        "data_quality": {
            "provider": provider_key,
            "provider_label": provider_label,
            "data_mode": str(data_mode).lower() if data_mode is not None else None,
            "age_minutes": data_age_minutes,
            "max_age_minutes": max_age_limit,
            "fresh": fresh,
            "history_rows": history_rows,
            "last_timestamp": last_timestamp,
        },
        "risk_alerts": risk_alerts,
        "latest_price": latest_price,
        "base_limits": {
            "max_position_oz": limits.max_position_oz,
            "stress_var_millions": limits.stress_var_millions,
            "daily_drawdown_pct": limits.daily_drawdown_pct,
        },
        "adjusted_limits": {
            "max_position_oz": effective_limits.max_position_oz,
            "stress_var_millions": effective_limits.stress_var_millions,
            "daily_drawdown_pct": effective_limits.daily_drawdown_pct,
        },
        "news_adjustment": news_adjustment,
        "snapshot_timestamp": snapshot_timestamp.isoformat(),
        "market_session": market_session,
    }
    return snapshot
