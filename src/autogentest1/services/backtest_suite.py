"""Parameter sweep utilities built on top of the lightweight backtester."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .backtest import run_backtest as _run_backtest
from .market_data import fetch_price_history


@dataclass(frozen=True)
class ParameterCombination:
    """Single parameter choice produced from a grid search."""

    values: Dict[str, Any]

    def merged(self, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(base or {})
        payload.update(self.values)
        return payload


@dataclass
class ParameterSweepRun:
    """Structured container for one backtest sweep result."""

    params: Dict[str, Any]
    metrics: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    score: Optional[float]
    score_metric: str
    used_fallback: bool
    error: Optional[str] = None
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "params": self.params,
            "metrics": self.metrics,
            "score": self.score,
            "score_metric": self.score_metric,
            "used_fallback": self.used_fallback,
        }
        if self.error:
            payload["error"] = self.error
        if self.rank is not None:
            payload["rank"] = self.rank
        if self.result is not None:
            payload["backtest"] = self.result
        return payload


def _expand_parameter_grid(grid: Optional[Mapping[str, Sequence[Any]]]) -> List[ParameterCombination]:
    if not grid:
        return [ParameterCombination(values={})]
    normalized: Dict[str, List[Any]] = {}
    for key, raw_values in grid.items():
        if raw_values is None:
            continue
        values = list(raw_values)
        if not values:
            continue
        normalized[key] = values
    if not normalized:
        return [ParameterCombination(values={})]
    keys = list(normalized.keys())
    combinations: List[ParameterCombination] = []
    for choice in product(*(normalized[key] for key in keys)):
        combinations.append(ParameterCombination(values=dict(zip(keys, choice))))
    return combinations


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _resolve_metric_goal(metric_name: str, explicit_goal: Optional[str]) -> str:
    if explicit_goal:
        goal = explicit_goal.lower()
        if goal in {"maximize", "minimize"}:
            return goal
    # Default to maximize for most metrics, including negative drawdowns (closer to zero is better).
    if metric_name in {"max_drawdown", "drawdown", "loss"}:
        return "maximize"
    return "maximize"


def _score_metrics(
    metrics: Mapping[str, Any],
    *,
    primary_metric: str,
    fallback_metric: str,
    metric_goal: str,
) -> Tuple[Optional[float], bool, float]:
    primary_value = _coerce_float(metrics.get(primary_metric))
    used_fallback = False
    if primary_value is None:
        fallback_value = _coerce_float(metrics.get(fallback_metric))
        primary_value = fallback_value
        used_fallback = True
    if primary_value is None:
        sentinel = float("-inf") if metric_goal == "maximize" else float("inf")
        return None, used_fallback, sentinel
    return primary_value, used_fallback, primary_value


def _history_summary(history: Any) -> Dict[str, Any]:
    if history is None:
        return {"rows": 0}
    df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    if df.empty:
        return {"rows": 0}
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:  # pragma: no cover - defensive conversion
            df.index = pd.RangeIndex(start=0, stop=len(df))
    df = df.sort_index()
    start = df.index[0]
    end = df.index[-1]
    return {
        "rows": int(len(df)),
        "start": start.strftime("%Y-%m-%d") if isinstance(start, pd.Timestamp) else str(start),
        "end": end.strftime("%Y-%m-%d") if isinstance(end, pd.Timestamp) else str(end),
    }


def run_parameter_sweep(
    history: Any,
    *,
    strategy: str,
    parameter_grid: Optional[Mapping[str, Sequence[Any]]],
    base_params: Optional[Mapping[str, Any]] = None,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.0,
    evaluation_metric: str = "sharpe_ratio",
    metric_goal: Optional[str] = None,
    fallback_metric: str = "total_return",
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Execute a parameter sweep across the supplied grid using cached history."""

    combinations = _expand_parameter_grid(parameter_grid)
    goal = _resolve_metric_goal(evaluation_metric, metric_goal)
    runs: List[ParameterSweepRun] = []
    errors: List[ParameterSweepRun] = []

    for combination in combinations:
        params = combination.merged(base_params)
        try:
            result = _run_backtest(
                history,
                strategy=strategy,
                initial_capital=initial_capital,
                params=params,
                risk_free_rate=risk_free_rate,
                slippage_bps=slippage_bps,
                commission_per_trade=commission_per_trade,
            )
        except ValueError as exc:
            errors.append(
                ParameterSweepRun(
                    params=params,
                    metrics=None,
                    result=None,
                    score=None,
                    score_metric=evaluation_metric,
                    used_fallback=False,
                    error=str(exc),
                )
            )
            continue

        metrics = result.get("metrics", {})
        score, used_fallback, sort_value = _score_metrics(
            metrics,
            primary_metric=evaluation_metric,
            fallback_metric=fallback_metric,
            metric_goal=goal,
        )
        runs.append(
            ParameterSweepRun(
                params=result.get("parameters", params),
                metrics=metrics,
                result=result,
                score=score,
                score_metric=evaluation_metric if not used_fallback else fallback_metric,
                used_fallback=used_fallback,
            )
        )
        runs[-1].result["sort_value"] = sort_value  # type: ignore[index]

    reverse_sort = goal == "maximize"
    successful_runs = sorted(
        runs,
        key=lambda entry: entry.result.get("sort_value", float("-inf") if reverse_sort else float("inf"))
        if entry.result
        else (float("-inf") if reverse_sort else float("inf")),
        reverse=reverse_sort,
    )

    for idx, run in enumerate(successful_runs):
        run.rank = idx + 1
        # Remove internal sort helper before returning to callers.
        if run.result is not None and "sort_value" in run.result:
            run.result.pop("sort_value", None)

    if top_n < 1:
        top_n = 1
    summary_runs = successful_runs[:top_n]

    history_meta = _history_summary(history)

    best_run = successful_runs[0] if successful_runs else None
    summary: Dict[str, Any] = {
        "total_runs": len(combinations),
        "successful_runs": len(successful_runs),
        "failed_runs": len(errors),
        "evaluation_metric": evaluation_metric,
        "metric_goal": goal,
        "fallback_metric": fallback_metric,
        "history": history_meta,
    }
    if best_run and best_run.score is not None:
        summary.update(
            {
                "best_score": best_run.score,
                "best_params": best_run.params,
                "best_rank": best_run.rank,
            }
        )

    output_runs: List[Dict[str, Any]] = [run.to_dict() for run in successful_runs]
    output_runs.extend(run.to_dict() for run in errors)

    return {
        "strategy": strategy,
        "runs": output_runs,
        "best": best_run.to_dict() if best_run else None,
        "top": [run.to_dict() for run in summary_runs],
        "summary": summary,
    }


def fetch_and_run_parameter_sweep(
    *,
    symbol: str,
    days: int = 365,
    strategy: str,
    parameter_grid: Optional[Mapping[str, Sequence[Any]]],
    base_params: Optional[Mapping[str, Any]] = None,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.0,
    evaluation_metric: str = "sharpe_ratio",
    metric_goal: Optional[str] = None,
    fallback_metric: str = "total_return",
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Fetch price history and execute a parameter sweep in one step."""

    history = fetch_price_history(symbol, days=days)
    if history.empty:
        raise ValueError(f"No historical data available for {symbol} ({days}d)")

    base = dict(base_params or {})
    base.setdefault("symbol", symbol)

    result = run_parameter_sweep(
        history,
        strategy=strategy,
        parameter_grid=parameter_grid,
        base_params=base,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        evaluation_metric=evaluation_metric,
        metric_goal=metric_goal,
        fallback_metric=fallback_metric,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
        top_n=top_n,
    )
    result.setdefault("summary", {}).update({"symbol": symbol, "days": days})
    return result
