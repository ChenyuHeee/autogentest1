"""Hard risk gate enforcement for trading plans.

This module inspects the final workflow payload alongside the
pre-computed risk snapshot and blocks execution when hard-coded
limits are breached. The checks are intentionally conservative so
that automated plans do not proceed when the desk would normally
escalate to humans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..config.settings import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


STOP_ORDER_TYPES = {
    "STOP",
    "STOP_LIMIT",
    "STOP_LOSS",
    "TRAILING_STOP",
    "STOP_MARKET",
    "TRAILING_STOP_LIMIT",
}


@dataclass(frozen=True)
class HardRiskViolation:
    """Structured metadata describing a breached hard limit."""

    code: str
    message: str
    metric: Optional[float] = None
    limit: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.metric is not None:
            payload["metric"] = self.metric
        if self.limit is not None:
            payload["limit"] = self.limit
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class HardRiskGateReport:
    """Evaluation outcome for the hard gate layer."""

    violations: List[HardRiskViolation]
    evaluated_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breached": self.breached,
            "violations": [violation.to_dict() for violation in self.violations],
            "evaluated_metrics": self.evaluated_metrics,
        }

    @property
    def breached(self) -> bool:
        return bool(self.violations)

    def summary(self) -> str:
        if not self.violations:
            return "No hard risk breaches detected."
        parts = [f"{violation.code}: {violation.message}" for violation in self.violations]
        return "; ".join(parts)


class HardRiskBreachError(RuntimeError):
    """Raised when the hard risk gate blocks the execution."""

    def __init__(self, report: HardRiskGateReport, *, result: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(report.summary())
        self.report = report
        self.partial_result = result

    def __str__(self) -> str:
        return f"Hard risk gate breached: {self.report.summary()}"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _normalize_side(raw: Any) -> Optional[str]:
    side = str(raw or "").upper()
    if side in {"BUY", "LONG", "BID"}:
        return "LONG"
    if side in {"SELL", "SHORT", "ASK"}:
        return "SHORT"
    return None


def _opposite_side(side: Optional[str]) -> Optional[str]:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return None


def _extract_order_size(order: Mapping[str, Any]) -> Optional[float]:
    size = order.get("size_oz") or order.get("size") or order.get("quantity")
    numeric = _safe_float(size)
    if numeric is None:
        return None
    return abs(numeric)


def _extract_price(order: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = _safe_float(order.get(key))
        if value is not None:
            return value
    return None


def _is_stop_order(order: Mapping[str, Any]) -> bool:
    order_type = str(order.get("type", "")).upper()
    classification = str(order.get("classification", "")).upper()
    intent = str(order.get("intent", "")).upper()
    return (
        order_type in STOP_ORDER_TYPES
        or classification in STOP_ORDER_TYPES
        or intent in STOP_ORDER_TYPES
    )


def _format_side_map(values: Mapping[tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
    formatted: Dict[str, Dict[str, float]] = {}
    for (instrument, side), amount in values.items():
        bucket = formatted.setdefault(instrument, {})
        bucket[side] = round(float(amount), 4)
    return formatted


def _largest_order_size(orders: Iterable[Mapping[str, Any]]) -> Optional[float]:
    sizes: List[float] = []
    for order in orders:
        size = order.get("size_oz") or order.get("size") or order.get("quantity")
        numeric = _safe_float(size)
        if numeric is not None:
            sizes.append(abs(numeric))
    return max(sizes) if sizes else None


def _has_stop_protection(orders: Iterable[Mapping[str, Any]]) -> bool:
    for order in orders:
        order_type = str(order.get("type", "")).upper()
        if order_type in {"STOP", "STOP_LIMIT", "STOP_LOSS"}:
            return True
        if _safe_float(order.get("stop")) is not None:
            return True
    return False


def _collect_orders(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    details = _as_dict(response.get("details"))
    execution = _as_dict(details.get("execution_checklist"))
    orders = execution.get("orders")
    if isinstance(orders, list):
        return [order for order in orders if isinstance(order, dict)]
    # fallback: some payloads surface orders directly under details
    direct_orders = details.get("orders")
    if isinstance(direct_orders, list):
        return [order for order in direct_orders if isinstance(order, dict)]
    return []


def _extract_primary_direction(response: Mapping[str, Any]) -> Optional[str]:
    details = _as_dict(response.get("details"))
    trading_plan = _as_dict(details.get("trading_plan"))
    base_plan = _as_dict(trading_plan.get("base_plan"))
    position = _safe_float(base_plan.get("position_oz") or base_plan.get("size_oz"))
    if position is None or position == 0:
        return None
    return "LONG" if position > 0 else "SHORT"


def _evaluate_stop_requirements(
    orders: Iterable[Mapping[str, Any]],
    *,
    settings: Settings,
    primary_direction: Optional[str],
) -> Dict[str, Any]:
    exposures: Dict[tuple[str, str], float] = {}
    coverage: Dict[tuple[str, str], float] = {}
    distance_samples: List[Dict[str, Any]] = []
    invalid_direction: List[Dict[str, Any]] = []

    for order in orders:
        size = _extract_order_size(order)
        if size is None or size == 0:
            continue
        instrument = str(order.get("instrument") or order.get("symbol") or "UNKNOWN").upper()
        side = _normalize_side(order.get("side")) or "LONG"
        order_type_stop = _is_stop_order(order)

        entry_price = _extract_price(order, ("entry", "price", "limit", "avg_price"))
        stop_price = _extract_price(order, ("stop", "stop_price", "trigger", "stopLevel"))

        key = (instrument, side)

        if not order_type_stop:
            if primary_direction and side != primary_direction:
                # Treat as exit or hedge orders; monitor but do not require coverage.
                continue
            exposures[key] = exposures.get(key, 0.0) + size
            if stop_price is not None:
                coverage[key] = coverage.get(key, 0.0) + size
                if entry_price and entry_price > 0:
                    distance_pct = abs(entry_price - stop_price) / entry_price * 100
                    expected_direction_valid = True
                    if side == "LONG" and stop_price >= entry_price:
                        expected_direction_valid = False
                    if side == "SHORT" and stop_price <= entry_price:
                        expected_direction_valid = False
                    sample = {
                        "instrument": instrument,
                        "side": side,
                        "distance_pct": distance_pct,
                    }
                    if expected_direction_valid:
                        distance_samples.append(sample)
                    else:
                        invalid_direction.append({**sample, "entry": entry_price, "stop": stop_price})
        else:
            stop_side = _normalize_side(order.get("side"))
            cover_side = _opposite_side(stop_side)
            if cover_side is None:
                continue
            if primary_direction and cover_side != primary_direction:
                continue
            cover_key = (instrument, cover_side)
            coverage[cover_key] = coverage.get(cover_key, 0.0) + size
            if entry_price and stop_price and entry_price > 0:
                distance_pct = abs(entry_price - stop_price) / entry_price * 100
                sample = {
                    "instrument": instrument,
                    "side": cover_side,
                    "distance_pct": distance_pct,
                    "stop_role": "standalone",
                }
                # For standalone stops we cannot reliably determine direction, so skip validation.
                distance_samples.append(sample)

    coverage_ratio_by_key: Dict[tuple[str, str], float] = {}
    uncovered_by_key: Dict[tuple[str, str], float] = {}
    total_exposure = 0.0
    total_coverage = 0.0

    for key, exposure in exposures.items():
        covered = coverage.get(key, 0.0)
        total_exposure += exposure
        total_coverage += min(covered, exposure)
        if exposure > 0:
            coverage_ratio_by_key[key] = covered / exposure if exposure else 0.0
            uncovered = max(0.0, exposure - covered)
            if uncovered > 0:
                uncovered_by_key[key] = uncovered

    min_ratio = min(coverage_ratio_by_key.values()) if coverage_ratio_by_key else None

    evaluated: Dict[str, Any] = {
        "stop_exposure_by_symbol": _format_side_map(exposures),
        "stop_coverage_by_symbol": _format_side_map(coverage),
        "stop_coverage_ratio": {
            instrument: {
                side: round(float(coverage_ratio_by_key[(instrument, side)]), 4)
                for side in mapping
                if (instrument, side) in coverage_ratio_by_key
            }
            for instrument, mapping in _format_side_map(exposures).items()
        },
        "stop_total_exposure_oz": total_exposure,
        "stop_total_coverage_oz": total_coverage,
        "stop_uncovered_oz": sum(uncovered_by_key.values()),
        "stop_distance_samples": len(distance_samples),
    }

    if distance_samples:
        distances = [sample["distance_pct"] for sample in distance_samples]
        evaluated["stop_distance_min_pct"] = min(distances)
        evaluated["stop_distance_max_pct"] = max(distances)

    violations: List[HardRiskViolation] = []

    required_ratio = settings.hard_gate_stop_coverage_ratio
    if required_ratio > 0 and total_exposure > 0 and (min_ratio is None or min_ratio < required_ratio):
        violations.append(
            HardRiskViolation(
                code="STOP_COVERAGE_SHORTFALL",
                message="Stop-loss coverage is below required ratio",
                details={
                    "required_ratio": required_ratio,
                    "observed_ratios": evaluated["stop_coverage_ratio"],
                    "uncovered_exposure": _format_side_map(uncovered_by_key),
                },
            )
        )

    max_distance = settings.hard_gate_max_stop_distance_pct
    if max_distance and distance_samples:
        too_wide = [sample for sample in distance_samples if sample["distance_pct"] > max_distance]
        if too_wide:
            violations.append(
                HardRiskViolation(
                    code="STOP_DISTANCE_TOO_WIDE",
                    message="Stop-loss distance exceeds maximum configured percentage",
                    limit=max_distance,
                    details={"samples": too_wide[:5]},
                )
            )

    min_distance = settings.hard_gate_min_stop_distance_pct
    if min_distance and distance_samples:
        too_tight = [sample for sample in distance_samples if sample["distance_pct"] < min_distance]
        if too_tight:
            violations.append(
                HardRiskViolation(
                    code="STOP_DISTANCE_TOO_TIGHT",
                    message="Stop-loss distance is tighter than configured minimum",
                    limit=min_distance,
                    details={"samples": too_tight[:5]},
                )
            )

    if invalid_direction:
        violations.append(
            HardRiskViolation(
                code="STOP_DIRECTION_INVALID",
                message="Stop-loss price is on the wrong side of the entry",
                details={"samples": invalid_direction[:5]},
            )
        )

    return {
        "evaluated": evaluated,
        "violations": violations,
        "exposures": exposures,
    }


def _evaluate_liquidity(
    *,
    exposures: Mapping[tuple[str, str], float],
    risk_snapshot: Mapping[str, Any],
    settings: Settings,
) -> Dict[str, Any]:
    liquidity_metrics = _as_dict(risk_snapshot.get("liquidity_metrics"))
    evaluated: Dict[str, Any] = {}
    violations: List[HardRiskViolation] = []

    total_exposure = sum(exposures.values())
    dominant_exposure = max(exposures.values()) if exposures else 0.0

    volume_ratio = _safe_float(liquidity_metrics.get("volume_ratio"))
    latest_volume = _safe_float(liquidity_metrics.get("latest_volume"))
    avg_volume = _safe_float(liquidity_metrics.get("avg_volume"))
    spread_bps = _safe_float(liquidity_metrics.get("spread_bps"))
    atr_slippage_bps = _safe_float(liquidity_metrics.get("atr_based_slippage_bps"))
    atr_pct = _safe_float(liquidity_metrics.get("atr_pct"))

    depth_ratio = settings.hard_gate_depth_volume_ratio
    estimated_depth_oz: Optional[float] = None
    if latest_volume is not None:
        estimated_depth_oz = latest_volume * depth_ratio
    if avg_volume is not None:
        baseline_depth = avg_volume * depth_ratio
        estimated_depth_oz = max(estimated_depth_oz or 0.0, baseline_depth)

    estimated_slippage_bps: Optional[float] = None
    if atr_slippage_bps is not None:
        scale = max(1.0, dominant_exposure / settings.hard_gate_liquidity_baseline_oz)
        estimated_slippage_bps = atr_slippage_bps * scale
    elif spread_bps is not None:
        scale = max(1.0, dominant_exposure / settings.hard_gate_liquidity_baseline_oz)
        estimated_slippage_bps = spread_bps * scale

    evaluated.update(
        {
            "liquidity_volume_ratio": volume_ratio,
            "liquidity_spread_bps": spread_bps,
            "liquidity_estimated_depth_oz": estimated_depth_oz,
            "liquidity_estimated_slippage_bps": estimated_slippage_bps,
            "liquidity_atr_pct": atr_pct,
            "liquidity_total_entry_oz": total_exposure,
            "liquidity_dominant_exposure_oz": dominant_exposure,
        }
    )

    depth_buffer = settings.hard_gate_depth_buffer_ratio
    required_depth = dominant_exposure * depth_buffer

    if settings.hard_gate_min_liquidity_depth_oz is not None:
        required_depth = max(required_depth, settings.hard_gate_min_liquidity_depth_oz)

    if estimated_depth_oz is not None and dominant_exposure > 0 and estimated_depth_oz < required_depth:
        violations.append(
            HardRiskViolation(
                code="LIQUIDITY_DEPTH_INSUFFICIENT",
                message="Estimated market depth is insufficient for planned size",
                metric=estimated_depth_oz,
                limit=required_depth,
                details={
                    "dominant_exposure": dominant_exposure,
                    "depth_ratio": depth_ratio,
                },
            )
        )

    if spread_bps is not None and spread_bps > settings.hard_gate_max_spread_bps:
        violations.append(
            HardRiskViolation(
                code="LIQUIDITY_SPREAD_EXCEEDED",
                message="Bid-ask spread exceeds configured maximum",
                metric=spread_bps,
                limit=settings.hard_gate_max_spread_bps,
            )
        )

    if (
        estimated_slippage_bps is not None
        and estimated_slippage_bps > settings.hard_gate_max_slippage_bps
    ):
        violations.append(
            HardRiskViolation(
                code="LIQUIDITY_SLIPPAGE_EXCEEDED",
                message="Projected slippage exceeds configured maximum",
                metric=estimated_slippage_bps,
                limit=settings.hard_gate_max_slippage_bps,
            )
        )

    return {
        "evaluated": evaluated,
        "violations": violations,
    }


def _extract_risk_metrics(response: Mapping[str, Any]) -> Dict[str, Any]:
    details = _as_dict(response.get("details"))
    risk_compliance = _as_dict(details.get("risk_compliance_signoff"))
    metrics = _as_dict(risk_compliance.get("risk_metrics"))
    return metrics


def enforce_hard_limits(
    response: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    settings: Settings,
) -> HardRiskGateReport:
    """Evaluate the final workflow response against hard risk limits."""

    if not settings.hard_gate_enabled:
        return HardRiskGateReport(violations=[], evaluated_metrics={})

    violations: List[HardRiskViolation] = []
    evaluated: Dict[str, Any] = {}

    risk_snapshot = _as_dict(context.get("risk_snapshot"))
    risk_metrics = _extract_risk_metrics(response)
    orders = _collect_orders(response)
    primary_direction = _extract_primary_direction(response)
    evaluated["primary_direction"] = primary_direction

    # -- Position utilization -------------------------------------------------
    position_utilization = _safe_float(risk_metrics.get("position_utilization"))
    if position_utilization is None:
        position_utilization = _safe_float(risk_snapshot.get("position_utilization"))
    evaluated["position_utilization"] = position_utilization

    utilization_limit = settings.hard_gate_max_position_utilization
    if utilization_limit is None:
        utilization_limit = 1.0
    evaluated["position_utilization_limit"] = utilization_limit

    if (
        position_utilization is not None
        and utilization_limit is not None
        and position_utilization > utilization_limit
    ):
        violations.append(
            HardRiskViolation(
                code="POSITION_UTILIZATION",
                message="Proposed plan exceeds hard position utilization limit",
                metric=position_utilization,
                limit=utilization_limit,
            )
        )

    # -- Single order sizing --------------------------------------------------
    largest_order = _largest_order_size(orders)
    evaluated["largest_order_oz"] = largest_order

    single_order_limit = settings.hard_gate_max_single_order_oz
    if single_order_limit is None:
        single_order_limit = settings.compliance_max_single_order_oz or settings.max_position_oz
    evaluated["single_order_limit_oz"] = single_order_limit

    if (
        largest_order is not None
        and single_order_limit is not None
        and largest_order > single_order_limit
    ):
        violations.append(
            HardRiskViolation(
                code="SINGLE_ORDER_EXPOSURE",
                message="Largest ticket size exceeds hard limit",
                metric=largest_order,
                limit=single_order_limit,
            )
        )

    # -- Stop-loss coverage ---------------------------------------------------
    evaluated["has_stop_protection"] = _has_stop_protection(orders)
    if settings.hard_gate_require_stop_loss and not evaluated["has_stop_protection"]:
        violations.append(
            HardRiskViolation(
                code="STOP_LOSS_MISSING",
                message="No stop-loss protection detected for execution orders",
                details={"orders_checked": len(orders)},
            )
        )

    stop_diagnostics = _evaluate_stop_requirements(
        orders,
        settings=settings,
        primary_direction=primary_direction,
    )
    evaluated.update(stop_diagnostics["evaluated"])
    violations.extend(stop_diagnostics["violations"])

    liquidity_diagnostics = _evaluate_liquidity(
        exposures=stop_diagnostics["exposures"],
        risk_snapshot=risk_snapshot,
        settings=settings,
    )
    evaluated.update(liquidity_diagnostics["evaluated"])
    violations.extend(liquidity_diagnostics["violations"])

    # -- Stress loss vs limit -------------------------------------------------
    stress_loss = _safe_float(risk_metrics.get("stress_test_worst_loss_millions"))
    evaluated["stress_test_worst_loss_millions"] = stress_loss

    stress_limit = settings.hard_gate_max_stress_loss_millions
    if stress_limit is None:
        stress_limit = settings.stress_var_millions
    evaluated["stress_loss_limit_millions"] = stress_limit

    if (
        stress_loss is not None
        and stress_limit is not None
        and stress_loss < 0
        and abs(stress_loss) > stress_limit
    ):
        violations.append(
            HardRiskViolation(
                code="STRESS_LOSS_LIMIT",
                message="Stress scenario loss exceeds configured maximum",
                metric=abs(stress_loss),
                limit=stress_limit,
            )
        )

    # -- Daily drawdown -------------------------------------------------------
    pnl_today = _safe_float(risk_snapshot.get("pnl_today_millions"))
    drawdown_floor = _safe_float(risk_snapshot.get("drawdown_threshold_millions"))
    evaluated["pnl_today_millions"] = pnl_today
    evaluated["drawdown_threshold_millions"] = drawdown_floor

    if (
        pnl_today is not None
        and drawdown_floor is not None
        and pnl_today <= drawdown_floor
    ):
        violations.append(
            HardRiskViolation(
                code="DAILY_DRAWDOWN",
                message="Daily PnL breaches drawdown floor",
                metric=pnl_today,
                limit=drawdown_floor,
            )
        )

    # -- Pre-computed risk alerts --------------------------------------------
    alerts = risk_snapshot.get("risk_alerts")
    fatal_alerts = {"position_limit_exceeded", "drawdown_limit_breached", "var_limit_exceeded", "scenario_loss_exceeds_limit"}
    if isinstance(alerts, list):
        blocking = [alert for alert in alerts if isinstance(alert, str) and alert in fatal_alerts]
        evaluated["blocking_alerts"] = blocking
        for alert in blocking:
            violations.append(
                HardRiskViolation(
                    code="RISK_ALERT",
                    message=f"Underlying risk snapshot flagged '{alert}'",
                )
            )
    else:
        evaluated["blocking_alerts"] = []

    # -- Cross-asset correlation ---------------------------------------------
    correlation_threshold = settings.hard_gate_correlation_threshold
    evaluated["correlation_threshold"] = correlation_threshold
    correlations = risk_snapshot.get("cross_asset_correlations")
    if isinstance(correlations, list) and correlation_threshold is not None:
        breached_pairs: List[Dict[str, Any]] = []
        for entry in correlations:
            if not isinstance(entry, Mapping):
                continue
            value = _safe_float(entry.get("value"))
            if value is None:
                continue
            if abs(value) >= correlation_threshold:
                breached_pairs.append(
                    {
                        "label": entry.get("label") or entry.get("symbol"),
                        "value": value,
                    }
                )
        evaluated["correlation_breaches"] = breached_pairs
        for pair in breached_pairs:
            violations.append(
                HardRiskViolation(
                    code="CORRELATION_LIMIT",
                    message=f"Correlation {pair['label']} {pair['value']:.2f} exceeds threshold {correlation_threshold:.2f}",
                    metric=pair["value"],
                    limit=correlation_threshold,
                )
            )
    else:
        evaluated["correlation_breaches"] = []

    if violations:
        for violation in violations:
            logger.error("硬风控约束触发：%s -> %s", violation.code, violation.message)

    return HardRiskGateReport(violations=violations, evaluated_metrics=evaluated)