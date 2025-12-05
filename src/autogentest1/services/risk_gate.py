"""Hard risk gate enforcement for trading plans.

This module inspects the final workflow payload alongside the
pre-computed risk snapshot and blocks execution when hard-coded
limits are breached. The checks are intentionally conservative so
that automated plans do not proceed when the desk would normally
escalate to humans.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from ..config.settings import Settings
from .audit import record_audit_event
from .circuit_breaker import evaluate_circuit_breaker
from ..utils.logging import get_logger

logger = get_logger(__name__)
def _emit_risk_gate_audit(
    *,
    settings: Settings,
    report: "HardRiskGateReport",
    response: Mapping[str, Any],
    context: Mapping[str, Any],
    message: Optional[str] = None,
) -> None:
    try:
        tags = [tag for tag in (context.get("symbol"), response.get("phase")) if tag]
        tags.append("BREACHED" if report.breached else "PASSED")
        payload: Dict[str, Any] = {
            "breached": report.breached,
            "evaluated_metrics": report.evaluated_metrics,
        }
        if report.violations:
            payload["violations"] = [violation.to_dict() for violation in report.violations]
        summary = {
            "symbol": context.get("symbol"),
            "phase": response.get("phase"),
            "status": response.get("status"),
        }
        record_audit_event(
            "risk_gate.evaluation",
            severity="ERROR" if report.breached else "INFO",
            message=message or "Hard risk gate evaluation completed",
            payload=payload,
            context={key: value for key, value in summary.items() if value is not None},
            tags=tags,
            component="risk_gate",
            settings=settings,
        )
    except Exception:  # pragma: no cover - logging failures are non-blocking
        logger.exception("Failed to emit risk gate audit event")


STOP_ORDER_TYPES = {
    "STOP",
    "STOP_LIMIT",
    "STOP_LOSS",
    "TRAILING_STOP",
    "STOP_MARKET",
    "TRAILING_STOP_LIMIT",
}

_DATA_PROVIDER_PROFILE_MAP: Dict[str, str] = {
    "polygon": "institutional",
    "polygon.io": "institutional",
    "ibkr": "institutional",
    "interactivebrokers": "institutional",
    "twelvedata": "professional",
    "twelve_data": "professional",
    "12data": "professional",
    "twelve": "professional",
    "tanshu": "professional",
    "tanshuapi": "professional",
    "tanshu_gold": "professional",
    "tanshu-gold": "professional",
    "yfinance": "retail",
    "yahoo": "retail",
    "alpha_vantage": "retail",
    "alphavantage": "retail",
    "alpha-vantage": "retail",
    "alpha": "retail",
}

_DATA_QUALITY_CALIBRATIONS: Dict[str, Dict[str, float]] = {
    "institutional": {
        "min_stop_distance_pct": 0.3,
        "max_spread_bps": 35.0,
        "max_slippage_bps": 40.0,
        "min_liquidity_depth_oz": 1000.0,
        "liquidity_baseline_oz": 2500.0,
    },
    "professional": {
        "min_stop_distance_pct": 0.4,
        "max_spread_bps": 45.0,
        "max_slippage_bps": 45.0,
        "min_liquidity_depth_oz": 700.0,
        "liquidity_baseline_oz": 2000.0,
    },
    "retail": {
        "min_stop_distance_pct": 0.5,
        "max_spread_bps": 50.0,
        "max_slippage_bps": 55.0,
        "min_liquidity_depth_oz": 500.0,
        "liquidity_baseline_oz": 1500.0,
    },
    "simulated": {
        "min_stop_distance_pct": 0.6,
        "max_spread_bps": 80.0,
        "max_slippage_bps": 70.0,
        "min_liquidity_depth_oz": 200.0,
        "liquidity_baseline_oz": 800.0,
    },
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


def _extract_target_position(response: Mapping[str, Any]) -> Optional[float]:
    details = _as_dict(response.get("details"))
    trading_plan = _as_dict(details.get("trading_plan"))
    base_plan = _as_dict(trading_plan.get("base_plan"))
    target = (
        base_plan.get("position_oz")
        or base_plan.get("net_position_oz")
        or base_plan.get("net_oz")
        or base_plan.get("position")
        or base_plan.get("size_oz")
        or base_plan.get("size")
    )
    return _safe_float(target)


def _normalize_tag_values(raw: Any) -> List[str]:
    tags: List[str] = []
    if raw is None:
        return tags
    if isinstance(raw, (list, tuple, set)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = [token for token in re.split(r"[\s,;|/]+", raw) if token]
    else:
        items = [raw]
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized = text.replace("-", "_").replace(" ", "_").upper()
        tags.append(normalized)
    return tags


def _extract_strategy_tags(response: Mapping[str, Any], context: Mapping[str, Any]) -> Set[str]:
    tags: Set[str] = set()

    details = _as_dict(response.get("details"))
    trading_plan = _as_dict(details.get("trading_plan"))
    base_plan = _as_dict(trading_plan.get("base_plan"))
    alternate_plan = _as_dict(trading_plan.get("alternate_plan"))
    risk_snapshot = _as_dict(context.get("risk_snapshot"))

    candidate_values: List[Any] = [
        details.get("strategy"),
        details.get("strategy_name"),
        details.get("strategy_tags"),
        details.get("risk_classification"),
        trading_plan.get("strategy"),
        trading_plan.get("strategy_name"),
        trading_plan.get("strategy_tags"),
        trading_plan.get("tags"),
        trading_plan.get("labels"),
        trading_plan.get("metadata"),
        base_plan.get("strategy"),
        base_plan.get("strategy_name"),
        base_plan.get("strategy_tags"),
        base_plan.get("tags"),
        base_plan.get("labels"),
        alternate_plan.get("strategy"),
        alternate_plan.get("strategy_tags"),
        alternate_plan.get("tags"),
        context.get("strategy_tags"),
        context.get("workflow_tags"),
        risk_snapshot.get("strategy_tags"),
    ]

    metadata_candidates: List[Any] = []
    for value in candidate_values:
        if isinstance(value, Mapping):
            for key in (
                "tags",
                "labels",
                "strategy_tags",
                "risk_exemptions",
                "exemptions",
                "category",
                "classification",
                "type",
            ):
                if key in value:
                    metadata_candidates.append(value[key])
    candidate_values.extend(metadata_candidates)

    for value in candidate_values:
        tags.update(_normalize_tag_values(value))

    return tags


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
    payload = _as_dict(details.get("payload"))
    if payload:
        payload_execution = _as_dict(payload.get("execution_checklist"))
        orders = payload_execution.get("orders")
        if isinstance(orders, list):
            return [order for order in orders if isinstance(order, dict)]
    # fallback: some payloads surface orders directly under details
    direct_orders = details.get("orders")
    if isinstance(direct_orders, list):
        return [order for order in direct_orders if isinstance(order, dict)]
    payload_plan = _as_dict(payload.get("trading_plan")) if payload else {}
    plan = _as_dict(details.get("plan"))
    if payload_plan and not plan:
        plan = payload_plan
    synthesized: List[Dict[str, Any]] = []
    for key in ("base_plan", "alternate_plan"):
        candidate = _as_dict(plan.get(key))
        size = candidate.get("position_oz") or candidate.get("size_oz") or candidate.get("net_oz")
        entry = candidate.get("entry") or candidate.get("entry_price")
        stop = candidate.get("stop") or candidate.get("stop_loss")
        if size is None and entry is None and stop is None:
            continue
        order: Dict[str, Any] = {
            "name": key,
        }
        if size is not None:
            order["size_oz"] = size
            try:
                qty = float(size)
            except (TypeError, ValueError):
                qty = None
            if qty is not None:
                order["side"] = "BUY" if qty >= 0 else "SELL"
        if entry is not None:
            order["entry"] = entry
        if stop is not None:
            order["stop"] = stop
        order.setdefault("type", "LIMIT")
        synthesized.append(order)
    if synthesized:
        return synthesized
    return []


def _extract_primary_direction(response: Mapping[str, Any]) -> Optional[str]:
    position = _extract_target_position(response)
    if position is None or position == 0:
        return None
    return "LONG" if position > 0 else "SHORT"


def _evaluate_stop_requirements(
    orders: Iterable[Mapping[str, Any]],
    *,
    settings: Settings,
    primary_direction: Optional[str],
    coverage_exempt: bool = False,
    min_distance_limit: Optional[float] = None,
    max_distance_limit: Optional[float] = None,
) -> Dict[str, Any]:
    exposures: Dict[tuple[str, str], float] = {}
    coverage: Dict[tuple[str, str], float] = {}
    distance_samples: List[Dict[str, Any]] = []
    invalid_direction: List[Dict[str, Any]] = []

    allow_trailing_override = settings.hard_gate_allow_trailing_stop_override
    allow_technical_override = settings.hard_gate_allow_technical_stop_override
    required_ratio = max(0.0, settings.hard_gate_stop_coverage_ratio)
    coverage_required = required_ratio > 0 and not coverage_exempt
    target_stop_min_ratio = settings.hard_gate_target_stop_min_ratio

    technical_keywords = {"technical", "chart", "structure", "support", "resistance", "fib", "trend"}

    def _extract_target_price(order: Mapping[str, Any]) -> Optional[float]:
        candidates = (
            "target",
            "target_price",
            "take_profit",
            "takeProfit",
            "tp",
            "limit_take_profit",
        )
        price = _extract_price(order, candidates)
        if price is not None:
            return price
        targets = order.get("targets")
        if isinstance(targets, (list, tuple)):
            for candidate in targets:
                candidate_value = _safe_float(candidate)
                if candidate_value is not None:
                    return candidate_value
        return None

    for order in orders:
        size = _extract_order_size(order)
        if size is None or size == 0:
            continue
        instrument = str(order.get("instrument") or order.get("symbol") or "UNKNOWN").upper()
        side = _normalize_side(order.get("side")) or "LONG"
        order_type_stop = _is_stop_order(order)

        entry_price = _extract_price(order, ("entry", "price", "limit", "avg_price"))
        stop_price = _extract_price(order, ("stop", "stop_price", "trigger", "stopLevel"))
        target_price = _extract_target_price(order)

        order_tags = set(_normalize_tag_values(order.get("tags") or order.get("labels")))
        order_tags.update(_normalize_tag_values(order.get("strategy_tags")))
        order_tags.update(_normalize_tag_values(order.get("risk_tags")))

        order_type = str(order.get("type", "")).upper()
        classification = str(order.get("classification", "")).upper()
        intent = str(order.get("intent", "")).upper()
        stop_basis = str(
            order.get("stop_basis")
            or order.get("stop_reason")
            or order.get("stop_type")
            or order.get("stop_strategy")
            or ""
        ).lower()

        is_trailing = any(
            tag in text
            for tag in ("TRAIL", "TRAILING")
            for text in (order_type, classification, intent)
        ) or any(
            key in order
            for key in (
                "trailing_amount",
                "trailing_percent",
                "trailing_distance",
                "trail_percent",
            )
        ) or "TRAILING" in order_tags

        is_technical = any(keyword in stop_basis for keyword in technical_keywords) or any(
            tag in order_tags for tag in {"TECHNICAL", "STRUCTURAL", "PRICE_ACTION", "LEVEL"}
        )

        min_distance_override = (is_trailing and allow_trailing_override) or (is_technical and allow_technical_override)
        ratio_override = is_technical and allow_technical_override

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
                    target_distance_pct: Optional[float] = None
                    risk_reward_ratio: Optional[float] = None
                    if target_price and entry_price:
                        target_distance = target_price - entry_price
                        if side == "LONG" and target_price > entry_price:
                            target_distance_pct = abs(target_distance) / entry_price * 100
                        elif side == "SHORT" and target_price < entry_price:
                            target_distance_pct = abs(target_distance) / entry_price * 100
                        if target_distance_pct and distance_pct > 0:
                            risk_reward_ratio = target_distance_pct / distance_pct

                    sample = {
                        "instrument": instrument,
                        "side": side,
                        "distance_pct": distance_pct,
                        "is_trailing": is_trailing,
                        "is_technical": is_technical,
                        "target_distance_pct": target_distance_pct,
                        "risk_reward_ratio": risk_reward_ratio,
                        "override_min_distance": min_distance_override,
                        "override_ratio": ratio_override,
                    }
                    if expected_direction_valid or min_distance_override:
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
                    "is_trailing": is_trailing,
                    "is_technical": is_technical,
                    "override_min_distance": min_distance_override,
                    "override_ratio": ratio_override,
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
        "stop_coverage_required": coverage_required,
        "stop_coverage_exempt": coverage_exempt,
        "stop_risk_reward_ratio_threshold": target_stop_min_ratio,
    }

    if distance_samples:
        distances = [sample["distance_pct"] for sample in distance_samples]
        evaluated["stop_distance_min_pct"] = min(distances)
        evaluated["stop_distance_max_pct"] = max(distances)
        evaluated["stop_distance_avg_pct"] = sum(distances) / len(distances)
        evaluated["stop_trailing_samples"] = sum(1 for sample in distance_samples if sample.get("is_trailing"))
        evaluated["stop_technical_samples"] = sum(1 for sample in distance_samples if sample.get("is_technical"))
        ratios = [sample["risk_reward_ratio"] for sample in distance_samples if sample.get("risk_reward_ratio")]
        if ratios:
            evaluated["stop_risk_reward_ratio_min"] = min(ratios)
            evaluated["stop_risk_reward_ratio_avg"] = sum(ratios) / len(ratios)

    effective_max_distance = max_distance_limit if max_distance_limit is not None else settings.hard_gate_max_stop_distance_pct
    effective_min_distance = min_distance_limit if min_distance_limit is not None else settings.hard_gate_min_stop_distance_pct
    if effective_max_distance is not None:
        evaluated["stop_distance_limit_max_pct"] = effective_max_distance
    if effective_min_distance is not None:
        evaluated["stop_distance_limit_min_pct"] = effective_min_distance

    violations: List[HardRiskViolation] = []

    if coverage_required and total_exposure > 0 and (min_ratio is None or min_ratio < required_ratio):
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

    max_distance = effective_max_distance
    if max_distance and distance_samples:
        too_wide = [
            sample
            for sample in distance_samples
            if sample["distance_pct"] > max_distance and not sample.get("override_min_distance")
        ]
        if too_wide:
            violations.append(
                HardRiskViolation(
                    code="STOP_DISTANCE_TOO_WIDE",
                    message="Stop-loss distance exceeds maximum configured percentage",
                    limit=max_distance,
                    details={"samples": too_wide[:5]},
                )
            )

    min_distance = effective_min_distance
    if min_distance and distance_samples:
        too_tight = [
            sample
            for sample in distance_samples
            if sample["distance_pct"] < min_distance and not sample.get("override_min_distance")
        ]
        if too_tight:
            violations.append(
                HardRiskViolation(
                    code="STOP_DISTANCE_TOO_TIGHT",
                    message="Stop-loss distance is tighter than configured minimum",
                    limit=min_distance,
                    details={"samples": too_tight[:5]},
                )
            )

    if target_stop_min_ratio is not None and target_stop_min_ratio > 0 and distance_samples:
        poor_ratios = [
            sample
            for sample in distance_samples
            if sample.get("risk_reward_ratio") is not None
            and sample["risk_reward_ratio"] < target_stop_min_ratio
            and not sample.get("override_ratio")
        ]
        if poor_ratios:
            violations.append(
                HardRiskViolation(
                    code="STOP_RISK_REWARD_TOO_LOW",
                    message="Risk/reward ratio between targets and stops is below minimum",
                    limit=target_stop_min_ratio,
                    details={"samples": poor_ratios[:5]},
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
    calibration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    liquidity_metrics = _as_dict(risk_snapshot.get("liquidity_metrics"))
    evaluated: Dict[str, Any] = {}
    violations: List[HardRiskViolation] = []

    total_exposure = sum(exposures.values())
    dominant_exposure = max(exposures.values()) if exposures else 0.0

    session_key = str(risk_snapshot.get("market_session") or "").lower()
    relaxation_map = {
        str(key).lower(): float(value)
        for key, value in settings.hard_gate_liquidity_session_relaxation.items()
        if value is not None
    }
    relaxation_factor = relaxation_map.get(session_key, 1.0)
    if relaxation_factor <= 0:
        relaxation_factor = 1.0

    calibration = _as_dict(calibration or {})

    liquidity_baseline_override = _safe_float(calibration.get("liquidity_baseline_oz"))
    liquidity_baseline = max(1.0, settings.hard_gate_liquidity_baseline_oz)
    if liquidity_baseline_override is not None and liquidity_baseline_override > 0:
        liquidity_baseline = max(1.0, float(liquidity_baseline_override))

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
    scale = 1.0
    if dominant_exposure > 0:
        scale = max(1.0, math.sqrt(dominant_exposure / liquidity_baseline))
    if atr_slippage_bps is not None:
        estimated_slippage_bps = atr_slippage_bps * scale
    elif spread_bps is not None:
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
            "liquidity_session": session_key or "unknown",
            "liquidity_session_relaxation_factor": relaxation_factor,
            "liquidity_profile_calibration": calibration,
            "liquidity_baseline_oz": liquidity_baseline,
        }
    )

    depth_buffer = settings.hard_gate_depth_buffer_ratio
    required_depth_base = dominant_exposure * depth_buffer

    min_depth_floor = _safe_float(calibration.get("min_liquidity_depth_oz"))
    min_depth_config = settings.hard_gate_min_liquidity_depth_oz
    if min_depth_floor is not None:
        if min_depth_config is None:
            min_depth_config = float(min_depth_floor)
        else:
            min_depth_config = min(min_depth_config, float(min_depth_floor))
    if min_depth_config is not None:
        required_depth_base = max(required_depth_base, min_depth_config)

    required_depth = required_depth_base / relaxation_factor if relaxation_factor else required_depth_base

    spread_limit_base = settings.hard_gate_max_spread_bps
    spread_floor = _safe_float(calibration.get("max_spread_bps"))
    if spread_floor is not None:
        spread_limit_base = max(spread_limit_base, float(spread_floor))
    spread_limit_adjusted = spread_limit_base * relaxation_factor

    slippage_limit_base = settings.hard_gate_max_slippage_bps
    slippage_floor = _safe_float(calibration.get("max_slippage_bps"))
    if slippage_floor is not None:
        slippage_limit_base = max(slippage_limit_base, float(slippage_floor))
    slippage_limit_adjusted = slippage_limit_base * relaxation_factor

    evaluated["liquidity_required_depth_oz"] = required_depth
    evaluated["liquidity_required_depth_base_oz"] = required_depth_base
    evaluated["liquidity_spread_limit_bps"] = spread_limit_adjusted
    evaluated["liquidity_slippage_limit_bps"] = slippage_limit_adjusted

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

    if spread_bps is not None and spread_bps > spread_limit_adjusted:
        violations.append(
            HardRiskViolation(
                code="LIQUIDITY_SPREAD_EXCEEDED",
                message="Bid-ask spread exceeds configured maximum",
                metric=spread_bps,
                limit=spread_limit_adjusted,
            )
        )

    if (
        estimated_slippage_bps is not None
        and estimated_slippage_bps > slippage_limit_adjusted
    ):
        violations.append(
            HardRiskViolation(
                code="LIQUIDITY_SLIPPAGE_EXCEEDED",
                message="Projected slippage exceeds configured maximum",
                metric=estimated_slippage_bps,
                limit=slippage_limit_adjusted,
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


def _evaluate_data_quality(settings: Settings, risk_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    quality_raw = _as_dict(risk_snapshot.get("data_quality"))

    provider = str(quality_raw.get("provider") or settings.data_provider or "").lower().strip()
    provider_label = quality_raw.get("provider_label") or provider or settings.data_provider
    data_mode = str(quality_raw.get("data_mode") or settings.data_mode or "live").lower().strip()

    age_minutes = _safe_float(quality_raw.get("age_minutes"))
    max_age_minutes = _safe_float(quality_raw.get("max_age_minutes"))
    if max_age_minutes is None and settings.market_data_max_age_minutes:
        max_age_minutes = float(settings.market_data_max_age_minutes)

    history_rows: Optional[int] = None
    rows_raw = quality_raw.get("history_rows")
    if rows_raw is not None:
        try:
            history_rows = int(rows_raw)
        except (TypeError, ValueError):
            history_rows = None

    profile = _DATA_PROVIDER_PROFILE_MAP.get(provider, "retail") if provider else "retail"
    if data_mode == "mock":
        profile = "simulated"

    calibration_template = _DATA_QUALITY_CALIBRATIONS.get(profile, _DATA_QUALITY_CALIBRATIONS["retail"])
    calibration = dict(calibration_template)

    stale = False
    fresh: Optional[bool] = None
    if age_minutes is not None and max_age_minutes is not None:
        stale = age_minutes > max_age_minutes
        fresh = not stale

    return {
        "provider": provider or None,
        "provider_label": provider_label,
        "data_mode": data_mode,
        "age_minutes": age_minutes,
        "max_age_minutes": max_age_minutes,
        "history_rows": history_rows,
        "last_timestamp": quality_raw.get("last_timestamp"),
        "profile": profile,
        "calibration": calibration,
        "stale": stale,
        "fresh": fresh,
    }


def enforce_hard_limits(
    response: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    settings: Settings,
) -> HardRiskGateReport:
    """Evaluate the final workflow response against hard risk limits."""

    if not settings.hard_gate_enabled:
        report = HardRiskGateReport(violations=[], evaluated_metrics={})
        _emit_risk_gate_audit(
            settings=settings,
            report=report,
            response=response,
            context=context,
            message="Hard risk gate disabled",
        )
        return report

    def _first_non_none(*values: Optional[float]) -> Optional[float]:
        for value in values:
            if value is not None:
                return value
        return None

    violations: List[HardRiskViolation] = []
    evaluated: Dict[str, Any] = {}

    risk_snapshot = _as_dict(context.get("risk_snapshot"))
    risk_metrics = _extract_risk_metrics(response)
    orders = _collect_orders(response)
    portfolio_state = _as_dict(context.get("portfolio_state"))

    data_quality = _evaluate_data_quality(settings, risk_snapshot)
    evaluated["data_quality"] = data_quality
    calibration = _as_dict(data_quality.get("calibration"))

    if data_quality.get("stale"):
        violations.append(
            HardRiskViolation(
                code="MARKET_DATA_STALE",
                message="Market data freshness window exceeded",
                details={
                    "provider": data_quality.get("provider_label") or data_quality.get("provider"),
                    "age_minutes": data_quality.get("age_minutes"),
                    "max_age_minutes": data_quality.get("max_age_minutes"),
                },
            )
        )
        if settings.hard_gate_fail_fast:
            report = HardRiskGateReport(violations=violations, evaluated_metrics=evaluated)
            _emit_risk_gate_audit(
                settings=settings,
                report=report,
                response=response,
                context=context,
                message="Market data failed freshness check",
            )
            return report

    primary_direction = _extract_primary_direction(response)
    target_position = _extract_target_position(response)
    strategy_tags = _extract_strategy_tags(response, context)

    evaluated["primary_direction"] = primary_direction
    evaluated["target_position_oz"] = target_position
    evaluated["strategy_tags"] = sorted(strategy_tags)

    coverage_config_tags: Set[str] = set()
    for value in settings.hard_gate_stop_coverage_exemptions:
        coverage_config_tags.update(_normalize_tag_values(value))
    coverage_config_tags = {tag for tag in coverage_config_tags if tag}
    evaluated["stop_coverage_configured_exemptions"] = sorted(coverage_config_tags)

    stop_coverage_exempt = False
    matching_coverage_exemptions: List[str] = []
    if coverage_config_tags:
        if {"ALL", "ANY", "UNIVERSAL"} & coverage_config_tags:
            stop_coverage_exempt = True
            matching_coverage_exemptions = sorted(coverage_config_tags)
        else:
            matching_coverage_exemptions = sorted(strategy_tags & coverage_config_tags)
            stop_coverage_exempt = bool(matching_coverage_exemptions)

    if not stop_coverage_exempt:
        for order in orders:
            order_exemption_tags = set(
                _normalize_tag_values(
                    order.get("risk_exemptions")
                    or order.get("risk_exemption")
                    or order.get("exemptions")
                )
            )
            matched = sorted(order_exemption_tags & coverage_config_tags)
            if matched:
                stop_coverage_exempt = True
                matching_coverage_exemptions = matched
                break

    evaluated["stop_coverage_exempt_match"] = matching_coverage_exemptions

    reported_utilization = _safe_float(risk_metrics.get("position_utilization"))
    if reported_utilization is None:
        reported_utilization = _safe_float(risk_snapshot.get("position_utilization"))
    evaluated["position_utilization_reported"] = reported_utilization

    current_position = _safe_float(risk_snapshot.get("current_position_oz"))
    if current_position is None:
        current_position = 0.0
    evaluated["current_position_oz"] = current_position

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

    evaluated["has_stop_protection"] = _has_stop_protection(orders)
    if settings.hard_gate_require_stop_loss and not evaluated["has_stop_protection"]:
        violations.append(
            HardRiskViolation(
                code="STOP_LOSS_MISSING",
                message="No stop-loss protection detected for execution orders",
                details={"orders_checked": len(orders)},
            )
        )

    stop_min_limit = _safe_float(calibration.get("min_stop_distance_pct"))
    stop_diagnostics = _evaluate_stop_requirements(
        orders,
        settings=settings,
        primary_direction=primary_direction,
        coverage_exempt=stop_coverage_exempt,
        min_distance_limit=stop_min_limit,
    )
    evaluated.update(stop_diagnostics["evaluated"])
    violations.extend(stop_diagnostics["violations"])

    exposures = stop_diagnostics["exposures"]
    planned_primary_exposure = 0.0
    if exposures:
        if primary_direction:
            planned_primary_exposure = sum(
                size for (instrument, side), size in exposures.items() if side == primary_direction
            )
        else:
            planned_primary_exposure = sum(exposures.values())
    evaluated["planned_primary_exposure_oz"] = planned_primary_exposure
    evaluated["planned_total_exposure_oz"] = sum(exposures.values())

    current_abs = abs(current_position)
    target_abs = abs(target_position) if target_position is not None else None
    if target_abs is None:
        target_abs = current_abs + planned_primary_exposure

    planned_incremental_from_target = max(0.0, target_abs - current_abs)
    planned_incremental_oz = max(planned_primary_exposure, planned_incremental_from_target)

    max_position_limit = settings.max_position_oz if settings.max_position_oz and settings.max_position_oz > 0 else None

    total_utilization_planned: Optional[float] = None
    incremental_utilization: Optional[float] = None
    if max_position_limit:
        total_utilization_planned = target_abs / max_position_limit if target_abs is not None else None
        incremental_utilization = planned_incremental_oz / max_position_limit

    evaluated["position_incremental_oz"] = planned_incremental_oz
    evaluated["position_incremental_utilization"] = incremental_utilization
    evaluated["position_utilization_planned"] = total_utilization_planned

    risk_regime = str(risk_snapshot.get("risk_regime") or context.get("risk_regime") or "").lower()
    regime_key = risk_regime or "routine"
    evaluated["position_regime"] = regime_key

    routine_cap = settings.hard_gate_max_position_utilization_routine
    elevated_cap = settings.hard_gate_max_position_utilization_elevated
    hard_cap = settings.hard_gate_max_position_utilization_hard_limit
    legacy_cap = settings.hard_gate_max_position_utilization
    incremental_limit = settings.hard_gate_incremental_position_utilization_limit

    evaluated["position_utilization_limit_routine"] = routine_cap
    evaluated["position_utilization_limit_elevated"] = elevated_cap
    evaluated["position_utilization_limit_hard"] = hard_cap
    evaluated["position_utilization_limit_legacy"] = legacy_cap
    evaluated["position_incremental_limit"] = incremental_limit

    if regime_key in {"stress", "halt", "crisis", "hard"}:
        active_cap = _first_non_none(hard_cap, elevated_cap, routine_cap, legacy_cap)
    elif regime_key in {"elevated", "heightened", "watch"}:
        active_cap = _first_non_none(elevated_cap, hard_cap, routine_cap, legacy_cap)
    else:
        active_cap = _first_non_none(routine_cap, legacy_cap, elevated_cap, hard_cap)
    if active_cap is None:
        active_cap = 1.0
    evaluated["position_utilization_limit"] = active_cap

    utilization_for_limit = total_utilization_planned if total_utilization_planned is not None else reported_utilization
    evaluated["position_utilization"] = utilization_for_limit

    if (
        utilization_for_limit is not None
        and active_cap is not None
        and utilization_for_limit > active_cap
    ):
        violations.append(
            HardRiskViolation(
                code="POSITION_UTILIZATION",
                message="Proposed plan exceeds active position utilization limit",
                metric=utilization_for_limit,
                limit=active_cap,
                details={"regime": regime_key},
            )
        )

    hard_cap_enforced = _first_non_none(hard_cap, legacy_cap)
    if (
        hard_cap_enforced is not None
        and utilization_for_limit is not None
        and utilization_for_limit > hard_cap_enforced
        and hard_cap_enforced > (active_cap or 0.0)
    ):
        violations.append(
            HardRiskViolation(
                code="POSITION_UTILIZATION_HARD_CAP",
                message="Proposed plan exceeds absolute position utilization limit",
                metric=utilization_for_limit,
                limit=hard_cap_enforced,
            )
        )

    if (
        incremental_limit is not None
        and incremental_utilization is not None
        and incremental_utilization > incremental_limit
    ):
        violations.append(
            HardRiskViolation(
                code="POSITION_INCREMENTAL_LIMIT",
                message="Incremental position sizing exceeds per-trade utilization limit",
                metric=incremental_utilization,
                limit=incremental_limit,
                details={"incremental_oz": planned_incremental_oz, "max_position_oz": settings.max_position_oz},
            )
        )

    liquidity_diagnostics = _evaluate_liquidity(
        exposures=exposures,
        risk_snapshot=risk_snapshot,
        settings=settings,
        calibration=calibration,
    )
    evaluated.update(liquidity_diagnostics["evaluated"])
    violations.extend(liquidity_diagnostics["violations"])

    stress_loss = _safe_float(risk_metrics.get("stress_test_worst_loss_millions"))
    evaluated["stress_test_worst_loss_millions"] = stress_loss

    stress_limit = settings.hard_gate_max_stress_loss_millions
    if stress_limit is None:
        stress_limit = settings.stress_var_millions
    stress_warning = settings.hard_gate_stress_loss_warning_millions
    stress_circuit = settings.hard_gate_stress_loss_circuit_breaker_millions

    evaluated["stress_loss_limit_millions"] = stress_limit
    evaluated["stress_loss_warning_threshold_millions"] = stress_warning
    evaluated["stress_loss_circuit_breaker_millions"] = stress_circuit
    evaluated["stress_loss_warning_triggered"] = False

    if stress_loss is not None and stress_loss < 0:
        stress_abs = abs(stress_loss)
        if stress_circuit is not None and stress_abs >= stress_circuit:
            violations.append(
                HardRiskViolation(
                    code="STRESS_LOSS_CIRCUIT_BREAKER",
                    message="Stress scenario loss breaches circuit breaker threshold",
                    metric=stress_abs,
                    limit=stress_circuit,
                )
            )
        elif stress_limit is not None and stress_abs > stress_limit:
            violations.append(
                HardRiskViolation(
                    code="STRESS_LOSS_LIMIT",
                    message="Stress scenario loss exceeds configured maximum",
                    metric=stress_abs,
                    limit=stress_limit,
                )
            )
        elif stress_warning is not None and stress_abs >= stress_warning:
            evaluated["stress_loss_warning_triggered"] = True

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

    warning_threshold = settings.hard_gate_correlation_warning_threshold
    limit_threshold = settings.hard_gate_correlation_limit_threshold
    block_threshold = settings.hard_gate_correlation_block_threshold
    legacy_threshold = settings.hard_gate_correlation_threshold

    if block_threshold is None:
        block_threshold = legacy_threshold
    if limit_threshold is None:
        limit_threshold = block_threshold
    if warning_threshold is None:
        warning_threshold = limit_threshold

    evaluated["correlation_threshold_warning"] = warning_threshold
    evaluated["correlation_threshold_limit"] = limit_threshold
    evaluated["correlation_threshold_block"] = block_threshold
    evaluated["correlation_threshold"] = block_threshold

    correlations = risk_snapshot.get("cross_asset_correlations")
    correlation_warnings: List[Dict[str, Any]] = []
    correlation_limits: List[Dict[str, Any]] = []
    correlation_blocks: List[Dict[str, Any]] = []
    if isinstance(correlations, list):
        for entry in correlations:
            if not isinstance(entry, Mapping):
                continue
            value = _safe_float(entry.get("value"))
            if value is None:
                continue
            label = entry.get("label") or entry.get("symbol")
            record = {"label": label, "value": value}
            abs_value = abs(value)
            if block_threshold is not None and abs_value >= block_threshold:
                correlation_blocks.append(record)
            elif limit_threshold is not None and abs_value >= limit_threshold:
                correlation_limits.append(record)
            elif warning_threshold is not None and abs_value >= warning_threshold:
                correlation_warnings.append(record)
    evaluated["correlation_warnings"] = correlation_warnings
    evaluated["correlation_limits"] = correlation_limits
    evaluated["correlation_blocks"] = correlation_blocks
    evaluated["correlation_breaches"] = correlation_blocks + correlation_limits

    if block_threshold is not None:
        for pair in correlation_blocks:
            violations.append(
                HardRiskViolation(
                    code="CORRELATION_BLOCK",
                    message=f"Correlation {pair['label']} {pair['value']:.2f} exceeds block threshold {block_threshold:.2f}",
                    metric=pair["value"],
                    limit=block_threshold,
                )
            )
    if limit_threshold is not None:
        for pair in correlation_limits:
            violations.append(
                HardRiskViolation(
                    code="CORRELATION_LIMIT",
                    message=f"Correlation {pair['label']} {pair['value']:.2f} exceeds limit threshold {limit_threshold:.2f}",
                    metric=pair["value"],
                    limit=limit_threshold,
                )
            )

    circuit_breaker = evaluate_circuit_breaker(
        settings=settings,
        risk_snapshot=risk_snapshot,
        portfolio_state=portfolio_state,
    )
    evaluated.update(circuit_breaker.evaluated)
    if circuit_breaker.state_patch:
        evaluated["circuit_breaker_state_patch"] = circuit_breaker.state_patch
    for violation in circuit_breaker.violations:
        violations.append(
            HardRiskViolation(
                code=violation.code,
                message=violation.message,
                metric=violation.metric,
                limit=violation.limit,
            )
        )

    if violations:
        for violation in violations:
            logger.error("硬风控约束触发：%s -> %s", violation.code, violation.message)

    report = HardRiskGateReport(violations=violations, evaluated_metrics=evaluated)
    _emit_risk_gate_audit(
        settings=settings,
        report=report,
        response=response,
        context=context,
    )
    return report