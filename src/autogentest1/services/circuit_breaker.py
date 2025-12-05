"""Circuit breaker evaluation utilities for risk management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

from ..config.settings import Settings
from .audit import record_audit_event
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class CircuitBreakerViolation:
    code: str
    message: str
    metric: Optional[float] = None
    limit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.metric is not None:
            payload["metric"] = self.metric
        if self.limit is not None:
            payload["limit"] = self.limit
        return payload


@dataclass(frozen=True)
class CircuitBreakerEvaluation:
    triggered: bool
    violations: Sequence[CircuitBreakerViolation]
    evaluated: Dict[str, Any]
    state_patch: Dict[str, Any]


def _emit_circuit_breaker_audit(
    *,
    settings: Settings,
    evaluation: CircuitBreakerEvaluation,
    message: Optional[str] = None,
) -> None:
    try:
        payload: Dict[str, Any] = {
            "triggered": evaluation.triggered,
            "evaluated": evaluation.evaluated,
        }
        if evaluation.violations:
            payload["violations"] = [violation.to_dict() for violation in evaluation.violations]
        if evaluation.state_patch:
            payload["state_patch"] = evaluation.state_patch
        record_audit_event(
            "circuit_breaker.evaluation",
            severity="ERROR" if evaluation.triggered else "INFO",
            message=message or "Circuit breaker evaluation completed",
            payload=payload,
            context={},
            component="circuit_breaker",
            settings=settings,
        )
    except Exception:  # pragma: no cover - logging failures should not break flow
        logger.exception("Failed to emit circuit breaker audit event")


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def evaluate_circuit_breaker(
    *,
    settings: Settings,
    risk_snapshot: Mapping[str, Any],
    portfolio_state: Mapping[str, Any] | None = None,
    current_time: Optional[datetime] = None,
) -> CircuitBreakerEvaluation:
    """Evaluate circuit breaker triggers and provide state updates."""

    evaluated: Dict[str, Any] = {}
    violations: list[CircuitBreakerViolation] = []

    if not settings.circuit_breaker_enabled:
        evaluation = CircuitBreakerEvaluation(False, tuple(), evaluated, {})
        _emit_circuit_breaker_audit(
            settings=settings,
            evaluation=evaluation,
            message="Circuit breaker disabled",
        )
        return evaluation

    now = current_time or datetime.now(timezone.utc)
    controls = dict(portfolio_state.get("risk_controls", {})) if portfolio_state else {}

    consecutive_losses = int(controls.get("consecutive_losses", 0) or 0)
    daily_pnl = risk_snapshot.get("pnl_today_millions")
    current_vol = risk_snapshot.get("realized_vol_annualized")
    baseline_vol = controls.get("baseline_vol_annualized")

    cooldown_until = _parse_timestamp(controls.get("cooldown_until"))
    cooldown_active = bool(cooldown_until and cooldown_until > now)

    evaluated.update(
        {
            "circuit_breaker_consecutive_losses": consecutive_losses,
            "circuit_breaker_daily_pnl_millions": daily_pnl,
            "circuit_breaker_current_vol": current_vol,
            "circuit_breaker_baseline_vol": baseline_vol,
            "circuit_breaker_cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
        }
    )

    if cooldown_active:
        violations.append(
            CircuitBreakerViolation(
                code="CIRCUIT_BREAKER_ACTIVE_COOLDOWN",
                message="Circuit breaker cooldown window still active",
            )
        )

    daily_limit = settings.circuit_breaker_daily_loss_limit_millions
    if daily_limit > 0 and isinstance(daily_pnl, (int, float)) and daily_pnl <= -float(daily_limit):
        violations.append(
            CircuitBreakerViolation(
                code="CIRCUIT_BREAKER_DAILY_LOSS",
                message="Daily PnL loss breached circuit breaker limit",
                metric=float(daily_pnl),
                limit=float(-daily_limit),
            )
        )

    loss_limit = settings.circuit_breaker_max_consecutive_losses
    if loss_limit > 0 and consecutive_losses >= loss_limit:
        violations.append(
            CircuitBreakerViolation(
                code="CIRCUIT_BREAKER_CONSECUTIVE_LOSSES",
                message="Consecutive loss count hit circuit breaker threshold",
                metric=float(consecutive_losses),
                limit=float(loss_limit),
            )
        )

    spike_multiple = settings.circuit_breaker_vol_spike_multiple
    if (
        spike_multiple > 0
        and isinstance(current_vol, (int, float))
        and isinstance(baseline_vol, (int, float))
        and baseline_vol > 0
        and current_vol >= float(baseline_vol) * float(spike_multiple)
    ):
        violations.append(
            CircuitBreakerViolation(
                code="CIRCUIT_BREAKER_VOL_SPIKE",
                message="Realized volatility spike breached circuit breaker multiple",
                metric=float(current_vol),
                limit=float(baseline_vol) * float(spike_multiple),
            )
        )

    triggered = bool(violations)

    new_controls = dict(controls)
    new_controls["last_evaluated_at"] = now.isoformat()

    if triggered and not cooldown_active:
        cooldown_duration = max(1, int(settings.circuit_breaker_cooldown_minutes))
        cooldown_target = now + timedelta(minutes=cooldown_duration)
        new_controls["last_triggered_at"] = now.isoformat()
        new_controls["cooldown_until"] = cooldown_target.isoformat()

    state_patch: Dict[str, Any] = {}
    if new_controls != controls:
        state_patch["risk_controls"] = new_controls

    if triggered:
        codes = [violation.code for violation in violations]
        logger.warning(
            "触发熔断：codes=%s daily_pnl=%s consecutive_losses=%s",
            codes,
            daily_pnl,
            consecutive_losses,
        )
    elif state_patch:
        cooldown_until = new_controls.get("cooldown_until")
        last_evaluated_at = new_controls.get("last_evaluated_at")
        logger.info(
            "熔断状态更新：cooldown_until=%s last_evaluated_at=%s",
            cooldown_until,
            last_evaluated_at,
        )

    evaluation = CircuitBreakerEvaluation(triggered, tuple(violations), evaluated, state_patch)
    _emit_circuit_breaker_audit(settings=settings, evaluation=evaluation)
    return evaluation