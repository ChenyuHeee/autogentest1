"""Structured audit logging utilities."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

from ..config.settings import Settings, get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

_AUDIT_LOCK = threading.Lock()
_DEFAULT_SEVERITY = "INFO"


@dataclass(frozen=True)
class AuditWriteResult:
    """Outcome returned when persisting an audit event."""

    event: Dict[str, Any]
    path: Optional[Path]


def _stringify(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive
        return repr(value)


def _normalize_tags(tags: Optional[Sequence[Any]]) -> list[str]:
    if not tags:
        return []
    normalized: list[str] = []
    for item in tags:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _resolve_directory(settings: Settings) -> Path:
    base = Path(settings.audit_log_directory).expanduser()
    if not base.is_absolute():
        package_root = Path(__file__).resolve().parent.parent
        base = package_root / base
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_event(
    *,
    event_type: str,
    severity: str,
    message: Optional[str],
    payload: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
    tags: Optional[Sequence[Any]],
    correlation_id: Optional[str],
    component: Optional[str],
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    event: Dict[str, Any] = {
        "event_id": str(uuid4()),
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "severity": severity,
    }
    if component:
        event["component"] = component
    if message:
        event["message"] = message
    if correlation_id:
        event["correlation_id"] = correlation_id
    serialized_payload = dict(payload) if isinstance(payload, Mapping) else None
    if serialized_payload:
        event["payload"] = serialized_payload
    serialized_context = dict(context) if isinstance(context, Mapping) else None
    if serialized_context:
        event["context"] = serialized_context
    normalized_tags = _normalize_tags(tags)
    if normalized_tags:
        event["tags"] = normalized_tags
    event["_ts_epoch"] = timestamp.timestamp()
    return event


def record_audit_event(
    event_type: str,
    *,
    severity: str = _DEFAULT_SEVERITY,
    message: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    tags: Optional[Sequence[Any]] = None,
    correlation_id: Optional[str] = None,
    component: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> AuditWriteResult:
    """Persist a structured audit record to the configured log sink."""

    active_settings = settings or get_settings()
    if not getattr(active_settings, "audit_log_enabled", True):
        event = _build_event(
            event_type=event_type,
            severity=severity.upper(),
            message=message,
            payload=payload,
            context=context,
            tags=tags,
            correlation_id=correlation_id,
            component=component,
        )
        return AuditWriteResult(event=event, path=None)

    severity_label = severity.upper() if severity else _DEFAULT_SEVERITY

    event = _build_event(
        event_type=event_type,
        severity=severity_label,
        message=message,
        payload=payload,
        context=context,
        tags=tags,
        correlation_id=correlation_id,
        component=component,
    )

    try:
        directory = _resolve_directory(active_settings)
        timestamp = datetime.fromisoformat(event["timestamp"])
        file_path = directory / f"audit-{timestamp:%Y%m%d}.jsonl"
        serialized = json.dumps(event, default=_stringify, ensure_ascii=True)
        with _AUDIT_LOCK:
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
        return AuditWriteResult(event=event, path=file_path)
    except Exception as exc:  # pragma: no cover - file system failures are rare
        logger.exception("Audit log write failed: %s", exc)
        return AuditWriteResult(event=event, path=None)