"""Structured validation helpers for agent responses."""

from __future__ import annotations

from typing import Any, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError


class ResponseDetails(BaseModel):
    """Container for nested details; allows arbitrary payloads."""

    model_config = ConfigDict(extra="allow")


class WorkflowResponseModel(BaseModel):
    """Expected shape for final workflow response objects."""

    phase: str
    status: str
    summary: str
    details: ResponseDetails

    model_config = ConfigDict(extra="allow")


def validate_workflow_response(payload: Any) -> Tuple[bool, str | None]:
    """Validate payload against the workflow schema.

    Returns ``(True, None)`` when valid; otherwise ``(False, error_message)``.
    """

    try:
        model = WorkflowResponseModel.model_validate(payload)
    except ValidationError as exc:
        return False, exc.errors(include_context=False, include_url=False).__repr__()
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    if model.status != "COMPLETE":
        return False, f"Expected final status 'COMPLETE' but received '{model.status}'"
    if "Phase 5" not in model.phase:
        return False, f"Expected final phase to reference 'Phase 5' but received '{model.phase}'"
    return True, None
