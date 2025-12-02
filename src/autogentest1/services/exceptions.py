"""Domain-specific exceptions for service layers."""

from __future__ import annotations


class DataProviderError(RuntimeError):
    """Raised when a market data provider fails irrecoverably."""


class DataStalenessError(RuntimeError):
    """Raised when fetched market data is older than the allowed freshness window."""


class WorkflowFormatError(RuntimeError):
    """Raised when an agent response fails structured validation after retries."""


class WorkflowEscalationRequired(RuntimeError):
    """Raised when automated negotiation cannot resolve conflicts and human input is required."""
