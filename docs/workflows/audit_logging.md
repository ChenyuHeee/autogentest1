# Structured Audit Logging

This workflow note summarizes how the system records structured audit events so that
risk, compliance, and operations teams can reconstruct the decision trail for every
run.

## Overview

- **Sink**: JSON Lines files written under `~/.autogentest1/audit/` by default
  (override via the `AUDIT_LOG_DIRECTORY` setting or environment variable).
- **Format**: Each event contains timestamped metadata including component name,
  severity, payload summary, and optional tags.
- **Rotation**: Files are grouped by calendar day using the pattern
  `audit-YYYYMMDD.jsonl`.
- **Enablement**: Controlled by Pydantic settings:
  - `audit_log_enabled` (default `True`)
  - `audit_log_directory` (default `~/.autogentest1/audit`, accepts absolute paths
    or project-relative paths)

## Event Structure

```json
{
  "event_id": "1699bde5-ce8b-4a2f-9d05-9fb30d29bf61",
  "timestamp": "2025-12-05T11:42:31.284921+00:00",
  "event_type": "risk_gate.evaluation",
  "component": "risk_gate",
  "severity": "ERROR",
  "message": "Hard risk gate evaluation completed",
  "tags": ["XAUUSD", "Phase 4", "BREACHED"],
  "payload": {
    "breached": true,
    "violations": [
      {"code": "STOP_COVERAGE_SHORTFALL", "message": "Stop coverage below limit", "metric": 0.72, "limit": 1.0}
    ],
    "evaluated_metrics": {"position_utilization": 0.82, "stress_test_worst_loss_millions": 3.1}
  },
  "context": {"symbol": "XAUUSD", "phase": "Phase 4", "status": "BLOCKED"}
}
```

## Producers

1. **Risk Gate** (`services.risk_gate.enforce_hard_limits`)
   - Emits after every evaluation, regardless of pass/fail.
   - Tags include the workflow phase (if present) and a `BREACHED`/`PASSED` flag.
   - Payload captures the violation list and the evaluated metrics used for the
     decision.
2. **Circuit Breaker** (`services.circuit_breaker.evaluate_circuit_breaker`)
   - Emits whether or not triggers fire, allowing operators to trace state patches.
   - Payload includes the latest risk controls snapshot and any state updates the
     module applied (e.g., cooldown windows).

## Operational Notes

- **Parsing**: The JSONL format is streaming-friendly; a single `jq` or pandas read
  can reconstruct multi-day audit trails.
- **Storage hygiene**: Rotate or archive the audit directory (`~/.autogentest1/audit/`
  by default) regularly if
  the workflow runs at high frequency.
- **Scenario testing**: Unit tests cover audit writes for both the risk gate and
  circuit breaker to prevent regressions when the logging contract changes.

With structured audit events in place, production incidents can be investigated or
handed to regulators without scraping free-form agent transcripts.
