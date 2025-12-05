"""Utility for exporting effective application settings.

This module can be invoked via ``python -m autogentest1.config.dump_settings``
so operators can snapshot configuration prior to deployment.  Secrets are
redacted by default to avoid leaking credentials into logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .settings import Settings, get_settings

_REDACTION_TOKENS: tuple[str, ...] = ("key", "secret", "token", "password", "credential")
_REDACTED_PLACEHOLDER = "<redacted>"


def _needs_redaction(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _REDACTION_TOKENS)


def _sanitize(value: Any, *, parent_key: str | None = None) -> Any:
    if parent_key and _needs_redaction(parent_key):
        # Always hide sensitive values unless explicitly requested otherwise.
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [_REDACTED_PLACEHOLDER if item is not None else None for item in value]
        if isinstance(value, Mapping):
            # Retain structure but redact every leaf.
            return {key: _REDACTED_PLACEHOLDER for key in value.keys()}
        return _REDACTED_PLACEHOLDER

    if isinstance(value, Mapping):
        return {key: _sanitize(sub_value, parent_key=key) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item, parent_key=parent_key) for item in value)
    return value


def dump_settings_dict(*, include_secrets: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    """Return the effective settings as a dictionary.

    Parameters
    ----------
    include_secrets:
        When ``True`` sensitive fields are returned verbatim.  Defaults to
        ``False`` which redacts values whose keys contain any of
        ``_REDACTION_TOKENS``.
    settings:
        Optionally provide a :class:`Settings` instance (mainly for testing).
    """

    effective_settings = settings or get_settings()
    payload = effective_settings.model_dump()
    if include_secrets:
        return payload
    return {key: _sanitize(value, parent_key=key) for key, value in payload.items()}


def _format_payload(payload: Mapping[str, Any], fmt: str, *, pretty: bool) -> str:
    if fmt == "json":
        indent = 2 if pretty else None
        return json.dumps(payload, indent=indent, sort_keys=pretty)
    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("PyYAML is required for --format yaml") from exc
        if pretty:
            return yaml.safe_dump(payload, sort_keys=False)
        return yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    raise ValueError(f"Unsupported format: {fmt}")  # pragma: no cover - guarded by argparse


def _write_output(text: str, destination: Path | None) -> None:
    if destination is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export effective AutoGenTest1 settings")
    parser.add_argument("--format", choices=("json", "yaml"), default="json", help="Output format")
    parser.add_argument("--output", type=Path, help="Optional file to write instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON/YAML output")
    parser.add_argument(
        "--include-secrets",
        action="store_true",
        help="Include sensitive values (API keys, tokens, passwords). Use with caution.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = dump_settings_dict(include_secrets=args.include_secrets)
    text = _format_payload(payload, args.format, pretty=args.pretty)
    _write_output(text, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
