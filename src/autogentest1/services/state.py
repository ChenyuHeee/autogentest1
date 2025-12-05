"""Persistent portfolio state helpers."""

from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Mapping

from ..utils.logging import get_logger

logger = get_logger(__name__)

_STATE_FILENAME = "portfolio_state.json"


def _state_file_path() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs" / _STATE_FILENAME


def _default_state() -> Dict[str, Any]:
    return {
        "positions": {
            "symbol": "XAUUSD",
            "net_oz": 0.0,
            "average_cost": None,
        },
        "pnl": {
            "realized_millions": 0.0,
            "unrealized_millions": 0.0,
        },
        "risk_controls": {
            "consecutive_losses": 0,
            "baseline_vol_annualized": None,
            "last_triggered_at": None,
            "last_evaluated_at": None,
            "cooldown_until": None,
        },
        "last_updated": None,
    }


def load_portfolio_state() -> Dict[str, Any]:
    """Load portfolio state from disk or return defaults."""

    path = _state_file_path()
    if not path.exists():
        logger.info("未找到状态文件：%s，使用默认参数", path)
        return _default_state()

    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.error("状态文件解析失败 %s：%s", path, exc)
        return _default_state()

    # Ensure missing keys are populated to avoid KeyError downstream.
    defaults = _default_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            for sub_key, sub_value in value.items():
                state[key].setdefault(sub_key, sub_value)
    return state


def save_portfolio_state(state: Dict[str, Any]) -> Path:
    """Persist the portfolio state to disk."""

    path = _state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    logger.info("组合状态已保存：%s", path)
    return path


def update_portfolio_state(update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge partial updates into the stored portfolio state and persist."""

    state = load_portfolio_state()
    state.update(update)
    save_portfolio_state(state)
    return state


def _deep_merge(target: Dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def apply_portfolio_state_patch(patch: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply a nested state patch and persist the merged portfolio state."""

    if not patch:
        logger.debug("空的组合状态补丁，直接返回当前状态")
        return load_portfolio_state()

    state = load_portfolio_state()
    merged = deepcopy(state)
    _deep_merge(merged, patch)
    logger.info("应用组合状态补丁：%s", json.dumps(patch, ensure_ascii=False))
    save_portfolio_state(merged)
    return merged