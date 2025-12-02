"""CLI entry point for running the gold outlook workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .config.settings import get_settings
from .services.news_watcher import run_default_watcher
from .utils.logging import configure_logging, get_logger
from .workflows.gold_outlook import run_gold_outlook

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the workflow."""

    parser = argparse.ArgumentParser(description="Run the AutoGen gold outlook workflow.")
    parser.add_argument("--symbol", default=None, help="Symbol or ticker to analyze (default from settings)")
    parser.add_argument("--days", type=int, default=None, help="Lookback window in days (default from settings)")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response instead of formatted text")
    parser.add_argument("--watch-news", action="store_true", help="Start the asynchronous news watcher")
    return parser.parse_args()


def main() -> None:
    """Bootstrap application settings and execute the workflow."""

    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    symbol = args.symbol or settings.default_symbol
    days = args.days or settings.default_days

    if args.watch_news:
        asyncio.run(run_default_watcher())
        return

    logger.info("Running workflow for symbol=%s days=%d", symbol, days)
    result: dict[str, Any] = run_gold_outlook(symbol=symbol, days=days, settings=settings)

    if args.raw:
        print(json.dumps(result, indent=2))
    else:
        print(result.get("response"))


if __name__ == "__main__":
    main()
