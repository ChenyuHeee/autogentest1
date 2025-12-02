"""Base interfaces for market and news data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from requests import Session


class MarketDataAdapter(ABC):
    """Abstract adapter for fetching OHLCV style market data."""

    @abstractmethod
    def fetch_price_history(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        session: Optional[Session] = None,
    ) -> pd.DataFrame:
        """Return price history for ``symbol`` in the given window."""


class NewsDataAdapter(ABC):
    """Abstract adapter for retrieving structured news or sentiment payloads."""

    @abstractmethod
    def fetch_sentiment(self, symbol: str, *, limit: int = 20) -> Dict[str, Any]:
        """Return sentiment payload for ``symbol``."""
