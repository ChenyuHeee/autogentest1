"""Adapter wrapping Yahoo Finance for market data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from requests import Session

from .base import MarketDataAdapter


class YahooFinanceAdapter(MarketDataAdapter):
    """Fetch OHLCV data using the ``yfinance`` package."""

    def fetch_price_history(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        session: Optional[Session] = None,
    ) -> pd.DataFrame:
        data = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            session=session,
            auto_adjust=False,
            progress=False,
        )
        data.index = pd.to_datetime(data.index)
        return data
