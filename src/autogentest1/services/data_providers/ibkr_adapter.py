"""Placeholder adapter for Interactive Brokers TWS market data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from requests import Session

from .base import MarketDataAdapter


class IBKRAdapter(MarketDataAdapter):
    """Stub adapter raising NotImplementedError until IBKR integration is configured."""

    def fetch_price_history(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        session: Optional[Session] = None,
    ) -> pd.DataFrame:  # pragma: no cover - integration placeholder
        raise NotImplementedError(
            "Interactive Brokers adapter not yet implemented. Configure TWS connection parameters before use."
        )
