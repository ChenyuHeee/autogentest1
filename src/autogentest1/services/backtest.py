"""Lightweight backtesting utilities for QuantResearchAgent and tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .sentiment import collect_sentiment_snapshot

TRADING_DAYS_PER_YEAR = 252


@dataclass(slots=True)
class Trade:
    """Structured representation of a single round-trip trade."""

    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "return_pct": self.return_pct,
            "pnl": self.pnl,
        }


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest output compatible with JSON serialization."""

    symbol: str
    strategy: str
    start: str
    end: str
    initial_capital: float
    equity_curve: List[Dict[str, Any]]
    metrics: Dict[str, Optional[float]]
    trades: List[Trade]
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "start": self.start,
            "end": self.end,
            "initial_capital": self.initial_capital,
            "equity_curve": self.equity_curve,
            "metrics": self.metrics,
            "trades": [trade.to_dict() for trade in self.trades],
            "parameters": self.parameters,
        }


def _ensure_history(history: Any) -> pd.DataFrame:
    if history is None:
        raise ValueError("Historical price data is required")
    if isinstance(history, pd.Series):
        df = history.to_frame(name="Close")
    else:
        df = pd.DataFrame(history).copy()
    if "Close" not in df.columns:
        raise ValueError("History must contain a 'Close' column")
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.loc[:, [col for col in df.columns if col == "Close"]]
    return df


def _initialise_outputs(
    index: pd.Index,
    equity: pd.Series,
    close: pd.Series,
    positions: pd.Series,
) -> List[Dict[str, Any]]:
    dt_index = pd.DatetimeIndex(index)
    drawdown = equity / equity.cummax() - 1
    records: List[Dict[str, Any]] = []
    for idx, timestamp in enumerate(dt_index):
        records.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "equity": float(equity.iloc[idx]),
                "drawdown": float(drawdown.iloc[idx]),
                "price": float(close.iloc[idx]),
                "position": float(positions.iloc[idx]),
            }
        )
    return records


def _collect_trades(
    close: pd.Series,
    equity: pd.Series,
    positions: pd.Series,
) -> List[Trade]:
    trades: List[Trade] = []
    prev_position = 0.0
    open_trade: Optional[Dict[str, Any]] = None

    for idx, timestamp in enumerate(positions.index):
        position = float(positions.iloc[idx])
        if position == prev_position:
            continue
        
        # When position changes at 'idx', it means the trade was entered/exited
        # based on the signal from the PREVIOUS bar (idx-1).
        # So the entry/exit price is the Close of the previous bar.
        # And the entry/exit equity is the Equity at the end of the previous bar.
        
        if idx > 0:
            trade_date = positions.index[idx-1]
            trade_price = float(close.iloc[idx-1])
            trade_equity = float(equity.iloc[idx-1])
        else:
            # Edge case: Position starts non-zero at index 0.
            # We assume entry at start of simulation.
            trade_date = positions.index[0]
            trade_price = float(close.iloc[0])
            trade_equity = float(equity.iloc[0]) # This is initial_capital if returns[0]=0

        ts = pd.Timestamp(trade_date)
        
        if position > prev_position:  # Enter long
            open_trade = {
                "entry_date": ts.strftime("%Y-%m-%d"),
                "entry_price": trade_price,
                "entry_equity": trade_equity,
            }
        elif position < prev_position and open_trade:
            # Exit long position
            return_pct = (trade_equity / open_trade["entry_equity"]) - 1.0
            pnl = trade_equity - open_trade["entry_equity"]
            trades.append(
                Trade(
                    entry_date=open_trade["entry_date"],
                    exit_date=ts.strftime("%Y-%m-%d"),
                    direction="long",
                    entry_price=open_trade["entry_price"],
                    exit_price=trade_price,
                    return_pct=float(return_pct),
                    pnl=float(pnl),
                )
            )
            open_trade = None
        prev_position = position

    # If a trade is still open, close it at the final bar
    if open_trade:
        last_timestamp = positions.index[-1]
        last_price = float(close.iloc[-1])
        last_equity = float(equity.iloc[-1])
        return_pct = (last_equity / open_trade["entry_equity"]) - 1.0
        pnl = last_equity - open_trade["entry_equity"]
        trades.append(
            Trade(
                entry_date=open_trade["entry_date"],
                exit_date=last_timestamp.strftime("%Y-%m-%d"),
                direction="long",
                entry_price=open_trade["entry_price"],
                exit_price=last_price,
                return_pct=float(return_pct),
                pnl=float(pnl),
            )
        )

    return trades


def _compute_metrics(
    equity: pd.Series,
    strategy_returns: pd.Series,
    trades: Iterable[Trade],
    initial_capital: float,
    *,
    risk_free_rate: float,
) -> Dict[str, Optional[float]]:
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    periods = len(strategy_returns)
    annualized_return: Optional[float] = None
    if periods > 0:
        years = periods / TRADING_DAYS_PER_YEAR
        if years > 0:
            annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    annualized_vol = float(strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if periods > 1 else None
    sharpe_ratio: Optional[float] = None
    if annualized_vol and annualized_vol > 0 and annualized_return is not None:
        sharpe_ratio = float((annualized_return - risk_free_rate) / annualized_vol)

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    trade_list = list(trades)
    win_rate: Optional[float] = None
    avg_trade_return: Optional[float] = None
    if trade_list:
        wins = sum(1 for trade in trade_list if trade.return_pct > 0)
        win_rate = wins / len(trade_list)
        avg_trade_return = float(np.mean([trade.return_pct for trade in trade_list]))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "trades": float(len(trade_list)),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }


def _buy_and_hold_strategy(close: pd.Series, initial_capital: float) -> Dict[str, Any]:
    returns = close.pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod() * initial_capital
    positions = pd.Series(data=1.0, index=close.index)
    trades = [
        Trade(
            entry_date=close.index[0].strftime("%Y-%m-%d"),
            exit_date=close.index[-1].strftime("%Y-%m-%d"),
            direction="long",
            entry_price=float(close.iloc[0]),
            exit_price=float(close.iloc[-1]),
            return_pct=float(close.iloc[-1] / close.iloc[0] - 1.0),
            pnl=float(equity.iloc[-1] - initial_capital),
        )
    ]
    return {
        "equity": equity,
        "strategy_returns": returns,
        "positions": positions,
        "trades": trades,
    }


def _sma_crossover_strategy(
    close: pd.Series,
    initial_capital: float,
    *,
    short_window: int,
    long_window: int,
) -> Dict[str, Any]:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    short_ma = close.rolling(window=short_window, min_periods=1).mean()
    long_ma = close.rolling(window=long_window, min_periods=1).mean()
    raw_signal = (short_ma > long_ma).astype(float)
    positions = raw_signal.shift(1).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    strategy_returns = returns * positions
    equity = (1.0 + strategy_returns).cumprod() * initial_capital
    trades = _collect_trades(close, equity, positions)
    return {
        "equity": equity,
        "strategy_returns": strategy_returns,
        "positions": positions,
        "trades": trades,
    }


def _mean_reversion_strategy(
    close: pd.Series,
    initial_capital: float,
    *,
    lookback: int,
    z_entry: float,
    z_exit: float,
) -> Dict[str, Any]:
    returns = close.pct_change().fillna(0.0)
    rolling_mean = returns.rolling(window=lookback, min_periods=1).mean()
    rolling_std = returns.rolling(window=lookback, min_periods=1).std().replace(0.0, np.nan)
    z_scores = (returns - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)

    signal = pd.Series(0.0, index=close.index)
    position = 0.0
    for idx, z in enumerate(z_scores):
        if position == 0 and z <= -abs(z_entry):
            position = 1.0
        elif position == 1 and z >= -abs(z_exit):
            position = 0.0
        signal.iloc[idx] = position

    shifted_positions = signal.shift(1).fillna(0.0)
    strategy_returns = returns * shifted_positions
    equity = (1.0 + strategy_returns).cumprod() * initial_capital
    trades = _collect_trades(close, equity, shifted_positions)

    return {
        "equity": equity,
        "strategy_returns": strategy_returns,
        "positions": shifted_positions,
        "trades": trades,
    }


def _sentiment_weighted_strategy(
    close: pd.Series,
    initial_capital: float,
    *,
    symbol: str,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Strategy that adjusts position based on historical sentiment scores."""
    
    positions = pd.Series(0.0, index=close.index)
    
    # Iterate through dates to fetch historical sentiment
    # Note: This is slow because it calls the sentiment service for each day
    for timestamp in close.index:
        date_str = timestamp.strftime("%Y-%m-%d")
        snapshot = collect_sentiment_snapshot(symbol=symbol, simulation_date=date_str)
        score = snapshot.get("score", 0.0)
        
        if score > threshold:
            positions.at[timestamp] = 1.0  # Long
        elif score < -threshold:
            positions.at[timestamp] = -1.0 # Short
        else:
            positions.at[timestamp] = 0.0  # Neutral

    shifted_positions = positions.shift(1).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    strategy_returns = returns * shifted_positions
    equity = (1.0 + strategy_returns).cumprod() * initial_capital
    trades = _collect_trades(close, equity, shifted_positions)

    return {
        "equity": equity,
        "strategy_returns": strategy_returns,
        "positions": shifted_positions,
        "trades": trades,
    }


def _positions_from_signals(index: pd.Index, signals: Sequence[Dict[str, Any]]) -> pd.Series:
    if not signals:
        raise ValueError("signals list cannot be empty for custom strategy")
    series = pd.Series(0.0, index=index)
    for entry in signals:
        if not isinstance(entry, dict):
            continue
        date_value = entry.get("date")
        if date_value is None:
            continue
        try:
            timestamp = pd.Timestamp(date_value)
        except Exception:
            continue
        position_value = float(entry.get("position", 0.0))
        if timestamp in series.index:
            series.at[timestamp] = position_value
        else:
            try:
                loc = series.index.get_indexer(pd.DatetimeIndex([timestamp]), method="nearest")
                idx = int(loc[0])
                if idx >= 0:
                    series.iloc[idx] = position_value
            except Exception:
                continue
    if (series != 0).sum() == 0:
        raise ValueError("No valid positions found in signals")
    return series.ffill().fillna(0.0)


def run_backtest(
    history: Any,
    *,
    strategy: str = "buy_and_hold",
    initial_capital: float = 1_000_000.0,
    params: Optional[Dict[str, Any]] = None,
    risk_free_rate: float = 0.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
) -> Dict[str, Any]:
    """Execute a lightweight backtest over the provided history."""

    df = _ensure_history(history)
    if df.empty or len(df) < 2:
        raise ValueError("History must contain at least two data points for backtesting")

    close = df["Close"].astype(float)
    params = params.copy() if params else {}
    strategy_name = strategy.lower()

    if strategy_name == "buy_and_hold":
        payload = _buy_and_hold_strategy(close, initial_capital)
    elif strategy_name in {"sma", "sma_crossover", "moving_average"}:
        short_window = int(params.get("short_window", 20))
        long_window = int(params.get("long_window", 50))
        payload = _sma_crossover_strategy(
            close,
            initial_capital,
            short_window=short_window,
            long_window=long_window,
        )
        params.update({"short_window": short_window, "long_window": long_window})
    elif strategy_name in {"mean_reversion", "zscore"}:
        lookback = int(params.get("lookback", 20))
        z_entry = float(params.get("z_entry", 1.0))
        z_exit = float(params.get("z_exit", 0.25))
        payload = _mean_reversion_strategy(
            close,
            initial_capital,
            lookback=lookback,
            z_entry=z_entry,
            z_exit=z_exit,
        )
        params.update({"lookback": lookback, "z_entry": z_entry, "z_exit": z_exit})
    elif strategy_name in {"signals", "custom_signals", "custom"}:
        signal_entries = params.get("signals")
        if not isinstance(signal_entries, Sequence):
            raise ValueError("Custom signals strategy requires a 'signals' sequence in params")
        raw_positions = _positions_from_signals(close.index, signal_entries)
        positions_shifted = raw_positions.shift(1).fillna(0.0)
        returns = close.pct_change().fillna(0.0)
        strategy_returns = returns * positions_shifted
        equity = (1.0 + strategy_returns).cumprod() * initial_capital
        trades = _collect_trades(close, equity, positions_shifted)
        payload = {
            "equity": equity,
            "strategy_returns": strategy_returns,
            "positions": positions_shifted,
            "trades": trades,
        }
        params["signals_count"] = len(signal_entries)
    elif strategy_name == "sentiment_weighted":
        threshold = float(params.get("threshold", 0.1))
        symbol = str(params.get("symbol", "XAUUSD"))
        payload = _sentiment_weighted_strategy(
            close,
            initial_capital,
            symbol=symbol,
            threshold=threshold,
        )
        params["threshold"] = threshold
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'")

    equity: pd.Series = payload["equity"]
    strategy_returns: pd.Series = payload["strategy_returns"]
    positions: pd.Series = payload.get("positions", pd.Series(0.0, index=close.index))
    trades: List[Trade] = payload.get("trades", [])

    if slippage_bps or commission_per_trade:
        if positions.empty:
            position_changes = pd.Series(0.0, index=positions.index)
        else:
            initial_change = abs(float(positions.iloc[0]))
            position_changes = positions.diff().abs().fillna(initial_change)
        slippage_return_cost = position_changes * (slippage_bps / 10_000.0)
        commission_return_cost = (
            position_changes * (commission_per_trade / initial_capital)
            if commission_per_trade
            else pd.Series(0.0, index=position_changes.index)
        )
        total_cost = slippage_return_cost + commission_return_cost
        strategy_returns = strategy_returns - total_cost
        equity = (1.0 + strategy_returns).cumprod() * initial_capital

    if strategy_name == "buy_and_hold":
        trades = [
            Trade(
                entry_date=close.index[0].strftime("%Y-%m-%d"),
                exit_date=close.index[-1].strftime("%Y-%m-%d"),
                direction="long",
                entry_price=float(close.iloc[0]),
                exit_price=float(close.iloc[-1]),
                return_pct=float(equity.iloc[-1] / initial_capital - 1.0),
                pnl=float(equity.iloc[-1] - initial_capital),
            )
        ]
    else:
        trades = _collect_trades(close, equity, positions)

    if slippage_bps:
        params["slippage_bps"] = slippage_bps
    if commission_per_trade:
        params["commission_per_trade"] = commission_per_trade

    equity_curve = _initialise_outputs(close.index, equity, close, positions)
    metrics = _compute_metrics(
        equity,
        strategy_returns,
        trades,
        initial_capital,
        risk_free_rate=risk_free_rate,
    )

    result = BacktestResult(
        symbol=params.get("symbol", ""),
        strategy=strategy_name,
        start=close.index[0].strftime("%Y-%m-%d"),
        end=close.index[-1].strftime("%Y-%m-%d"),
        initial_capital=initial_capital,
        equity_curve=equity_curve,
        metrics=metrics,
        trades=trades,
        parameters=params,
    )
    return result.to_dict()