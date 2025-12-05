import pandas as pd
from autogentest1.services.backtest import run_backtest
from autogentest1.services.news_ingest import _NEWS_ARCHIVE_FILE
import json
import os

def test_sentiment_backtest():
    # 1. Ensure we have the mock news data
    print(f"Checking news archive at: {_NEWS_ARCHIVE_FILE}")
    if not _NEWS_ARCHIVE_FILE.exists():
        print("News archive not found, creating temporary one...")
        # (This part is just a safeguard, I already created the file)
        
    # 2. Create mock price history matching the news dates
    # We have real news for 2025-11-06
    dates = pd.date_range(start="2025-11-01", end="2025-11-10")
    # Mock prices: Gold usually moves around 2600-3000 in 2025 (hypothetically)
    prices = [2600.0, 2610.0, 2605.0, 2620.0, 2625.0, 2630.0, 2628.0, 2635.0, 2640.0, 2650.0]
    
    history = pd.DataFrame({"Close": prices}, index=dates)
    
    print("\nRunning Backtest with Strategy='sentiment_weighted'...")
    
    # Run backtest with a very low threshold to ensure we trigger a trade
    # The actual sentiment score for 2025-11-06 is around 0.006 (weakly positive)
    result = run_backtest(
        history=history,
        strategy="sentiment_weighted",
        initial_capital=100_000.0,
        params={
            "symbol": "XAUUSD",
            "threshold": 0.001  # Lowered threshold to capture weak sentiment
        }
    )
    
    # 4. Analyze results
    print(f"Total Return: {result['metrics']['total_return']:.2%}")
    print(f"Trades Executed: {result['metrics']['trades']}")
    
    print("\n--- Trade Log ---")
    for trade in result['trades']:
        print(f"{trade['entry_date']} -> {trade['exit_date']} | {trade['direction'].upper()} | PnL: ${trade['pnl']:.2f}")

    print("\n--- Daily Positions ---")
    for day in result['equity_curve'][:5]: # Show first 5 days
        print(f"{day['date']}: Position={day['position']} Price={day['price']}")

if __name__ == "__main__":
    test_sentiment_backtest()
