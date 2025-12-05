import argparse
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ARCHIVE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "rag" / "news_archive.json"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

DEFAULT_NEWS_QUERY = "gold price OR XAUUSD OR federal reserve"

def load_archive() -> Dict[str, List[Dict[str, Any]]]:
    if not ARCHIVE_FILE_PATH.exists():
        return {}
    try:
        with ARCHIVE_FILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading archive: {e}")
        return {}

def save_archive(data: Dict[str, List[Dict[str, Any]]]):
    ARCHIVE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE_FILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved archive to {ARCHIVE_FILE_PATH}")

def fetch_alpha_vantage(
    api_key: str,
    date_str: str,
    tickers: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Fetch news from Alpha Vantage for a specific date.
    """
    time_from = f"{date_str.replace('-', '')}T0000"
    time_to = f"{date_str.replace('-', '')}T2359"
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "time_from": time_from,
        "time_to": time_to,
        "limit": limit,
        "apikey": api_key,
    }
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "Note" in data:
            print(f"Alpha Vantage notice for {date_str}: {data['Note']}")
            return []

        if "feed" not in data:
            # print(f"Alpha Vantage: No feed found for {date_str}. Response keys: {list(data.keys())}")
            return []
            
        articles = []
        for item in data["feed"]:
            articles.append({
                "source": item.get("source", "AlphaVantage"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "weight": float(item.get("overall_sentiment_score", 0)) + 1.0,
                "published": item.get("time_published", "")
            })
        return articles
    except Exception as e:
        print(f"Alpha Vantage Error for {date_str}: {e}")
        return []

def fetch_newsapi(api_key: str, date_str: str, query: str, page_size: int) -> List[Dict[str, Any]]:
    """
    Fetch news from NewsAPI for a specific date.
    """
    params = {
        "q": query,
        "from": date_str,
        "to": date_str,
        "sortBy": "popularity",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "articles" not in data:
            print(f"NewsAPI: No articles found for {date_str}")
            return []
            
        articles = []
        for item in data["articles"]:
            articles.append({
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "title": item.get("title", ""),
                "summary": item.get("description", ""),
                "url": item.get("url", ""),
                "weight": 1.0,
                "published": item.get("publishedAt", "")
            })
        return articles
    except Exception as e:
        print(f"NewsAPI Error for {date_str}: {e}")
        return []

def iter_days(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical gold-related news into the RAG archive.")
    parser.add_argument("--start-date", dest="start_date", type=str, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", dest="end_date", type=str, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--days",
        dest="days",
        type=int,
        default=31,
        help="Number of trailing days to backfill when start-date is omitted.",
    )
    parser.add_argument(
        "--tickers",
        dest="tickers",
        type=str,
        default="XAUUSD",
        help="Comma-separated symbols for Alpha Vantage NEWS_SENTIMENT queries.",
    )
    parser.add_argument(
        "--query",
        dest="query",
        type=str,
        default=DEFAULT_NEWS_QUERY,
        help="Keyword query for NewsAPI articles.",
    )
    parser.add_argument(
        "--alpha-wait",
        dest="alpha_wait",
        type=float,
        default=12.0,
        help="Seconds to pause between Alpha Vantage requests (rate limiting).",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=50,
        help="Maximum articles per provider per day.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Replace existing archive entries for a day instead of appending.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Fetch data but skip writing the archive to disk.",
    )
    return parser.parse_args()


def normalize_signature(article: Dict[str, Any]) -> Tuple[str, str]:
    title = article.get("title") or ""
    url = article.get("url") or ""
    return title.strip(), url.strip()


def main():
    args = parse_args()

    # 1. Get API Keys from environment
    alpha_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    newsapi_key = os.environ.get("NEWS_API_KEY")
    
    if not alpha_key and not newsapi_key:
        print("Error: No API keys found in .env (ALPHA_VANTAGE_API_KEY or NEWS_API_KEY).")
        return

    print(f"Loaded keys - Alpha: {'Yes' if alpha_key else 'No'}, NewsAPI: {'Yes' if newsapi_key else 'No'}")
    
    today = datetime.utcnow().date()
    end_date = today
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SystemExit(f"Invalid end-date: {args.end_date}. Use YYYY-MM-DD.") from exc
    end_date = min(end_date, today)

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SystemExit(f"Invalid start-date: {args.start_date}. Use YYYY-MM-DD.") from exc
    else:
        span = max(1, args.days)
        start_date = end_date - timedelta(days=span - 1)

    if start_date > end_date:
        raise SystemExit("Start date must be on or before end date.")

    tickers = ",".join(sorted({token.strip().upper() for token in args.tickers.split(',') if token.strip()}))
    query = args.query.strip() or DEFAULT_NEWS_QUERY
    limit = max(1, min(1000, args.limit))

    archive = load_archive()
    total_added = 0
    
    for day in iter_days(datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time())):
        date_str = day.strftime("%Y-%m-%d")
        print(f"\nFetching news for {date_str}...")
        
        daily_articles: List[Dict[str, Any]] = []
        
        # Fetch Alpha Vantage
        if alpha_key:
            # print("  Querying Alpha Vantage...")
            av_articles = fetch_alpha_vantage(alpha_key, date_str, tickers, limit)
            daily_articles.extend(av_articles)
            # print(f"  Found {len(av_articles)} articles from Alpha Vantage.")
            if args.alpha_wait > 0:
                time.sleep(args.alpha_wait)  # Rate limit protection
            
        # Fetch NewsAPI
        if newsapi_key:
            # print("  Querying NewsAPI...")
            na_articles = fetch_newsapi(newsapi_key, date_str, query, limit)
            daily_articles.extend(na_articles)
            # print(f"  Found {len(na_articles)} articles from NewsAPI.")
        
        # Merge into archive
        if daily_articles:
            if args.overwrite or date_str not in archive:
                archive[date_str] = []

            existing_signatures = {normalize_signature(a) for a in archive.get(date_str, [])}
            added_count = 0
            for art in daily_articles:
                signature = normalize_signature(art)
                if signature in existing_signatures:
                    continue
                if not signature[0] and not signature[1]:
                    continue
                archive[date_str].append(art)
                existing_signatures.add(signature)
                added_count += 1
            print(f"  -> Added {added_count} new articles. (Total for day: {len(archive[date_str])})")
            total_added += added_count
        else:
            print("  -> No articles found.")
            
    if args.dry_run:
        print("Dry run complete; archive not updated on disk.")
    else:
        save_archive(archive)
    print(f"\nDone! Total new articles added: {total_added}")

if __name__ == "__main__":
    main()

