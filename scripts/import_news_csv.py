import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configuration
CSV_FILE_PATH = "historical_news.csv"  # Default input file
ARCHIVE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "rag" / "news_archive.json"

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
    print(f"Successfully saved {len(data)} dates to {ARCHIVE_FILE_PATH}")

def import_csv(csv_path: str):
    archive = load_archive()
    count = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Validate required fields
                if not all(k in row for k in ['date', 'title']):
                    print(f"Skipping invalid row: {row}")
                    continue

                date_str = row['date'].strip() # Expected YYYY-MM-DD
                
                # Construct article object
                article = {
                    "source": row.get('source', 'Unknown'),
                    "title": row['title'].strip(),
                    "summary": row.get('summary', row['title']).strip(),
                    "url": row.get('url', ''),
                    "weight": 1.0, # Default weight
                    "published": f"{date_str}T12:00:00Z" # Default time
                }

                # Add to archive
                if date_str not in archive:
                    archive[date_str] = []
                
                # Avoid duplicates based on title
                existing_titles = {a['title'] for a in archive[date_str]}
                if article['title'] not in existing_titles:
                    archive[date_str].append(article)
                    count += 1
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    save_archive(archive)
    print(f"Imported {count} new articles.")

if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE_PATH
    print(f"Importing from {target_csv}...")
    import_csv(target_csv)
