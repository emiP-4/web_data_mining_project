"""
Lab Session 1: From the Unstructured Web to Structured Entities
Course: Web Mining & Semantics

Phase 1 - Web Crawling & Cleaning  → crawler_output.jsonl

Usage:
  python crawl.py                        # uses default seed URLs (Summer Olympics)
  python crawl.py --urls urls.txt        # load seed URLs from a text file (one per line)
  python crawl.py --domain winter        # switch to a preset domain
"""

import json
import time
import argparse
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

import httpx
import trafilatura
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_WORD_COUNT  = 500   # pages below this are discarded
REQUEST_DELAY   = 1.5   # seconds between requests (be polite)
REQUEST_TIMEOUT = 15    # seconds before giving up on a URL
OUTPUT_JSONL    = Path("./data/samples/crawler_output.jsonl")

headers = {
    'User-Agent': 'WDM_Project_Bot/1.0 (emilie.poupat@edu.devinci.fr)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

# Preset seed URLs by domain (edit / extend freely)
DOMAIN_SEEDS = {
    "summer": [
        "https://en.wikipedia.org/wiki/Summer_Olympic_Games",
        "https://en.wikipedia.org/wiki/2024_Summer_Olympics",
        "https://en.wikipedia.org/wiki/2020_Summer_Olympics",
        "https://en.wikipedia.org/wiki/2016_Summer_Olympics",
        "https://en.wikipedia.org/wiki/International_Olympic_Committee",
        "https://en.wikipedia.org/wiki/Olympic_symbols",
        "https://en.wikipedia.org/wiki/Athletics_at_the_Summer_Olympics",
        "https://en.wikipedia.org/wiki/Swimming_at_the_Summer_Olympics",
        "https://en.wikipedia.org/wiki/Gymnastics_at_the_Summer_Olympics",
        "https://en.wikipedia.org/wiki/Olympic_Games",
    ],
    "winter": [
        "https://en.wikipedia.org/wiki/Winter_Olympic_Games",
        "https://en.wikipedia.org/wiki/2022_Winter_Olympics",
        "https://en.wikipedia.org/wiki/2018_Winter_Olympics",
        "https://en.wikipedia.org/wiki/Alpine_skiing_at_the_Winter_Olympics",
        "https://en.wikipedia.org/wiki/Figure_skating_at_the_Winter_Olympics",
        "https://en.wikipedia.org/wiki/Biathlon_at_the_Winter_Olympics",
        "https://en.wikipedia.org/wiki/Ice_hockey_at_the_Winter_Olympics",
        "https://en.wikipedia.org/wiki/Cross-country_skiing_at_the_Winter_Olympics",
    ],
    "athletes": [
        "https://en.wikipedia.org/wiki/Usain_Bolt",
        "https://en.wikipedia.org/wiki/Michael_Phelps",
        "https://en.wikipedia.org/wiki/Simone_Biles",
        "https://en.wikipedia.org/wiki/Eliud_Kipchoge",
        "https://en.wikipedia.org/wiki/Cathy_Freeman",
        "https://en.wikipedia.org/wiki/Carl_Lewis",
        "https://en.wikipedia.org/wiki/Nadia_Comaneci",
        "https://en.wikipedia.org/wiki/Jesse_Owens",
        "https://en.wikipedia.org/wiki/Yelena_Isinbayeva",
        "https://en.wikipedia.org/wiki/Mikaela_Shiffrin",
    ],
    "host_cities": [
        "https://en.wikipedia.org/wiki/Paris",
        "https://en.wikipedia.org/wiki/Los_Angeles",
        "https://en.wikipedia.org/wiki/Brisbane",
        "https://en.wikipedia.org/wiki/Tokyo",
        "https://en.wikipedia.org/wiki/London",
        "https://en.wikipedia.org/wiki/Beijing",
        "https://en.wikipedia.org/wiki/Athens",
        "https://en.wikipedia.org/wiki/Sydney",
        "https://en.wikipedia.org/wiki/Barcelona",
    ],
}

# ─────────────────────────────────────────────
# CRAWLING & CLEANING
# ─────────────────────────────────────────────

def can_fetch(url: str, client: httpx.Client) -> bool:
    """Check robots.txt before fetching."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    
    try:
        # Fetch robots.txt manually using your allowed client
        response = client.get(robots_url, timeout=5, follow_redirects=True)
        
        if response.status_code == 200:
            # Feed the text lines directly into the parser
            rp.parse(response.text.splitlines())
            
            # Check the rules specifically against your bot's name
            user_agent = client.headers.get("User-Agent", "*")
            return rp.can_fetch(user_agent, url)
        else:
            return True   # if robots.txt is unreachable/404, proceed cautiously
    except Exception:
        return True

def fetch_and_clean(url: str, client: httpx.Client) -> dict | None:
    """
    Fetch a URL, extract main text with trafilatura, and return a record.
    Returns None if the page is not useful (< MIN_WORD_COUNT words).
    """
    if not can_fetch(url, client):
        print(f"  [BLOCKED by robots.txt] {url}")
        return None

    try:
        response = client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        print(f"  [FETCH ERROR] {url} → {e}")
        return None

    html = response.text

    # trafilatura strips boilerplate (navbars, footers, ads, etc.)
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    if not text:
        print(f"  [NO CONTENT] {url}")
        return None

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        print(f"  [TOO SHORT] {url}  ({word_count} words)")
        return None

    print(f"  [OK] {url}  ({word_count} words)")
    return {
        "url":        url,
        "fetched_at": datetime.utcnow().isoformat(),
        "word_count": word_count,
        "text":       text,
    }


def crawl(seed_urls: list[str]) -> list[dict]:
    """Crawl all seed URLs and write results to JSONL."""
    records = []
    with httpx.Client(headers=headers) as client: 
        for url in seed_urls:
            print(f"Fetching: {url}")
            record = fetch_and_clean(url, client)
            if record:
                records.append(record)
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(REQUEST_DELAY)

    print(f"\nPhase 1 done. {len(records)} useful pages saved → {OUTPUT_JSONL}\n")
    return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Olympics Web Mining — Phase 1: Crawling")
    parser.add_argument(
        "--domain",
        default="summer",
        choices=list(DOMAIN_SEEDS.keys()),
        help="Preset domain to crawl: summer | winter | athletes | host_cities",
    )
    parser.add_argument(
        "--urls",
        type=str,
        default=None,
        help="Path to a text file with one seed URL per line",
    )
    args = parser.parse_args()

    OUTPUT_JSONL.unlink(missing_ok=True)

    if args.urls:
        with open(args.urls) as f:
            seeds = [line.strip() for line in f if line.strip()]
    else:
        seeds = DOMAIN_SEEDS[args.domain]

    print(f"=== PHASE 1: Crawling {len(seeds)} seed URLs (domain: {args.domain}) ===\n")
    records = crawl(seeds)

    if not records:
        print("No usable pages found. Check your URLs or network connection.")

    print("=== DONE — run ie.py to extract entities and relations ===")


if __name__ == "__main__":
    main()
