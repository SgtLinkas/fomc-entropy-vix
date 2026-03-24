"""
fetch_fomc_statements.py
------------------------
Downloads FOMC post-meeting press statements from the Federal Reserve website
(https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) and saves them
as a tidy CSV with columns: date, url, text.

Usage
-----
    python data/fetch_fomc_statements.py [--output data/fomc_statements.csv]
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
HEADERS = {"User-Agent": "FOMC-research-bot/1.0 (academic)"}


def _get_statement_links(session: requests.Session) -> list[dict]:
    """Parse the FOMC calendar page and return a list of {date, url} dicts."""
    resp = session.get(CALENDAR_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Statement links contain 'monetary20' (year) and end with '.htm'
        if re.search(r"monetarypolicy/monetary\d{8}a\.htm", href):
            full_url = BASE_URL + href if href.startswith("/") else href
            # Extract date from URL, e.g. monetary20230201a.htm -> 2023-02-01
            m = re.search(r"monetary(\d{4})(\d{2})(\d{2})a", href)
            if m:
                date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                links.append({"date": date_str, "url": full_url})

    # De-duplicate and sort chronologically
    seen = set()
    unique = []
    for item in sorted(links, key=lambda x: x["date"]):
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    return unique


def _fetch_statement_text(session: requests.Session, url: str) -> str:
    """Download a single statement page and return its plain text."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The main body text is usually inside <div id="article"> or <div class="col-xs-12">
    for selector in [
        soup.find("div", id="article"),
        soup.find("div", class_="col-xs-12"),
        soup.find("td", class_="content-area"),
    ]:
        if selector:
            return selector.get_text(separator=" ", strip=True)

    return soup.get_text(separator=" ", strip=True)


def fetch_statements(output_path: str | Path = "data/fomc_statements.csv",
                     delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch all available FOMC statements and save to *output_path*.

    Parameters
    ----------
    output_path : str or Path
        Destination CSV file.
    delay : float
        Seconds to wait between HTTP requests (be polite to Fed servers).

    Returns
    -------
    pd.DataFrame with columns: date, url, text
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching FOMC calendar...")
    links = _get_statement_links(session)
    print(f"Found {len(links)} statement links.")

    records = []
    for i, item in enumerate(links, 1):
        print(f"[{i}/{len(links)}] {item['date']}  {item['url']}")
        try:
            text = _fetch_statement_text(session, item["url"])
            records.append({"date": item["date"], "url": item["url"], "text": text})
        except requests.RequestException as exc:
            print(f"  WARNING: could not fetch {item['url']}: {exc}")
        time.sleep(delay)

    df = pd.DataFrame(records, columns=["date", "url", "text"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} statements to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch FOMC press statements.")
    parser.add_argument(
        "--output",
        default="data/fomc_statements.csv",
        help="Output CSV path (default: data/fomc_statements.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (default: 1.0)",
    )
    args = parser.parse_args()
    fetch_statements(output_path=args.output, delay=args.delay)
