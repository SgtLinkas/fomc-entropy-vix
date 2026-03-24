"""
fetch_vix.py
------------
Downloads CBOE VIX daily close prices via Yahoo Finance (^VIX) and/or FRED
(VIXCLS) and saves them as a tidy CSV with columns: date, vix.

Usage
-----
    python data/fetch_vix.py [--source yahoo|fred] [--api-key <FRED_KEY>] \
        [--start 1990-01-02] [--end 2024-12-31] \
        [--output data/vix.csv]
"""

import argparse
from pathlib import Path

import pandas as pd


def fetch_vix_yahoo(
    start: str = "1990-01-02",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch VIX from Yahoo Finance using yfinance.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame with columns: date, vix
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError("Install yfinance: pip install yfinance") from exc

    ticker = yf.Ticker("^VIX")
    hist = ticker.history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        raise ValueError("No VIX data returned from Yahoo Finance.")

    df = hist[["Close"]].reset_index()
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)
    return df


def fetch_vix_fred(
    api_key: str,
    start: str = "1990-01-02",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch VIX from FRED (series: VIXCLS).

    Parameters
    ----------
    api_key : str
        FRED API key.
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD). Defaults to latest available.

    Returns
    -------
    pd.DataFrame with columns: date, vix
    """
    try:
        import fredapi  # type: ignore
    except ImportError as exc:
        raise ImportError("Install fredapi: pip install fredapi") from exc

    fred = fredapi.Fred(api_key=api_key)
    series = fred.get_series("VIXCLS", observation_start=start, observation_end=end)
    df = series.reset_index()
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)
    df.sort_values("date", inplace=True)
    return df


def fetch_vix(
    source: str = "yahoo",
    api_key: str | None = None,
    start: str = "1990-01-02",
    end: str | None = None,
    output_path: str | Path = "data/vix.csv",
) -> pd.DataFrame:
    """
    Download VIX data and save to *output_path*.

    Parameters
    ----------
    source : {'yahoo', 'fred'}
        Data source.
    api_key : str, optional
        Required when source='fred'.
    start : str
        Start date.
    end : str or None
        End date.
    output_path : str or Path
        Destination CSV.

    Returns
    -------
    pd.DataFrame with columns: date, vix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "yahoo":
        print("Fetching VIX from Yahoo Finance (^VIX)...")
        df = fetch_vix_yahoo(start=start, end=end)
    elif source == "fred":
        if not api_key:
            raise ValueError("--api-key is required when --source=fred")
        print("Fetching VIX from FRED (VIXCLS)...")
        df = fetch_vix_fred(api_key=api_key, start=start, end=end)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'yahoo' or 'fred'.")

    print(
        f"  Got {len(df)} observations "
        f"({df['date'].min().date()} – {df['date'].max().date()})"
    )
    df.to_csv(output_path, index=False)
    print(f"Saved VIX data to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CBOE VIX data.")
    parser.add_argument(
        "--source",
        choices=["yahoo", "fred"],
        default="yahoo",
        help="Data source (default: yahoo)",
    )
    parser.add_argument("--api-key", default=None, help="FRED API key (required for --source=fred)")
    parser.add_argument("--start", default="1990-01-02", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output", default="data/vix.csv", help="Output CSV path")
    args = parser.parse_args()
    fetch_vix(
        source=args.source,
        api_key=args.api_key,
        start=args.start,
        end=args.end,
        output_path=args.output,
    )
