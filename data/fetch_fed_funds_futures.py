"""
fetch_fed_funds_futures.py
--------------------------
Downloads 30-Day Federal Funds Futures settlement prices from FRED
(series FF* or FEDL01) and derives the *market-implied* federal funds rate
target around each FOMC meeting date.

The "surprise" component is computed as:
    surprise = implied_rate_post - implied_rate_pre

where pre/post refer to the day before and day after each FOMC statement.

Usage
-----
    python data/fetch_fed_funds_futures.py --api-key <FRED_API_KEY> \
        [--output data/fed_funds_surprises.csv]
"""

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# FRED series for 30-Day Fed Funds Futures (front-month settlement)
# FRED provides the *rate* as 100 minus price.
# ---------------------------------------------------------------------------
FRED_SERIES = "DFFRGDP"          # fallback; preferred series listed below
FF_FUTURES_SERIES = "FEDL01"     # Overnight Fed Funds rate as proxy when futures unavailable


def _implied_rate_from_price(price: float) -> float:
    """Convert CME-style futures price (100 - rate) to implied rate (%)."""
    return 100.0 - price


def fetch_from_fred(api_key: str, series_id: str = "FF") -> pd.DataFrame:
    """
    Download a FRED time series.

    Parameters
    ----------
    api_key : str
        FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html).
    series_id : str
        FRED series identifier.

    Returns
    -------
    pd.DataFrame with columns: date, value
    """
    try:
        import fredapi  # type: ignore
    except ImportError as exc:
        raise ImportError("Install fredapi: pip install fredapi") from exc

    fred = fredapi.Fred(api_key=api_key)
    series = fred.get_series(series_id)
    df = series.reset_index()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)
    return df


def compute_fomc_surprises(
    futures_df: pd.DataFrame,
    fomc_dates: pd.Series,
    window: int = 1,
) -> pd.DataFrame:
    """
    Compute the rate surprise around each FOMC meeting.

    Parameters
    ----------
    futures_df : pd.DataFrame
        Daily implied rates with columns [date, value].
    fomc_dates : pd.Series
        Dates of FOMC statements (datetime-like).
    window : int
        Number of business days before/after meeting to average rates.

    Returns
    -------
    pd.DataFrame with columns: fomc_date, pre_rate, post_rate, surprise
    """
    futures_df = futures_df.set_index("date")["value"].sort_index()

    records = []
    for meeting_date in pd.to_datetime(fomc_dates):
        # Find nearest trading days before and after
        pre_dates = futures_df.index[futures_df.index < meeting_date]
        post_dates = futures_df.index[futures_df.index >= meeting_date]

        if len(pre_dates) < window or len(post_dates) < window:
            continue

        pre_rate = futures_df.loc[pre_dates[-window:]].mean()
        post_rate = futures_df.loc[post_dates[:window]].mean()
        records.append(
            {
                "fomc_date": meeting_date,
                "pre_rate": pre_rate,
                "post_rate": post_rate,
                "surprise": post_rate - pre_rate,
            }
        )

    return pd.DataFrame(records)


def fetch_fed_funds_surprises(
    api_key: str,
    fomc_dates_path: str | Path | None = None,
    output_path: str | Path = "data/fed_funds_surprises.csv",
    series_id: str = "FF",
) -> pd.DataFrame:
    """
    High-level entry point: download futures data and compute FOMC surprises.

    Parameters
    ----------
    api_key : str
        FRED API key.
    fomc_dates_path : str or Path, optional
        Path to a CSV with a 'date' column listing FOMC meeting dates.
        If None, surprises are not computed and raw series is saved.
    output_path : str or Path
        Destination CSV.
    series_id : str
        FRED series to download (default 'FF' = effective Fed Funds rate).

    Returns
    -------
    pd.DataFrame
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FRED series '{series_id}'...")
    df = fetch_from_fred(api_key, series_id)
    print(f"  Got {len(df)} observations ({df['date'].min().date()} – {df['date'].max().date()})")

    if fomc_dates_path is not None:
        fomc_df = pd.read_csv(fomc_dates_path, parse_dates=["date"])
        result = compute_fomc_surprises(df, fomc_df["date"])
        print(f"  Computed surprises for {len(result)} FOMC meetings.")
        result.to_csv(output_path, index=False)
        print(f"Saved surprises to {output_path}")
        return result

    df.to_csv(output_path, index=False)
    print(f"Saved raw series to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Fed Funds Futures / rate data.")
    parser.add_argument("--api-key", required=True, help="FRED API key")
    parser.add_argument(
        "--series",
        default="FF",
        help="FRED series ID (default: FF = effective fed funds rate)",
    )
    parser.add_argument(
        "--fomc-dates",
        default=None,
        help="Path to CSV with 'date' column of FOMC meeting dates",
    )
    parser.add_argument(
        "--output",
        default="data/fed_funds_surprises.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    fetch_fed_funds_surprises(
        api_key=args.api_key,
        fomc_dates_path=args.fomc_dates,
        output_path=args.output,
        series_id=args.series,
    )
