"""
event_study.py
--------------
Event study analysis around FOMC meeting dates.

For each FOMC meeting (event date t = 0) we compute:
  - Cumulative Abnormal VIX Change (CAVC) over windows [-n, +m]
  - Abnormal change = actual change minus expected (pre-event drift)

Methodology
-----------
1. Estimate a "normal" model for ΔlnVIX using a pre-event estimation
   window (default: 120 trading days before the event window).
2. Compute abnormal returns (residuals) in the event window.
3. Aggregate across events (average + t-test).

Usage
-----
    python analysis/event_study.py \
        --vix        data/vix.csv \
        --fomc-dates data/fomc_entropy.csv \
        --output     data/event_study_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _log_change(series: pd.Series) -> pd.Series:
    """Compute log first-difference of a series."""
    return np.log(series).diff()


def compute_event_windows(
    vix: pd.Series,
    event_dates: list[pd.Timestamp],
    pre_window: int = 5,
    post_window: int = 10,
    estimation_window: int = 120,
) -> pd.DataFrame:
    """
    Compute abnormal VIX changes around each FOMC event.

    Parameters
    ----------
    vix : pd.Series
        Daily VIX levels indexed by date.
    event_dates : list[Timestamp]
        FOMC meeting dates.
    pre_window : int
        Days before event to include (e.g. 5 → [-5, 0]).
    post_window : int
        Days after event to include (e.g. 10 → [0, +10]).
    estimation_window : int
        Days before the pre-event window to use for normal-model estimation.

    Returns
    -------
    pd.DataFrame with columns:
        event_date, t, vix_level, delta_log_vix, abnormal, car
    where t is the event-time index (0 = event day).
    """
    dlvix = _log_change(vix).dropna()
    trading_days = dlvix.index.sort_values()

    records = []
    for event_date in event_dates:
        # Find position of event date in trading calendar
        positions = np.searchsorted(trading_days, event_date, side="left")
        if positions >= len(trading_days):
            continue
        t0 = positions  # index of event day (or next trading day)

        # Estimation window indices
        est_start = t0 - pre_window - estimation_window
        est_end = t0 - pre_window - 1

        if est_start < 0:
            continue

        # Normal model: mean of ΔlnVIX in estimation window
        est_returns = dlvix.iloc[est_start : est_end + 1]
        mu = est_returns.mean()

        # Event window: [t0 - pre_window, t0 + post_window]
        ev_start = max(t0 - pre_window, 0)
        ev_end = min(t0 + post_window, len(trading_days) - 1)

        event_slice = dlvix.iloc[ev_start : ev_end + 1]
        if event_slice.empty:
            continue

        t_indices = range(ev_start - t0, ev_end - t0 + 1)

        abnormal = event_slice.values - mu
        car = np.cumsum(abnormal)

        # VIX levels over event window
        vix_slice = vix.reindex(event_slice.index)

        for t_idx, (date, ab, c, vix_lev, dlv) in enumerate(
            zip(event_slice.index, abnormal, car, vix_slice.values, event_slice.values)
        ):
            records.append(
                {
                    "event_date": event_date,
                    "calendar_date": date,
                    "t": list(t_indices)[t_idx],
                    "vix_level": vix_lev,
                    "delta_log_vix": dlv,
                    "abnormal": ab,
                    "car": c,
                }
            )

    return pd.DataFrame(records)


def aggregate_event_study(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate abnormal VIX changes across events by event-time t.

    Returns mean, std, t-statistic, p-value and 95% CI per event-time bin.
    """
    grouped = event_df.groupby("t")["abnormal"]

    rows = []
    for t, group in grouped:
        values = group.dropna().values
        if len(values) == 0:
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        n = len(values)
        se = std / np.sqrt(n) if n > 0 else np.nan
        t_stat, p_val = stats.ttest_1samp(values, 0.0) if n > 1 else (np.nan, np.nan)
        rows.append(
            {
                "t": t,
                "n_events": n,
                "mean_abnormal": mean,
                "std_abnormal": std,
                "se_abnormal": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "ci_lower": mean - 1.96 * se,
                "ci_upper": mean + 1.96 * se,
            }
        )

    result = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return result


def run(
    vix_path: str | Path = "data/vix.csv",
    fomc_dates_path: str | Path = "data/fomc_entropy.csv",
    output_path: str | Path = "data/event_study_results.csv",
    pre_window: int = 5,
    post_window: int = 10,
    estimation_window: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full event study pipeline.

    Returns
    -------
    (event_df, aggregated_df)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vix_df = pd.read_csv(vix_path, parse_dates=["date"])
    vix = vix_df.set_index("date")["vix"].sort_index()

    fomc_df = pd.read_csv(fomc_dates_path, parse_dates=["date"])
    event_dates = fomc_df["date"].sort_values().tolist()

    print(f"Running event study for {len(event_dates)} FOMC meetings...")
    event_df = compute_event_windows(
        vix,
        event_dates,
        pre_window=pre_window,
        post_window=post_window,
        estimation_window=estimation_window,
    )
    print(f"  Event panel: {len(event_df)} rows")

    agg = aggregate_event_study(event_df)

    # Save both outputs
    raw_out = output_path.with_name(output_path.stem + "_raw" + output_path.suffix)
    event_df.to_csv(raw_out, index=False)
    agg.to_csv(output_path, index=False)
    print(f"Saved aggregated results to {output_path}")
    print(f"Saved raw event panel to {raw_out}")
    return event_df, agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event study around FOMC meetings.")
    parser.add_argument("--vix", default="data/vix.csv")
    parser.add_argument("--fomc-dates", default="data/fomc_entropy.csv",
                        help="CSV with 'date' column of FOMC meeting dates")
    parser.add_argument("--output", default="data/event_study_results.csv")
    parser.add_argument("--pre-window", type=int, default=5)
    parser.add_argument("--post-window", type=int, default=10)
    parser.add_argument("--estimation-window", type=int, default=120)
    args = parser.parse_args()
    run(
        vix_path=args.vix,
        fomc_dates_path=args.fomc_dates,
        output_path=args.output,
        pre_window=args.pre_window,
        post_window=args.post_window,
        estimation_window=args.estimation_window,
    )
