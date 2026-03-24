"""
tests/test_event_study.py
--------------------------
Unit tests for analysis/event_study.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.event_study import aggregate_event_study, compute_event_windows


def _make_vix_series(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    vix = 20 + np.cumsum(rng.normal(0, 0.5, n))
    vix = np.clip(vix, 9, 80)
    return pd.Series(vix, index=dates, name="vix")


class TestComputeEventWindows:
    def test_returns_dataframe(self):
        vix = _make_vix_series()
        dates = [pd.Timestamp("2001-03-15"), pd.Timestamp("2002-06-20")]
        df = compute_event_windows(vix, dates, pre_window=3, post_window=5,
                                   estimation_window=60)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        vix = _make_vix_series()
        dates = [pd.Timestamp("2001-03-15")]
        df = compute_event_windows(vix, dates, pre_window=3, post_window=5,
                                   estimation_window=60)
        for col in ["event_date", "t", "vix_level", "delta_log_vix", "abnormal", "car"]:
            assert col in df.columns

    def test_t_range(self):
        vix = _make_vix_series()
        dates = [pd.Timestamp("2001-06-01")]
        pre, post = 3, 5
        df = compute_event_windows(vix, dates, pre_window=pre, post_window=post,
                                   estimation_window=60)
        if not df.empty:
            assert df["t"].min() >= -pre
            assert df["t"].max() <= post

    def test_no_events_outside_data(self):
        vix = _make_vix_series(n=200)
        # Event date far in the future
        dates = [pd.Timestamp("2050-01-01")]
        df = compute_event_windows(vix, dates)
        assert df.empty

    def test_multiple_events(self):
        vix = _make_vix_series(n=1000)
        dates = pd.bdate_range("2001-01-01", periods=5, freq="60B").tolist()
        df = compute_event_windows(vix, dates, pre_window=2, post_window=3,
                                   estimation_window=30)
        # Should have rows for at least some events
        assert len(df) > 0


class TestAggregateEventStudy:
    def _make_event_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        rows = []
        for event_idx in range(10):
            for t in range(-3, 6):
                rows.append({"event_date": f"2001-{event_idx+1:02d}-01",
                             "t": t, "abnormal": rng.normal(0, 0.01)})
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        df = self._make_event_df()
        result = aggregate_event_study(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        df = self._make_event_df()
        result = aggregate_event_study(df)
        for col in ["t", "n_events", "mean_abnormal", "se_abnormal",
                    "t_stat", "p_value", "ci_lower", "ci_upper"]:
            assert col in result.columns

    def test_sorted_by_t(self):
        df = self._make_event_df()
        result = aggregate_event_study(df)
        assert result["t"].is_monotonic_increasing

    def test_ci_width_positive(self):
        df = self._make_event_df()
        result = aggregate_event_study(df)
        assert ((result["ci_upper"] - result["ci_lower"]) >= 0).all()
