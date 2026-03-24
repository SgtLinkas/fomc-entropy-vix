"""
tests/test_iv_estimation.py
----------------------------
Unit tests for analysis/iv_estimation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.iv_estimation import _manual_2sls
from data.fetch_fed_funds_futures import compute_fomc_surprises


class TestComputeFomcSurprises:
    def _make_futures_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2000-01-03", periods=500)
        values = np.linspace(5.5, 4.0, 500) + np.random.default_rng(0).normal(0, 0.05, 500)
        return pd.DataFrame({"date": dates, "value": values})

    def test_returns_dataframe(self):
        futures = self._make_futures_df()
        fomc_dates = pd.Series(pd.bdate_range("2001-01-01", periods=5, freq="60B"))
        result = compute_fomc_surprises(futures, fomc_dates)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        futures = self._make_futures_df()
        fomc_dates = pd.Series(pd.bdate_range("2001-01-01", periods=5, freq="60B"))
        result = compute_fomc_surprises(futures, fomc_dates)
        for col in ["fomc_date", "pre_rate", "post_rate", "surprise"]:
            assert col in result.columns

    def test_surprise_equals_diff(self):
        futures = self._make_futures_df()
        fomc_dates = pd.Series(pd.bdate_range("2001-01-01", periods=3, freq="60B"))
        result = compute_fomc_surprises(futures, fomc_dates)
        diff = (result["post_rate"] - result["pre_rate"]).round(10).reset_index(drop=True)
        surprise = result["surprise"].round(10).reset_index(drop=True)
        pd.testing.assert_series_equal(diff, surprise, check_names=False)

    def test_no_data_returns_empty(self):
        futures = self._make_futures_df()
        # Dates before any data
        fomc_dates = pd.Series([pd.Timestamp("1990-01-01")])
        result = compute_fomc_surprises(futures, fomc_dates)
        assert len(result) == 0


class TestManual2SLS:
    def _make_regression_data(self, n: int = 60, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        # True DGP: vix_post = 2 + 3*entropy + noise
        surprise = rng.normal(0, 1, n)
        entropy = 5.0 + 0.8 * surprise + rng.normal(0, 0.5, n)
        vix_pre = rng.uniform(10, 25, n)
        vix_post = 2 + 3 * entropy + 0.5 * vix_pre + rng.normal(0, 2, n)
        return pd.DataFrame({
            "entropy": entropy,
            "surprise": surprise,
            "vix_pre": vix_pre,
            "vix_post": vix_post,
        })

    def test_returns_dict(self):
        df = self._make_regression_data()
        result = _manual_2sls(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = self._make_regression_data()
        result = _manual_2sls(df)
        for key in ["params", "std_errors", "t_stats", "p_values", "f_stat", "n"]:
            assert key in result

    def test_n_matches_input(self):
        df = self._make_regression_data(n=50)
        result = _manual_2sls(df)
        assert result["n"] == 50

    def test_f_stat_positive(self):
        df = self._make_regression_data()
        result = _manual_2sls(df)
        assert result["f_stat"] > 0

    def test_entropy_hat_coef_reasonable(self):
        # With strong instrument, 2SLS should recover approx true slope of 3
        df = self._make_regression_data(n=200)
        result = _manual_2sls(df)
        coef = result["params"].get("entropy_hat", None)
        if coef is not None:
            assert 1.0 < coef < 6.0
