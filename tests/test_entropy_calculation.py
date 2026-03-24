"""
tests/test_entropy_calculation.py
----------------------------------
Unit tests for analysis/entropy_calculation.py
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.entropy_calculation import (
    LM_UNCERTAINTY_WORDS,
    _tokenize,
    compute_entropy,
    compute_entropy_for_statements,
    compute_uncertainty_share,
)


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenize("interest rates; inflation.")
        assert "rates" in tokens
        assert "inflation" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_extra_whitespace(self):
        tokens = _tokenize("  a  b  c  ")
        assert tokens == ["a", "b", "c"]


class TestComputeEntropy:
    def test_uniform_distribution(self):
        # 4 equally likely tokens → entropy = log2(4) = 2.0
        tokens = ["a", "b", "c", "d"]
        assert math.isclose(compute_entropy(tokens), 2.0)

    def test_all_same_token(self):
        # Single unique token → entropy = 0
        tokens = ["rate"] * 100
        assert math.isclose(compute_entropy(tokens), 0.0)

    def test_empty(self):
        assert compute_entropy([]) == 0.0

    def test_positive(self):
        tokens = ["inflation", "rates", "fomc", "uncertainty"]
        assert compute_entropy(tokens) > 0.0

    def test_more_diversity_more_entropy(self):
        tokens_diverse = list("abcdefgh")   # 8 unique
        tokens_less = list("aabb")          # 2 unique
        assert compute_entropy(tokens_diverse) > compute_entropy(tokens_less)


class TestUncertaintyShare:
    def test_known_words(self):
        tokens = ["may", "increase", "rates", "uncertain"]
        count, share = compute_uncertainty_share(tokens)
        # "may" and "uncertain" are in the LM list
        assert count >= 2
        assert 0 < share <= 1.0

    def test_no_uncertainty_words(self):
        tokens = ["federal", "reserve", "bank", "policy"]
        count, share = compute_uncertainty_share(tokens)
        assert count == 0
        assert share == 0.0

    def test_empty(self):
        count, share = compute_uncertainty_share([])
        assert count == 0
        assert share == 0.0

    def test_custom_word_list(self):
        tokens = ["foo", "bar", "baz"]
        count, share = compute_uncertainty_share(tokens, uncertainty_words={"foo"})
        assert count == 1
        assert math.isclose(share, 1 / 3)


class TestComputeEntropyForStatements:
    def _make_df(self):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-03-15"]),
                "text": [
                    "The committee decided to raise rates by 25 basis points.",
                    "There may be uncertainty in the inflation outlook. Policy could change.",
                ],
            }
        )

    def test_returns_correct_columns(self):
        df = compute_entropy_for_statements(self._make_df())
        for col in ["date", "n_tokens", "vocab_size", "entropy",
                    "uncertainty_count", "uncertainty_share"]:
            assert col in df.columns

    def test_length(self):
        df_in = self._make_df()
        df_out = compute_entropy_for_statements(df_in)
        assert len(df_out) == len(df_in)

    def test_entropy_positive(self):
        df_out = compute_entropy_for_statements(self._make_df())
        assert (df_out["entropy"] >= 0).all()

    def test_sorted_by_date(self):
        df_out = compute_entropy_for_statements(self._make_df())
        assert df_out["date"].is_monotonic_increasing

    def test_uncertainty_share_in_range(self):
        df_out = compute_entropy_for_statements(self._make_df())
        assert (df_out["uncertainty_share"] >= 0).all()
        assert (df_out["uncertainty_share"] <= 1).all()
