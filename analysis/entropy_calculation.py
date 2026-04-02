"""
entropy_calculation.py
-----------------------
Computes text-based entropy measures for FOMC press statements.

Two entropy variants are computed:

1. Unigram Shannon entropy
   H = -Σ p_i * log2(p_i)
   where p_i is the relative frequency of word i in the statement.

2. Sentence-level uncertainty index (following Loughran-McDonald
   uncertainty word list, optionally).

The main entry point reads a CSV of FOMC statements (produced by
data/fetch_fomc_statements.py) and outputs a CSV with columns:
    date, n_tokens, vocab_size, entropy, uncertainty_count,
    uncertainty_share

Usage
-----
    python analysis/entropy_calculation.py \
        --input data/fomc_statements.csv \
        --output data/fomc_entropy.csv
"""

import argparse
import math
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Loughran-McDonald uncertainty word list (abbreviated subset)
# Full list: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# ---------------------------------------------------------------------------
LM_UNCERTAINTY_WORDS = {
    "approximate", "approximately", "arbitrarily", "certain", "contingency",
    "contingent", "depend", "depends", "doubt", "doubts", "estimate",
    "estimated", "estimates", "fluctuate", "fluctuation", "indefinite",
    "indefinitely", "indeterminate", "may", "might", "possible", "possibly",
    "potential", "potentially", "rough", "roughly", "uncertain", "uncertainly",
    "uncertainties", "uncertainty", "unpredictable", "vague", "could",
    "should", "likely", "unlikely", "ambiguous", "ambiguity",
}


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, return word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.split(r"\s+", text.strip())
    return [t for t in tokens if t]


def compute_entropy(tokens: list[str]) -> float:
    """
    Compute Shannon entropy (bits) of the token distribution.

    Parameters
    ----------
    tokens : list[str]
        List of word tokens.

    Returns
    -------
    float : entropy in bits (log base 2).  Returns 0.0 for empty input.
    """
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def compute_uncertainty_share(
    tokens: list[str],
    uncertainty_words: set[str] | None = None,
) -> tuple[int, float]:
    """
    Count how many tokens are in the uncertainty word list.

    Parameters
    ----------
    tokens : list[str]
        Pre-tokenised words (lower-cased).
    uncertainty_words : set[str], optional
        Custom word list.  Defaults to LM_UNCERTAINTY_WORDS.

    Returns
    -------
    (count, share) : (int, float)
        Raw count and fraction of total tokens.
    """
    if uncertainty_words is None:
        uncertainty_words = LM_UNCERTAINTY_WORDS
    if not tokens:
        return 0, 0.0
    count = sum(1 for t in tokens if t in uncertainty_words)
    return count, count / len(tokens)


def compute_entropy_for_statements(statements: pd.DataFrame) -> pd.DataFrame:
    """
    Compute entropy metrics for a DataFrame of FOMC statements.

    Parameters
    ----------
    statements : pd.DataFrame
        Must have columns: date, text.

    Returns
    -------
    pd.DataFrame with columns:
        date, n_tokens, vocab_size, entropy,
        uncertainty_count, uncertainty_share
    """
    records = []
    for _, row in statements.iterrows():
        tokens = _tokenize(str(row["text"]))
        entropy = compute_entropy(tokens)
        u_count, u_share = compute_uncertainty_share(tokens)
        records.append(
            {
                "date": row["date"],
                "n_tokens": len(tokens),
                "vocab_size": len(set(tokens)),
                "entropy": entropy,
                "uncertainty_count": u_count,
                "uncertainty_share": u_share,
            }
        )
    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    result.sort_values("date", inplace=True)
    return result


def run(
    input_path: str | Path = "data/fomc_statements.csv",
    output_path: str | Path = "data/fomc_entropy.csv",
) -> pd.DataFrame:
    """
    Load statements CSV, compute entropy metrics, save results.

    Parameters
    ----------
    input_path : str or Path
    output_path : str or Path

    Returns
    -------
    pd.DataFrame
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"  {len(df)} statements loaded.")

    result = compute_entropy_for_statements(df)

    result.to_csv(output_path, index=False)
    print(f"Saved entropy data to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute text entropy of FOMC statements.")
    parser.add_argument("--input", default="data/fomc_statements.csv")
    parser.add_argument("--output", default="data/fomc_entropy.csv")
    args = parser.parse_args()
    run(input_path=args.input, output_path=args.output)
