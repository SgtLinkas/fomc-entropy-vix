"""
iv_estimation.py
----------------
Instrumental Variables (2SLS) estimation of the causal effect of
FOMC communication uncertainty (entropy) on market volatility (VIX).

Model
-----
    Stage 1 (first stage):
        entropy_t = α + β * instrument_t + γ * controls_t + ε_t

    Stage 2 (second stage):
        vix_t = δ + θ * entropy_hat_t + λ * controls_t + u_t

Instrument candidates
---------------------
- Fed Funds rate surprise (pre/post FOMC): exogenous shock to market
  expectations that is correlated with communication uncertainty but
  uncorrelated with contemporaneous VIX shocks.

Usage
-----
    python analysis/iv_estimation.py \
        --entropy   data/fomc_entropy.csv \
        --vix       data/vix.csv \
        --surprises data/fed_funds_surprises.csv \
        --output    data/iv_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_and_merge(
    entropy_path: str | Path,
    vix_path: str | Path,
    surprises_path: str | Path,
    vix_window: int = 5,
) -> pd.DataFrame:
    """
    Load and merge entropy, VIX, and surprise data onto FOMC meeting dates.

    Parameters
    ----------
    entropy_path : str or Path
        CSV with columns: date, entropy, uncertainty_share, …
    vix_path : str or Path
        CSV with columns: date, vix
    surprises_path : str or Path
        CSV with columns: fomc_date, surprise
    vix_window : int
        Number of business days after the FOMC meeting over which to
        average the VIX (captures post-meeting volatility response).

    Returns
    -------
    pd.DataFrame with one row per FOMC meeting, columns:
        date, entropy, uncertainty_share, surprise, vix_post, vix_pre
    """
    entropy = pd.read_csv(entropy_path, parse_dates=["date"])
    vix = pd.read_csv(vix_path, parse_dates=["date"]).set_index("date")["vix"].sort_index()
    surprises = pd.read_csv(surprises_path, parse_dates=["fomc_date"])
    surprises.rename(columns={"fomc_date": "date"}, inplace=True)

    merged = entropy.merge(surprises[["date", "surprise"]], on="date", how="inner")

    # Compute average VIX in [t, t+vix_window] and [t-vix_window, t-1]
    post_vix, pre_vix = [], []
    for meeting_date in merged["date"]:
        post_dates = vix.index[(vix.index >= meeting_date)][:vix_window]
        pre_dates = vix.index[(vix.index < meeting_date)][-vix_window:]
        post_vix.append(vix.loc[post_dates].mean() if len(post_dates) else np.nan)
        pre_vix.append(vix.loc[pre_dates].mean() if len(pre_dates) else np.nan)

    merged["vix_post"] = post_vix
    merged["vix_pre"] = pre_vix
    merged.dropna(inplace=True)
    return merged


def run_2sls(df: pd.DataFrame) -> dict:
    """
    Run 2SLS estimation of entropy → VIX.

    Parameters
    ----------
    df : pd.DataFrame
        Merged data with columns:
        entropy, uncertainty_share, surprise, vix_post, vix_pre

    Returns
    -------
    dict with keys: first_stage, second_stage, f_stat, n
        first_stage and second_stage are dicts with keys:
        coef, std_err, t_stat, p_value (as arrays/dicts)
    """
    try:
        from linearmodels.iv import IV2SLS  # type: ignore
    except ImportError:
        # Fallback: manual 2SLS using statsmodels OLS
        return _manual_2sls(df)

    import statsmodels.api as sm  # type: ignore

    df = df.copy()
    df["const"] = 1.0

    endog = df[["vix_post"]]
    exog = df[["const", "vix_pre"]]           # included exogenous
    instruments = df[["const", "surprise", "vix_pre"]]  # instruments
    endog_regressor = df[["entropy"]]

    model = IV2SLS(endog, exog, endog_regressor, instruments)
    res = model.fit(cov_type="robust")

    return {
        "params": res.params.to_dict(),
        "std_errors": res.std_errors.to_dict(),
        "t_stats": res.tstats.to_dict(),
        "p_values": res.pvalues.to_dict(),
        "f_stat": float(res.first_stage.diagnostics.loc["f.stat", "IV(1)"]),
        "n": int(res.nobs),
        "summary": str(res.summary),
    }


def _manual_2sls(df: pd.DataFrame) -> dict:
    """Fallback 2SLS without linearmodels (uses statsmodels OLS)."""
    import statsmodels.api as sm  # type: ignore

    X_first = sm.add_constant(df[["surprise", "vix_pre"]])
    first_stage = sm.OLS(df["entropy"], X_first).fit()

    df = df.copy()
    df["entropy_hat"] = first_stage.fittedvalues

    X_second = sm.add_constant(df[["entropy_hat", "vix_pre"]])
    second_stage = sm.OLS(df["vix_post"], X_second).fit()

    # F-stat on excluded instrument in first stage
    n, k = X_first.shape
    f_stat = (first_stage.rsquared / 1) / ((1 - first_stage.rsquared) / (n - k))

    return {
        "params": second_stage.params.to_dict(),
        "std_errors": second_stage.bse.to_dict(),
        "t_stats": second_stage.tvalues.to_dict(),
        "p_values": second_stage.pvalues.to_dict(),
        "f_stat": float(f_stat),
        "n": int(n),
        "summary": str(second_stage.summary()),
    }


def run(
    entropy_path: str | Path = "data/fomc_entropy.csv",
    vix_path: str | Path = "data/vix.csv",
    surprises_path: str | Path = "data/fed_funds_surprises.csv",
    output_path: str | Path = "data/iv_results.csv",
    vix_window: int = 5,
) -> dict:
    """
    Full pipeline: load data → merge → 2SLS → save results.

    Returns
    -------
    dict with estimation results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Merging data...")
    df = _load_and_merge(entropy_path, vix_path, surprises_path, vix_window=vix_window)
    print(f"  {len(df)} FOMC meetings with complete data.")

    print("Running 2SLS estimation...")
    results = run_2sls(df)
    print(f"  N = {results['n']}, First-stage F = {results['f_stat']:.2f}")
    print(results["summary"])

    # Save parameter table
    param_df = pd.DataFrame(
        {
            "coef": results["params"],
            "std_err": results["std_errors"],
            "t_stat": results["t_stats"],
            "p_value": results["p_values"],
        }
    )
    param_df.to_csv(output_path)
    print(f"Saved IV results to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IV (2SLS) estimation.")
    parser.add_argument("--entropy", default="data/fomc_entropy.csv")
    parser.add_argument("--vix", default="data/vix.csv")
    parser.add_argument("--surprises", default="data/fed_funds_surprises.csv")
    parser.add_argument("--output", default="data/iv_results.csv")
    parser.add_argument("--vix-window", type=int, default=5,
                        help="Business days after FOMC to average VIX (default: 5)")
    args = parser.parse_args()
    run(
        entropy_path=args.entropy,
        vix_path=args.vix,
        surprises_path=args.surprises,
        output_path=args.output,
        vix_window=args.vix_window,
    )
