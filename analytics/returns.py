# analytics/returns.py
import numpy as np
import pandas as pd
from typing import Tuple

TRADING_DAYS_NSE = 252


def calculate_nse_metrics(
    price_df: pd.DataFrame,
    cov_halflife: int = 126,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Calculates annualized returns, volatility, and covariance matrix from
    historical NSE price data.

    Parameters
    ----------
    price_df     : DataFrame of adjusted close prices (may contain _Volume columns).
    cov_halflife : Half-life in trading days for EWMA covariance estimation.
                   Default 126 (~6 months) gives recent regimes ~4x more weight
                   than data from one year ago, down-weighting COVID-era extremes
                   (March-April 2020) relative to the full-window sample covariance.

    Returns
    -------
    annual_returns  : Annualized CAGR per asset (geometric, log-return basis).
    annual_volatility: Annualized volatility per asset.
    cov_matrix      : Annualized EWMA covariance matrix (forward-looking bias reduced).

    Notes
    -----
    CAGR uses the full available price window (geometric mean of daily log returns),
    which is unaffected by EWMA weighting — only the covariance / volatility estimate
    switches to EWMA so that the Monte Carlo shock magnitude reflects recent
    correlation structure rather than the COVID outlier cluster.
    """
    # Filter volume columns
    price_cols  = [c for c in price_df.columns if not c.endswith("_Volume")]
    clean_prices = price_df[price_cols]

    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()

    # Annualized CAGR: geometric mean over the full window (unweighted)
    mean_daily  = log_returns.mean()
    annual_returns   = np.exp(mean_daily * TRADING_DAYS_NSE) - 1.0

    # EWMA annualized volatility: exponentially weighted std, annualized
    ewma_vol    = log_returns.ewm(halflife=cov_halflife, adjust=True).std()
    annual_volatility = ewma_vol.iloc[-1] * np.sqrt(TRADING_DAYS_NSE)

    # EWMA annualized covariance matrix (RiskMetrics-style, halflife=126 trading days)
    # ewm().cov() returns a MultiIndex DataFrame; extract the last time-slice
    ewma_cov_daily = log_returns.ewm(halflife=cov_halflife, adjust=True).cov().iloc[
        -len(price_cols):
    ]
    # Reset MultiIndex to get a plain (A x A) matrix
    if isinstance(ewma_cov_daily.index, pd.MultiIndex):
        ewma_cov_daily = ewma_cov_daily.droplevel(0)
    cov_matrix = ewma_cov_daily * TRADING_DAYS_NSE

    return annual_returns, annual_volatility, cov_matrix
