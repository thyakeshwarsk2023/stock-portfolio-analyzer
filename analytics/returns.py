# analytics/returns.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional

TRADING_DAYS_NSE = 252

# Standard date boundary for the Indian market COVID-19 outlier shock
_COVID_START = "2020-02-01"
_COVID_END   = "2020-05-31"


def calculate_nse_metrics(
    price_df: pd.DataFrame,
    cov_halflife: int = 126,
    filter_covid: bool = True,
    drift_halflife: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Calculates annualized returns (drift), volatility, and covariance matrix
    from historical NSE price data with institutional regime filtering.

    Parameters
    ----------
    price_df       : DataFrame of adjusted close prices (may contain _Volume columns).
    cov_halflife   : Half-life in trading days for EWMA covariance estimation.
                     Default 126 (~6 months) gives recent regimes ~4x more weight
                     than data from one year ago, down-weighting distant regime extremes
                     relative to full-window sample covariance.
    filter_covid   : bool, default True. If the dataset spans early 2020, masks out
                     the March-May 2020 COVID-19 crash and liquidity shock window
                     (2020-02-01 to 2020-05-31) so that abnormal -40% crash returns
                     and anomalous bounce spikes do not distort drift and covariance.
    drift_halflife : Optional[int], default None. If specified, estimates expected
                     annualized return (drift) using EWMA weighting over the clean window,
                     providing regime-adaptive forward drift for Monte Carlo simulations.
                     If None, uses unweighted geometric CAGR over the clean window.

    Returns
    -------
    annual_returns   : Annualized CAGR per asset (used as drift vector in Monte Carlo).
    annual_volatility: Annualized volatility per asset.
    cov_matrix       : Annualized EWMA covariance matrix (forward-looking shock matrix).
    """
    # Filter volume columns
    price_cols   = [c for c in price_df.columns if not c.endswith("_Volume")]
    clean_prices = price_df[price_cols]

    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()

    # Mask out COVID-19 shock window if present in historical data
    if filter_covid and not log_returns.empty:
        # Normalize index to tz-naive for date comparison
        idx = log_returns.index.tz_localize(None) if log_returns.index.tz is not None else log_returns.index
        covid_mask = (idx >= _COVID_START) & (idx <= _COVID_END)
        covid_bars = int(covid_mask.sum())
        if covid_bars > 0:
            print(f"[DATA] COVID-19 outlier regime detected ({covid_bars} bars, {_COVID_START} to {_COVID_END}).")
            print(f"       Filtering COVID window to prevent anomalous shock distortions in drift & covariance.")
            log_returns = log_returns.loc[~covid_mask]

    # Annualized Drift (CAGR)
    if drift_halflife is not None and drift_halflife > 0:
        mean_daily = log_returns.ewm(halflife=drift_halflife, adjust=True).mean().iloc[-1]
    else:
        mean_daily = log_returns.mean()

    annual_returns = np.exp(mean_daily * TRADING_DAYS_NSE) - 1.0

    # EWMA annualized volatility: exponentially weighted std, annualized
    ewma_vol = log_returns.ewm(halflife=cov_halflife, adjust=True).std()
    annual_volatility = ewma_vol.iloc[-1] * np.sqrt(TRADING_DAYS_NSE)

    # EWMA annualized covariance matrix (RiskMetrics-style, halflife=126 trading days)
    ewma_cov_daily = log_returns.ewm(halflife=cov_halflife, adjust=True).cov().iloc[
        -len(price_cols):
    ]
    if isinstance(ewma_cov_daily.index, pd.MultiIndex):
        ewma_cov_daily = ewma_cov_daily.droplevel(0)
    cov_matrix = ewma_cov_daily * TRADING_DAYS_NSE

    return annual_returns, annual_volatility, cov_matrix
