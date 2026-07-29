import numpy as np
import pandas as pd
from typing import Tuple

TRADING_DAYS_NSE = 252

def calculate_nse_metrics(price_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Calculates annualized returns, annualized volatility, and annualized covariance matrix 
    from historical price data, filtering out volume columns to prevent zero-log warnings.
    """
    # 1. Filter out volume columns (keep only stock price columns)
    price_cols = [col for col in price_df.columns if not col.endswith("_Volume")]
    clean_prices = price_df[price_cols]

    # 2. Calculate daily log returns
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()

    # 3. Calculate annualized returns (CAGR using compound exponential)
    mean_daily_returns = log_returns.mean()
    annual_returns = np.exp(mean_daily_returns * TRADING_DAYS_NSE) - 1.0

    # 4. Calculate annualized volatility (Standard Deviation * sqrt(252))
    annual_volatility = log_returns.std() * np.sqrt(TRADING_DAYS_NSE)

    # 5. Calculate annualized covariance matrix (Daily Covariance * 252)
    cov_matrix = log_returns.cov() * TRADING_DAYS_NSE

    return annual_returns, annual_volatility, cov_matrix