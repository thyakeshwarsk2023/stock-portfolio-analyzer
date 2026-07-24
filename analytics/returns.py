import numpy as np
import pandas as pd
from typing import Tuple

TRADING_DAYS_NSE = 252
def calculate_nse_metrics(price_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    mean_daily_returns = log_returns.mean()
    annual_returns = np.exp(mean_daily_returns * TRADING_DAYS_NSE) - 1
    annual_volatility = log_returns.std() * np.sqrt(TRADING_DAYS_NSE)
    cov_matrix = log_returns.cov()
    return annual_returns, annual_volatility, cov_matrix
