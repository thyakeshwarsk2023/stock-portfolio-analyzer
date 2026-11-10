# analytics/risk.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

RISK_FREE_RATE_INDIA = 0.065  # 6.5% standard RBI G-Sec yield

def calculate_portfolio_series(price_df: pd.DataFrame, weights_dict: Dict[str, float]) -> pd.Series:
    """Calculates historical normalized portfolio daily value indexed to 1.0."""
    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers])
    
    # Normalized prices starting at 1.0
    normalized_prices = price_df[tickers] / price_df[tickers].iloc[0]
    portfolio_series = (normalized_prices * weights).sum(axis=1)
    return portfolio_series

def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculates peak-to-trough maximum drawdown."""
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return float(drawdown.min())

def calculate_sortino_ratio(
    daily_log_returns: pd.Series, 
    annual_return: float, 
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
    target_return: float = 0.0
) -> float:
    """
    Calculates annualized Sortino Ratio focusing strictly on downside deviation.
    Uses root-mean-square downside deviation (RMSDD) below target_return (default 0.0).
    """
    if len(daily_log_returns) == 0:
        return 0.0

    # Convert daily log returns to daily simple returns
    simple_returns = np.exp(daily_log_returns) - 1.0
    
    # Daily target threshold (default 0.0 for negative returns/losses)
    daily_target = target_return / 252.0
    downside = np.minimum(simple_returns - daily_target, 0.0)
    
    # Root-mean-square downside deviation across all N periods
    downside_variance = np.mean(downside ** 2)
    if downside_variance <= 0.0:
        return 0.0
        
    downside_std_annual = np.sqrt(downside_variance) * np.sqrt(252)
    if downside_std_annual == 0.0:
        return 0.0
        
    return float((annual_return - risk_free_rate) / downside_std_annual)

def calculate_risk_metrics(
    portfolio_series: pd.Series, 
    annual_return: float, 
    annual_volatility: float,
    risk_free_rate: float = RISK_FREE_RATE_INDIA
) -> Dict[str, float]:
    """Computes Sharpe Ratio, Sortino Ratio, and Max Drawdown."""
    # Calculate daily log returns of portfolio
    daily_log_returns = np.log(portfolio_series / portfolio_series.shift(1)).dropna()
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
    
    # Sortino Ratio
    sortino = calculate_sortino_ratio(daily_log_returns, annual_return, risk_free_rate)
    
    # Max Drawdown
    mdd = calculate_max_drawdown(portfolio_series)
    
    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd
    }
