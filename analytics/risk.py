# analytics/risk.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

RISK_FREE_RATE_INDIA = 0.065  # 6.5% standard RBI G-Sec yield

def calculate_portfolio_series(
    price_df: pd.DataFrame, 
    weights_dict: Dict[str, float],
    rebalance_freq: Optional[str] = None
) -> pd.Series:
    """
    Calculates historical portfolio daily value indexed to 1.0.
    Supports static Buy-and-Hold or true periodic rebalancing.
    
    Parameters:
    - price_df: DataFrame of historical asset closing prices
    - weights_dict: Target allocations per ticker (e.g., {'BEL': 0.25, ...})
    - rebalance_freq: Rebalancing frequency:
        - None or 'none': Static Buy-and-Hold (weights drift with asset price growth)
        - 'daily': Rebalanced daily back to target weights
        - 'monthly': Rebalanced at the start of every calendar month
        - 'quarterly': Rebalanced at the start of every calendar quarter
        - 'annually': Rebalanced at the start of every calendar year
    """
    tickers = list(weights_dict.keys())
    target_weights = np.array([weights_dict[t] for t in tickers], dtype=float)
    weight_sum = np.sum(target_weights)
    if weight_sum > 0:
        target_weights = target_weights / weight_sum  # Normalize
        
    clean_prices = price_df[tickers].copy()
    if len(clean_prices) == 0:
        return pd.Series(dtype=float)
        
    freq = (rebalance_freq or "none").strip().lower()
    
    # 1. Static Buy-and-Hold (No Rebalancing: weights drift)
    if freq in ("none", "buy_and_hold", "hold"):
        normalized_prices = clean_prices / clean_prices.iloc[0]
        portfolio_series = (normalized_prices * target_weights).sum(axis=1)
        portfolio_series.name = "Portfolio_BuyAndHold"
        return portfolio_series

    # 2. Daily Rebalanced Portfolio (Fixed daily weight exposure)
    if freq == "daily":
        daily_asset_returns = clean_prices.pct_change().fillna(0.0)
        daily_port_return = (daily_asset_returns * target_weights).sum(axis=1)
        portfolio_series = (1.0 + daily_port_return).cumprod()
        portfolio_series.name = "Portfolio_DailyRebalanced"
        return portfolio_series

    # 3. Periodic Rebalancing (Monthly, Quarterly, Annually)
    dates = pd.to_datetime(clean_prices.index)
    n_days = len(dates)
    portfolio_values = np.zeros(n_days, dtype=float)
    portfolio_values[0] = 1.0
    
    # Determine rebalance trigger dates
    is_rebalance_day = np.zeros(n_days, dtype=bool)
    for i in range(1, n_days):
        prev_d = dates[i - 1]
        curr_d = dates[i]
        
        if freq == "monthly":
            if curr_d.month != prev_d.month or curr_d.year != prev_d.year:
                is_rebalance_day[i] = True
        elif freq == "quarterly":
            curr_q = (curr_d.month - 1) // 3
            prev_q = (prev_d.month - 1) // 3
            if curr_q != prev_q or curr_d.year != prev_d.year:
                is_rebalance_day[i] = True
        elif freq in ("annually", "annual", "yearly"):
            if curr_d.year != prev_d.year:
                is_rebalance_day[i] = True
        else:
            raise ValueError(f"Unknown rebalance_freq: '{rebalance_freq}'. Expected 'monthly', 'quarterly', 'annually', 'daily', or 'none'.")

    # Simulate wealth evolution with unit-based rebalancing
    prices_matrix = clean_prices.values
    current_units = (portfolio_values[0] * target_weights) / prices_matrix[0]
    
    for t in range(1, n_days):
        port_val = float(np.sum(current_units * prices_matrix[t]))
        portfolio_values[t] = port_val
        
        if is_rebalance_day[t]:
            current_units = (port_val * target_weights) / prices_matrix[t]
            
    portfolio_series = pd.Series(portfolio_values, index=clean_prices.index, name=f"Portfolio_{freq.capitalize()}Rebalanced")
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
