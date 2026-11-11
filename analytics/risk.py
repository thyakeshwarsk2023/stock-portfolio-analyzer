# analytics/risk.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Set

RISK_FREE_RATE_INDIA = 0.065  # 6.5% standard RBI G-Sec yield


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> Set:
    """
    Returns a set of dates on which a periodic rebalance should occur,
    derived from true calendar boundaries rather than fixed trading-day counts.

    Rules
    -----
    daily     : every trading day rebalances (all dates).
    monthly   : last trading day of each calendar month.
    quarterly : last trading day of March, June, September, December.
    annually  : last trading day of December each year.
    """
    freq = freq.lower()
    if freq == "daily":
        return set(index)

    s = pd.Series(index, index=index)
    shifted = s.shift(-1)

    if freq == "monthly":
        mask = ((s.dt.month != shifted.dt.month) | (s.dt.year != shifted.dt.year)) & shifted.notna()
        return set(index[mask.values])

    if freq == "quarterly":
        mask = ((s.dt.quarter != shifted.dt.quarter) | (s.dt.year != shifted.dt.year)) & shifted.notna()
        return set(index[mask.values])

    if freq == "annually":
        mask = (s.dt.year != shifted.dt.year) & shifted.notna()
        return set(index[mask.values])

    # Unknown frequency — fall back to quarterly
    return _rebalance_dates(index, "quarterly")


def calculate_portfolio_series(
    price_df: pd.DataFrame,
    weights_dict: Dict[str, float],
    rebalance_freq: Optional[str] = None,
) -> pd.Series:
    """
    Builds a normalised (base = 1.0) portfolio value series.

    Parameters
    ----------
    price_df       : DataFrame of adjusted close prices.
    weights_dict   : {ticker: target_weight} allocation.
    rebalance_freq : One of 'none', 'daily', 'monthly', 'quarterly', 'annually',
                     or None.  When omitted / 'none' returns a static
                     buy-and-hold series where weights drift freely.

    Implementation notes
    --------------------
    Rebalancing uses a *unit-based* simulation: holds fractional asset units
    between rebalance events.  On each true calendar-boundary rebalance date
    the engine resets units so current market values match target weights.
    Calendar boundaries (month-end, quarter-end, year-end) are derived from
    the actual trading-date index — no fixed-day-count approximation.
    """
    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers], dtype=np.float64)
    prices = price_df[tickers].copy()

    if rebalance_freq is None or str(rebalance_freq).lower() in ("none", ""):
        normalized = prices / prices.iloc[0]
        return (normalized * weights).sum(axis=1)

    reb_dates = _rebalance_dates(prices.index, str(rebalance_freq))

    n_days = len(prices)
    portfolio_values = np.empty(n_days, dtype=np.float64)

    p0 = prices.iloc[0].values
    units = weights / p0
    portfolio_values[0] = (units * p0).sum()  # == 1.0

    for day_idx in range(1, n_days):
        p_today = prices.iloc[day_idx].values
        port_val = (units * p_today).sum()
        portfolio_values[day_idx] = port_val

        date_today = prices.index[day_idx]
        if date_today in reb_dates and day_idx < n_days - 1:
            units = (weights * port_val) / p_today

    return pd.Series(portfolio_values, index=prices.index)


def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculates peak-to-trough maximum drawdown."""
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return float(drawdown.min())


def calculate_sortino_ratio(
    daily_log_returns: pd.Series,
    annual_return: float,
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
    target_return: float = 0.0,
) -> float:
    """
    Calculates annualized Sortino Ratio.
    Uses root-mean-square downside deviation (RMSDD) below target_return.
    """
    if len(daily_log_returns) == 0:
        return 0.0

    simple_returns = np.exp(daily_log_returns) - 1.0
    daily_target = target_return / 252.0
    downside = np.minimum(simple_returns - daily_target, 0.0)

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
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Dict[str, float]:
    """Computes Sharpe Ratio, Sortino Ratio, and Max Drawdown."""
    daily_log_returns = np.log(portfolio_series / portfolio_series.shift(1)).dropna()
    sharpe  = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
    sortino = calculate_sortino_ratio(daily_log_returns, annual_return, risk_free_rate)
    mdd     = calculate_max_drawdown(portfolio_series)
    return {"Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Drawdown": mdd}
