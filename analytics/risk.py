# analytics/risk.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

RISK_FREE_RATE_INDIA = 0.065  # 6.5% standard RBI G-Sec yield

# Rebalancing frequency -> approximate trading days per period
_REBALANCE_PERIODS = {
    "daily":     1,
    "monthly":   21,
    "quarterly": 63,
    "annually":  252,
}


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
                     or None.  When omitted / 'none', returns a static
                     buy-and-hold series where weights drift freely with prices.

    Returns
    -------
    pd.Series of daily portfolio values indexed to 1.0 on the first day.

    Implementation notes
    --------------------
    The rebalancing engine works in *unit space* rather than compounding
    daily-return multipliers.  On each rebalance date the engine resets the
    number of units held in every asset so that their current market values
    match the target weights exactly.  Between rebalance dates asset values
    drift naturally with their price series.  This avoids the well-known
    path-dependency error that arises when you apply weight vectors to
    cumulatively-compounded return arrays.
    """
    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers], dtype=np.float64)
    prices = price_df[tickers].copy()

    # ------------------------------------------------------------------ #
    # Static buy-and-hold: allocate fractional units on day 0, never trade
    # ------------------------------------------------------------------ #
    if rebalance_freq is None or str(rebalance_freq).lower() in ("none", ""):
        normalized = prices / prices.iloc[0]
        return (normalized * weights).sum(axis=1)

    # ------------------------------------------------------------------ #
    # Periodic rebalancing: unit-based simulation
    # ------------------------------------------------------------------ #
    period_days = _REBALANCE_PERIODS.get(str(rebalance_freq).lower(), 63)

    n_days = len(prices)
    portfolio_values = np.empty(n_days, dtype=np.float64)

    # Day 0: buy initial units proportional to target weights.
    # Treat total portfolio as 1.0 notional.
    p0 = prices.iloc[0].values                          # shape (A,)
    units = weights / p0                                # fractional units per asset
    portfolio_values[0] = (units * p0).sum()            # == 1.0 by construction

    for day_idx in range(1, n_days):
        p_today = prices.iloc[day_idx].values
        port_val = (units * p_today).sum()
        portfolio_values[day_idx] = port_val

        # Rebalance at every period boundary (but not on the final day)
        if day_idx % period_days == 0 and day_idx < n_days - 1:
            # Redistribute total value back to target weights
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
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Dict[str, float]:
    """Computes Sharpe Ratio, Sortino Ratio, and Max Drawdown."""
    daily_log_returns = np.log(portfolio_series / portfolio_series.shift(1)).dropna()

    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
    sortino = calculate_sortino_ratio(daily_log_returns, annual_return, risk_free_rate)
    mdd = calculate_max_drawdown(portfolio_series)

    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
    }
