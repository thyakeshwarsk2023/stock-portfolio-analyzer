# analytics/benchmark.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from analytics.risk import calculate_risk_metrics, RISK_FREE_RATE_INDIA

def fetch_nifty50_benchmark(period: str = "5y") -> pd.DataFrame:
    """Fetches historical Nifty 50 (^NSEI) index data."""
    df = yf.download("^NSEI", period=period, progress=False)
    if "Adj Close" in df:
        prices = df["Adj Close"]
    else:
        prices = df["Close"]
        
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        
    prices = prices.rename(columns={prices.columns[0]: "NIFTY50"})
    return prices.dropna().ffill().bfill()

def evaluate_benchmark_comparison(
    portfolio_prices: pd.Series,
    portfolio_ann_return: Optional[float] = None,
    portfolio_ann_vol: Optional[float] = None,
    period: str = "5y",
    risk_free_rate: float = RISK_FREE_RATE_INDIA
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compares custom portfolio metrics directly against Nifty 50.
    Calculates portfolio CAGR directly from the date-aligned portfolio series
    using geometric compounding for exact mathematical consistency with the benchmark.
    Returns:
    - comparison_df: Summary DataFrame
    - nifty_norm: Date-aligned Nifty 50 normalized series (indexed to 1.0)
    """
    nifty_df = fetch_nifty50_benchmark(period=period)
    nifty_prices = nifty_df["NIFTY50"]
    
    # Align dates between portfolio and benchmark
    common_dates = portfolio_prices.index.intersection(nifty_prices.index)
    if len(common_dates) < 2:
        raise ValueError("Insufficient overlapping trading dates between portfolio and benchmark.")
        
    port_series = portfolio_prices.loc[common_dates]
    nifty_series = nifty_prices.loc[common_dates]
    
    # Normalize series starting at 1.0 on common start date
    port_norm = port_series / port_series.iloc[0]
    nifty_norm = nifty_series / nifty_series.iloc[0]
    
    # Calculate aligned geometric CAGR and annualized volatility for Portfolio
    port_log_returns = np.log(port_series / port_series.shift(1)).dropna()
    aligned_port_cagr = float(np.exp(port_log_returns.mean() * 252) - 1.0)
    aligned_port_vol = float(port_log_returns.std() * np.sqrt(252))
    
    # Calculate aligned geometric CAGR and annualized volatility for Nifty 50
    nifty_log_returns = np.log(nifty_series / nifty_series.shift(1)).dropna()
    nifty_ann_return = float(np.exp(nifty_log_returns.mean() * 252) - 1.0)
    nifty_ann_vol = float(nifty_log_returns.std() * np.sqrt(252))
    
    # Compute institutional risk metrics over the aligned window
    port_metrics = calculate_risk_metrics(port_norm, aligned_port_cagr, aligned_port_vol, risk_free_rate=risk_free_rate)
    nifty_metrics = calculate_risk_metrics(nifty_norm, nifty_ann_return, nifty_ann_vol, risk_free_rate=risk_free_rate)
    
    comparison_df = pd.DataFrame({
        "Metric": [
            "Annual Return (CAGR)",
            "Annual Volatility",
            f"Sharpe Ratio (Rf={risk_free_rate:.1%})",
            "Sortino Ratio",
            "Max Drawdown"
        ],
        "Your Portfolio": [
            f"{aligned_port_cagr:.2%}",
            f"{aligned_port_vol:.2%}",
            f"{port_metrics['Sharpe Ratio']:.2f}",
            f"{port_metrics['Sortino Ratio']:.2f}",
            f"{port_metrics['Max Drawdown']:.2%}"
        ],
        "Nifty 50 Index": [
            f"{nifty_ann_return:.2%}",
            f"{nifty_ann_vol:.2%}",
            f"{nifty_metrics['Sharpe Ratio']:.2f}",
            f"{nifty_metrics['Sortino Ratio']:.2f}",
            f"{nifty_metrics['Max Drawdown']:.2%}"
        ]
    })
    
    return comparison_df, nifty_norm
