# analytics/benchmark.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
from analytics.risk import calculate_risk_metrics

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
    portfolio_ann_return: float,
    portfolio_ann_vol: float,
    period: str = "5y"
) -> pd.DataFrame:
    """Compares custom portfolio metrics directly against Nifty 50."""
    nifty_df = fetch_nifty50_benchmark(period=period)
    nifty_prices = nifty_df["NIFTY50"]
    
    # Align dates between portfolio and benchmark
    common_dates = portfolio_prices.index.intersection(nifty_prices.index)
    port_series = portfolio_prices.loc[common_dates]
    nifty_series = nifty_prices.loc[common_dates]
    
    # Calculate Nifty statistics
    nifty_log_returns = np.log(nifty_series / nifty_series.shift(1)).dropna()
    nifty_ann_return = float(np.exp(nifty_log_returns.mean() * 252) - 1)
    nifty_ann_vol = float(nifty_log_returns.std() * np.sqrt(252))
    
    # Normalize Nifty series to start at 1.0
    nifty_norm = nifty_series / nifty_series.iloc[0]
    
    port_metrics = calculate_risk_metrics(port_series, portfolio_ann_return, portfolio_ann_vol)
    nifty_metrics = calculate_risk_metrics(nifty_norm, nifty_ann_return, nifty_ann_vol)
    
    comparison_df = pd.DataFrame({
        "Metric": ["Annual Return (CAGR)", "Annual Volatility", "Sharpe Ratio (Rf=6.5%)", "Sortino Ratio", "Max Drawdown"],
        "Your Portfolio": [
            f"{portfolio_ann_return:.2%}",
            f"{portfolio_ann_vol:.2%}",
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
    
    return comparison_df