# analytics/benchmark.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from analytics.risk import calculate_risk_metrics, RISK_FREE_RATE_INDIA

# NIFTYBEES.NS is the Nippon India ETF Nifty BeES — a Total Return (TRI) proxy
# for the Nifty 50. Dividends are reinvested into NAV, eliminating the
# +1.2–1.5% annual CAGR understatement in the raw ^NSEI Price Return index.
# Expense ratio ~0.05% — negligible vs the PR/TRI gap.
_BENCHMARK_TICKER = "NIFTYBEES.NS"
_BENCHMARK_LABEL  = "Nifty 50 TRI (NIFTYBEES)"


def fetch_nifty50_benchmark(period: str = "5y") -> pd.DataFrame:
    """
    Fetches Nifty 50 Total Return benchmark via NIFTYBEES.NS ETF.
    Returns adjusted close prices as a single-column DataFrame labelled 'NIFTY50'.
    """
    df = yf.download(_BENCHMARK_TICKER, period=period, progress=False, auto_adjust=True)

    if ("Close", _BENCHMARK_TICKER) in df.columns:
        prices = df[("Close", _BENCHMARK_TICKER)]
    elif "Close" in df.columns:
        prices = df["Close"]
    else:
        raise ValueError(f"Could not extract Close prices for {_BENCHMARK_TICKER}.")

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    return pd.DataFrame({"NIFTY50": prices}).dropna().ffill()


def evaluate_benchmark_comparison(
    portfolio_prices: pd.Series,
    portfolio_ann_return: Optional[float] = None,
    portfolio_ann_vol: Optional[float] = None,
    period: str = "5y",
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compares custom portfolio metrics against the Nifty 50 TRI (NIFTYBEES.NS).
    Both series are aligned on common trading dates before computing any metric.

    Returns
    -------
    comparison_df : pd.DataFrame — side-by-side metric table.
    nifty_norm    : pd.Series   — Nifty 50 TRI normalized to 1.0 on common start.
    """
    nifty_df = fetch_nifty50_benchmark(period=period)
    nifty_prices = nifty_df["NIFTY50"]

    common_dates = portfolio_prices.index.intersection(nifty_prices.index)
    if len(common_dates) < 2:
        raise ValueError("Insufficient overlapping trading dates between portfolio and benchmark.")

    port_series = portfolio_prices.loc[common_dates]
    nifty_series = nifty_prices.loc[common_dates]

    # Normalize to 1.0 on common start
    port_norm  = port_series  / port_series.iloc[0]
    nifty_norm = nifty_series / nifty_series.iloc[0]

    # Geometric CAGR and annualized vol (portfolio)
    port_log  = np.log(port_series  / port_series.shift(1)).dropna()
    aligned_port_cagr = float(np.exp(port_log.mean() * 252) - 1.0)
    aligned_port_vol  = float(port_log.std() * np.sqrt(252))

    # Geometric CAGR and annualized vol (benchmark)
    nifty_log = np.log(nifty_series / nifty_series.shift(1)).dropna()
    nifty_ann_return = float(np.exp(nifty_log.mean() * 252) - 1.0)
    nifty_ann_vol    = float(nifty_log.std() * np.sqrt(252))

    port_metrics  = calculate_risk_metrics(port_norm,  aligned_port_cagr, aligned_port_vol, risk_free_rate)
    nifty_metrics = calculate_risk_metrics(nifty_norm, nifty_ann_return,  nifty_ann_vol,   risk_free_rate)

    comparison_df = pd.DataFrame({
        "Metric": [
            "Annual Return (CAGR)",
            "Annual Volatility",
            f"Sharpe Ratio (Rf={risk_free_rate:.1%})",
            "Sortino Ratio",
            "Max Drawdown",
        ],
        "Your Portfolio": [
            f"{aligned_port_cagr:.2%}",
            f"{aligned_port_vol:.2%}",
            f"{port_metrics['Sharpe Ratio']:.2f}",
            f"{port_metrics['Sortino Ratio']:.2f}",
            f"{port_metrics['Max Drawdown']:.2%}",
        ],
        _BENCHMARK_LABEL: [
            f"{nifty_ann_return:.2%}",
            f"{nifty_ann_vol:.2%}",
            f"{nifty_metrics['Sharpe Ratio']:.2f}",
            f"{nifty_metrics['Sortino Ratio']:.2f}",
            f"{nifty_metrics['Max Drawdown']:.2%}",
        ],
    })

    return comparison_df, nifty_norm
