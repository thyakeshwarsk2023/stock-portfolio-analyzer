# data/fetch.py
import sys

# UTF-8 encoding safeguard for Windows console environments
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import yfinance as yf
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional

DEFAULT_INDIAN_GSEC_RF = 0.065  # 6.50% standard 10-Year RBI G-Sec benchmark yield

def format_nse_symbol(symbol: str) -> str:
    """Ensures symbol has .NS suffix for NSE India data."""
    symbol = symbol.strip().upper()
    if symbol.startswith("^") or "." in symbol:
        return symbol
    return f"{symbol}.NS"


def fetch_nse_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Fetches historical closing prices and daily volume for Indian equities.
    Truncates the historical window to the common inception date of all assets
    to eliminate artificial 0% return/volatility distortions from backward-filling recent IPOs.
    """
    symbol_map = {t: format_nse_symbol(t) for t in tickers}
    yf_symbols = list(symbol_map.values())
    
    print(f"[DATA] Fetching {period} historical data from NSE: {list(symbol_map.keys())}")
    
    data = yf.download(yf_symbols, period=period, progress=False)
    combined_df = pd.DataFrame()

    # Case 1: Single Ticker Download
    if len(tickers) == 1:
        ticker = tickers[0]
        if "Adj Close" in data:
            combined_df[ticker] = data["Adj Close"]
        elif "Close" in data:
            combined_df[ticker] = data["Close"]

        if "Volume" in data:
            combined_df[f"{ticker}_Volume"] = data["Volume"]

    # Case 2: Multiple Tickers Download
    else:
        for orig_ticker, yf_symbol in symbol_map.items():
            if ("Adj Close", yf_symbol) in data.columns:
                combined_df[orig_ticker] = data[("Adj Close", yf_symbol)]
            elif ("Close", yf_symbol) in data.columns:
                combined_df[orig_ticker] = data[("Close", yf_symbol)]

            if ("Volume", yf_symbol) in data.columns:
                combined_df[f"{orig_ticker}_Volume"] = data[("Volume", yf_symbol)]

    combined_df = combined_df.dropna(how="all")
    if combined_df.empty:
        return combined_df

    # Common Inception Date Alignment
    price_cols = [t for t in tickers if t in combined_df.columns]
    first_valid_dates = {}
    for t in price_cols:
        f_idx = combined_df[t].first_valid_index()
        if f_idx is not None:
            first_valid_dates[t] = f_idx
        else:
            print(f"[WARN] No price history returned for ticker '{t}'.")

    if first_valid_dates:
        latest_inception_date = max(first_valid_dates.values())
        earliest_date = combined_df.index[0]
        
        if (latest_inception_date - earliest_date).days > 7:
            recent_stocks = [t for t, d in first_valid_dates.items() if d == latest_inception_date]
            print(f"[NOTICE] Recent listing detected: {recent_stocks} inception date is {latest_inception_date.strftime('%Y-%m-%d')}.")
            print(f"         Restricting window to {latest_inception_date.strftime('%Y-%m-%d')} to prevent 0% return distortion.")
            combined_df = combined_df.loc[latest_inception_date:]

    combined_df = combined_df.ffill().bfill().dropna()
    return combined_df


def fetch_fundamental_info(ticker: str) -> Dict[str, Any]:
    """
    Fetches fundamental financial metrics for an NSE stock 
    (Market Cap, ROE, Trailing P/E, Sector) with fallback calculations.
    """
    yf_symbol = format_nse_symbol(ticker)
    data_warning = None
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        info = ticker_obj.info or {}
        
        market_cap = info.get("marketCap", 0)
        roe = info.get("returnOnEquity")
        trailing_pe = info.get("trailingPE")
        sector = info.get("sector", "Unknown")
        long_name = info.get("longName", ticker)
        
        # Fallback 1: Calculate Trailing P/E from current price and trailing EPS
        if (trailing_pe is None or trailing_pe == 0.0):
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            eps = info.get("trailingEps")
            if current_price and eps and eps > 0:
                trailing_pe = round(current_price / eps, 2)
            elif info.get("forwardPE"):
                trailing_pe = round(info.get("forwardPE"), 2)

        # Fallback 2: Check alternative ROE fields or ROA
        if roe is None or roe == 0.0:
            if "financialData" in info and isinstance(info["financialData"], dict):
                roe = info["financialData"].get("returnOnEquity")
            if (roe is None or roe == 0.0) and info.get("returnOnAssets"):
                roe = info.get("returnOnAssets")

        missing_fields = []
        if not market_cap:
            missing_fields.append("Market Cap")
        if not roe:
            missing_fields.append("ROE")
        if not trailing_pe:
            missing_fields.append("Trailing P/E")
            
        if missing_fields:
            data_warning = f"Metrics {missing_fields} omitted in feed"

        return {
            "marketCap": market_cap or 0,
            "returnOnEquity": float(roe) if roe else 0.0,
            "trailingPE": float(trailing_pe) if trailing_pe else 0.0,
            "sector": sector,
            "longName": long_name,
            "data_warning": data_warning
        }
    except Exception as e:
        return {
            "marketCap": 0,
            "returnOnEquity": 0.0,
            "trailingPE": 0.0,
            "sector": "Unknown",
            "longName": ticker,
            "data_warning": f"Data error: {str(e)}"
        }


def fetch_indian_risk_free_rate(custom_rf: Optional[float] = None) -> float:
    """
    Determines the risk-free rate for Indian equity analytics:
    - If custom_rf is provided, validates and applies it.
    - Otherwise, defaults to the standard RBI 10Y G-Sec benchmark rate (6.50%).
    """
    if custom_rf is not None and 0.01 <= custom_rf <= 0.20:
        print(f"[CONFIG] Risk-Free Rate: {custom_rf:.2%} (Custom Override)")
        return custom_rf

    rf = DEFAULT_INDIAN_GSEC_RF
    print(f"[CONFIG] Risk-Free Rate: {rf:.2%} (RBI 10Y G-Sec Benchmark)")
    return rf
