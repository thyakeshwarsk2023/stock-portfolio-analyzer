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
from typing import List, Dict, Tuple, Any

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
    (Market Cap, ROE, Trailing P/E, Sector) for fundamental quality checks.
    """
    yf_symbol = format_nse_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        info = ticker_obj.info
        return {
            "marketCap": info.get("marketCap", 0),
            "returnOnEquity": info.get("returnOnEquity", 0.0),
            "trailingPE": info.get("trailingPE", 0.0),
            "sector": info.get("sector", "Unknown"),
            "longName": info.get("longName", ticker)
        }
    except Exception as e:
        print(f"[WARN] Could not fetch fundamental info for {ticker}: {e}")
        return {
            "marketCap": 0,
            "returnOnEquity": 0.0,
            "trailingPE": 0.0,
            "sector": "Unknown",
            "longName": ticker
        }
