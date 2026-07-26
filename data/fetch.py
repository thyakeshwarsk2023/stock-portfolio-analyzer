# data/fetch.py
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
    Fetches historical closing prices AND daily volume for Indian equities.
    Output DataFrame columns:
      - Price columns : Clean ticker names (e.g., 'BEL', 'RELIANCE')
      - Volume columns: Ticker with '_Volume' suffix (e.g., 'BEL_Volume', 'RELIANCE_Volume')
    """
    symbol_map = {t: format_nse_symbol(t) for t in tickers}
    yf_symbols = list(symbol_map.values())
    
    print(f"Fetching {period} historical price & volume data from NSE for: {list(symbol_map.keys())}...")
    
    data = yf.download(yf_symbols, period=period, progress=False)
    combined_df = pd.DataFrame()

    # Case 1: Single Ticker Download (yfinance returns single-level columns)
    if len(tickers) == 1:
        ticker = tickers[0]
        if "Adj Close" in data:
            combined_df[ticker] = data["Adj Close"]
        elif "Close" in data:
            combined_df[ticker] = data["Close"]

        if "Volume" in data:
            combined_df[f"{ticker}_Volume"] = data["Volume"]

    # Case 2: Multiple Tickers Download (yfinance returns MultiIndex columns)
    else:
        for orig_ticker, yf_symbol in symbol_map.items():
            # Extract Adjusted Close / Close Price
            if ("Adj Close", yf_symbol) in data.columns:
                combined_df[orig_ticker] = data[("Adj Close", yf_symbol)]
            elif ("Close", yf_symbol) in data.columns:
                combined_df[orig_ticker] = data[("Close", yf_symbol)]

            # Extract Daily Trading Volume
            if ("Volume", yf_symbol) in data.columns:
                combined_df[f"{orig_ticker}_Volume"] = data[("Volume", yf_symbol)]

    # Clean missing dates & forward/backward fill holiday gaps
    combined_df = combined_df.dropna(how="all").ffill().bfill()
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
        print(f"⚠️ Could not fetch fundamental info for {ticker}: {e}")
        return {
            "marketCap": 0,
            "returnOnEquity": 0.0,
            "trailingPE": 0.0,
            "sector": "Unknown",
            "longName": ticker
        }