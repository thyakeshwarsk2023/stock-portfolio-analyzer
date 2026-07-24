# data/fetch.py
import yfinance as yf
import pandas as pd
from typing import List, Dict

def format_nse_symbol(symbol: str) -> str:
    """Ensures symbol has .NS suffix for NSE India data."""
    symbol = symbol.strip().upper()
    if symbol.startswith("^") or "." in symbol:
        return symbol
    return f"{symbol}.NS"

def fetch_nse_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """Fetches historical price data for Indian equities."""
    symbol_map = {t: format_nse_symbol(t) for t in tickers}
    yf_symbols = list(symbol_map.values())
    
    print(f"Fetching {period} historical data from NSE for: {list(symbol_map.keys())}...")
    
    data = yf.download(yf_symbols, period=period, progress=False)
    
    if "Adj Close" in data:
        df = data["Adj Close"]
    elif "Close" in data:
        df = data["Close"]
    else:
        df = data
        
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Clean data
    df = df.dropna(how="all").ffill().bfill()
    
    # Rename back to clean ticker names (e.g. 'BEL.NS' -> 'BEL')
    reverse_map = {v: k for k, v in symbol_map.items()}
    df = df.rename(columns=reverse_map)
    
    return df