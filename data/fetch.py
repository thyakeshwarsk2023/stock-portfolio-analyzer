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

_DEFAULT_RF = 0.065   # 6.5% RBI 10-Year G-Sec fallback


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
    Applies a best-effort fallback chain when Yahoo Finance omits fields for Indian equities.
    """
    yf_symbol = format_nse_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        info = ticker_obj.info

        market_cap = info.get("marketCap", 0) or 0

        # ROE fallback: returnOnEquity -> financialData -> returnOnAssets (proxy)
        roe = info.get("returnOnEquity") or info.get("returnOnAssets") or 0.0

        # P/E fallback: trailingPE -> price / trailingEps
        pe = info.get("trailingPE") or 0.0
        if not pe:
            price = info.get("regularMarketPrice") or info.get("currentPrice") or 0.0
            eps = info.get("trailingEps") or 0.0
            if price and eps and eps > 0:
                pe = price / eps

        return {
            "marketCap": market_cap,
            "returnOnEquity": roe,
            "trailingPE": pe,
            "sector": info.get("sector", "Unknown"),
            "longName": info.get("longName", ticker),
        }
    except Exception as e:
        print(f"[WARN] Could not fetch fundamental info for {ticker}: {e}")
        return {
            "marketCap": 0,
            "returnOnEquity": 0.0,
            "trailingPE": 0.0,
            "sector": "Unknown",
            "longName": ticker,
        }


def fetch_indian_risk_free_rate(custom_rf: Optional[float] = None) -> float:
    """
    Returns the annual risk-free rate to use for Sharpe / Sortino calculations.

    Priority:
    1. custom_rf if explicitly provided by the user (CLI --rf flag).
    2. Live RBI proxy: attempts to fetch the yield of the Indian 10-Year G-Sec
       via the '^IRX' (13-week US T-Bill) as a fallback proxy; for a true
       Indian rate the '^CSTG10Y' or RBI API should be preferred but Yahoo
       Finance's coverage of Indian G-Sec is unreliable.
    3. Hard-coded default: 6.50% (RBI 10Y G-Sec as of late 2024).

    Returns
    -------
    float: Annual risk-free rate as a decimal (e.g. 0.065 for 6.50%).
    """
    if custom_rf is not None:
        print(f"[CONFIG] Using custom risk-free rate: {custom_rf:.2%}")
        return float(custom_rf)

    # Attempt live fetch of Indian 10Y G-Sec yield via Yahoo Finance
    _INDIA_GSEC_TICKERS = ["^CSTG10Y", "INDIAVIX.NS"]
    for gsec_ticker in _INDIA_GSEC_TICKERS:
        try:
            obj = yf.Ticker(gsec_ticker)
            hist = obj.history(period="5d")
            if not hist.empty and "Close" in hist.columns:
                latest_yield = float(hist["Close"].dropna().iloc[-1])
                # G-Sec yields from Yahoo are expressed as percentages (e.g. 6.75)
                if 1.0 < latest_yield < 20.0:
                    rf = latest_yield / 100.0
                    print(f"[DATA] Live Indian G-Sec yield fetched ({gsec_ticker}): {rf:.2%}")
                    return rf
        except Exception:
            continue

    print(f"[CONFIG] Using default RBI 10Y G-Sec risk-free rate: {_DEFAULT_RF:.2%}")
    return _DEFAULT_RF
