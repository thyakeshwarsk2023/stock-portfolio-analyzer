# data/fetch.py
import sys

# UTF-8 encoding safeguard for Windows console environments
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import logging
import time
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


def sanitize_bad_ticks(prices: pd.Series, label: str = "") -> pd.Series:
    """
    Detects and repairs transient Yahoo Finance feed glitches (e.g. 10x decimal
    omissions) where price temporarily drops >65% or spikes >200% and reverts
    within 1-3 trading days.
    """
    clean = prices.dropna().sort_index()
    if len(clean) < 5:
        return prices

    bad_tick_mask = pd.Series(False, index=clean.index)
    for lag in (1, 2, 3):
        prev_ok = clean.shift(lag)
        for lead in (1, 2, 3):
            next_ok = clean.shift(-lead)
            bad_tick_mask |= (clean < prev_ok * 0.35) & (next_ok > clean * 3.0)
            bad_tick_mask |= (clean > prev_ok * 3.0) & (next_ok < clean * 0.35)

    if bad_tick_mask.any():
        bad_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in clean.index[bad_tick_mask]]
        tag = f" in {label}" if label else ""
        print(f"[DATA] Repaired {int(bad_tick_mask.sum())} Yahoo Finance bad tick(s){tag}: {bad_dates}")
        clean = clean.mask(bad_tick_mask).interpolate(method="linear")
        return clean.reindex(prices.index)

    return prices


def _extract_ticker_series(data: pd.DataFrame, yf_symbol: str, is_single: bool) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Extracts Close and Volume series cleanly regardless of single vs MultiIndex columns."""
    price_col = None
    vol_col = None

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", yf_symbol) in data.columns:
            price_col = data[("Close", yf_symbol)]
        elif ("Adj Close", yf_symbol) in data.columns:
            price_col = data[("Adj Close", yf_symbol)]
        if ("Volume", yf_symbol) in data.columns:
            vol_col = data[("Volume", yf_symbol)]
    elif is_single:
        if "Close" in data.columns:
            price_col = data["Close"]
        elif "Adj Close" in data.columns:
            price_col = data["Adj Close"]
        if "Volume" in data.columns:
            vol_col = data["Volume"]

    if isinstance(price_col, pd.DataFrame):
        price_col = price_col.iloc[:, 0]
    if isinstance(vol_col, pd.DataFrame):
        vol_col = vol_col.iloc[:, 0]

    return price_col, vol_col


def fetch_nse_data(tickers: List[str], period: str = "5y", max_retries: int = 2) -> pd.DataFrame:
    """
    Fetches adjusted historical closing prices and daily volume for Indian equities.
    Uses explicit auto_adjust=True, exponential-backoff retries for transient Yahoo
    failures, shared bad-tick repair, and common-inception truncation.
    """
    symbol_map = {t: format_nse_symbol(t) for t in tickers}
    yf_symbols = list(symbol_map.values())

    print(f"[DATA] Fetching {period} historical data from NSE: {list(symbol_map.keys())}")

    combined_df = pd.DataFrame()
    pending = dict(symbol_map)

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = 2 ** (attempt - 1)
            print(f"[DATA] Retrying missing ticker(s) {list(pending.keys())} (attempt {attempt}/{max_retries}, wait {delay}s)...")
            time.sleep(delay)

        req_symbols = list(pending.values())
        data = yf.download(req_symbols, period=period, progress=False, auto_adjust=True)
        if data.empty:
            continue

        is_single = len(req_symbols) == 1
        still_missing = {}
        for orig_ticker, yf_symbol in pending.items():
            p_col, v_col = _extract_ticker_series(data, yf_symbol, is_single)
            if p_col is not None and not p_col.dropna().empty:
                combined_df[orig_ticker] = sanitize_bad_ticks(p_col, label=yf_symbol)
                if v_col is not None:
                    combined_df[f"{orig_ticker}_Volume"] = v_col
            else:
                still_missing[orig_ticker] = yf_symbol

        pending = still_missing
        if not pending:
            break

    combined_df = combined_df.dropna(how="all")
    if combined_df.empty:
        return combined_df

    if combined_df.index.tz is not None:
        combined_df.index = combined_df.index.tz_localize(None)

    # Common Inception Date Alignment
    price_cols = [t for t in tickers if t in combined_df.columns]
    first_valid_dates = {}
    for t in price_cols:
        f_idx = combined_df[t].first_valid_index()
        if f_idx is not None:
            first_valid_dates[t] = f_idx
        else:
            print(f"[WARN] No price history returned for ticker '{t}'.")

    for missing_t in pending:
        print(f"[WARN] No price history returned for ticker '{missing_t}' after {max_retries + 1} attempts.")

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
    Applies a best-effort multi-stage fallback chain:
    1. Direct yfinance info attributes.
    2. Primary statement recovery (balance sheet Net Income / Stockholder Equity)
       for Indian PSUs and conglomerates where Yahoo info omits ROE.
    3. Trailing P/E calculation from currentPrice / trailingEps if trailingPE is missing.
    """
    yf_symbol = format_nse_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(yf_symbol)
        info = ticker_obj.info

        market_cap = info.get("marketCap", 0) or 0

        # Primary ROE lookup
        roe = info.get("returnOnEquity")

        # Fallback 1: Balance sheet / Financials statement calculation (recovers HAL, defence PSUs)
        if roe is None or roe == 0.0:
            try:
                bs = ticker_obj.balance_sheet
                fin = ticker_obj.financials
                if not bs.empty and not fin.empty:
                    eq = 0
                    if "Common Stock Equity" in bs.index:
                        eq = bs.loc["Common Stock Equity"].iloc[0]
                    elif "Stockholders Equity" in bs.index:
                        eq = bs.loc["Stockholders Equity"].iloc[0]

                    ni = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else 0
                    if eq and eq > 0 and ni:
                        roe = float(ni / eq)
            except Exception:
                pass

        # Fallback 2: ROA proxy if ROE still unresolvable
        if roe is None or roe == 0.0:
            roe = info.get("returnOnAssets") or 0.0

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
       via '^CSTG10Y'.
    3. Institutional default: 6.50% (RBI 10Y Benchmark G-Sec yield).

    Returns
    -------
    float: Annual risk-free rate as a decimal (e.g. 0.065 for 6.50%).
    """
    if custom_rf is not None:
        print(f"[CONFIG] Using custom risk-free rate: {custom_rf:.2%}")
        return float(custom_rf)

    # Attempt live fetch of Indian 10Y G-Sec yield quietly
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    try:
        yf_logger.setLevel(logging.CRITICAL)
        obj = yf.Ticker("^CSTG10Y")
        hist = obj.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            latest_yield = float(hist["Close"].dropna().iloc[-1])
            if 1.0 < latest_yield < 20.0:
                rf = latest_yield / 100.0
                print(f"[DATA] Live Indian 10Y G-Sec yield fetched (^CSTG10Y): {rf:.2%}")
                return rf
    except Exception:
        pass
    finally:
        yf_logger.setLevel(prev_level)

    print(f"[CONFIG] Using standard RBI 10Y G-Sec risk-free rate: {_DEFAULT_RF:.2%}")
    return _DEFAULT_RF
