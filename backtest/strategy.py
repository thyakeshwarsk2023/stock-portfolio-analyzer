# backtest/strategy.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

def generate_multi_factor_quant_signals(
    price_series: pd.Series,
    volume_series: pd.Series = None,
    rsi_period: int = 14,
    rsi_buy_threshold: float = 45.0,
    rsi_sell_threshold: float = 70.0,
    trend_filter_window: int = 200,
    volume_mult: float = 1.1
) -> pd.Series:
    """
    Multi-Factor Technical Strategy:
    1. Macro Trend Filter: Price MUST be above 200-day SMA.
    2. Momentum Trigger: RSI recovering (< 45).
    3. Volume Confirmation: Daily volume > 1.1x 20-day Volume SMA.
    """
    # 1. Macro Trend Guard
    sma_200 = price_series.rolling(window=trend_filter_window).mean()
    macro_uptrend = price_series > sma_200

    # 2. RSI Calculation
    delta = price_series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 3. Volume Filter
    if volume_series is not None:
        vol_sma = volume_series.rolling(window=20).mean()
        volume_confirm = volume_series > (vol_sma * volume_mult)
    else:
        volume_confirm = pd.Series(True, index=price_series.index)

    # Signal Generation State Machine
    signals = pd.Series(0.0, index=price_series.index)
    position = 0.0

    for i in range(len(price_series)):
        curr_rsi = rsi.iloc[i]
        curr_macro = macro_uptrend.iloc[i]
        curr_vol = volume_confirm.iloc[i]

        if np.isnan(curr_rsi) or np.isnan(curr_macro):
            continue

        # Buy Condition: In macro uptrend + RSI value area + Volume confirmation
        if curr_macro and (curr_rsi < rsi_buy_threshold) and curr_vol:
            position = 1.0
        # Sell Condition: RSI Overbought OR Macro Trend Breaks
        elif (curr_rsi > rsi_sell_threshold) or (not curr_macro):
            position = 0.0

        signals.iloc[i] = position

    # Shift by 1 day to eliminate Lookahead Bias
    return signals.shift(1).fillna(0.0)


def generate_portfolio_signals(
    prices_df: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Generates multi-factor quant signals for all portfolio assets.
    Returns a DataFrame of signals (0.0 or 1.0) indexed by date.
    """
    signals_df = pd.DataFrame(index=prices_df.index)
    for ticker in tickers:
        if ticker in prices_df.columns:
            price_col = prices_df[ticker]
            vol_col = prices_df.get(f"{ticker}_Volume")
            signals_df[ticker] = generate_multi_factor_quant_signals(price_col, volume_series=vol_col)
    return signals_df


def apply_fundamental_quality_filter(
    info: Dict[str, Any],
    min_market_cap_cr: float = 1000.0,
    min_roe: float = 0.10,
    max_pe: float = 80.0
) -> Tuple[bool, str]:
    """
    Fundamental Gatekeeper: Evaluates financial quality of an individual stock.
    Handles missing or fallback metrics with informative notices.
    """
    mkt_cap_cr = info.get("marketCap", 0) / 10000000.0 if info.get("marketCap") else 0.0
    roe = info.get("returnOnEquity")
    pe = info.get("trailingPE")
    name = info.get("longName", "Asset")

    # If data is completely missing or unpopulated from Yahoo Finance
    if info.get("data_warning"):
        return True, f"PASSED WITH CAUTION ({name}: {info['data_warning']})"

    if roe is None or roe == 0.0:
        return True, f"PASSED (Notice: ROE unavailable in data feed for {name})"

    if mkt_cap_cr > 0 and mkt_cap_cr < min_market_cap_cr:
        return False, f"REJECTED ({name}): Market Cap (₹{mkt_cap_cr:.1f} Cr) below ₹{min_market_cap_cr:.0f} Cr limit."
        
    if roe < min_roe:
        return False, f"REJECTED ({name}): ROE ({roe:.1%}) below {min_roe:.0%} threshold."
        
    if pe and (pe > max_pe or pe <= 0):
        return False, f"REJECTED ({name}): Trailing P/E ({pe:.1f}) outside valid range (0 - {max_pe})."

    pe_str = f"{pe:.1f}" if pe else "N/A"
    return True, f"PASSED ({name}): Market Cap ₹{mkt_cap_cr:,.0f} Cr | ROE {roe:.1%} | P/E {pe_str}"


def screen_portfolio_fundamentals(
    tickers: List[str],
    fundamental_info_dict: Dict[str, Dict[str, Any]],
    min_market_cap_cr: float = 1000.0,
    min_roe: float = 0.10,
    max_pe: float = 80.0
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Screens the entire portfolio against fundamental quality standards.
    Returns:
    - summary_df: DataFrame with MktCap, ROE, P/E, Status, and Reason
    - passed_tickers: list of tickers that passed
    - rejected_tickers: list of tickers that failed
    """
    rows = []
    passed_tickers = []
    rejected_tickers = []

    for t in tickers:
        info = fundamental_info_dict.get(t, {})
        passed, reason = apply_fundamental_quality_filter(
            info,
            min_market_cap_cr=min_market_cap_cr,
            min_roe=min_roe,
            max_pe=max_pe
        )
        
        mkt_cap_cr = info.get("marketCap", 0) / 10000000.0 if info.get("marketCap") else 0.0
        roe = info.get("returnOnEquity")
        pe = info.get("trailingPE")
        
        rows.append({
            "Ticker": t,
            "Market Cap (₹ Cr)": f"₹{mkt_cap_cr:,.0f}" if mkt_cap_cr > 0 else "N/A",
            "ROE": f"{roe:.1%}" if (roe is not None and roe != 0.0) else "N/A",
            "Trailing P/E": f"{pe:.1f}" if (pe is not None and pe > 0) else "N/A",
            "Gate Status": "PASSED" if passed else "REJECTED",
            "Details": reason
        })
        
        if passed:
            passed_tickers.append(t)
        else:
            rejected_tickers.append(t)

    summary_df = pd.DataFrame(rows)
    return summary_df, passed_tickers, rejected_tickers
