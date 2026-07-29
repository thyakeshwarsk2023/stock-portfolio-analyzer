# backtest/strategies.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

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


# backtest/strategies.py
def apply_fundamental_quality_filter(
    info: Dict[str, Any],
    min_market_cap_cr: float = 1000.0,
    min_roe: float = 0.10,
    max_pe: float = 80.0
) -> Tuple[bool, str]:
    """
    Fundamental Gatekeeper: Rejects stocks failing financial health checks.
    Handles missing API fields gracefully.
    """
    mkt_cap_cr = info.get("marketCap", 0) / 10000000.0 if info.get("marketCap") else 0.0
    roe = info.get("returnOnEquity")
    pe = info.get("trailingPE")

    # If Yahoo Finance returns None/0 for ROE or Market Cap, issue warning and pass
    if roe is None or roe == 0.0:
        return True, f"PASSED (API Notice: ROE unavailable via Yahoo Finance feed for {info.get('longName', 'Asset')})"

    if mkt_cap_cr > 0 and mkt_cap_cr < min_market_cap_cr:
        return False, f"REJECTED: Market Cap (₹{mkt_cap_cr:.1f} Cr) below ₹{min_market_cap_cr:.0f} Cr limit."
        
    if roe < min_roe:
        return False, f"REJECTED: ROE ({roe:.1%}) below {min_roe:.0%} limit."
        
    if pe and (pe > max_pe or pe <= 0):
        return False, f"REJECTED: Trailing P/E ({pe:.1f}) outside valid range (0 - {max_pe})."

    return True, f"PASSED: Market Cap ₹{mkt_cap_cr:,.0f} Cr | ROE {roe:.1%} | P/E {pe:.1f}"