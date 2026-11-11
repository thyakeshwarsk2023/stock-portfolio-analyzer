# backtest/strategy.py
import pandas as pd
import numpy as np
import scipy.signal as sig
from typing import Dict, Any, Tuple, List, Optional


def calculate_wilder_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates RSI using J. Welles Wilder's original exponential smoothing (RMA).

    Wilder initialization (1978 spec):
      - AvgGain[period] = simple mean of first `period` gains.
      - Subsequent bars: AvgGain[i] = AvgGain[i-1] * (1 - 1/N) + Gain[i] * (1/N).

    Implemented via scipy.signal.lfilter — a compiled C-level 1st-order IIR
    recursion. Eliminates both the Python-level for-loop and the numerical
    underflow risks of explicit power division on long time horizons.
    """
    if len(price_series) <= period:
        return pd.Series(np.nan, index=price_series.index)

    prices = price_series.values.astype(np.float64)
    deltas = np.diff(prices)                                      # (n-1,)

    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    n     = len(prices)
    alpha = 1.0 / period
    decay = 1.0 - alpha

    # Seed: simple mean of first `period` changes (index 0..period-1 of deltas)
    seed_g = float(np.mean(gains[:period]))
    seed_l = float(np.mean(losses[:period]))

    tail_g = gains[period:]
    tail_l = losses[period:]

    # Compiled C IIR recursion: y[k] = decay * y[k-1] + alpha * x[k]
    # Setting initial delay condition zi = [seed * decay] guarantees y[0] matches Wilder recursion
    b = [alpha]
    a = [1.0, -decay]

    if len(tail_g) > 0:
        tail_avg_g, _ = sig.lfilter(b, a, tail_g, zi=[seed_g * decay])
        tail_avg_l, _ = sig.lfilter(b, a, tail_l, zi=[seed_l * decay])
        avg_gains  = np.concatenate([[seed_g], tail_avg_g])
        avg_losses = np.concatenate([[seed_l], tail_avg_l])
    else:
        avg_gains  = np.array([seed_g])
        avg_losses = np.array([seed_l])

    # Pad with NaN for the warm-up window
    full_avg_g = np.full(n, np.nan)
    full_avg_l = np.full(n, np.nan)
    full_avg_g[period:] = avg_gains
    full_avg_l[period:] = avg_losses

    # Vectorized RS + RSI + boundary conditions
    with np.errstate(divide="ignore", invalid="ignore"):
        rs  = np.where(full_avg_l > 0, full_avg_g / full_avg_l, 0.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = np.where((full_avg_g == 0) & (full_avg_l == 0), 50.0, rsi)
    rsi = np.where((full_avg_l == 0) & (full_avg_g > 0),  100.0, rsi)
    rsi = np.where((full_avg_g == 0) & (full_avg_l > 0),    0.0, rsi)
    rsi[:period] = np.nan

    return pd.Series(rsi, index=price_series.index)


def generate_multi_factor_quant_signals(
    price_series: pd.Series,
    volume_series: Optional[pd.Series] = None,
    rsi_period: int = 14,
    rsi_buy_threshold: float = 45.0,
    rsi_sell_threshold: float = 70.0,
    trend_filter_window: int = 200,
    volume_mult: float = 1.1,
) -> pd.Series:
    """
    Multi-Factor Technical Strategy:
    1. Macro Trend Filter: Price MUST be above 200-day SMA.
    2. Momentum Trigger: Wilder's RSI recovering (< 45).
    3. Volume Confirmation: Daily volume > 1.1x 20-day Volume SMA.

    Signal state machine is fully vectorized — no Python-level row iteration.
    """
    # 1. Macro Trend Guard
    sma_200       = price_series.rolling(window=trend_filter_window).mean()
    macro_uptrend = price_series > sma_200

    # 2. Wilder RSI
    rsi = calculate_wilder_rsi(price_series, period=rsi_period)

    # 3. Volume Filter
    if volume_series is not None:
        vol_sma        = volume_series.rolling(window=20).mean()
        volume_confirm = volume_series > (vol_sma * volume_mult)
    else:
        volume_confirm = pd.Series(True, index=price_series.index)

    # Warm-up mask: both RSI and 200-SMA must be valid (post-warmup)
    valid = rsi.notna() & sma_200.notna()

    buy_cond  = valid & (macro_uptrend == True) & (rsi < rsi_buy_threshold) & (volume_confirm == True)
    sell_cond = valid & ((rsi > rsi_sell_threshold) | (macro_uptrend == False))

    # Vectorized state propagation: +1 on buy trigger, 0 on sell trigger, forward-fill holds
    raw = pd.Series(np.nan, index=price_series.index, dtype=float)
    raw[~valid]   = 0.0   # pre-warmup: flat
    raw[buy_cond]  = 1.0
    raw[sell_cond] = 0.0  # sell overrides buy if both trigger simultaneously

    signals = raw.ffill().fillna(0.0)

    # Shift by 1 day to strictly eliminate lookahead bias
    return signals.shift(1).fillna(0.0)


def generate_portfolio_signals(
    prices_df: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """
    Generates vectorized multi-factor signals for all portfolio assets.
    Returns a DataFrame of signals (0.0 or 1.0) indexed by date.
    """
    signals_df = pd.DataFrame(index=prices_df.index)
    for ticker in tickers:
        if ticker in prices_df.columns:
            price_col = prices_df[ticker]
            vol_col   = prices_df.get(f"{ticker}_Volume")
            signals_df[ticker] = generate_multi_factor_quant_signals(
                price_col, volume_series=vol_col
            )
    return signals_df


def apply_fundamental_quality_filter(
    info: Dict[str, Any],
    min_market_cap_cr: float = 1000.0,
    min_roe: float = 0.10,
    max_pe: float = 80.0,
) -> Tuple[bool, str]:
    """
    Fundamental Gatekeeper: Evaluates financial quality of an individual stock.
    """
    mkt_cap_cr = info.get("marketCap", 0) / 10_000_000.0 if info.get("marketCap") else 0.0
    roe  = info.get("returnOnEquity")
    pe   = info.get("trailingPE")
    name = info.get("longName", "Asset")

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
    max_pe: float = 80.0,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Screens the entire portfolio against fundamental quality standards.
    Returns (summary_df, passed_tickers, rejected_tickers).
    """
    rows, passed_tickers, rejected_tickers = [], [], []

    for t in tickers:
        info = fundamental_info_dict.get(t, {})
        passed, reason = apply_fundamental_quality_filter(
            info, min_market_cap_cr=min_market_cap_cr, min_roe=min_roe, max_pe=max_pe
        )
        mkt_cap_cr = info.get("marketCap", 0) / 10_000_000.0 if info.get("marketCap") else 0.0
        roe = info.get("returnOnEquity")
        pe  = info.get("trailingPE")

        rows.append({
            "Ticker":           t,
            "Market Cap (₹ Cr)": f"₹{mkt_cap_cr:,.0f}" if mkt_cap_cr > 0 else "N/A",
            "ROE":              f"{roe:.1%}" if (roe is not None and roe != 0.0) else "N/A",
            "Trailing P/E":     f"{pe:.1f}"  if (pe  is not None and pe  > 0)   else "N/A",
            "Gate Status":      "PASSED" if passed else "REJECTED",
            "Details":          reason,
        })
        (passed_tickers if passed else rejected_tickers).append(t)

    return pd.DataFrame(rows), passed_tickers, rejected_tickers
