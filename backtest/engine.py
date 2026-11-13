# backtest/engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from analytics.risk import (
    calculate_max_drawdown,
    calculate_sortino_ratio,
    _rebalance_dates,
    RISK_FREE_RATE_INDIA,
)

# Indian equity delivery transaction cost breakdown (NSE/Zerodha flat-fee scale):
#   Buy side  : Stamp duty 0.015% + NSE exchange 0.003% + SEBI 0.0001% + brokerage/GST (~0.02%) ≈ 0.04%
#   Sell side : STT 0.10% (sell-only for delivery) + exchange/SEBI/brokerage/DP (~0.04%) ≈ 0.14%
_STT_SELL   = 0.0010               # 0.10% STT — charged on sell turnover only
_BUY_COST   = 0.0004               # ~0.04% buy-side statutory & brokerage cost
_SELL_COST  = _STT_SELL + 0.0004   # ~0.14% sell-side total cost
_ROUND_TRIP = _BUY_COST + _SELL_COST


def _trade_pnls_from_signals(
    price_col: pd.Series,
    sig_col: pd.Series,
) -> List[float]:
    """
    Extracts discrete trade P&L percentages from a price series and binary signal column.
    Uses STT-asymmetric friction: _BUY_COST on entry, _SELL_COST on exit.
    Closes any open position at the final bar.
    """
    pnls: List[float] = []
    in_pos   = False
    entry_px = 0.0

    prices = price_col.values
    sigs   = sig_col.values
    n      = len(sigs)

    for i in range(n):
        s = sigs[i]
        if s == 1.0 and not in_pos:
            in_pos   = True
            raw      = prices[i - 1] if i > 0 else prices[0]
            entry_px = raw * (1.0 + _BUY_COST)
        elif s == 0.0 and in_pos:
            in_pos   = False
            raw      = prices[i - 1] if i > 0 else prices[0]
            exit_px  = raw * (1.0 - _SELL_COST)
            pnls.append((exit_px - entry_px) / entry_px)

    if in_pos:
        exit_px = prices[-1] * (1.0 - _SELL_COST)
        pnls.append((exit_px - entry_px) / entry_px)

    return pnls


def _trade_stats(pnls: np.ndarray) -> Dict[str, float]:
    total  = len(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr     = len(wins) / total if total > 0 else 0.0
    aw     = float(wins.mean())        if len(wins)   > 0 else 0.0
    al     = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    gp, gl = wins.sum(), abs(losses.sum())
    pf     = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {
        "total": total,
        "win_rate": wr,
        "avg_win": aw,
        "avg_loss": al,
        "profit_factor": pf,
        "expectancy": wr * aw - (1.0 - wr) * al,
    }


def _rebalance_buckets_with_friction(
    buckets: np.ndarray,
    weights: np.ndarray,
    is_invested: np.ndarray,
) -> np.ndarray:
    """
    Rebalances capital buckets back to target `weights` while charging
    STT-asymmetric turnover friction on shares actually bought or sold:
      - Trimming an invested asset (delta < 0 and invested) -> _SELL_COST (0.14%)
      - Adding to an invested asset (delta > 0 and invested) -> _BUY_COST (0.04%)
      - Cash transfers incur zero stock brokerage/STT.
    """
    total_val = buckets.sum()
    if total_val <= 0.0:
        return buckets

    targets = weights * total_val
    deltas  = targets - buckets

    sell_turnover = np.where((deltas < 0) & is_invested, -deltas, 0.0)
    buy_turnover  = np.where((deltas > 0) & is_invested,  deltas, 0.0)

    total_fee = float((sell_turnover * _SELL_COST + buy_turnover * _BUY_COST).sum())
    net_val   = max(total_val - total_fee, 0.0)
    return weights * net_val


def run_vectorized_backtest(
    price_series: pd.Series,
    signals: pd.Series,
    initial_capital: float = 500_000.0,
    execution_timing: str = "close",
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Dict[str, Any]:
    """
    Single-asset vectorized backtest with STT-asymmetric friction.
    Delegates to run_portfolio_vectorized_backtest to ensure identical
    cash-yield compounding (rf_daily) and transaction-cost mechanics.
    """
    ticker = str(price_series.name) if hasattr(price_series, "name") and price_series.name else "ASSET"
    prices_df = pd.DataFrame({ticker: price_series})
    signals_df = pd.DataFrame({ticker: signals})
    weights_dict = {ticker: 1.0}

    return run_portfolio_vectorized_backtest(
        prices_df=prices_df,
        signals_df=signals_df,
        weights_dict=weights_dict,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        rebalance_freq="none",
    )


def run_portfolio_vectorized_backtest(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    weights_dict: Dict[str, float],
    initial_capital: float = 500_000.0,
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
    rebalance_freq: Optional[str] = "quarterly",
) -> Dict[str, Any]:
    """
    Multi-asset portfolio backtest using a capital-bucket model with optional
    calendar-boundary rebalancing and STT-asymmetric friction.

    Capital model (eliminates implicit daily rebalancing)
    -----------------------------------------------------
    Each asset i starts with capital w_i * initial_capital.
    Between rebalance dates, buckets evolve independently: invested positions
    earn the asset market return, cash positions earn daily Rf.

    If `rebalance_freq` is specified (e.g. 'quarterly', 'monthly', 'annually'),
    both the active strategy and the Buy & Hold benchmark rebalance buckets back
    to target weights at true NSE calendar boundaries using _rebalance_dates(),
    paying asymmetric STT friction on rebalancing turnover.
    If `rebalance_freq` is 'none' or None, buckets drift freely.
    """
    tickers  = [t for t in weights_dict if t in prices_df.columns]
    weights  = np.array([weights_dict[t] for t in tickers], dtype=np.float64)
    weights  = weights / weights.sum()
    rf_daily = risk_free_rate / 252.0

    common_idx = prices_df.index.intersection(signals_df.index)
    prices = prices_df[tickers].loc[common_idx]
    sigs   = signals_df[tickers].loc[common_idx].fillna(0.0)
    n      = len(common_idx)

    use_rebalance = rebalance_freq is not None and str(rebalance_freq).lower() not in ("none", "")
    reb_dates = _rebalance_dates(common_idx, str(rebalance_freq)) if use_rebalance else set()

    # --- Capital buckets: Active Strategy & Buy & Hold ---
    bucket      = weights * initial_capital
    bh_bucket   = weights * initial_capital
    port_values = np.empty(n, dtype=np.float64)
    bh_values   = np.empty(n, dtype=np.float64)

    daily_ret = prices.pct_change(fill_method=None).fillna(0.0).values
    sigs_arr  = sigs.values

    # Initial day 0 setup
    if (sigs_arr[0] == 1.0).any():
        bucket = np.where(sigs_arr[0] == 1.0, bucket * (1.0 - _BUY_COST), bucket)
    port_values[0] = bucket.sum()
    bh_values[0]   = bh_bucket.sum()

    all_invested = np.ones(len(tickers), dtype=bool)

    for day in range(1, n):
        prev_sig = sigs_arr[day - 1]
        curr_sig = sigs_arr[day]
        r        = daily_ret[day]

        # 1. Signal transition asymmetric friction (at market open/entry/exit)
        buy_mask  = (curr_sig > prev_sig)
        sell_mask = (curr_sig < prev_sig)

        bucket = np.where(buy_mask,  bucket * (1.0 - _BUY_COST),  bucket)
        bucket = np.where(sell_mask, bucket * (1.0 - _SELL_COST), bucket)

        # 2. Daily asset return or cash Rf yield
        bucket    = bucket * np.where(curr_sig == 1.0, 1.0 + r, 1.0 + rf_daily)
        bh_bucket = bh_bucket * (1.0 + r)

        # 3. Record end-of-day mark-to-market value
        port_values[day] = bucket.sum()
        bh_values[day]   = bh_bucket.sum()

        # 4. Periodic calendar-boundary rebalancing (after close, for next day)
        if use_rebalance and common_idx[day] in reb_dates and day < n - 1:
            is_inv    = (curr_sig == 1.0)
            bucket    = _rebalance_buckets_with_friction(bucket, weights, is_inv)
            bh_bucket = _rebalance_buckets_with_friction(bh_bucket, weights, all_invested)

    # Mark-to-market liquidation friction on terminal bar:
    # Deducts _SELL_COST from open invested holdings so terminal equity reflects
    # net realizable cash value, matching the closed trade ledger P&L
    if n > 0:
        final_sigs = sigs_arr[-1]
        active_liquidation_fees = float((bucket * np.where(final_sigs == 1.0, _SELL_COST, 0.0)).sum())
        bh_liquidation_fees     = float((bh_bucket * _SELL_COST).sum())
        port_values[-1] -= active_liquidation_fees
        bh_values[-1]   -= bh_liquidation_fees

    equity_curve = pd.Series(port_values, index=common_idx)
    bh_curve     = pd.Series(bh_values, index=common_idx)

    # --- Performance metrics ---
    years      = max(n / 252, 0.1)
    final_val  = float(equity_curve.iloc[-1])
    bh_final   = float(bh_curve.iloc[-1])
    strat_cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0
    bh_cagr    = (bh_final  / initial_capital) ** (1.0 / years) - 1.0

    port_dr   = equity_curve.pct_change(fill_method=None).fillna(0.0)
    bh_dr     = bh_curve.pct_change(fill_method=None).fillna(0.0)
    strat_vol = float(port_dr.std() * np.sqrt(252))
    bh_vol    = float(bh_dr.std() * np.sqrt(252))
    sharpe    = (strat_cagr - risk_free_rate) / strat_vol if strat_vol > 0 else 0.0
    sortino   = calculate_sortino_ratio(np.log(1 + port_dr.clip(lower=-0.9999)), strat_cagr, risk_free_rate)
    mdd       = calculate_max_drawdown(equity_curve)
    bh_mdd    = calculate_max_drawdown(bh_curve)
    calmar    = strat_cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    # --- Per-asset trade ledger ---
    asset_rows: list = []
    all_pnls:   list = []

    for t, w in zip(tickers, weights):
        raw_pnls = _trade_pnls_from_signals(prices[t], sigs[t])
        arr      = np.array(raw_pnls)
        all_pnls.extend(raw_pnls)
        s = _trade_stats(arr) if len(arr) > 0 else {
            "total": 0, "win_rate": 0, "avg_win": 0,
            "avg_loss": 0, "profit_factor": 0, "expectancy": 0
        }
        wins   = arr[arr > 0] if len(arr) else np.array([])
        losses = arr[arr < 0] if len(arr) else np.array([])
        asset_rows.append({
            "Ticker":        t,
            "Weight":        f"{w:.0%}",
            "Trades":        s["total"],
            "Win Rate":      f"{s['win_rate']:.1%}"      if s["total"] > 0 else "N/A",
            "Profit Factor": f"{s['profit_factor']:.2f}" if s["total"] > 0 else "N/A",
            "Avg Win":       f"{wins.mean():.2%}"        if len(wins)   > 0 else "N/A",
            "Avg Loss":      f"{abs(losses.mean()):.2%}" if len(losses) > 0 else "N/A",
        })

    agg = _trade_stats(np.array(all_pnls)) if all_pnls else {
        "total": 0, "win_rate": 0, "avg_win": 0,
        "avg_loss": 0, "profit_factor": 0, "expectancy": 0
    }

    return {
        "Strategy Final Value":    final_val,
        "Buy & Hold Final Value":  bh_final,
        "Strategy CAGR":           strat_cagr,
        "Buy & Hold CAGR":         bh_cagr,
        "Strategy Volatility":     strat_vol,
        "Buy & Hold Volatility":   bh_vol,
        "Sharpe Ratio":            sharpe,
        "Sortino Ratio":           sortino,
        "Calmar Ratio":            calmar,
        "Max Drawdown":            mdd,
        "Buy & Hold Max Drawdown": bh_mdd,
        "Total Trades":            agg["total"],
        "Win Rate":                agg["win_rate"],
        "Profit Factor":           agg["profit_factor"],
        "Expectancy Per Trade":    agg["expectancy"],
        "Avg Win":                 agg["avg_win"],
        "Avg Loss":                agg["avg_loss"],
        "Equity Curve":            equity_curve,
        "Buy & Hold Curve":        bh_curve,
        "Asset Breakdown":         pd.DataFrame(asset_rows),
    }
