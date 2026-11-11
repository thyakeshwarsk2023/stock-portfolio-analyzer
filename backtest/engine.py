# backtest/engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from analytics.risk import calculate_max_drawdown, calculate_sortino_ratio, RISK_FREE_RATE_INDIA

# Indian equity delivery transaction cost breakdown (Zerodha flat-fee scale):
#   Buy side  : brokerage 0.03% + NSE exchange fee 0.00335% + GST 18% on brokerage
#   Sell side : above + STT 0.1% (Securities Transaction Tax — sell-only for delivery)
# Conservative round-trip approximation used below:
_STT_SELL  = 0.0010   # 0.10% STT — charged on sell value only
_BUY_COST  = 0.0004   # ~0.04% brokerage + exchange fees on buy side
_SELL_COST = _STT_SELL + 0.0004   # ~0.14% total on sell side
_ROUND_TRIP = _BUY_COST + _SELL_COST   # ~0.18% realistic round-trip


def _trade_pnls_from_signals(
    price_col: pd.Series,
    sig_col: pd.Series,
) -> List[float]:
    """
    Extracts discrete trade P&L percentages from a price series and binary signal column.
    Uses STT-asymmetric friction: buy_cost on entry, sell_cost on exit.
    Closes any open position at the final bar.
    """
    pnls: List[float] = []
    in_pos    = False
    entry_px  = 0.0

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
    aw     = float(wins.mean())   if len(wins)   > 0 else 0.0
    al     = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    gp, gl = wins.sum(), abs(losses.sum())
    pf     = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {"total": total, "win_rate": wr, "avg_win": aw, "avg_loss": al, "profit_factor": pf,
            "expectancy": wr * aw - (1 - wr) * al}


def run_vectorized_backtest(
    price_series: pd.Series,
    signals: pd.Series,
    initial_capital: float = 500_000.0,
    execution_timing: str = "close",
) -> Dict[str, Any]:
    """
    Single-asset vectorized backtest with STT-asymmetric friction.
    """
    daily_returns = price_series.pct_change(fill_method=None).fillna(0.0)

    # Friction: buy_cost on entry bar, sell_cost on exit bar
    sig_diff   = signals.diff().fillna(0.0)
    buy_events  = (sig_diff > 0).astype(float) * _BUY_COST
    sell_events = (sig_diff < 0).astype(float) * _SELL_COST

    strategy_daily_returns = (signals * daily_returns) - buy_events - sell_events
    equity_curve   = initial_capital * (1.0 + strategy_daily_returns).cumprod()
    buy_hold_curve = initial_capital * (1.0 + daily_returns).cumprod()

    pnls    = np.array(_trade_pnls_from_signals(price_series, signals))
    stats   = _trade_stats(pnls)

    years       = max(len(price_series) / 252, 0.1)
    final_val   = float(equity_curve.iloc[-1])
    bh_final    = float(buy_hold_curve.iloc[-1])
    strat_cagr  = (final_val / initial_capital) ** (1 / years) - 1.0
    bh_cagr     = (bh_final  / initial_capital) ** (1 / years) - 1.0
    strat_vol   = float(strategy_daily_returns.std() * np.sqrt(252))
    sharpe      = (strat_cagr - RISK_FREE_RATE_INDIA) / strat_vol if strat_vol > 0 else 0.0
    sortino     = calculate_sortino_ratio(np.log(1 + strategy_daily_returns), strat_cagr)
    mdd         = calculate_max_drawdown(equity_curve)
    calmar      = strat_cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    return {
        "Strategy Final Value":   final_val,
        "Buy & Hold Final Value": bh_final,
        "Strategy CAGR":          strat_cagr,
        "Buy & Hold CAGR":        bh_cagr,
        "Strategy Volatility":    strat_vol,
        "Sharpe Ratio":           sharpe,
        "Sortino Ratio":          sortino,
        "Calmar Ratio":           calmar,
        "Max Drawdown":           mdd,
        "Total Trades":           stats["total"],
        "Win Rate":               stats["win_rate"],
        "Profit Factor":          stats["profit_factor"],
        "Expectancy Per Trade":   stats["expectancy"],
        "Avg Win":                stats["avg_win"],
        "Avg Loss":               stats["avg_loss"],
        "Equity Curve":           equity_curve,
        "Buy & Hold Curve":       buy_hold_curve,
    }


def run_portfolio_vectorized_backtest(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    weights_dict: Dict[str, float],
    initial_capital: float = 500_000.0,
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Dict[str, Any]:
    """
    Multi-asset portfolio backtest using a capital-bucket model.

    Capital model (fixes implicit daily rebalancing)
    ------------------------------------------------
    Each asset i is allocated a fixed capital bucket of w_i * initial_capital.
    Within its bucket, the asset grows by its daily market return when
    signal=1 (invested), or by the daily Rf rate when signal=0 (cash).
    Buckets never cross-rebalance — their sizes drift independently.
    Total portfolio equity = sum of all bucket values.

    This eliminates the daily-rebalancing artifact of the previous
    return-based model (which applied fixed weights every day, implying
    a continuous rebalance not present in real execution).

    Friction model: STT-asymmetric
    --------------------------------
    - Entry  : _BUY_COST  applied to the bucket value on signal 0->1 transition.
    - Exit   : _SELL_COST applied to the bucket value on signal 1->0 transition.
    """
    tickers = [t for t in weights_dict if t in prices_df.columns]
    weights = np.array([weights_dict[t] for t in tickers])
    rf_daily = risk_free_rate / 252.0

    common_idx = prices_df.index.intersection(signals_df.index)
    prices = prices_df[tickers].loc[common_idx]
    sigs   = signals_df[tickers].loc[common_idx].fillna(0.0)

    n      = len(common_idx)
    n_a    = len(tickers)

    # --- Capital buckets ---
    # bucket[i] = INR value of asset i's allocation
    bucket      = weights * initial_capital      # (A,) initial allocation
    port_values = np.empty(n, dtype=np.float64)

    daily_ret = prices.pct_change(fill_method=None).fillna(0.0).values   # (T, A)
    sigs_arr  = sigs.values                                               # (T, A)

    for day in range(n):
        if day > 0:
            prev_sig = sigs_arr[day - 1]
            curr_sig = sigs_arr[day]
            r        = daily_ret[day]              # (A,) today's market return

            # Transition frictions applied to bucket value at start of day
            buy_mask  = (curr_sig > prev_sig)      # 0->1
            sell_mask = (curr_sig < prev_sig)      # 1->0

            # Deduct friction from bucket before growth
            bucket = np.where(buy_mask,  bucket * (1.0 - _BUY_COST),  bucket)
            bucket = np.where(sell_mask, bucket * (1.0 - _SELL_COST), bucket)

            # Grow each bucket: invested earns market return, cash earns Rf
            bucket = bucket * np.where(curr_sig == 1.0, 1.0 + r, 1.0 + rf_daily)

        port_values[day] = bucket.sum()

    equity_curve = pd.Series(port_values, index=common_idx)

    # Buy-and-hold benchmark: same weights, no signals, no rebalancing
    bh_bucket      = weights * initial_capital
    bh_values      = np.empty(n, dtype=np.float64)
    bh_values[0]   = bh_bucket.sum()
    for day in range(1, n):
        bh_bucket    *= (1.0 + daily_ret[day])
        bh_values[day] = bh_bucket.sum()
    bh_curve = pd.Series(bh_values, index=common_idx)

    # --- Performance metrics ---
    years       = max(n / 252, 0.1)
    final_val   = float(equity_curve.iloc[-1])
    bh_final    = float(bh_curve.iloc[-1])
    strat_cagr  = (final_val / initial_capital) ** (1.0 / years) - 1.0
    bh_cagr     = (bh_final  / initial_capital) ** (1.0 / years) - 1.0

    port_dr    = equity_curve.pct_change(fill_method=None).fillna(0.0)
    bh_dr      = bh_curve.pct_change(fill_method=None).fillna(0.0)
    strat_vol  = float(port_dr.std() * np.sqrt(252))
    bh_vol     = float(bh_dr.std() * np.sqrt(252))
    sharpe     = (strat_cagr - risk_free_rate) / strat_vol if strat_vol > 0 else 0.0
    sortino    = calculate_sortino_ratio(np.log(1 + port_dr.clip(lower=-0.9999)), strat_cagr, risk_free_rate)
    mdd        = calculate_max_drawdown(equity_curve)
    bh_mdd     = calculate_max_drawdown(bh_curve)
    calmar     = strat_cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    # --- Per-asset trade ledger ---
    asset_rows:   list = []
    all_pnls:     list = []

    for t, w in zip(tickers, weights):
        raw_pnls = _trade_pnls_from_signals(prices[t], sigs[t])
        arr      = np.array(raw_pnls)
        all_pnls.extend(raw_pnls)
        s = _trade_stats(arr) if len(arr) > 0 else {"total": 0, "win_rate": 0, "avg_win": 0,
                                                     "avg_loss": 0, "profit_factor": 0, "expectancy": 0}
        wins   = arr[arr > 0] if len(arr) else np.array([])
        losses = arr[arr < 0] if len(arr) else np.array([])
        asset_rows.append({
            "Ticker":        t,
            "Weight":        f"{w:.0%}",
            "Trades":        s["total"],
            "Win Rate":      f"{s['win_rate']:.1%}" if s["total"] > 0 else "N/A",
            "Profit Factor": f"{s['profit_factor']:.2f}" if s["total"] > 0 else "N/A",
            "Avg Win":       f"{wins.mean():.2%}"        if len(wins)   > 0 else "N/A",
            "Avg Loss":      f"{abs(losses.mean()):.2%}" if len(losses) > 0 else "N/A",
        })

    # Aggregate stats
    agg = _trade_stats(np.array(all_pnls)) if all_pnls else \
          {"total": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "expectancy": 0}

    return {
        "Strategy Final Value":   final_val,
        "Buy & Hold Final Value": bh_final,
        "Strategy CAGR":          strat_cagr,
        "Buy & Hold CAGR":        bh_cagr,
        "Strategy Volatility":    strat_vol,
        "Buy & Hold Volatility":  bh_vol,
        "Sharpe Ratio":           sharpe,
        "Sortino Ratio":          sortino,
        "Calmar Ratio":           calmar,
        "Max Drawdown":           mdd,
        "Buy & Hold Max Drawdown": bh_mdd,
        "Total Trades":           agg["total"],
        "Win Rate":               agg["win_rate"],
        "Profit Factor":          agg["profit_factor"],
        "Expectancy Per Trade":   agg["expectancy"],
        "Avg Win":                agg["avg_win"],
        "Avg Loss":               agg["avg_loss"],
        "Equity Curve":           equity_curve,
        "Buy & Hold Curve":       bh_curve,
        "Asset Breakdown":        pd.DataFrame(asset_rows),
    }
