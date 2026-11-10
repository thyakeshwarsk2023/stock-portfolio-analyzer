# backtest/engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from analytics.risk import calculate_max_drawdown, calculate_sortino_ratio, RISK_FREE_RATE_INDIA


def run_vectorized_backtest(
    price_series: pd.Series,
    signals: pd.Series,
    initial_capital: float = 500000.0,
    cost_per_trade: float = 0.0010,  # 0.10% covers Indian STT, exchange charges & slippage
    execution_timing: str = "close",
) -> Dict[str, Any]:
    """
    Executes a vectorized backtest for a single asset trading strategy.
    Aligns vectorized daily returns and discrete trade ledger execution timing
    (signals generated at bar t-1 close execute at bar t-1 close / bar t open).
    Calculates institutional risk, performance, and trade distribution metrics.
    """
    # 1. Daily Asset Simple Returns
    daily_returns = price_series.pct_change(fill_method=None).fillna(0.0)

    # 2. Identify Trade Entries and Exits (Transitions)
    trade_actions = signals.diff().fillna(0.0)
    is_trade_day = trade_actions != 0.0

    # 3. Apply Transaction Friction
    friction = is_trade_day.astype(float) * cost_per_trade
    strategy_daily_returns = (signals * daily_returns) - friction

    # 4. Wealth Curves
    equity_curve = initial_capital * (1.0 + strategy_daily_returns).cumprod()
    buy_hold_curve = initial_capital * (1.0 + daily_returns).cumprod()

    # 5. Extract Individual Trade Distributions
    trade_pnls = []
    in_position = False
    entry_price = 0.0

    for i in range(len(signals)):
        sig = signals.iloc[i]

        # Buy Entry Transition
        if sig == 1.0 and not in_position:
            in_position = True
            raw_entry = price_series.iloc[i - 1] if i > 0 else price_series.iloc[0]
            entry_price = raw_entry * (1.0 + cost_per_trade)

        # Sell Exit Transition
        elif sig == 0.0 and in_position:
            in_position = False
            raw_exit = price_series.iloc[i - 1] if i > 0 else price_series.iloc[0]
            exit_price = raw_exit * (1.0 - cost_per_trade)
            trade_pnl_pct = (exit_price - entry_price) / entry_price
            trade_pnls.append(trade_pnl_pct)

    # Close any open trade at the end of the backtest period using the final bar's closing price
    if in_position and len(price_series) > 0:
        in_position = False
        exit_price = price_series.iloc[-1] * (1.0 - cost_per_trade)
        trade_pnl_pct = (exit_price - entry_price) / entry_price
        trade_pnls.append(trade_pnl_pct)

    trade_pnls = np.array(trade_pnls)

    # 6. Trade Distribution Stats
    total_trades = len(trade_pnls)
    winning_trades = trade_pnls[trade_pnls > 0]
    losing_trades = trade_pnls[trade_pnls < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

    gross_profits = winning_trades.sum()
    gross_losses = abs(losing_trades.sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (99.0 if gross_profits > 0 else 0.0)

    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

    # 7. Portfolio Horizon Metrics
    years = max(len(price_series) / 252, 0.1)
    final_val = equity_curve.iloc[-1]
    bh_final_val = buy_hold_curve.iloc[-1]

    strat_cagr = (final_val / initial_capital) ** (1 / years) - 1.0
    bh_cagr = (bh_final_val / initial_capital) ** (1 / years) - 1.0

    strat_vol = strategy_daily_returns.std() * np.sqrt(252)
    sharpe = (strat_cagr - RISK_FREE_RATE_INDIA) / strat_vol if strat_vol > 0 else 0.0

    log_returns = np.log(1.0 + strategy_daily_returns)
    sortino = calculate_sortino_ratio(log_returns, strat_cagr)
    mdd = calculate_max_drawdown(equity_curve)
    calmar = strat_cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    return {
        "Strategy Final Value": final_val,
        "Buy & Hold Final Value": bh_final_val,
        "Strategy CAGR": strat_cagr,
        "Buy & Hold CAGR": bh_cagr,
        "Strategy Volatility": strat_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": mdd,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Expectancy Per Trade": expectancy,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Equity Curve": equity_curve,
        "Buy & Hold Curve": buy_hold_curve,
    }


def run_portfolio_vectorized_backtest(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    weights_dict: Dict[str, float],
    initial_capital: float = 500000.0,
    cost_per_trade: float = 0.0010,
    risk_free_rate: float = RISK_FREE_RATE_INDIA,
) -> Dict[str, Any]:
    """
    Multi-asset portfolio backtest with dynamic cash de-risking.

    When a signal for asset i is 0, the capital allocated to that asset
    earns the daily risk-free rate instead of market returns.  When the
    signal is 1, it earns the asset's simple daily return.

    Portfolio daily return at day t:
        R_p(t) = sum_i  w_i * [ sig_i(t) * r_i(t) + (1 - sig_i(t)) * Rf_daily ]
                 - cost * |sig_i(t) - sig_i(t-1)|   (transaction friction)

    Parameters
    ----------
    prices_df       : DataFrame of adjusted close prices (date index, ticker cols).
    signals_df      : DataFrame of 0/1 signals aligned with prices_df index.
    weights_dict    : {ticker: target_weight} allocation.
    initial_capital : Starting portfolio value in INR.
    cost_per_trade  : Round-trip friction per trade event (default 0.10%).
    risk_free_rate  : Annual Rf rate; converted to daily for cash return.

    Returns
    -------
    Dict containing portfolio equity curve, buy-and-hold curve, all risk
    metrics, per-asset trade breakdown, and aggregate trade statistics.
    """
    tickers = [t for t in weights_dict if t in prices_df.columns]
    weights = np.array([weights_dict[t] for t in tickers])
    rf_daily = risk_free_rate / 252.0

    # Align signals and prices on common index
    common_idx = prices_df.index.intersection(signals_df.index)
    prices = prices_df[tickers].loc[common_idx]
    sigs = signals_df[tickers].loc[common_idx].fillna(0.0)

    daily_asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    # Transaction friction: triggered whenever signal changes state per asset
    sig_changes = sigs.diff().fillna(0.0).abs()
    friction_df = sig_changes * cost_per_trade

    # Strategy return per asset per day: signal * market_return + (1-signal) * Rf
    strat_asset_returns = (
        sigs * daily_asset_returns
        + (1.0 - sigs) * rf_daily
        - friction_df
    )

    # Weighted portfolio daily return
    port_daily_returns = (strat_asset_returns * weights).sum(axis=1)

    # Buy-and-hold (fully invested, no signals): just the weighted market return
    bh_daily_returns = (daily_asset_returns * weights).sum(axis=1)

    # Wealth curves
    equity_curve = initial_capital * (1.0 + port_daily_returns).cumprod()
    bh_curve = initial_capital * (1.0 + bh_daily_returns).cumprod()

    # ---- Portfolio-level performance metrics ----
    years = max(len(common_idx) / 252, 0.1)
    final_val = float(equity_curve.iloc[-1])
    bh_final_val = float(bh_curve.iloc[-1])

    strat_cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0
    bh_cagr = (bh_final_val / initial_capital) ** (1.0 / years) - 1.0

    strat_vol = float(port_daily_returns.std() * np.sqrt(252))
    bh_vol = float(bh_daily_returns.std() * np.sqrt(252))
    sharpe = (strat_cagr - risk_free_rate) / strat_vol if strat_vol > 0 else 0.0

    log_returns = np.log(1.0 + port_daily_returns.clip(lower=-0.9999))
    sortino = calculate_sortino_ratio(log_returns, strat_cagr, risk_free_rate)
    mdd = calculate_max_drawdown(equity_curve)
    bh_mdd = calculate_max_drawdown(bh_curve)
    calmar = strat_cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    # ---- Per-asset trade breakdown ----
    asset_rows = []
    all_trade_pnls: list = []

    for t, w in zip(tickers, weights):
        price_col = prices[t]
        sig_col = sigs[t]
        trade_pnls: list = []
        in_pos = False
        entry_px = 0.0

        for i in range(len(sig_col)):
            s = sig_col.iloc[i]
            if s == 1.0 and not in_pos:
                in_pos = True
                entry_px = (price_col.iloc[i - 1] if i > 0 else price_col.iloc[0]) * (1 + cost_per_trade)
            elif s == 0.0 and in_pos:
                in_pos = False
                exit_px = (price_col.iloc[i - 1] if i > 0 else price_col.iloc[0]) * (1 - cost_per_trade)
                trade_pnls.append((exit_px - entry_px) / entry_px)

        # Close open position at final bar
        if in_pos:
            exit_px = price_col.iloc[-1] * (1 - cost_per_trade)
            trade_pnls.append((exit_px - entry_px) / entry_px)

        arr = np.array(trade_pnls)
        all_trade_pnls.extend(trade_pnls)

        wins = arr[arr > 0]
        losses = arr[arr < 0]
        gp = wins.sum()
        gl = abs(losses.sum())

        asset_rows.append({
            "Ticker": t,
            "Weight": f"{w:.0%}",
            "Trades": len(arr),
            "Win Rate": f"{len(wins) / len(arr):.1%}" if len(arr) > 0 else "N/A",
            "Profit Factor": f"{gp / gl:.2f}" if gl > 0 else ("N/A" if gp == 0 else "inf"),
            "Avg Win": f"{wins.mean():.2%}" if len(wins) > 0 else "N/A",
            "Avg Loss": f"{abs(losses.mean()):.2%}" if len(losses) > 0 else "N/A",
        })

    asset_breakdown_df = pd.DataFrame(asset_rows)

    # Aggregate trade statistics
    all_arr = np.array(all_trade_pnls)
    total_trades = len(all_arr)
    agg_wins = all_arr[all_arr > 0]
    agg_losses = all_arr[all_arr < 0]
    agg_win_rate = len(agg_wins) / total_trades if total_trades > 0 else 0.0
    agg_avg_win = float(agg_wins.mean()) if len(agg_wins) > 0 else 0.0
    agg_avg_loss = float(abs(agg_losses.mean())) if len(agg_losses) > 0 else 0.0
    agg_gp = agg_wins.sum()
    agg_gl = abs(agg_losses.sum())
    agg_pf = agg_gp / agg_gl if agg_gl > 0 else (99.0 if agg_gp > 0 else 0.0)
    agg_expectancy = (agg_win_rate * agg_avg_win) - ((1 - agg_win_rate) * agg_avg_loss)

    return {
        "Strategy Final Value": final_val,
        "Buy & Hold Final Value": bh_final_val,
        "Strategy CAGR": strat_cagr,
        "Buy & Hold CAGR": bh_cagr,
        "Strategy Volatility": strat_vol,
        "Buy & Hold Volatility": bh_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": mdd,
        "Buy & Hold Max Drawdown": bh_mdd,
        "Total Trades": total_trades,
        "Win Rate": agg_win_rate,
        "Profit Factor": agg_pf,
        "Expectancy Per Trade": agg_expectancy,
        "Avg Win": agg_avg_win,
        "Avg Loss": agg_avg_loss,
        "Equity Curve": equity_curve,
        "Buy & Hold Curve": bh_curve,
        "Asset Breakdown": asset_breakdown_df,
    }
