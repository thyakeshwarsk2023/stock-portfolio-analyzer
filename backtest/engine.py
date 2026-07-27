# backtest/engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from analytics.risk import calculate_max_drawdown, calculate_sortino_ratio, RISK_FREE_RATE_INDIA

def run_vectorized_backtest(
    price_series: pd.Series, 
    signals: pd.Series, 
    initial_capital: float = 500000.0,
    cost_per_trade: float = 0.0010  # 0.10% covers Indian STT, exchange charges & slippage
) -> Dict[str, Any]:
    """
    Executes a vectorized backtest for a single asset trading strategy.
    Calculates institutional risk, performance, and trade distribution metrics.
    """
    # 1. Daily Asset Simple Returns
    daily_returns = price_series.pct_change().fillna(0.0)
    
    # 2. Identify Trade Entries and Exits (Transitions)
    # diff() != 0 marks every buy (+1) or sell (-1) trade action
    trade_actions = signals.diff().fillna(0.0)
    is_trade_day = trade_actions != 0.0
    
    # 3. Apply Transaction Friction
    # Net Daily Strategy Return = (Position * Asset Return) - (Trade Action * Friction)
    friction = is_trade_day.astype(float) * cost_per_trade
    strategy_daily_returns = (signals * daily_returns) - friction
    
    # 4. Wealth Curves
    equity_curve = initial_capital * (1.0 + strategy_daily_returns).cumprod()
    buy_hold_curve = initial_capital * (1.0 + daily_returns).cumprod()
    
    # 5. Extract Individual Trade Distributions
    # Reconstruct trade trade-by-trade PnLs
    trade_pnls = []
    in_position = False
    entry_price = 0.0
    
    for i in range(len(signals)):
        sig = signals.iloc[i]
        price = price_series.iloc[i]
        
        if sig == 1.0 and not in_position:
            in_position = True
            entry_price = price * (1.0 + cost_per_trade)  # Add slippage on buy
        elif sig == 0.0 and in_position:
            in_position = False
            exit_price = price * (1.0 - cost_per_trade)   # Add slippage on sell
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
        "Avg Loss": avg_loss
    }