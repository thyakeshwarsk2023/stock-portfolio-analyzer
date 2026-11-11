# main.py
import sys
import argparse

# UTF-8 encoding safeguard for Windows console environments
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

from data.fetch import fetch_nse_data, fetch_fundamental_info, fetch_indian_risk_free_rate
from analytics.returns import calculate_nse_metrics
from portfolio.checks import run_indian_portfolio_checks
from simulation.engine import run_monte_carlo_nse
from analytics.risk import calculate_portfolio_series
from analytics.benchmark import evaluate_benchmark_comparison
from backtest.strategy import (
    generate_portfolio_signals, 
    screen_portfolio_fundamentals
)
from backtest.engine import run_portfolio_vectorized_backtest
from analytics.visualization import plot_portfolio_dashboard

DEFAULT_PORTFOLIO = {
    "BEL": 0.25,
    "RELIANCE": 0.20,
    "HDFCBANK": 0.20,
    "TCS": 0.15,
    "HAL": 0.20
}

def format_inr(val: float) -> str:
    """Formats numbers to Indian currency (Lakhs / Crores)."""
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakh"
    else:
        return f"₹{val:,.2f}"

def parse_portfolio_string(raw_str: str) -> Dict[str, float]:
    """Parses 'TICKER:WEIGHT,TICKER:WEIGHT' into a dictionary."""
    config = {}
    try:
        entries = raw_str.split(",")
        for entry in entries:
            ticker, weight = entry.split(":")
            config[ticker.strip().upper()] = float(weight.strip())
        return config
    except Exception as e:
        print(f"[WARN] Format error in portfolio entry: '{e}'. Using default test portfolio.")
        return DEFAULT_PORTFOLIO.copy()

def get_user_inputs_interactive() -> Tuple[Dict[str, float], float, int, int, str, Optional[float]]:
    """Interactively collects portfolio settings, tickers, and rebalancing preferences from the user."""
    print("--------------------------------------------------")
    print("         CUSTOM INVESTMENT CONFIGURATION          ")
    print("--------------------------------------------------")
    
    # 1. Initial Investment
    inv_input = input("1. Enter starting capital in ₹ (Press Enter for default ₹5,00,000): ").strip()
    initial_investment = float(inv_input) if inv_input else 500000.0
    
    # 2. Years
    years_input = input("2. Enter investment horizon in years (Press Enter for default 10): ").strip()
    years = int(years_input) if years_input else 10
    
    # 3. Simulations
    sim_input = input("3. Enter number of simulations (Press Enter for default 10,000): ").strip()
    simulations = int(sim_input) if sim_input else 10000
    
    # 4. Custom Tickers & Weights
    print("\n4. Enter stocks and allocations as 'TICKER:WEIGHT' separated by commas.")
    print("   Example: BEL:0.25, RELIANCE:0.25, HDFCBANK:0.50")
    print("   (Or press Enter to run the default test portfolio: BEL, RELIANCE, HDFCBANK, TCS, HAL)")
    
    raw_portfolio = input("\nYour Portfolio Allocation: ").strip()
    portfolio_config = parse_portfolio_string(raw_portfolio) if raw_portfolio else DEFAULT_PORTFOLIO.copy()

    # 5. Rebalancing Frequency
    print("\n5. Select Portfolio Rebalancing Strategy:")
    print("   [1] None (Static Buy & Hold - weights drift)")
    print("   [2] Quarterly (Rebalance every 3 months)")
    print("   [3] Monthly (Rebalance at beginning of each month)")
    print("   [4] Annually (Rebalance once per year)")
    reb_input = input("   Enter choice 1-4 (Press Enter for default [2] Quarterly): ").strip()
    reb_map = {"1": "none", "2": "quarterly", "3": "monthly", "4": "annually"}
    rebalance_freq = reb_map.get(reb_input, "quarterly")

    # 6. Risk-Free Rate
    rf_input = input("\n6. Enter benchmark Risk-Free Rate % (Press Enter for RBI 10Y G-Sec: 6.50%): ").strip()
    custom_rf = float(rf_input) / 100.0 if rf_input else None

    return portfolio_config, initial_investment, years, simulations, rebalance_freq, custom_rf

def build_cli_parser() -> argparse.ArgumentParser:
    """Builds robust CLI argument parser for non-interactive scripting and automation."""
    parser = argparse.ArgumentParser(
        description="Indian Stock Portfolio Quantitative Analyzer & Monte Carlo Simulator (NSE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--capital", type=float, default=500000.0, help="Starting capital in INR (₹)")
    parser.add_argument("--years", type=int, default=10, help="Investment horizon in years")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of Monte Carlo simulations")
    parser.add_argument("--tickers", type=str, default="", help="Stock allocations format 'TICKER:WEIGHT,TICKER:WEIGHT'")
    parser.add_argument("--rebalance", type=str, default="quarterly", choices=["none", "quarterly", "monthly", "annually", "daily"], help="Periodic rebalancing frequency")
    parser.add_argument("--rf", type=float, default=None, help="Custom annual Risk-Free Rate (e.g. 0.065 for 6.5%%)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating and saving chart figures")
    parser.add_argument("--plot-file", type=str, default="portfolio_report.png", help="Path to save high-res PNG visual report")
    parser.add_argument("--html-file", type=str, default="portfolio_dashboard.html", help="Path to save interactive HTML dashboard")
    parser.add_argument("--interactive", action="store_true", help="Force interactive terminal configuration prompts")
    return parser

def main():
    print("==================================================")
    print("   INDIAN PORTFOLIO MONTE CARLO SIMULATOR (NSE)   ")
    print("==================================================\n")
    
    parser = build_cli_parser()
    args = parser.parse_args()
    
    is_interactive = args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty())
    
    if is_interactive:
        portfolio_config, initial_investment, years, simulations, rebalance_freq, custom_rf = get_user_inputs_interactive()
    else:
        initial_investment = args.capital
        years = args.years
        simulations = args.simulations
        rebalance_freq = args.rebalance
        custom_rf = args.rf
        portfolio_config = parse_portfolio_string(args.tickers) if args.tickers else DEFAULT_PORTFOLIO.copy()

    # Determine Risk-Free Rate
    rf = fetch_indian_risk_free_rate(custom_rf)
    
    # 1. Fetch Market Data
    tickers = list(portfolio_config.keys())
    prices = fetch_nse_data(tickers, period="5y")
    
    # 2. Portfolio Health Diagnostics
    is_valid, warnings = run_indian_portfolio_checks(prices, portfolio_config)
    
    print("\n--- PORTFOLIO HEALTH & RISK DIAGNOSTICS ---")
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
    else:
        print("[OK] Portfolio allocations balanced within risk bounds.")
        
    if not is_valid:
        print("\n[ERROR] Stopping execution due to critical weight validation errors.")
        return

    # 3. Individual Stock Historical Statistics
    ann_returns, ann_vol, cov_matrix = calculate_nse_metrics(prices)
    
    print("\n--- HISTORICAL ANNUALIZED STATS (NSE DATA) ---")
    stats_df = pd.DataFrame({
        "Annual Return": ann_returns[tickers].map("{:.2%}".format),
        "Annual Volatility": ann_vol[tickers].map("{:.2%}".format)
    })
    print(stats_df)
    
    # Portfolio series construction with user-selected rebalancing strategy
    portfolio_series = calculate_portfolio_series(prices, portfolio_config, rebalance_freq=rebalance_freq)

    # 4. Symmetrical Benchmark Comparison
    reb_label = "Static Buy & Hold" if rebalance_freq in (None, "none") else f"Periodic Rebalanced ({rebalance_freq.capitalize()})"
    print(f"\n================ HISTORICAL RISK & BENCHMARK (NSE - {reb_label}) ================")
    benchmark_df, nifty_norm = evaluate_benchmark_comparison(
        portfolio_prices=portfolio_series,
        period="5y",
        risk_free_rate=rf
    )
    print(benchmark_df.to_string(index=False))
    print("=================================================================================")

    # 5. Multi-Asset Portfolio Fundamental Screener
    print("\n================ MULTI-ASSET FUNDAMENTAL QUALITY SCREENER ================")
    fundamental_info_dict = {t: fetch_fundamental_info(t) for t in tickers}
    fund_summary_df, passed_tickers, rejected_tickers = screen_portfolio_fundamentals(
        tickers=tickers,
        fundamental_info_dict=fundamental_info_dict
    )
    print(fund_summary_df[["Ticker", "Market Cap (₹ Cr)", "ROE", "Trailing P/E", "Gate Status"]].to_string(index=False))
    
    if rejected_tickers:
        print(f"\n[NOTICE] Assets failing fundamental quality gate: {rejected_tickers}")
    else:
        print("[OK] All portfolio assets passed the institutional fundamental quality gate.")
    print("===========================================================================")

    # 6. Multi-Asset Portfolio Active Strategy Backtest Engine
    print("\n================ MULTI-ASSET ACTIVE STRATEGY BACKTEST (NSE) ================")
    print("Strategy: 200-SMA Macro Trend Filter + RSI Oversold Trigger + Volume Confirmation")
    print(f"Allocation: Cash de-risking enabled at {rf:.2%} Risk-Free yield.")
    
    portfolio_signals = generate_portfolio_signals(prices, tickers)
    bt = run_portfolio_vectorized_backtest(
        prices_df=prices,
        signals_df=portfolio_signals,
        weights_dict=portfolio_config,
        initial_capital=initial_investment,
        risk_free_rate=rf,
        rebalance_freq=rebalance_freq,
    )
    
    print("\n--- MULTI-ASSET PORTFOLIO PERFORMANCE & RISK ---")
    print(f"- Active Strategy Final Value : {format_inr(bt['Strategy Final Value'])}")
    print(f"- Buy & Hold Final Value      : {format_inr(bt['Buy & Hold Final Value'])}")
    print(f"- Active Strategy CAGR         : {bt['Strategy CAGR']:.2%}")
    print(f"- Buy & Hold CAGR             : {bt['Buy & Hold CAGR']:.2%}")
    print(f"- Strategy Volatility         : {bt['Strategy Volatility']:.2%}")
    print(f"- Buy & Hold Volatility       : {bt['Buy & Hold Volatility']:.2%}")
    print(f"- Sharpe Ratio (Rf={rf:.1%})   : {bt['Sharpe Ratio']:.2f}")
    print(f"- Sortino Ratio               : {bt['Sortino Ratio']:.2f}")
    print(f"- Calmar Ratio                : {bt['Calmar Ratio']:.2f}")
    print(f"- Strategy Max Drawdown       : {bt['Max Drawdown']:.2%}")
    print(f"- Buy & Hold Max Drawdown     : {bt['Buy & Hold Max Drawdown']:.2%}")

    print("\n--- PORTFOLIO-WIDE TRADE DISTRIBUTION & EFFICIENCY ---")
    print(f"- Total Executed Trades       : {bt['Total Trades']}")
    print(f"- Aggregate Win Rate          : {bt['Win Rate']:.2%}")
    print(f"- Profit Factor               : {bt['Profit Factor']:.2f}")
    print(f"- Expectancy per Trade        : {bt['Expectancy Per Trade']:.2%}")
    print(f"- Average Win                 : {bt['Avg Win']:.2%}")
    print(f"- Average Loss                : {bt['Avg Loss']:.2%}")

    print("\n--- INDIVIDUAL ASSET TRADE CONTRIBUTION ---")
    print(bt["Asset Breakdown"].to_string(index=False))
    print("============================================================================")

    # 7. Vectorized Correlated Monte Carlo Simulation Engine
    mc_mode_label = "Static Drift" if rebalance_freq in (None, "none") else f"Rebalanced ({rebalance_freq.capitalize()})"
    print(f"\nRunning {simulations:,} Correlated Monte Carlo Simulations over {years} Years  [{mc_mode_label}]...")
    sim_results = run_monte_carlo_nse(
        annual_returns=ann_returns,
        cov_matrix=cov_matrix,
        weights_dict=portfolio_config,
        initial_investment_inr=initial_investment,
        years=years,
        n_simulations=simulations,
        rebalance_freq=rebalance_freq,
    )
    
    # 8. Output Monte Carlo Percentile Projections
    final_values = sim_results.iloc[-1]
    percentiles = np.percentile(final_values, [10, 25, 50, 75, 90])
    cagrs = ((percentiles / initial_investment) ** (1 / years)) - 1
    
    labels = ["10th (Severe Bear)", "25th (Conservative)", "50th (Median Expectation)", "75th (Growth)", "90th (Bull Scenario)"]
    
    print(f"\n================ {years}-YEAR PROJECTIONS ================")
    print(f"Initial Capital: {format_inr(initial_investment)} (₹{initial_investment:,.2f})\n")
    
    out_table = pd.DataFrame({
        "Scenario": labels,
        "Projected Portfolio Value": [format_inr(v) for v in percentiles],
        "Implied CAGR": [f"{c:.2%}" for c in cagrs]
    })
    print(out_table.to_string(index=False))
    print("=====================================================")

    # 9. Institutional Visualizations & Chart Dashboard
    if not args.no_plot:
        asset_stats_raw = pd.DataFrame({
            "Annual Return": ann_returns[tickers],
            "Annual Volatility": ann_vol[tickers]
        })
        
        plot_portfolio_dashboard(
            portfolio_series=portfolio_series,
            benchmark_series=nifty_norm,
            strategy_series=bt["Equity Curve"],
            monte_carlo_df=sim_results,
            asset_stats_df=asset_stats_raw,
            weights_dict=portfolio_config,
            initial_capital=initial_investment,
            rebalance_label=reb_label,
            output_png_path=args.plot_file,
            output_html_path=args.html_file
        )

if __name__ == "__main__":
    main()
