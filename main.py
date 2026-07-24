# main.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from data.fetch import fetch_nse_data
from analytics.returns import calculate_nse_metrics
from portfolio.checks import run_indian_portfolio_checks
from simulation.engine import run_monte_carlo_nse

def format_inr(val: float) -> str:
    """Formats numbers to Indian currency (Lakhs / Crores)."""
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakh"
    else:
        return f"₹{val:,.2f}"

def get_user_inputs() -> Tuple[Dict[str, float], float, int, int]:
    """Interactively collects portfolio settings and tickers from the user."""
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
    
    if not raw_portfolio:
        # Default test portfolio
        portfolio_config = {
            "BEL": 0.25,
            "RELIANCE": 0.20,
            "HDFCBANK": 0.20,
            "TCS": 0.15,
            "HAL": 0.20
        }
    else:
        portfolio_config = {}
        try:
            entries = raw_portfolio.split(",")
            for entry in entries:
                ticker, weight = entry.split(":")
                portfolio_config[ticker.strip().upper()] = float(weight.strip())
        except Exception as e:
            print(f"\n⚠️ Format error in portfolio entry: '{e}'. Using default test portfolio.")
            portfolio_config = {
                "BEL": 0.25,
                "RELIANCE": 0.20,
                "HDFCBANK": 0.20,
                "TCS": 0.15,
                "HAL": 0.20
            }

    return portfolio_config, initial_investment, years, simulations

def main():
    print("==================================================")
    print("   INDIAN PORTFOLIO MONTE CARLO SIMULATOR (NSE)   ")
    print("==================================================\n")
    
    # Dynamically fetch user inputs
    portfolio_config, initial_investment, years, simulations = get_user_inputs()
    
    # 1. Fetch Data
    tickers = list(portfolio_config.keys())
    prices = fetch_nse_data(tickers, period="5y")
    
    # 2. Run Risk Sanity Checks
    is_valid, warnings = run_indian_portfolio_checks(prices, portfolio_config)
    
    print("\n--- PORTFOLIO HEALTH & RISK DIAGNOSTICS ---")
    if warnings:
        for w in warnings:
            print(f"⚠️  {w}")
    else:
        print("✅ Portfolio health looks balanced! No critical warnings.")
        
    if not is_valid:
        print("\n❌ Stopping execution due to critical weight errors.")
        return

    # 3. Calculate Historical Metrics
    ann_returns, ann_vol, cov_matrix = calculate_nse_metrics(prices)
    
    print("\n--- HISTORICAL ANNUALIZED STATS (5Y NSE DATA) ---")
    stats_df = pd.DataFrame({
        "Annual Return": ann_returns.map("{:.2%}".format),
        "Annual Volatility": ann_vol.map("{:.2%}".format)
    })
    print(stats_df)
    
    # 4. Run Monte Carlo Simulation
    print(f"\nRunning {simulations:,} Monte Carlo Simulations over {years} Years...")
    sim_results = run_monte_carlo_nse(
        annual_returns=ann_returns,
        cov_matrix=cov_matrix,
        weights_dict=portfolio_config,
        initial_investment_inr=initial_investment,
        years=years,
        n_simulations=simulations
    )
    
    # 5. Output Results
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

if __name__ == "__main__":
    main()