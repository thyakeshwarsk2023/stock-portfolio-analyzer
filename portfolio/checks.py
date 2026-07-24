import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def run_indian_portfolio_checks(
    price_df: pd.DataFrame, 
    weights_dict: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """Runs automated risk checks tailored for Indian equity portfolios."""
    warnings = []
    is_valid = True

    # Guard 1: Empty weights check
    if not weights_dict:
        return False, ["CRITICAL: Portfolio weights dictionary is empty."]

    # Guard 2: Missing ticker guard (Prevents KeyError)
    missing_tickers = [t for t in weights_dict.keys() if t not in price_df.columns]
    if missing_tickers:
        return False, [f"CRITICAL: Tickers {missing_tickers} not found in price DataFrame columns."]

    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers])

    # Check 1: Weight Sum Validation
    if not np.isclose(np.sum(weights), 1.0, atol=0.01):
        is_valid = False
        warnings.append(f"CRITICAL: Portfolio weights sum to {np.sum(weights):.2f}. Must equal 1.00 (100%).")

    # Check 2: Negative Weights (Short positions)
    if np.any(weights < 0):
        warnings.append("WARNING: Negative weights detected. Ensure long-only allocation if intended.")

    # Check 3: Concentration Risk (Herfindahl Index)
    hhi = np.sum(weights ** 2)
    if hhi > 0.25:
        warnings.append(f"HIGH CONCENTRATION RISK (HHI: {hhi:.2f}): Portfolio is heavily weighted in top holdings.")
    
    for stock, w in weights_dict.items():
        if w >= 0.35:
            warnings.append(f"SINGLE-STOCK RISK: '{stock}' comprises {w:.0%} of portfolio (>35% limit).")

    # Slice only portfolio tickers for return calculations
    sliced_prices = price_df[tickers]
    log_returns = np.log(sliced_prices / sliced_prices.shift(1)).dropna()

    if len(log_returns) < 2:
        warnings.append("WARNING: Insufficient price history to calculate reliable correlation/volatility.")
        return is_valid, warnings

    # Check 4: High Correlation Alert (> 0.75)
    corr_matrix = log_returns.corr()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            corr = corr_matrix.loc[t1, t2]
            if not np.isnan(corr) and corr > 0.75:
                warnings.append(f"REDUNDANCY ALERT: '{t1}' and '{t2}' have high correlation ({corr:.2f}).")

    # Check 5: Volatility Check
    ann_vol = log_returns.std() * np.sqrt(252)
    for stock in tickers:
        vol = ann_vol[stock]
        if not np.isnan(vol) and vol > 0.40:
            warnings.append(f"HIGH VOLATILITY: '{stock}' has annualized volatility of {vol:.1%}.")

    return is_valid, warnings