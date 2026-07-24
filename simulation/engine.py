# simulation/engine.py
import numpy as np
import pandas as pd
from typing import Dict, Optional

def run_monte_carlo_nse(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights_dict: Dict[str, float],
    initial_investment_inr: float = 500000.0,
    years: int = 10,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Runs a high-performance vectorized Monte Carlo simulation with correlated asset returns.
    Uses float32 precision for memory optimization and supports reproducible random seeds.
    """
    if seed is not None:
        np.random.seed(seed)
        
    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers], dtype=np.float32).reshape(-1, 1)
    
    # Daily parameters (converted to float32)
    daily_mu = (np.log(1 + annual_returns[tickers].values) / 252).astype(np.float32)
    daily_cov = cov_matrix.loc[tickers, tickers].values.astype(np.float32)
    
    # Cholesky Decomposition with matrix regularization
    try:
        L = np.linalg.cholesky(daily_cov)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(daily_cov)
        evals = np.maximum(evals, 1e-7)
        daily_cov = (evecs @ np.diag(evals) @ evecs.T).astype(np.float32)
        L = np.linalg.cholesky(daily_cov)

    trading_days = years * 252
    num_assets = len(tickers)
    
    # Generate random standard normal numbers in float32 (50% less RAM usage)
    Z = np.random.normal(size=(trading_days, num_assets, n_simulations)).astype(np.float32)
    
    # Apply Cholesky matrix to correlate returns across stocks
    correlated_Z = np.einsum('ij, tjk -> tik', L, Z)
    
    # Calculate Itô's Lemma drift: mu - 0.5 * sigma^2
    asset_variances = np.diag(daily_cov).reshape(1, -1, 1)
    drift = daily_mu.reshape(1, -1, 1) - (0.5 * asset_variances)
    
    # Compute daily log returns and accumulate growth
    daily_log_returns = drift + correlated_Z
    cum_asset_returns = np.exp(np.cumsum(daily_log_returns, axis=0))
    
    # Calculate total portfolio wealth path
    portfolio_growth = np.einsum('tik, i -> tk', cum_asset_returns, weights.flatten())
    portfolio_growth = np.vstack([np.ones((1, n_simulations), dtype=np.float32), portfolio_growth])
    
    # Extract annual checkpoints
    annual_indices = [int(i * 252) for i in range(years + 1)]
    annual_paths = portfolio_growth[annual_indices, :] * initial_investment_inr
    
    return pd.DataFrame(
        annual_paths, 
        index=[f"Year {i}" for i in range(years + 1)]
    )