# simulation/engine.py
import numpy as np
import pandas as pd
from typing import Dict, Optional

# Rebalancing frequency -> approximate trading days per period
_REBALANCE_PERIODS: Dict[str, int] = {
    "daily":     1,
    "monthly":   21,
    "quarterly": 63,
    "annually":  252,
}


def run_monte_carlo_nse(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights_dict: Dict[str, float],
    initial_investment_inr: float = 500_000.0,
    years: int = 10,
    n_simulations: int = 10_000,
    seed: Optional[int] = None,
    rebalance_freq: Optional[str] = None,
    batch_size: int = 2_000,
) -> pd.DataFrame:
    """
    Chunked, float64 correlated GBM Monte Carlo simulation with optional rebalancing.

    Parameters
    ----------
    annual_returns         : Annualized geometric CAGR per asset.
    cov_matrix             : Annualized covariance matrix across assets.
    weights_dict           : {ticker: target_weight} allocation.
    initial_investment_inr : Starting portfolio value in INR.
    years                  : Forward projection horizon in years.
    n_simulations          : Number of independent Monte Carlo paths.
    seed                   : Optional RNG seed for reproducibility.
    rebalance_freq         : 'none'/None (static buy-and-hold drift), 'daily',
                             'monthly', 'quarterly', or 'annually'.
    batch_size             : Number of paths simulated per vector batch to bound
                             peak RAM allocation across long horizons.

    Returns
    -------
    pd.DataFrame of shape (years + 1, n_simulations) containing portfolio INR
    wealth at annual checkpoints (Year 0 ... Year N).

    Implementation notes
    --------------------
    - Simulates in double-precision (float64) batches of size `batch_size` and
      retains only annual checkpoints, cutting peak RAM allocation by ~80-90%.
    - Uses log-space GBM drift ln(1 + CAGR_annual) / 252, which already reflects
      the Itô-adjusted geometric expectation (mu_arith - 0.5 * sigma^2), paired
      with Cholesky-correlated standard normal shocks.
    """
    rng = np.random.default_rng(seed)

    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers], dtype=np.float64)  # (A,)
    weights = weights / weights.sum()

    # Daily parameters (float64)
    daily_mu = (np.log(1.0 + annual_returns[tickers].values) / 252.0).astype(np.float64)
    daily_cov = (cov_matrix.loc[tickers, tickers].values / 252.0).astype(np.float64)

    # Cholesky decomposition with spectral regularization on failure
    try:
        L = np.linalg.cholesky(daily_cov)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(daily_cov)
        evals = np.maximum(evals, 1e-9)
        daily_cov = evecs @ np.diag(evals) @ evecs.T
        L = np.linalg.cholesky(daily_cov)

    trading_days = years * 252
    num_assets = len(tickers)

    # annual_returns is geometric CAGR = exp(E[log_ret] * 252) - 1, so daily_mu
    # already equals (mu_arith - 0.5 * sigma^2); avoid double-subtracting 0.5 * var.
    drift = daily_mu  # (A,)

    is_static = rebalance_freq is None or str(rebalance_freq).lower() in ("none", "")
    period_days = None if is_static else _REBALANCE_PERIODS.get(str(rebalance_freq).lower(), 63)

    annual_paths = np.empty((years + 1, n_simulations), dtype=np.float64)
    annual_paths[0, :] = initial_investment_inr

    eff_batch = max(1, min(batch_size, n_simulations))

    for start in range(0, n_simulations, eff_batch):
        end = min(start + eff_batch, n_simulations)
        b_size = end - start

        # Correlated daily log returns for this batch: shape (T, A, B)
        Z = rng.standard_normal(size=(trading_days, num_assets, b_size))
        daily_log_ret = drift[np.newaxis, :, np.newaxis] + np.einsum("ij,tjk->tik", L, Z)
        del Z

        if is_static:
            # Sum log returns directly into annual blocks: (years, A, B)
            annual_log_steps = daily_log_ret.reshape(years, 252, num_assets, b_size).sum(axis=1)
            cum_asset_years = np.exp(np.cumsum(annual_log_steps, axis=0))
            annual_paths[1:, start:end] = np.einsum("yab,a->yb", cum_asset_years, weights) * initial_investment_inr
        else:
            # Bucket-based simulation matching rebalancing boundaries without allocating cum_price
            buckets = np.repeat((weights * initial_investment_inr)[:, np.newaxis], b_size, axis=1)  # (A, B)
            daily_factors = np.exp(daily_log_ret)

            for day in range(1, trading_days + 1):
                buckets *= daily_factors[day - 1]
                port_val = buckets.sum(axis=0)

                if day % 252 == 0:
                    annual_paths[day // 252, start:end] = port_val

                if day % period_days == 0 and day < trading_days:
                    buckets = weights[:, np.newaxis] * port_val[np.newaxis, :]

    return pd.DataFrame(annual_paths, index=[f"Year {i}" for i in range(years + 1)])
