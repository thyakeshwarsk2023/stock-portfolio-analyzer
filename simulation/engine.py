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
) -> pd.DataFrame:
    """
    Vectorized correlated GBM Monte Carlo simulation with optional periodic rebalancing.

    Parameters
    ----------
    annual_returns         : Annualized expected return per asset.
    cov_matrix             : Annualized covariance matrix across assets.
    weights_dict           : {ticker: target_weight} allocation.
    initial_investment_inr : Starting portfolio value in INR.
    years                  : Forward projection horizon in years.
    n_simulations          : Number of independent Monte Carlo paths.
    seed                   : Optional RNG seed for reproducibility.
    rebalance_freq         : One of 'none', 'daily', 'monthly', 'quarterly',
                             'annually', or None.
                             - None / 'none': static-weight GBM (weights drift).
                             - Otherwise: portfolio is rebalanced back to target
                               weights at every period boundary, matching the
                               historical attribution engine in analytics/risk.py.

    Returns
    -------
    pd.DataFrame of shape (years + 1, n_simulations) containing portfolio INR
    wealth at annual checkpoints (Year 0 … Year N).

    Implementation notes
    --------------------
    Static mode (no rebalancing)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Asset-level cumulative returns are computed via vectorized cumsum of log
    returns.  The portfolio value at day t is the weighted sum of asset
    growth factors:

        V(t) = sum_i [ w_i * exp(sum_{s=1}^{t} r_{i,s}) ]

    Rebalancing mode
    ~~~~~~~~~~~~~~~~
    The same correlated GBM shocks are used.  However, instead of a single
    weighted-sum at each day, the engine tracks per-asset *unit holdings*
    (analogous to the historical engine in risk.py).

    At each rebalance boundary day d the total portfolio value is redistributed
    back to target weights:

        units_i(d) = w_i * V(d) / price_i(d)

    where price_i(d) = exp( sum_{s=1}^{d} r_{i,s} ).

    Between rebalance events asset values drift freely:

        V(t) = sum_i [ units_i * price_i(t) ]

    This matches the unit-based logic in calculate_portfolio_series() exactly,
    so historical attribution and forward projection are mechanically consistent.

    Both modes use Itô-corrected drift:  mu_daily - 0.5 * sigma^2_daily
    and Cholesky-correlated standard normal shocks.
    Memory is managed with float32 arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers], dtype=np.float32)  # (A,)

    # Daily parameters (float32)
    daily_mu = (np.log(1.0 + annual_returns[tickers].values) / 252.0).astype(np.float32)
    daily_cov = (cov_matrix.loc[tickers, tickers].values / 252.0).astype(np.float32)

    # Cholesky decomposition with spectral regularisation on failure
    try:
        L = np.linalg.cholesky(daily_cov).astype(np.float32)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(daily_cov)
        evals = np.maximum(evals, 1e-7)
        daily_cov = (evecs @ np.diag(evals) @ evecs.T).astype(np.float32)
        L = np.linalg.cholesky(daily_cov).astype(np.float32)

    trading_days = years * 252
    num_assets = len(tickers)

    # Correlated standard normal shocks: shape (T, A, N)
    Z = np.random.normal(size=(trading_days, num_assets, n_simulations)).astype(np.float32)
    correlated_Z = np.einsum("ij,tjk->tik", L, Z)   # (T, A, N)

    # Itô drift per asset: mu_i - 0.5 * sigma_i^2
    asset_variances = np.diag(daily_cov).astype(np.float32)   # (A,)
    drift = (daily_mu - 0.5 * asset_variances).astype(np.float32)   # (A,)

    # Daily log returns: shape (T, A, N)
    daily_log_ret = drift[np.newaxis, :, np.newaxis] + correlated_Z   # broadcast

    # ------------------------------------------------------------------ #
    # Static buy-and-hold (original behaviour, vectorized)
    # ------------------------------------------------------------------ #
    if rebalance_freq is None or str(rebalance_freq).lower() in ("none", ""):
        # Cumulative asset price factors: (T, A, N)
        cum_asset = np.exp(np.cumsum(daily_log_ret, axis=0))

        # Portfolio value: weighted sum across assets -> (T, N)
        port_growth = np.einsum("tan,a->tn", cum_asset, weights)

        # Prepend Year-0 row (all ones * initial capital handled below)
        port_growth = np.vstack(
            [np.ones((1, n_simulations), dtype=np.float32), port_growth]
        )

        annual_indices = [i * 252 for i in range(years + 1)]
        annual_paths = port_growth[annual_indices, :] * initial_investment_inr

        return pd.DataFrame(annual_paths, index=[f"Year {i}" for i in range(years + 1)])

    # ------------------------------------------------------------------ #
    # Rebalancing-aware simulation (unit-based, matches risk.py engine)
    # ------------------------------------------------------------------ #
    period_days = _REBALANCE_PERIODS.get(str(rebalance_freq).lower(), 63)

    # Pre-compute cumulative log price factor for each asset at each day
    # cum_log[t] = sum_{s=0}^{t-1} daily_log_ret[s]  -> asset "price" relative to day 0
    # cum_log shape: (T+1, A, N)  (index 0 = day 0, price = 1.0)
    cum_log = np.zeros((trading_days + 1, num_assets, n_simulations), dtype=np.float32)
    cum_log[1:] = np.cumsum(daily_log_ret, axis=0)
    cum_price = np.exp(cum_log)   # (T+1, A, N)   -- relative price paths

    # Allocate output: portfolio wealth at every day, shape (T+1, N)
    port_wealth = np.empty((trading_days + 1, n_simulations), dtype=np.float32)
    port_wealth[0] = initial_investment_inr   # day-0 wealth

    # Initial units: w_i * V(0) / price_i(0).
    # Since cum_price[0] == 1.0, units_i = w_i * initial_investment_inr
    units = (weights[:, np.newaxis] * initial_investment_inr).astype(np.float32)  # (A, N)

    for day in range(1, trading_days + 1):
        p_today = cum_price[day]          # (A, N) -- relative price at this day

        # Portfolio value = sum_i units_i * p_today_i
        V = (units * p_today).sum(axis=0)   # (N,)
        port_wealth[day] = V

        # Rebalance at period boundaries (not on the final day)
        if day % period_days == 0 and day < trading_days:
            # Reset units so each asset value = w_i * V
            units = (weights[:, np.newaxis] * V[np.newaxis, :]) / p_today   # (A, N)

    # Extract annual checkpoints
    annual_indices = [i * 252 for i in range(years + 1)]
    annual_paths = port_wealth[annual_indices, :]   # already in INR

    return pd.DataFrame(annual_paths, index=[f"Year {i}" for i in range(years + 1)])
