# analytics/visualization.py
import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import os
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for deterministic server/CLI execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

def format_inr_axis(val: float, pos=None) -> str:
    """Formats numbers into Indian currency notation for chart axes."""
    if abs(val) >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif abs(val) >= 100000:
        return f"₹{val / 100000:.1f} L"
    else:
        return f"₹{val:,.0f}"

def format_inr_pill(val: float) -> str:
    """Formats compact currency values for chart annotations."""
    if abs(val) >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif abs(val) >= 100000:
        return f"₹{val / 100000:.2f} Lakh"
    else:
        return f"₹{val:,.0f}"

def plot_portfolio_dashboard(
    portfolio_series: pd.Series,
    benchmark_series: pd.Series,
    strategy_series: Optional[pd.Series] = None,
    buy_hold_strategy_series: Optional[pd.Series] = None,
    monte_carlo_df: Optional[pd.DataFrame] = None,
    asset_stats_df: Optional[pd.DataFrame] = None,
    weights_dict: Optional[Dict[str, float]] = None,
    initial_capital: float = 500000.0,
    rebalance_label: str = "Quarterly",
    output_png_path: str = "portfolio_report.png",
    output_html_path: str = "portfolio_dashboard.html"
) -> str:
    """
    Renders an institutional quantitative portfolio report:
    1. Historical Wealth Trajectory (Portfolio vs. Nifty 50 vs. Active Strategy)
    2. Underwater Drawdown Profiles
    3. Monte Carlo 10-Year Stochastic Simulation Fan Chart
    4. Asset Risk vs. Return Frontier & Portfolio Allocation
    """
    # -------------------------------------------------------------
    # 1. MATPLOTLIB INSTITUTIONAL DARK REPORT
    # -------------------------------------------------------------
    bg_canvas = "#0d1117"     # Deep slate canvas
    bg_card = "#161b22"       # Card surface
    border_color = "#30363d"  # Subtle border line
    grid_color = "#21262d"    # Low-contrast grid
    text_primary = "#f0f6fc"  # Clean off-white
    text_secondary = "#8b949e" # Muted metadata
    
    # Quantitative financial palette
    c_green = "#2ea043"   # Portfolio Primary Emerald
    c_gold = "#d29922"    # Nifty 50 Benchmark Amber
    c_blue = "#58a6ff"    # Strategy Cyan
    c_purple = "#bc8cff"  # Component Assets
    c_red = "#f85149"     # Drawdowns / Bear Cones

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica Neue", "Arial"],
        "axes.edgecolor": border_color,
        "axes.linewidth": 0.8,
        "xtick.color": text_secondary,
        "ytick.color": text_secondary,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": text_primary
    })

    fig = plt.figure(figsize=(18, 14), dpi=200)
    fig.patch.set_facecolor(bg_canvas)
    
    # 3-Row Grid:
    # Row 0: Full width historical wealth trajectory
    # Row 1: Drawdown (left) + Monte Carlo Fan Chart (right)
    # Row 2: Risk-Return Scatter (left) + Asset Allocation Breakdown (right)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.95], hspace=0.36, wspace=0.22)

    # Normalize indices to tz-naive to prevent intersection mismatches
    p_idx = portfolio_series.index.tz_localize(None) if portfolio_series.index.tz is not None else portfolio_series.index
    portfolio_series = pd.Series(portfolio_series.values, index=p_idx)
    b_idx = benchmark_series.index.tz_localize(None) if benchmark_series.index.tz is not None else benchmark_series.index
    benchmark_series = pd.Series(benchmark_series.values, index=b_idx)
    if strategy_series is not None and not strategy_series.empty:
        s_idx = strategy_series.index.tz_localize(None) if strategy_series.index.tz is not None else strategy_series.index
        strategy_series = pd.Series(strategy_series.values, index=s_idx)

    # Common index alignment
    common_idx = portfolio_series.index.intersection(benchmark_series.index)
    port_aligned = portfolio_series.loc[common_idx]
    bench_aligned = benchmark_series.loc[common_idx]
    
    port_wealth = initial_capital * (port_aligned / port_aligned.iloc[0])
    bench_wealth = initial_capital * (bench_aligned / bench_aligned.iloc[0])

    years_hist = max(len(common_idx) / 252, 0.1)
    port_cagr = (port_wealth.iloc[-1] / initial_capital) ** (1 / years_hist) - 1.0
    bench_cagr = (bench_wealth.iloc[-1] / initial_capital) ** (1 / years_hist) - 1.0

    # =========================================================
    # PANEL 1: HISTORICAL WEALTH TRAJECTORY (Full Width)
    # =========================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(bg_card)

    ax1.plot(common_idx, port_wealth, label=f"Portfolio ({rebalance_label})", color=c_green, linewidth=2.4, zorder=4)
    ax1.plot(common_idx, bench_wealth, label="Nifty 50 TRI (NIFTYBEES)", color=c_gold, linewidth=1.8, linestyle="--", zorder=3)

    if strategy_series is not None and not strategy_series.empty:
        strat_aligned = strategy_series.loc[strategy_series.index.intersection(common_idx)]
        strat_wealth = initial_capital * (strat_aligned / strat_aligned.iloc[0])
        ax1.plot(strat_aligned.index, strat_wealth, label="Multi-Asset Active Strategy (Dynamic Cash De-Risking)", color=c_blue, linewidth=2.0, zorder=5)

    ax1.set_title("Historical Portfolio Wealth Trajectory (5-Year Aligned NSE Data)", fontsize=13, fontweight="bold", pad=12, loc="left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_inr_axis))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.grid(True, linestyle=":", alpha=0.5, color=grid_color)
    
    # Statistical Callout Card inside Panel 1
    callout_text = (
        f"Initial Capital : {format_inr_pill(initial_capital)}\n"
        f"Portfolio End   : {format_inr_pill(port_wealth.iloc[-1])} ({port_cagr:+.1%} CAGR)\n"
        f"Benchmark (TRI) : {format_inr_pill(bench_wealth.iloc[-1])} ({bench_cagr:+.1%} CAGR)"
    )
    ax1.text(
        0.015, 0.93, callout_text,
        transform=ax1.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=bg_canvas, edgecolor=border_color, alpha=0.92)
    )
    
    ax1.legend(loc="lower right", framealpha=0.65, facecolor=bg_canvas, edgecolor=border_color, fontsize=9)

    # =========================================================
    # PANEL 2: UNDERWATER DRAWDOWN PROFILES
    # =========================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(bg_card)

    port_peak = port_wealth.cummax()
    port_dd = (port_wealth - port_peak) / port_peak
    
    bench_peak = bench_wealth.cummax()
    bench_dd = (bench_wealth - bench_peak) / bench_peak

    ax2.fill_between(common_idx, port_dd * 100, 0, color=c_red, alpha=0.25, label=f"Portfolio Drawdown (Max: {port_dd.min():.1%})", zorder=3)
    ax2.plot(common_idx, port_dd * 100, color=c_red, linewidth=1.4, zorder=4)
    ax2.plot(common_idx, bench_dd * 100, color=c_gold, linewidth=1.2, linestyle="--", label=f"Nifty 50 TRI Drawdown (Max: {bench_dd.min():.1%})", zorder=3)

    # Stress Threshold Reference Lines
    ax2.axhline(-10, color="#484f58", linestyle=":", linewidth=0.9)
    ax2.axhline(-20, color="#484f58", linestyle=":", linewidth=0.9)
    ax2.axhline(-30, color="#484f58", linestyle=":", linewidth=0.9)
    ax2.text(common_idx[0], -9.5, "-10% Correction", color=text_secondary, fontsize=7.5)
    ax2.text(common_idx[0], -19.5, "-20% Bear Threshold", color=text_secondary, fontsize=7.5)

    ax2.set_title("Underwater Peak-to-Trough Drawdown (%)", fontsize=11.5, fontweight="bold", pad=10, loc="left")
    ax2.set_ylabel("Drawdown Depth (%)", fontsize=9, color=text_secondary)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.grid(True, linestyle=":", alpha=0.5, color=grid_color)
    ax2.legend(loc="lower left", framealpha=0.65, facecolor=bg_canvas, edgecolor=border_color, fontsize=8.5)

    # =========================================================
    # PANEL 3: MONTE CARLO PERCENTILE FAN CHART
    # =========================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(bg_card)

    if monte_carlo_df is not None and not monte_carlo_df.empty:
        sim_data = monte_carlo_df.values
        years_idx = np.arange(len(monte_carlo_df))
        
        p10 = np.percentile(sim_data, 10, axis=1)
        p25 = np.percentile(sim_data, 25, axis=1)
        p50 = np.percentile(sim_data, 50, axis=1)
        p75 = np.percentile(sim_data, 75, axis=1)
        p90 = np.percentile(sim_data, 90, axis=1)

        # Multi-layer probabilistic fan cone
        ax3.fill_between(years_idx, p10, p90, color=c_blue, alpha=0.12, label="10th – 90th Outer Range", zorder=2)
        ax3.fill_between(years_idx, p25, p75, color=c_blue, alpha=0.28, label="25th – 75th Core Probability", zorder=3)
        ax3.plot(years_idx, p50, color=c_blue, linewidth=2.6, label="50th Percentile (Median Expectation)", zorder=5)
        ax3.plot(years_idx, p90, color=c_green, linewidth=1.3, linestyle="--", label="90th Percentile (Bull)", zorder=4)
        ax3.plot(years_idx, p10, color=c_red, linewidth=1.3, linestyle="--", label="10th Percentile (Severe Bear)", zorder=4)
        
        # Initial capital baseline
        ax3.axhline(initial_capital, color=c_gold, linestyle=":", linewidth=1.3, label=f"Capital Baseline ({format_inr_pill(initial_capital)})", zorder=3)

        # Right-edge terminal value tags
        final_yr = years_idx[-1]
        ax3.text(final_yr + 0.15, p90[-1], f"P90: {format_inr_pill(p90[-1])}", color=c_green, fontsize=8, fontweight="bold", verticalalignment="center")
        ax3.text(final_yr + 0.15, p50[-1], f"P50: {format_inr_pill(p50[-1])}", color=c_blue, fontsize=8, fontweight="bold", verticalalignment="center")
        ax3.text(final_yr + 0.15, p10[-1], f"P10: {format_inr_pill(p10[-1])}", color=c_red, fontsize=8, fontweight="bold", verticalalignment="center")

        ax3.set_xlim(years_idx[0], years_idx[-1] + 1.8)
        ax3.set_xticks(years_idx)
        ax3.set_xticklabels([f"Y{i}" for i in years_idx])
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_inr_axis))

    ax3.set_title("Monte Carlo 10-Year Projections (Percentile Fan Chart)", fontsize=11.5, fontweight="bold", pad=10, loc="left")
    ax3.set_xlabel("Horizon (Years)", fontsize=9, color=text_secondary)
    ax3.grid(True, linestyle=":", alpha=0.5, color=grid_color)
    ax3.legend(loc="upper left", framealpha=0.65, facecolor=bg_canvas, edgecolor=border_color, fontsize=8)

    # =========================================================
    # PANEL 4: ASSET RISK-RETURN EFFICIENT PROFILE
    # =========================================================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(bg_card)

    if asset_stats_df is not None and not asset_stats_df.empty:
        for ticker in asset_stats_df.index:
            r = asset_stats_df.loc[ticker, "Annual Return"]
            v = asset_stats_df.loc[ticker, "Annual Volatility"]
            ax4.scatter(v * 100, r * 100, color=c_purple, s=110, edgecolors=text_primary, linewidth=1.1, zorder=5)
            ax4.annotate(ticker, (v * 100, r * 100), textcoords="offset points", xytext=(8, 4), color=text_primary, fontsize=8.5, fontweight="bold")

        # Total portfolio point
        port_ret = float(np.exp(np.log(port_aligned / port_aligned.shift(1)).dropna().mean() * 252) - 1.0)
        port_vol = float(np.log(port_aligned / port_aligned.shift(1)).dropna().std() * np.sqrt(252))
        ax4.scatter(port_vol * 100, port_ret * 100, color=c_green, s=180, marker="*", edgecolors=text_primary, linewidth=1.4, zorder=6, label=f"Portfolio ({port_ret:.1%}, {port_vol:.1%})")

        # Benchmark Nifty 50 TRI point
        bench_ret = float(np.exp(np.log(bench_aligned / bench_aligned.shift(1)).dropna().mean() * 252) - 1.0)
        bench_vol = float(np.log(bench_aligned / bench_aligned.shift(1)).dropna().std() * np.sqrt(252))
        ax4.scatter(bench_vol * 100, bench_ret * 100, color=c_gold, s=140, marker="D", edgecolors=text_primary, linewidth=1.1, zorder=6, label=f"Nifty 50 TRI ({bench_ret:.1%}, {bench_vol:.1%})")

    ax4.set_title("Risk-Return Characteristics (Annualized)", fontsize=11.5, fontweight="bold", pad=10, loc="left")
    ax4.set_xlabel("Annual Volatility (%)", fontsize=9, color=text_secondary)
    ax4.set_ylabel("Annualized Return (CAGR %)", fontsize=9, color=text_secondary)
    ax4.grid(True, linestyle=":", alpha=0.5, color=grid_color)
    ax4.legend(loc="upper left", framealpha=0.65, facecolor=bg_canvas, edgecolor=border_color, fontsize=8.5)

    # =========================================================
    # PANEL 5: PORTFOLIO ASSET ALLOCATION BREAKDOWN
    # =========================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(bg_card)

    if weights_dict:
        stocks = list(weights_dict.keys())
        w_vals = [weights_dict[s] * 100 for s in stocks]
        y_pos = np.arange(len(stocks))
        
        palette_bars = [c_green, c_blue, c_gold, c_purple, "#ec4899", "#14b8a6", "#f97316"]
        bar_colors = [palette_bars[i % len(palette_bars)] for i in range(len(stocks))]
        
        bars = ax5.barh(y_pos, w_vals, color=bar_colors, height=0.55, edgecolor=border_color, linewidth=0.8, zorder=3)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(stocks, fontsize=9, fontweight="bold", color=text_primary)
        ax5.invert_yaxis()  # Top to bottom order
        
        for bar in bars:
            width = bar.get_width()
            ax5.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", color=text_primary, fontsize=8.5, fontweight="bold")
            
        ax5.set_xlim(0, max(w_vals) * 1.30)
        ax5.set_xlabel("Target Portfolio Weight (%)", fontsize=9, color=text_secondary)
    
    ax5.set_title(f"Target Portfolio Asset Allocation ({rebalance_label})", fontsize=11.5, fontweight="bold", pad=10, loc="left")
    ax5.grid(True, axis="x", linestyle=":", alpha=0.5, color=grid_color)

    # Clean header badge
    fig.text(0.08, 0.985, "PORTFOLIO QUANTITATIVE RISK & PERFORMANCE REPORT", fontsize=15, fontweight="heavy", color=text_primary)
    fig.text(0.08, 0.968, "National Stock Exchange of India (NSE) | Multi-Factor Backtest & Correlated Monte Carlo Projections", fontsize=9.5, color=text_secondary)

    # Save high-resolution publication image
    plt.savefig(output_png_path, dpi=200, bbox_inches="tight", facecolor=bg_canvas, edgecolor="none")
    plt.close(fig)
    print(f"[REPORT] Visual analytics report saved: {os.path.abspath(output_png_path)}")

    # -------------------------------------------------------------
    # 2. PLOTLY INTERACTIVE FINANCIAL DASHBOARD
    # -------------------------------------------------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        pfig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Historical Wealth Growth (₹ Starting Capital)",
                "Underwater Drawdown (%)",
                "Monte Carlo 10-Year Percentile Fan Chart",
                "Risk-Return Profile (Annualized %)"
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.10
        )

        # 1. Equity Curves
        pfig.add_trace(
            go.Scatter(
                x=common_idx, y=port_wealth, mode="lines",
                name=f"Portfolio ({rebalance_label})",
                line=dict(color="#2ea043", width=2.5),
                hovertemplate="<b>Portfolio</b><br>Date: %{x|%d %b %Y}<br>Value: ₹%{y:,.0f}<extra></extra>"
            ), row=1, col=1
        )
        pfig.add_trace(
            go.Scatter(
                x=common_idx, y=bench_wealth, mode="lines",
                name="Nifty 50 TRI (NIFTYBEES)",
                line=dict(color="#d29922", width=1.8, dash="dash"),
                hovertemplate="<b>Nifty 50 TRI</b><br>Date: %{x|%d %b %Y}<br>Value: ₹%{y:,.0f}<extra></extra>"
            ), row=1, col=1
        )
        if strategy_series is not None and not strategy_series.empty:
            strat_aligned = strategy_series.loc[strategy_series.index.intersection(common_idx)]
            strat_wealth = initial_capital * (strat_aligned / strat_aligned.iloc[0])
            pfig.add_trace(
                go.Scatter(
                    x=strat_aligned.index, y=strat_wealth, mode="lines",
                    name="Multi-Asset Strategy",
                    line=dict(color="#58a6ff", width=2.0),
                    hovertemplate="<b>Strategy</b><br>Date: %{x|%d %b %Y}<br>Value: ₹%{y:,.0f}<extra></extra>"
                ), row=1, col=1
            )

        # 2. Drawdowns
        pfig.add_trace(
            go.Scatter(
                x=common_idx, y=port_dd * 100, fill="tozeroy",
                name="Portfolio Drawdown",
                line=dict(color="#f85149", width=1.5),
                fillcolor="rgba(248, 81, 73, 0.22)",
                hovertemplate="<b>Portfolio Drawdown</b><br>Date: %{x|%d %b %Y}<br>Depth: %{y:.2f}%<extra></extra>"
            ), row=1, col=2
        )
        pfig.add_trace(
            go.Scatter(
                x=common_idx, y=bench_dd * 100, mode="lines",
                name="Nifty 50 TRI Drawdown",
                line=dict(color="#d29922", width=1.2, dash="dash"),
                hovertemplate="<b>Nifty 50 TRI Drawdown</b><br>Date: %{x|%d %b %Y}<br>Depth: %{y:.2f}%<extra></extra>"
            ), row=1, col=2
        )

        # 3. Monte Carlo Percentile Cones
        if monte_carlo_df is not None and not monte_carlo_df.empty:
            x_yrs = [f"Year {i}" for i in range(len(monte_carlo_df))]
            
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=p90, mode="lines", name="90th Bull",
                    line=dict(color="#2ea043", width=1.5, dash="dot"),
                    hovertemplate="90th Percentile: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=p75, mode="lines", name="75th Growth",
                    line=dict(color="#58a6ff", width=1.2, dash="dot"),
                    hovertemplate="75th Percentile: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=p50, mode="lines", name="50th Median",
                    line=dict(color="#58a6ff", width=3.0),
                    hovertemplate="50th Median: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=p25, mode="lines", name="25th Conservative",
                    line=dict(color="#58a6ff", width=1.2, dash="dot"),
                    hovertemplate="25th Percentile: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=p10, mode="lines", name="10th Bear",
                    line=dict(color="#f85149", width=1.5, dash="dot"),
                    hovertemplate="10th Percentile: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )
            pfig.add_trace(
                go.Scatter(
                    x=x_yrs, y=[initial_capital] * len(x_yrs), mode="lines", name="Initial Capital",
                    line=dict(color="#d29922", width=1.2, dash="dash"),
                    hovertemplate="Capital: ₹%{y:,.0f}<extra></extra>"
                ), row=2, col=1
            )

        # 4. Risk-Return Scatter
        if asset_stats_df is not None and not asset_stats_df.empty:
            for ticker in asset_stats_df.index:
                r = asset_stats_df.loc[ticker, "Annual Return"] * 100
                v = asset_stats_df.loc[ticker, "Annual Volatility"] * 100
                pfig.add_trace(
                    go.Scatter(
                        x=[v], y=[r], mode="markers+text", name=ticker,
                        text=[ticker], textposition="top right",
                        marker=dict(size=11, color="#bc8cff"),
                        hovertemplate=f"<b>{ticker}</b><br>CAGR: %{{y:.2f}}%<br>Volatility: %{{x:.2f}}%<extra></extra>"
                    ), row=2, col=2
                )
            pfig.add_trace(
                go.Scatter(
                    x=[port_vol * 100], y=[port_ret * 100], mode="markers+text", name="Portfolio",
                    text=["Portfolio"], textposition="top right",
                    marker=dict(size=16, symbol="star", color="#2ea043"),
                    hovertemplate="<b>Portfolio Total</b><br>CAGR: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>"
                ), row=2, col=2
            )
            pfig.add_trace(
                go.Scatter(
                    x=[bench_vol * 100], y=[bench_ret * 100], mode="markers+text", name="Nifty 50 TRI",
                    text=["Nifty 50 TRI"], textposition="top right",
                    marker=dict(size=14, symbol="diamond", color="#d29922"),
                    hovertemplate="<b>Nifty 50 TRI Benchmark</b><br>CAGR: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>"
                ), row=2, col=2
            )

        pfig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(family="Segoe UI, sans-serif", color="#f0f6fc", size=11),
            title=dict(
                text="<b>Quantitative Portfolio Analytics & Risk Dashboard</b><br><span style='font-size:12px;color:#8b949e'>NSE India Historical Backtest & Correlated Stochastic Projections</span>",
                x=0.03, y=0.97
            ),
            margin=dict(l=60, r=40, t=100, b=50),
            height=920,
            showlegend=False
        )
        pfig.write_html(output_html_path)
        print(f"[REPORT] Interactive HTML dashboard saved: {os.path.abspath(output_html_path)}")
    except Exception as e:
        print(f"[WARN] Interactive HTML dashboard could not be written: {e}")

    return output_png_path
