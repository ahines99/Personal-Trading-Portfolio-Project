"""
dashboard.py
------------
Interactive Plotly dashboard for the personal trading portfolio.

Layout:
  Row 1: KPI cards (CAGR, Alpha, Sharpe, Max DD, etc.)
  Row 2: Wealth growth (strategy vs SPY)
  Row 3: Annual returns comparison | Monthly returns heatmap
  Row 4: Drawdown | Rolling Sharpe
  Row 5: Current holdings | Sector allocation
  Row 6: Return distribution | Feature importance
  Row 7: Regime performance | OOS split | Factor correlation
  Row 8: Bootstrap CI
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. Run: pip install plotly")


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "strategy":   "#2563EB",
    "benchmark":  "#9CA3AF",
    "positive":   "#10B981",
    "negative":   "#EF4444",
    "neutral":    "#6B7280",
    "highlight":  "#F59E0B",
    "bg":         "#FFFFFF",
    "grid":       "#F3F4F6",
    "text":       "#111827",
    "text_light": "#6B7280",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["grid"],
    font=dict(family="Inter, sans-serif", size=11, color=COLORS["text"]),
    margin=dict(l=50, r=30, t=40, b=40),
)


def _apply_defaults(fig):
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", gridwidth=0.5,
                     zeroline=False, linecolor="#D1D5DB")
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", gridwidth=0.5,
                     zeroline=False, linecolor="#D1D5DB")
    return fig


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def equity_curve_chart(result, benchmark_equity=None, initial_capital=100_000):
    equity_norm = result.equity_curve / initial_capital

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_norm.index, y=equity_norm.values,
        name="Strategy",
        line=dict(color=COLORS["strategy"], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.2f}x<extra></extra>",
    ))

    if benchmark_equity is not None:
        bench_norm = benchmark_equity / initial_capital
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            name="SPY Buy & Hold",
            line=dict(color=COLORS["benchmark"], width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.2f}x<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"],
                  line_width=0.8, opacity=0.5)
    fig.update_layout(
        title="Wealth Growth (Normalized to $1)",
        yaxis_title="Growth Multiple",
        legend=dict(orientation="h", y=1.02, x=0),
        **LAYOUT_DEFAULTS,
    )
    return fig


def drawdown_chart(result):
    from metrics import drawdown_series
    dd = drawdown_series(result.equity_curve) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=COLORS["negative"], width=1),
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Drawdown (%)", yaxis_title="%",
                      yaxis_ticksuffix="%", **LAYOUT_DEFAULTS)
    return fig


def rolling_sharpe_chart(result, window=126):
    from metrics import rolling_sharpe
    rs = rolling_sharpe(result.daily_returns, window=window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        name=f"Rolling {window//21}m Sharpe",
        line=dict(color=COLORS["strategy"], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8)
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["positive"],
                  line_width=0.8, opacity=0.6, annotation_text="Sharpe=1")
    fig.update_layout(title=f"Rolling {window//21}-Month Sharpe Ratio",
                      yaxis_title="Sharpe", **LAYOUT_DEFAULTS)
    return fig


def return_distribution_chart(result):
    r = result.daily_returns.dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r.values, nbinsx=80, name="Daily returns",
        marker_color=COLORS["strategy"], opacity=0.75,
    ))
    fig.add_vline(x=0, line_color=COLORS["text"], line_width=1, line_dash="dash", opacity=0.5)
    fig.add_vline(x=r.mean(), line_color=COLORS["positive"], line_width=1,
                  line_dash="dot", annotation_text=f"Mean: {r.mean():.3f}%")
    fig.update_layout(title="Daily Return Distribution",
                      xaxis_title="Daily Return (%)", yaxis_title="Count",
                      showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def annual_returns_chart(annual_df):
    """Side-by-side annual returns: Strategy vs SPY."""
    if annual_df is None or annual_df.empty:
        return _empty_fig("No annual return data")

    fig = go.Figure()
    years = annual_df.index.astype(str).tolist()

    if "Strategy" in annual_df.columns:
        colors_s = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                    for v in annual_df["Strategy"]]
        fig.add_trace(go.Bar(
            x=years, y=annual_df["Strategy"].values,
            name="Strategy", marker_color=colors_s,
            text=[f"{v:.1f}%" for v in annual_df["Strategy"]], textposition="outside",
        ))

    if "SPY" in annual_df.columns:
        fig.add_trace(go.Bar(
            x=years, y=annual_df["SPY"].values,
            name="SPY", marker_color=COLORS["benchmark"], opacity=0.6,
        ))

    fig.update_layout(
        title="Annual Returns: Strategy vs SPY",
        yaxis_title="Return (%)", yaxis_ticksuffix="%",
        barmode="group", **LAYOUT_DEFAULTS,
    )
    return fig


def monthly_heatmap_chart(monthly_df):
    """Calendar heatmap of monthly returns."""
    if monthly_df is None or monthly_df.empty:
        return _empty_fig("No monthly return data")

    months = [c for c in monthly_df.columns if c != "YTD"]
    z = monthly_df[months].values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=months,
        y=monthly_df.index.astype(str).tolist(),
        colorscale=[
            [0.0, "#EF4444"],
            [0.5, "#FFFFFF"],
            [1.0, "#10B981"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap (%)",
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def holdings_chart(holdings_df):
    """Bar chart of current portfolio holdings."""
    if holdings_df is None or holdings_df.empty:
        return _empty_fig("No holdings data")

    df = holdings_df.sort_values("weight", ascending=True).tail(25)

    fig = go.Figure(go.Bar(
        x=df["weight"].values * 100,
        y=df["ticker"].values,
        orientation="h",
        marker_color=COLORS["strategy"],
        text=[f"{w:.1f}%" for w in df["weight"] * 100],
        textposition="outside",
    ))
    fig.update_layout(
        title="Current Holdings (Weight %)",
        xaxis_title="Weight (%)", xaxis_ticksuffix="%",
        yaxis=dict(autorange="reversed"),
        showlegend=False, **LAYOUT_DEFAULTS,
    )
    return fig


def sector_pie_chart(sector_alloc_df):
    """Pie chart of latest sector allocation."""
    if sector_alloc_df is None or sector_alloc_df.empty:
        return _empty_fig("No sector data")

    latest = sector_alloc_df.iloc[-1]
    latest = latest[latest > 0.005].sort_values(ascending=False)

    fig = go.Figure(go.Pie(
        labels=latest.index.tolist(),
        values=latest.values,
        textinfo="label+percent",
        hole=0.4,
    ))
    fig.update_layout(title="Sector Allocation (Latest)", **LAYOUT_DEFAULTS)
    return fig


def regime_performance_chart(perf_by_regime):
    if perf_by_regime is None or perf_by_regime.empty:
        return _empty_fig("No regime data")

    def parse_pct(s):
        try:
            return float(str(s).replace("%", ""))
        except:
            return 0.0

    regimes = perf_by_regime.index.tolist()
    ann_rets = [parse_pct(perf_by_regime.loc[r, "ann_return"]) for r in regimes]
    sharpes  = [parse_pct(perf_by_regime.loc[r, "sharpe"]) for r in regimes]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Ann. Return by Regime", "Sharpe by Regime"))

    colors_bar = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in ann_rets]
    fig.add_trace(go.Bar(x=regimes, y=ann_rets, marker_color=colors_bar,
                         text=[f"{v:.1f}%" for v in ann_rets], textposition="outside",
                         showlegend=False), row=1, col=1)

    colors_s = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in sharpes]
    fig.add_trace(go.Bar(x=regimes, y=sharpes, marker_color=colors_s,
                         text=[f"{v:.2f}" for v in sharpes], textposition="outside",
                         showlegend=False), row=1, col=2)

    fig.update_layout(title="Performance by Market Regime", **LAYOUT_DEFAULTS)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    return fig


def feature_importance_chart(importance):
    if importance is None:
        return _empty_fig("Feature importance requires LightGBM model")

    top_n = importance.head(15)
    fig = go.Figure(go.Bar(
        x=top_n.values, y=top_n.index, orientation="h",
        marker_color=COLORS["strategy"],
    ))
    fig.update_layout(title="Top Feature Importances (LightGBM gain)",
                      xaxis_title="Gain", yaxis=dict(autorange="reversed"),
                      showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def bootstrap_chart(bootstrap_df):
    if bootstrap_df is None or bootstrap_df.empty:
        return _empty_fig("No bootstrap data")

    def pf(s):
        try: return float(str(s))
        except: return 0.0

    metrics  = list(bootstrap_df.index)
    observed = [pf(bootstrap_df.loc[m, "observed"]) for m in metrics]
    p5       = [pf(bootstrap_df.loc[m, "p5"])       for m in metrics]
    p95      = [pf(bootstrap_df.loc[m, "p95"])      for m in metrics]

    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=[p5[i], p95[i]], y=[metric, metric],
            mode="lines", line=dict(color=COLORS["neutral"], width=6),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[observed[i]], y=[metric], mode="markers",
            marker=dict(color=COLORS["strategy"], size=12, symbol="diamond"),
            name=metric,
            hovertemplate=f"{metric}<br>Observed: {observed[i]:.3f}<br>"
                          f"90% CI: [{p5[i]:.3f}, {p95[i]:.3f}]<extra></extra>",
        ))

    fig.update_layout(title="Bootstrap Confidence Intervals (90% CI)",
                      xaxis_title="Value", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def oos_split_chart(oos_df):
    if oos_df is None or oos_df.empty:
        return _empty_fig("No OOS data")

    def parse_pct(s):
        try: return float(str(s).replace("%", ""))
        except: return 0.0

    periods = oos_df.index.tolist()
    metrics = ["Ann. Return", "Sharpe", "Max DD", "Win Rate"]
    titles = ["Ann. Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]

    fig = make_subplots(rows=1, cols=4, subplot_titles=titles)
    for col_idx, metric in enumerate(metrics):
        vals = [parse_pct(oos_df.loc[p, metric]) for p in periods]
        colors = [COLORS["strategy"] if p == "In-Sample" else COLORS["highlight"]
                  for p in periods]
        fig.add_trace(go.Bar(
            x=periods, y=vals, marker_color=colors,
            text=[f"{v:.2f}" for v in vals], textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx + 1)

    fig.update_layout(title="In-Sample vs Out-of-Sample", **LAYOUT_DEFAULTS)
    return fig


def factor_correlation_chart(corr_df):
    if corr_df is None or corr_df.empty or "error" in corr_df.columns:
        return _empty_fig("Factor correlation data unavailable")

    factors = corr_df.index.tolist()
    corrs = corr_df["Correlation"].values
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in corrs]

    fig = go.Figure(go.Bar(
        x=corrs, y=factors, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in corrs], textposition="outside",
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(title="Strategy Correlation to Known Factors",
                      xaxis_title="Correlation", yaxis=dict(autorange="reversed"),
                      showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def _empty_fig(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False)
    return fig


# ---------------------------------------------------------------------------
# Assemble full dashboard
# ---------------------------------------------------------------------------

def build_dashboard(
    result,
    benchmark_equity   = None,
    benchmark_returns  = None,
    perf_by_regime     = None,
    feature_importance = None,
    bootstrap_df       = None,
    oos_df             = None,
    factor_corr_df     = None,
    monthly_returns    = None,
    annual_df          = None,
    holdings_df        = None,
    sector_alloc_df    = None,
    initial_capital    = 100_000,
    strategy_name      = "ML-Ranked Concentrated Portfolio",
    output_path        = "results/dashboard.html",
    **kwargs,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required: pip install plotly")

    from metrics import (
        sharpe_ratio, sortino_ratio, annualized_return,
        max_drawdown, win_rate, information_ratio,
    )

    r     = result.daily_returns
    eq    = result.equity_curve
    bench = benchmark_returns.reindex(r.index).fillna(0) if benchmark_returns is not None \
            else pd.Series(0.0, index=r.index)

    ann_ret = annualized_return(r)
    spy_ret = annualized_return(bench)
    alpha = ann_ret - spy_ret

    final_value = eq.iloc[-1]
    total_ret = (final_value / initial_capital - 1) * 100

    kpis = {
        "CAGR":           f"{ann_ret*100:.1f}%",
        "Alpha vs SPY":   f"{alpha*100:+.1f}%",
        "Sharpe Ratio":   f"{sharpe_ratio(r):.2f}",
        "Max Drawdown":   f"{max_drawdown(eq)*100:.1f}%",
        "Total Return":   f"{total_ret:.0f}%",
        "Final Value":    f"${final_value:,.0f}",
    }

    # Build all charts
    charts = {
        "equity":       equity_curve_chart(result, benchmark_equity, initial_capital),
        "annual":       annual_returns_chart(annual_df),
        "monthly":      monthly_heatmap_chart(monthly_returns),
        "drawdown":     drawdown_chart(result),
        "rolling_sr":   rolling_sharpe_chart(result),
        "dist":         return_distribution_chart(result),
        "holdings":     holdings_chart(holdings_df),
        "sector":       sector_pie_chart(sector_alloc_df),
        "regime":       regime_performance_chart(perf_by_regime) if perf_by_regime is not None else None,
        "importance":   feature_importance_chart(feature_importance),
        "bootstrap":    bootstrap_chart(bootstrap_df) if bootstrap_df is not None else None,
        "oos_split":    oos_split_chart(oos_df) if oos_df is not None else None,
        "factor_corr":  factor_correlation_chart(factor_corr_df) if factor_corr_df is not None else None,
    }

    # KPI header HTML
    kpi_html = "".join(
        f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
        f'<div class="kpi-label">{k}</div></div>'
        for k, v in kpis.items()
    )

    def chart_div(fig, div_id, height=380):
        if fig is None:
            return ""
        return fig.to_html(
            full_html=False, include_plotlyjs=False,
            div_id=div_id, default_height=height,
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: "Inter", -apple-system, sans-serif;
    background: #F9FAFB;
    color: #111827;
    margin: 0;
    padding: 20px;
  }}
  .header {{
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 600; }}
  .header p  {{ margin: 0; font-size: 13px; opacity: 0.75; }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: white;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .kpi-value {{
    font-size: 22px;
    font-weight: 600;
    color: #2563EB;
    margin-bottom: 4px;
  }}
  .kpi-label {{
    font-size: 11px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .chart-grid {{
    display: grid;
    gap: 16px;
  }}
  .chart-full {{ grid-column: 1 / -1; }}
  .chart-grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .chart-wrap {{
    background: white;
    border-radius: 10px;
    padding: 8px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  @media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(3, 1fr); }}
    .chart-grid-2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <h1>{strategy_name}</h1>
    <p>ML-ranked concentrated long-only | {args_n_positions(holdings_df)} stocks | Monthly rebalance | ${initial_capital:,.0f} starting capital</p>
  </div>

  <div class="kpi-row">{kpi_html}</div>

  <div class="chart-grid">
    <!-- Wealth Growth -->
    <div class="chart-wrap chart-full">
      {chart_div(charts["equity"], "equity", 400)}
    </div>

    <!-- OOS Split -->
    {"" if charts.get("oos_split") is None else f'<div class="chart-wrap chart-full">{chart_div(charts["oos_split"], "oos_split", 320)}</div>'}

    <!-- Annual Returns + Monthly Heatmap -->
    <div class="chart-grid-2">
      <div class="chart-wrap">
        {chart_div(charts["annual"], "annual", 350)}
      </div>
      <div class="chart-wrap">
        {chart_div(charts["monthly"], "monthly", 350)}
      </div>
    </div>

    <!-- Drawdown + Rolling Sharpe -->
    <div class="chart-grid-2">
      <div class="chart-wrap">
        {chart_div(charts["drawdown"], "drawdown", 300)}
      </div>
      <div class="chart-wrap">
        {chart_div(charts["rolling_sr"], "rolling_sr", 300)}
      </div>
    </div>

    <!-- Holdings + Sector -->
    <div class="chart-grid-2">
      <div class="chart-wrap">
        {chart_div(charts["holdings"], "holdings", 400)}
      </div>
      <div class="chart-wrap">
        {chart_div(charts["sector"], "sector", 400)}
      </div>
    </div>

    <!-- Distribution + Feature Importance -->
    <div class="chart-grid-2">
      <div class="chart-wrap">
        {chart_div(charts["dist"], "dist", 300)}
      </div>
      <div class="chart-wrap">
        {chart_div(charts["importance"], "importance", 300)}
      </div>
    </div>

    <!-- Regime + Factor Correlation -->
    {"" if charts.get("regime") is None else f'<div class="chart-wrap chart-full">{chart_div(charts["regime"], "regime", 320)}</div>'}

    {"" if charts.get("factor_corr") is None else f'<div class="chart-wrap chart-full">{chart_div(charts["factor_corr"], "factor_corr", 300)}</div>'}

    <!-- Bootstrap -->
    {"" if charts.get("bootstrap") is None else f'<div class="chart-wrap chart-full">{chart_div(charts["bootstrap"], "bootstrap", 280)}</div>'}
  </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[dashboard] Saved -> {output_path}  ({len(html)//1024}KB)")
    return output_path


def args_n_positions(holdings_df):
    """Helper to show position count in header."""
    if holdings_df is not None and not holdings_df.empty:
        return len(holdings_df)
    return "~20"
