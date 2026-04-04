"""
dashboard_app.py
----------------
Quant Strategy Control Panel — Dash + Bootstrap dark theme.

Pages:
  1. Executive Overview
  2. Portfolio / Positions
  3. Performance Analytics
  4. Research & Signals
  5. Test Suite Results
  6. Time Travel (replay historical portfolio state)

Usage:
    python dashboard_app.py
    -> opens at http://localhost:8050
"""

import sys
from pathlib import Path

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

RESULTS = Path("results")

def _read(name, index_col=0):
    fp = RESULTS / name
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, index_col=index_col)

def _read_raw(name):
    fp = RESULTS / name
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp)

tearsheet       = _read("tearsheet.csv", index_col="Metric")
wealth          = _read("wealth_growth.csv", index_col=0)
holdings        = _read_raw("current_holdings.csv")
annual          = _read_raw("annual_returns.csv")
monthly         = _read_raw("monthly_returns.csv")
sector_alloc    = _read_raw("sector_allocation.csv")
regime_perf     = _read_raw("regime_performance.csv")
oos_tear        = _read_raw("oos_tearsheet.csv")
factor_decay    = _read_raw("factor_decay.csv")
feat_imp        = _read("feature_importance.csv", index_col=0)
factor_corr     = _read("factor_correlation.csv", index_col=0)
fama_french     = _read("fama_french.csv", index_col=0)
stress          = _read_raw("stress_test.csv")
bootstrap       = _read_raw("bootstrap_ci.csv")
test_results    = _read_raw("test_suite_results.csv")

# Weights history (for time-travel)
weights_hist = pd.DataFrame()
_wh_path = RESULTS / "weights_history.parquet"
if _wh_path.exists():
    weights_hist = pd.read_parquet(_wh_path)
    weights_hist.index = pd.to_datetime(weights_hist.index)

if not wealth.empty:
    wealth.index = pd.to_datetime(wealth.index)
    wealth.index.name = "date"

if not sector_alloc.empty and "date" in sector_alloc.columns:
    sector_alloc["date"] = pd.to_datetime(sector_alloc["date"])

if not test_results.empty:
    test_results = test_results.drop_duplicates(subset="test", keep="last")

# ─────────────────────────────────────────────────────────────────────────────
# Dark Slate Theme
# ─────────────────────────────────────────────────────────────────────────────

BG         = "#0F1117"
CARD_BG    = "#1A1D27"
SURFACE    = "#242836"
TEXT       = "#E8EAED"
TEXT_MUTED = "#8B8FA3"
ACCENT     = "#6C9FFF"
GREEN      = "#4ADE80"
RED        = "#F87171"
AMBER      = "#FBBF24"
BORDER     = "#2D3140"
FONT       = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# Dark plotly template
_tmpl = go.layout.Template(layout=go.Layout(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family=FONT, color=TEXT, size=12),
    xaxis=dict(gridcolor=SURFACE, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=TEXT_MUTED)),
    yaxis=dict(gridcolor=SURFACE, linecolor=BORDER, zeroline=False,
               tickfont=dict(color=TEXT_MUTED)),
    margin=dict(l=50, r=20, t=40, b=40),
    colorway=[ACCENT, "#9CA3AF", GREEN, RED, AMBER, "#A78BFA", "#F472B6"],
    legend=dict(font=dict(color=TEXT_MUTED)),
    title=dict(font=dict(color=TEXT, size=14)),
))
pio.templates["dark_custom"] = _tmpl
pio.templates.default = "dark_custom"

# Dark data-table styles
TBL_HEADER = {
    "backgroundColor": SURFACE, "fontWeight": "600",
    "borderBottom": f"2px solid {BORDER}", "fontSize": "12px",
    "color": TEXT, "textAlign": "left", "padding": "10px 12px",
}
TBL_CELL = {
    "fontSize": "13px", "color": TEXT, "textAlign": "left",
    "padding": "8px 12px", "borderBottom": f"1px solid {BORDER}",
    "fontFamily": FONT, "backgroundColor": CARD_BG,
}
TBL_COND = [
    {"if": {"row_index": "odd"}, "backgroundColor": SURFACE},
]

# ─────────────────────────────────────────────────────────────────────────────
# Reusable components
# ─────────────────────────────────────────────────────────────────────────────

def card(children, **kwargs):
    style = {
        "backgroundColor": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "16px",
        **kwargs.get("style", {}),
    }
    return html.Div(children, style=style)

def kpi_card(label, value, subtitle=None, color=None):
    val_color = color or TEXT
    children = [
        html.Div(label, style={
            "fontSize": "11px", "fontWeight": "600", "color": TEXT_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "4px",
        }),
        html.Div(value, style={
            "fontSize": "26px", "fontWeight": "600", "color": val_color,
            "lineHeight": "1.2",
        }),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={
            "fontSize": "11px", "color": TEXT_MUTED, "marginTop": "4px",
        }))
    return card(children, style={"textAlign": "center", "minWidth": "130px"})

def section_title(text):
    return html.H5(text, style={
        "fontWeight": "600", "color": TEXT, "marginBottom": "12px",
        "marginTop": "8px", "fontSize": "16px",
    })

def graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# Tearsheet helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts_val(metric):
    if tearsheet.empty or metric not in tearsheet.index:
        return "N/A"
    return str(tearsheet.loc[metric, "Value"])

def ts_num(metric):
    v = ts_val(metric)
    try:
        return float(v.replace("%", "").replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def build_overview():
    sharpe = ts_val("Sharpe Ratio")
    cagr = ts_val("CAGR")
    maxdd = ts_val("Max Drawdown")
    beta = ts_val("Beta to Market")
    sortino = ts_val("Sortino Ratio")
    win_rate = ts_val("Daily Win Rate")
    total_ret = ts_val("Total Return")
    costs = ts_val("Total Transaction Costs")

    oos_sharpe, oos_ret = "N/A", "N/A"
    if not oos_tear.empty and len(oos_tear) >= 2:
        oos_row = oos_tear.iloc[1]
        oos_sharpe = f"{oos_row.get('Sharpe', 'N/A')}"
        oos_ret = f"{oos_row.get('Ann. Return', 'N/A')}"

    sh_c = GREEN if ts_num("Sharpe Ratio") > 0.5 else (AMBER if ts_num("Sharpe Ratio") > 0 else RED)
    dd_c = RED if ts_num("Max Drawdown") < -20 else AMBER

    kpi1 = dbc.Row([
        dbc.Col(kpi_card("Sharpe Ratio", sharpe, "full period", sh_c), width=2),
        dbc.Col(kpi_card("CAGR", cagr, "annualized", ACCENT), width=2),
        dbc.Col(kpi_card("OOS Sharpe", oos_sharpe, "2024+", GREEN if oos_sharpe != "N/A" else None), width=2),
        dbc.Col(kpi_card("Max Drawdown", maxdd, "peak-to-trough", dd_c), width=2),
        dbc.Col(kpi_card("Beta", beta, "to market"), width=2),
        dbc.Col(kpi_card("Win Rate", win_rate, "daily"), width=2),
    ], className="g-3 mb-3")

    kpi2 = dbc.Row([
        dbc.Col(kpi_card("Total Return", total_ret), width=2),
        dbc.Col(kpi_card("Sortino", sortino), width=2),
        dbc.Col(kpi_card("OOS Return", oos_ret, "annualized"), width=2),
        dbc.Col(kpi_card("Costs", costs, "total txn"), width=2),
        dbc.Col(kpi_card("Positions", str(len(holdings)) if not holdings.empty else "0", "current"), width=2),
        dbc.Col(kpi_card("Calmar", ts_val("Calmar Ratio")), width=2),
    ], className="g-3 mb-3")

    # Equity curve
    eq_fig = go.Figure()
    if not wealth.empty:
        eq_fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth["Strategy"], name="Strategy",
            line=dict(color=ACCENT, width=2.5),
        ))
        eq_fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth["SPY Buy & Hold"], name="SPY",
            line=dict(color="#6B7280", width=1.5, dash="dash"),
        ))
        eq_fig.update_layout(title="Equity Curve", yaxis_title="Portfolio Value ($)",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                             height=400)

    # Drawdown
    dd_fig = go.Figure()
    if not wealth.empty:
        eq = wealth["Strategy"]
        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        dd_fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color=RED, width=1),
            fillcolor="rgba(248, 113, 113, 0.15)",
        ))
        dd_fig.update_layout(title="Underwater Drawdown", yaxis_title="Drawdown (%)", height=250)

    # Annual returns
    ann_fig = go.Figure()
    if not annual.empty:
        colors = [GREEN if v > 0 else RED for v in annual["Strategy"]]
        ann_fig.add_trace(go.Bar(x=annual["Year"], y=annual["Strategy"], name="Strategy",
                                 marker_color=colors, opacity=0.85))
        ann_fig.add_trace(go.Scatter(x=annual["Year"], y=annual["SPY"], name="SPY",
                                     line=dict(color="#6B7280", width=2, dash="dot"),
                                     mode="lines+markers", marker=dict(size=5)))
        ann_fig.update_layout(title="Annual Returns (%)", height=320, barmode="relative",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    return html.Div([
        kpi1, kpi2,
        dbc.Row([dbc.Col(card([graph(eq_fig)]), width=12)]),
        dbc.Row([
            dbc.Col(card([graph(dd_fig)]), width=6),
            dbc.Col(card([graph(ann_fig)]), width=6),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Page 2: Portfolio / Positions
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio():
    holdings_table = html.Div("No holdings data.", style={"color": TEXT_MUTED})
    if not holdings.empty:
        h = holdings.copy()
        h["weight_pct"] = (h["weight"] * 100).round(2)
        h["signal_score"] = h["signal_score"].round(4)
        h = h.sort_values("weight", ascending=False)
        holdings_table = dash_table.DataTable(
            data=h[["ticker", "weight_pct", "signal_score", "sector"]].to_dict("records"),
            columns=[
                {"name": "Ticker", "id": "ticker"},
                {"name": "Weight (%)", "id": "weight_pct", "type": "numeric"},
                {"name": "Signal", "id": "signal_score", "type": "numeric"},
                {"name": "Sector", "id": "sector"},
            ],
            style_table={"overflowX": "auto"},
            style_header=TBL_HEADER, style_cell=TBL_CELL,
            style_data_conditional=TBL_COND,
            sort_action="native", page_size=25,
        )

    # Sector donut
    sector_fig = go.Figure()
    if not holdings.empty:
        sw = holdings.groupby("sector")["weight"].sum().sort_values(ascending=False)
        sector_fig = go.Figure(data=[go.Pie(
            labels=sw.index, values=sw.values, hole=0.55, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Pastel),
            textfont=dict(size=11, color=TEXT),
        )])
        sector_fig.update_layout(title="Sector Allocation", height=380,
                                 showlegend=False, margin=dict(l=20, r=20, t=40, b=20))

    # Weight bars
    wt_fig = go.Figure()
    if not holdings.empty:
        hs = holdings.sort_values("weight", ascending=True)
        colors = [ACCENT if w > 0.05 else "#4A5568" for w in hs["weight"]]
        wt_fig.add_trace(go.Bar(y=hs["ticker"], x=hs["weight"]*100, orientation="h", marker_color=colors))
        wt_fig.update_layout(title="Position Weights (%)", height=max(350, len(holdings)*22),
                             xaxis_title="Weight (%)", margin=dict(l=60, r=20, t=40, b=30))

    # Sector over time
    st_fig = go.Figure()
    if not sector_alloc.empty:
        for s in [c for c in sector_alloc.columns if c != "date"]:
            if sector_alloc[s].sum() > 0:
                st_fig.add_trace(go.Scatter(x=sector_alloc["date"], y=sector_alloc[s]*100,
                                            name=s, stackgroup="one", mode="none"))
        st_fig.update_layout(title="Sector Allocation Over Time", yaxis_title="Allocation (%)",
                             height=380, legend=dict(font=dict(size=10), orientation="h", y=-0.15))

    return html.Div([
        dbc.Row([
            dbc.Col([section_title("Current Holdings"), card([holdings_table])], width=7),
            dbc.Col([card([graph(sector_fig)])], width=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([graph(wt_fig)]), width=5),
            dbc.Col(card([graph(st_fig)]), width=7),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Page 3: Performance Analytics
# ─────────────────────────────────────────────────────────────────────────────

def build_performance():
    # Heatmap
    hm_fig = go.Figure()
    if not monthly.empty:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        z = monthly[months].values.astype(float)
        hm_fig = go.Figure(data=go.Heatmap(
            z=z, x=months, y=monthly["Year"].astype(str),
            colorscale=[[0, RED], [0.5, "#1A1D27"], [1, GREEN]],
            zmid=0, text=np.where(np.isnan(z), "", z.round(1).astype(str)),
            texttemplate="%{text}%", textfont=dict(size=10, color=TEXT),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>",
        ))
        hm_fig.update_layout(title="Monthly Returns Heatmap (%)", height=380,
                             yaxis=dict(autorange="reversed"))

    # Regime
    reg_fig = go.Figure()
    if not regime_perf.empty:
        cm = {"bull_calm": GREEN, "bull_volatile": "#34D399",
              "bear_calm": AMBER, "bear_volatile": RED}
        rp = regime_perf.copy()
        rp["ann_return_num"] = rp["ann_return"].apply(
            lambda x: float(str(x).replace("%", "")) if pd.notna(x) else 0)
        reg_fig.add_trace(go.Bar(
            x=rp["regime"], y=rp["ann_return_num"],
            marker_color=[cm.get(r, ACCENT) for r in rp["regime"]],
            text=[f"{v:.1f}%" for v in rp["ann_return_num"]], textposition="outside",
            textfont=dict(color=TEXT),
        ))
        reg_fig.update_layout(title="Annualized Return by Regime", height=320,
                              yaxis_title="Ann. Return (%)")

    # IS vs OOS
    oos_fig = go.Figure()
    if not oos_tear.empty and len(oos_tear) >= 2:
        for m in ["Sharpe", "Ann. Return", "Ann. Vol", "Max DD"]:
            if m not in oos_tear.columns:
                continue
            try:
                is_v = float(str(oos_tear.iloc[0][m]).replace("%", ""))
                oos_v = float(str(oos_tear.iloc[1][m]).replace("%", ""))
                oos_fig.add_trace(go.Bar(x=[m], y=[is_v], name="IS" if m == "Sharpe" else None,
                                         marker_color="#4A5568", showlegend=(m == "Sharpe")))
                oos_fig.add_trace(go.Bar(x=[m], y=[oos_v], name="OOS" if m == "Sharpe" else None,
                                         marker_color=ACCENT, showlegend=(m == "Sharpe")))
            except (ValueError, TypeError):
                pass
        oos_fig.update_layout(title="In-Sample vs Out-of-Sample", height=320, barmode="group",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    # Stress
    stress_tbl = html.Div()
    if not stress.empty:
        stress_tbl = dash_table.DataTable(
            data=stress.to_dict("records"),
            columns=[{"name": c, "id": c} for c in stress.columns],
            style_header=TBL_HEADER, style_cell=TBL_CELL,
        )

    return html.Div([
        dbc.Row([
            dbc.Col(card([graph(hm_fig)]), width=7),
            dbc.Col(card([graph(reg_fig)]), width=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([graph(oos_fig)]), width=6),
            dbc.Col([section_title("Stress Tests"), card([stress_tbl])], width=6),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Page 4: Research & Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_research():
    # IC decay
    ic_fig = go.Figure()
    if not factor_decay.empty:
        fd = factor_decay.copy()
        fd.columns = [c.strip() for c in fd.columns]
        idx_col = fd.columns[0]
        labels = [h.replace("_IC", "") for h in fd[idx_col].tolist()]
        ics = fd["mean_IC"].values
        ic_fig.add_trace(go.Bar(x=labels, y=ics,
                                marker_color=[GREEN if v > 0 else RED for v in ics],
                                text=[f"{v:.4f}" for v in ics], textposition="outside",
                                textfont=dict(color=TEXT)))
        ic_fig.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED)
        ic_fig.update_layout(title="Signal IC by Horizon", height=320, yaxis_title="Mean IC")

    # Feature importance
    fi_fig = go.Figure()
    if not feat_imp.empty:
        fi = feat_imp.head(20).sort_values(feat_imp.columns[0], ascending=True)
        fi_fig.add_trace(go.Bar(y=fi.index, x=fi.iloc[:, 0], orientation="h",
                                marker_color=ACCENT, opacity=0.85))
        fi_fig.update_layout(title="Top 20 Feature Importance", height=500,
                             margin=dict(l=150, r=20, t=40, b=30))

    # Factor correlation
    fc_fig = go.Figure()
    if not factor_corr.empty:
        fc = factor_corr.sort_values("Correlation", ascending=True)
        fc_fig.add_trace(go.Bar(y=fc.index, x=fc["Correlation"], orientation="h",
                                marker_color=[GREEN if v > 0 else RED for v in fc["Correlation"]]))
        fc_fig.update_layout(title="Factor Correlations", height=300,
                             margin=dict(l=150, r=20, t=40, b=30), xaxis_title="Correlation")

    # FF table
    ff_tbl = html.Div()
    if not fama_french.empty:
        ff = fama_french.reset_index()
        ff.columns = [str(c) for c in ff.columns]
        for col in ff.columns[1:]:
            try:
                ff[col] = ff[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) and x != "" else "")
            except (ValueError, TypeError):
                pass
        ff_tbl = dash_table.DataTable(
            data=ff.to_dict("records"),
            columns=[{"name": c, "id": c} for c in ff.columns],
            style_header=TBL_HEADER, style_cell=TBL_CELL,
        )

    # Bootstrap CI
    bs_fig = go.Figure()
    if not bootstrap.empty:
        for _, row in bootstrap.iterrows():
            m = row["metric"]
            bs_fig.add_trace(go.Scatter(
                x=[float(row["p5"]), float(row["observed"]), float(row["p95"])],
                y=[m, m, m], mode="markers+lines",
                marker=dict(size=[8, 14, 8], color=[TEXT_MUTED, ACCENT, TEXT_MUTED]),
                line=dict(color=TEXT_MUTED, width=2), showlegend=False,
            ))
        bs_fig.update_layout(title="Bootstrap CI (5th / Obs / 95th)", height=220,
                             margin=dict(l=120, r=20, t=40, b=30))

    return html.Div([
        dbc.Row([
            dbc.Col(card([graph(ic_fig)]), width=5),
            dbc.Col(card([graph(fc_fig)]), width=4),
            dbc.Col(card([graph(bs_fig)]), width=3),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(card([graph(fi_fig)]), width=6),
            dbc.Col([section_title("Fama-French 5+Mom Regression"), card([ff_tbl])], width=6),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Page 5: Test Suite Results
# ─────────────────────────────────────────────────────────────────────────────

def build_test_suite():
    if test_results.empty:
        return card([html.P("No test suite results found.", style={"color": TEXT_MUTED})])

    tr = test_results.copy().sort_values("sharpe", ascending=False)
    tr["category"] = tr["test"].str.extract(r"^([A-Z])\d")[0]
    cat_labels = {"B": "ML/Momentum Blend", "C": "Portfolio Construction",
                  "A": "Holding Period", "D": "Hyperparameters"}
    tr["category_label"] = tr["category"].map(cat_labels).fillna("Other")

    baseline_mean = tr[tr["test"].str.contains("baseline", na=False)]["sharpe"].mean()

    # Sharpe bars
    s_fig = go.Figure()
    ts = tr.sort_values("sharpe", ascending=True)
    colors = [AMBER if "baseline" in t else (GREEN if s > baseline_mean else "#4A5568")
              for t, s in zip(ts["test"], ts["sharpe"])]
    s_fig.add_trace(go.Bar(
        y=ts["test"], x=ts["sharpe"], orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in ts["sharpe"]], textposition="outside",
        textfont=dict(size=10, color=TEXT),
    ))
    s_fig.update_layout(title="Sharpe by Config", height=max(400, len(tr)*25),
                        margin=dict(l=200, r=60, t=40, b=30), xaxis_title="Sharpe")

    # Scatter
    sc_fig = go.Figure()
    for cat in tr["category_label"].unique():
        sub = tr[tr["category_label"] == cat]
        sc_fig.add_trace(go.Scatter(
            x=sub["ann_vol"]*100 if "ann_vol" in sub.columns else [0],
            y=sub["ann_return"]*100, mode="markers+text",
            text=sub["test"], textposition="top center", textfont=dict(size=9),
            name=cat, marker=dict(size=10),
        ))
    sc_fig.update_layout(title="Return vs Volatility", height=400,
                         xaxis_title="Ann. Volatility (%)", yaxis_title="Ann. Return (%)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    # Results table
    disp = ["test", "sharpe", "ann_return", "sortino", "max_dd", "total_return", "final_nav"]
    avail = [c for c in disp if c in tr.columns]
    td = tr[avail].copy()
    for c in avail:
        if c in ["ann_return", "max_dd", "total_return"]:
            td[c] = td[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
        elif c in ["sharpe", "sortino"]:
            td[c] = td[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        elif c == "final_nav":
            td[c] = td[c].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    tbl = dash_table.DataTable(
        data=td.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in avail],
        style_header=TBL_HEADER, style_cell=TBL_CELL,
        style_data_conditional=TBL_COND + [
            {"if": {"filter_query": '{test} contains "baseline"'},
             "backgroundColor": "#2D2A1A"},
        ],
        sort_action="native", page_size=25,
    )

    return html.Div([
        dbc.Row([
            dbc.Col(card([graph(s_fig)]), width=6),
            dbc.Col(card([graph(sc_fig)]), width=6),
        ], className="g-3"),
        dbc.Row([dbc.Col([section_title("All Results"), card([tbl])], width=12)]),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Page 6: Time Travel
# ─────────────────────────────────────────────────────────────────────────────

def build_time_travel_layout():
    """Static layout with a date slider — callbacks fill the content."""
    if wealth.empty:
        return card([html.P("No wealth data available.", style={"color": TEXT_MUTED})])

    dates = wealth.index
    min_date = dates.min()
    max_date = dates.max()

    # Date marks for the slider (yearly)
    years = sorted(set(d.year for d in dates))
    marks = {}
    for y in years:
        idx = dates[dates.year == y]
        if len(idx) > 0:
            pos = dates.get_loc(idx[0])
            marks[int(pos)] = {"label": str(y), "style": {"color": TEXT_MUTED, "fontSize": "11px"}}

    return html.Div([
        dbc.Row([
            dbc.Col([
                section_title("Time Travel — Portfolio Replay"),
                html.Div([
                    html.Div("Select a date to see the portfolio state at that point in time.",
                             style={"color": TEXT_MUTED, "fontSize": "13px", "marginBottom": "12px"}),
                    dcc.Slider(
                        id="tt-slider",
                        min=0, max=len(dates) - 1,
                        value=len(dates) - 1,
                        marks=marks,
                        tooltip={"placement": "top", "always_visible": False},
                        updatemode="mouseup",
                    ),
                    html.Div(id="tt-date-label", style={
                        "textAlign": "center", "fontSize": "20px", "fontWeight": "600",
                        "color": ACCENT, "marginTop": "8px", "marginBottom": "16px",
                    }),
                ]),
            ], width=12),
        ]),
        # KPIs that update
        html.Div(id="tt-kpis"),
        # Charts row
        dbc.Row([
            dbc.Col(html.Div(id="tt-equity"), width=7),
            dbc.Col(html.Div(id="tt-sector"), width=5),
        ], className="g-3"),
        # Holdings row
        dbc.Row([
            dbc.Col(html.Div(id="tt-holdings"), width=12),
        ]),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "Quant Strategy Dashboard"

nav = dbc.Tabs(
    id="tabs", active_tab="overview",
    children=[
        dbc.Tab(label="Overview",    tab_id="overview"),
        dbc.Tab(label="Portfolio",   tab_id="portfolio"),
        dbc.Tab(label="Performance", tab_id="performance"),
        dbc.Tab(label="Research",    tab_id="research"),
        dbc.Tab(label="Test Suite",  tab_id="tests"),
        dbc.Tab(label="Time Travel", tab_id="timetravel"),
    ],
    style={"marginBottom": "20px"},
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H4("Quant Strategy Dashboard", style={
            "fontWeight": "700", "color": TEXT, "margin": 0, "fontSize": "22px",
            "display": "inline-block",
        }),
        html.Span("ML-Ranked Concentrated Long-Only", style={
            "color": TEXT_MUTED, "fontSize": "13px", "marginLeft": "16px",
        }),
    ], style={
        "backgroundColor": CARD_BG, "padding": "16px 32px",
        "borderBottom": f"1px solid {BORDER}", "marginBottom": "20px",
    }),
    # Content
    html.Div([nav, html.Div(id="page-content")], style={
        "maxWidth": "1600px", "margin": "0 auto", "padding": "0 24px 40px",
    }),
], style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": FONT})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("page-content", "children"), Input("tabs", "active_tab"))
def render_page(tab):
    if tab == "overview":    return build_overview()
    if tab == "portfolio":   return build_portfolio()
    if tab == "performance": return build_performance()
    if tab == "research":    return build_research()
    if tab == "tests":       return build_test_suite()
    if tab == "timetravel":  return build_time_travel_layout()
    return html.Div("Select a tab.")


@callback(
    Output("tt-date-label", "children"),
    Output("tt-kpis", "children"),
    Output("tt-equity", "children"),
    Output("tt-sector", "children"),
    Output("tt-holdings", "children"),
    Input("tt-slider", "value"),
)
def update_time_travel(slider_val):
    if wealth.empty:
        empty = html.Div()
        return "N/A", empty, empty, empty, empty

    dates = wealth.index
    as_of = dates[min(slider_val, len(dates) - 1)]
    as_of_str = as_of.strftime("%B %d, %Y")

    # --- KPIs up to as_of ---
    w_slice = wealth.loc[:as_of]
    strat = w_slice["Strategy"]
    spy = w_slice["SPY Buy & Hold"]

    total_ret = (strat.iloc[-1] / strat.iloc[0] - 1) * 100 if len(strat) > 1 else 0
    spy_ret = (spy.iloc[-1] / spy.iloc[0] - 1) * 100 if len(spy) > 1 else 0
    nav = strat.iloc[-1]
    dd_series = (strat - strat.expanding().max()) / strat.expanding().max()
    max_dd = dd_series.min() * 100

    # Rolling Sharpe (252d)
    daily_ret = strat.pct_change().dropna()
    if len(daily_ret) > 60:
        rolling_sharpe = daily_ret.rolling(252, min_periods=60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True
        ).iloc[-1]
    else:
        rolling_sharpe = 0

    n_days = (as_of - dates[0]).days

    ret_color = GREEN if total_ret > 0 else RED
    kpis = dbc.Row([
        dbc.Col(kpi_card("NAV", f"${nav:,.0f}", f"as of {as_of.strftime('%Y-%m-%d')}"), width=2),
        dbc.Col(kpi_card("Total Return", f"{total_ret:+.1f}%", "strategy", ret_color), width=2),
        dbc.Col(kpi_card("SPY Return", f"{spy_ret:+.1f}%", "benchmark"), width=2),
        dbc.Col(kpi_card("Alpha", f"{total_ret - spy_ret:+.1f}%", "vs SPY",
                         GREEN if total_ret > spy_ret else RED), width=2),
        dbc.Col(kpi_card("Max Drawdown", f"{max_dd:.1f}%", "peak-to-trough", RED), width=2),
        dbc.Col(kpi_card("Rolling Sharpe", f"{rolling_sharpe:.2f}", "252d",
                         GREEN if rolling_sharpe > 0.5 else AMBER), width=2),
    ], className="g-3 mb-3")

    # --- Equity curve up to as_of ---
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=w_slice.index, y=w_slice["Strategy"], name="Strategy",
                                line=dict(color=ACCENT, width=2.5)))
    eq_fig.add_trace(go.Scatter(x=w_slice.index, y=w_slice["SPY Buy & Hold"], name="SPY",
                                line=dict(color="#6B7280", width=1.5, dash="dash")))
    # Vertical line at as_of
    eq_fig.add_vline(x=as_of, line_dash="dot", line_color=AMBER, line_width=1)
    eq_fig.update_layout(title=f"Equity Curve (through {as_of.strftime('%Y-%m-%d')})",
                         yaxis_title="$", height=380,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    # --- Sector allocation at as_of ---
    sec_fig = go.Figure()
    if not sector_alloc.empty:
        row = sector_alloc[sector_alloc["date"] <= as_of]
        if not row.empty:
            row = row.iloc[-1]
            sectors = {c: row[c] for c in sector_alloc.columns
                       if c != "date" and pd.notna(row[c]) and row[c] > 0.001}
            if sectors:
                sec_fig = go.Figure(data=[go.Pie(
                    labels=list(sectors.keys()), values=list(sectors.values()),
                    hole=0.55, textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    textfont=dict(size=11, color=TEXT),
                )])
                sec_fig.update_layout(title=f"Sector Allocation", height=380,
                                      showlegend=False, margin=dict(l=20, r=20, t=40, b=20))

    # --- Holdings at as_of (from weights_history if available) ---
    holdings_content = html.Div()
    if not weights_hist.empty:
        # Find the closest date <= as_of
        valid_dates = weights_hist.index[weights_hist.index <= as_of]
        if len(valid_dates) > 0:
            snap_date = valid_dates[-1]
            snap = weights_hist.loc[snap_date].dropna()
            snap = snap[snap > 0.001].sort_values(ascending=False)

            if len(snap) > 0:
                snap_df = pd.DataFrame({
                    "ticker": snap.index,
                    "weight_pct": (snap.values * 100).round(2),
                })

                holdings_content = card([
                    section_title(f"Holdings on {snap_date.strftime('%Y-%m-%d')} ({len(snap)} positions)"),
                    dash_table.DataTable(
                        data=snap_df.to_dict("records"),
                        columns=[
                            {"name": "Ticker", "id": "ticker"},
                            {"name": "Weight (%)", "id": "weight_pct", "type": "numeric"},
                        ],
                        style_header=TBL_HEADER, style_cell=TBL_CELL,
                        style_data_conditional=TBL_COND,
                        sort_action="native", page_size=30,
                    ),
                ])
    else:
        holdings_content = card([
            html.Div([
                html.Div("Position-level time travel requires weights_history.parquet",
                         style={"color": AMBER, "fontWeight": "600", "marginBottom": "4px"}),
                html.Div("Run the strategy once with the updated run_strategy.py to generate it. "
                         "Sector-level and equity data are available now.",
                         style={"color": TEXT_MUTED, "fontSize": "13px"}),
            ], style={"padding": "12px"}),
        ])

    return (as_of_str,
            kpis,
            card([graph(eq_fig)]),
            card([graph(sec_fig)]),
            holdings_content)


if __name__ == "__main__":
    print("\n  Dashboard starting at http://localhost:8050\n")
    app.run(debug=False, port=8050)
