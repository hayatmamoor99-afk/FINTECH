"""
StockLens - AI-Powered Fintech Stock Analyzer
=============================================
Requirements:
    pip install streamlit yfinance plotly pandas numpy anthropic

Run:
    streamlit run stocklens_app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import anthropic
import os

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockLens — AI Market Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1e2d45;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label { color: #94a3b8; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
        width: 100%;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85; border: none; }

    /* Text input */
    .stTextInput > div > div > input {
        background: #0a0e1a;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 600;
    }
    .stTextInput > div > div > input:focus { border-color: #3b82f6; }

    /* Section headers */
    .section-header {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #1e2d45;
    }

    /* Verdict cards */
    .verdict-buy {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 12px;
        padding: 20px;
    }
    .verdict-sell {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 20px;
    }
    .verdict-hold {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 12px;
        padding: 20px;
    }

    /* Info boxes */
    .info-box {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .info-title {
        font-size: 12px;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 4px;
    }
    .info-body {
        font-size: 11px;
        color: #64748b;
        line-height: 1.6;
    }

    /* Investment cards */
    .inv-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .inv-period { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }
    .inv-value  { font-size: 22px; font-weight: 800; margin-top: 8px; }
    .inv-gain   { font-size: 13px; font-weight: 600; margin-top: 4px; }
    .inv-pct    { font-size: 11px; margin-top: 2px; }

    /* Disclaimer */
    .disclaimer {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 11px;
        color: #92400e;
        margin-top: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    """Fetch all stock data from yfinance."""
    tk = yf.Ticker(ticker)
    info = tk.info

    end = datetime.today()
    start_5yr = end - timedelta(days=5 * 365)

    hist = tk.history(start=start_5yr, end=end, interval="1mo")
    if hist.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol and try again.")

    hist = hist[["Close"]].dropna()

    periods = {
        "5yr": hist,
        "4yr": hist[hist.index >= end - timedelta(days=4 * 365)],
        "3yr": hist[hist.index >= end - timedelta(days=3 * 365)],
        "2yr": hist[hist.index >= end - timedelta(days=2 * 365)],
        "1yr": hist[hist.index >= end - timedelta(days=365)],
    }

    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
    prev_close    = info.get("previousClose") or info.get("regularMarketPreviousClose") or hist["Close"].iloc[-2]

    return {
        "ticker":        ticker.upper(),
        "company_name":  info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(current_price), 2),
        "prev_close":    round(float(prev_close), 2),
        "market_cap":    info.get("marketCap"),
        "pe_ratio":      info.get("trailingPE") or info.get("forwardPE"),
        "week52_high":   info.get("fiftyTwoWeekHigh"),
        "week52_low":    info.get("fiftyTwoWeekLow"),
        "avg_volume":    info.get("averageVolume"),
        "dividend":      info.get("dividendYield"),
        "sector":        info.get("sector", "N/A"),
        "ma50":          info.get("fiftyDayAverage"),
        "ma200":         info.get("twoHundredDayAverage"),
        "periods":       periods,
        "full_hist":     hist,
    }


def compute_technicals(hist: pd.DataFrame) -> dict:
    """Compute RSI, trend, volatility from price history."""
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi": 50, "trend": "Neutral", "volatility": "Unknown", "vol_pct": 0}

    # RSI (14-period simplified)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g  = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_l  = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi    = 100 - (100 / (1 + avg_g / avg_l)) if avg_l != 0 else 50

    # Trend
    trend_pct = (closes[-1] - closes[0]) / closes[0] * 100
    trend     = "Strong Uptrend" if trend_pct > 20 else \
                "Uptrend"        if trend_pct > 5  else \
                "Strong Downtrend" if trend_pct < -20 else \
                "Downtrend"      if trend_pct < -5 else "Sideways"

    # Volatility (annualised std of returns)
    returns  = np.diff(closes) / closes[:-1]
    vol_pct  = float(np.std(returns) * np.sqrt(12) * 100)  # monthly → annual
    volatility = "Very High" if vol_pct > 60 else \
                 "High"      if vol_pct > 35 else \
                 "Medium"    if vol_pct > 15 else "Low"

    return {
        "rsi":        round(float(rsi), 1),
        "trend":      trend,
        "trend_pct":  round(float(trend_pct), 2),
        "volatility": volatility,
        "vol_pct":    round(vol_pct, 1),
    }


def compute_recommendation(tech: dict, data: dict) -> dict:
    """Simple scoring model → BUY / HOLD / SELL with confidence."""
    score = 50  # neutral baseline

    rsi   = tech["rsi"]
    trend = tech["trend"]

    if rsi < 30:   score += 20
    elif rsi < 45: score += 10
    elif rsi > 70: score -= 20
    elif rsi > 55: score -= 5

    if "Strong Uptrend" in trend:   score += 20
    elif "Uptrend" in trend:        score += 10
    elif "Strong Downtrend" in trend: score -= 20
    elif "Downtrend" in trend:      score -= 10

    # Price vs MA50
    cp  = data["current_price"]
    m50 = data.get("ma50") or cp
    m200 = data.get("ma200") or cp
    if cp > m50:  score += 8
    else:          score -= 8
    if cp > m200: score += 8
    else:          score -= 8

    score = max(0, min(100, score))

    if score >= 62:
        rec = "BUY"
        color = "#10b981"
        icon  = "🚀"
    elif score <= 38:
        rec = "SELL"
        color = "#ef4444"
        icon  = "⛔"
    else:
        rec = "HOLD"
        color = "#f59e0b"
        icon  = "⏸️"

    hist_5yr = data["periods"]["5yr"]["Close"].values
    support    = round(float(np.percentile(hist_5yr, 15)), 2)
    resistance = round(float(np.percentile(hist_5yr, 85)), 2)

    return {
        "rec":        rec,
        "confidence": score,
        "color":      color,
        "icon":       icon,
        "support":    support,
        "resistance": resistance,
    }


def simulate_investment(hist: pd.DataFrame, amount: float = 100) -> dict:
    """Calculate hypothetical investment returns."""
    closes = hist["Close"].dropna().values.astype(float)
    if len(closes) < 2:
        return {"value": amount, "gain": 0, "pct": 0}
    shares = amount / closes[0]
    value  = round(shares * closes[-1], 2)
    gain   = round(value - amount, 2)
    pct    = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
    return {"value": value, "gain": gain, "pct": pct}


def format_large(n) -> str:
    if n is None: return "N/A"
    n = float(n)
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def get_ai_analysis(data: dict, tech: dict, rec: dict) -> str:
    """Call Claude API for a professional stock analysis paragraph."""
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return "💡 Add your Anthropic API key in the sidebar to unlock AI-generated analysis."

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        inv_5yr  = simulate_investment(data["periods"]["5yr"])
        inv_1yr  = simulate_investment(data["periods"]["1yr"])
        prompt   = f"""You are a senior equity analyst. Write a concise 3-sentence professional analysis for {data['ticker']} ({data['company_name']}).

Key data:
- Current price: ${data['current_price']}
- RSI: {tech['rsi']}
- Trend: {tech['trend']} ({tech['trend_pct']}% over 5yr period)
- Volatility: {tech['volatility']} ({tech['vol_pct']}% annualised)
- Recommendation: {rec['rec']} with {rec['confidence']}% confidence
- $100 invested 5 years ago would now be ${inv_5yr['value']} ({inv_5yr['pct']}%)
- $100 invested 1 year ago would now be ${inv_1yr['value']} ({inv_1yr['pct']}%)
- Support: ${rec['support']}, Resistance: ${rec['resistance']}

Write 3 punchy sentences: (1) overall performance narrative, (2) current technical picture, (3) forward-looking outlook.
Keep it factual, professional, and data-driven. End with a one-line risk reminder."""

        msg = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 300,
            messages   = [{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI analysis unavailable: {e}"


def make_chart(hist: pd.DataFrame, ticker: str, period_label: str) -> go.Figure:
    """Build a beautiful Plotly price chart."""
    closes = hist["Close"].dropna()
    dates  = closes.index

    is_up   = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color   = "#10b981" if is_up else "#ef4444"
    fill_c  = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        fill="tozeroy",
        fillcolor=fill_c,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>",
        name=ticker,
    ))

    # Moving average
    if len(closes) >= 6:
        ma = closes.rolling(3).mean()
        fig.add_trace(go.Scatter(
            x=dates, y=ma,
            line=dict(color="rgba(139,92,246,0.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip",
        ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {period_label} Price History", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(
            gridcolor="#1e2d45", showgrid=True, zeroline=False,
            tickfont=dict(size=10), showline=False,
        ),
        yaxis=dict(
            gridcolor="#1e2d45", showgrid=True, zeroline=False,
            tickprefix="$", tickfont=dict(size=10), showline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#94a3b8"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")),
        height=320,
    )
    return fig


# ─── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
        <div style="font-size:32px">📡</div>
        <div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">StockLens</div>
        <div style="font-size:11px;color:#64748b">AI-Powered Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    ticker_input = st.text_input(
        "Stock Ticker",
        placeholder="e.g. AAPL, TSLA, MSFT...",
        label_visibility="collapsed",
        key="ticker_input",
    ).upper().strip()

    analyze_btn = st.button("🔍 Analyze Stock", use_container_width=True)

    st.markdown('<div class="section-header">⚡ Quick Picks</div>', unsafe_allow_html=True)
    qp_cols = st.columns(2)
    quick_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM"]
    for i, qt in enumerate(quick_tickers):
        with qp_cols[i % 2]:
            if st.button(qt, key=f"qp_{qt}"):
                st.session_state["selected_ticker"] = qt
                st.rerun()

    st.markdown('<div class="section-header">🔑 API Key (Optional)</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed",
        help="Required for AI-generated analysis paragraphs",
    )
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown('<div class="section-header">📚 Beginner\'s Guide</div>', unsafe_allow_html=True)
    guides = [
        ("📊 Histogram", "A line chart showing stock price over time. Rising = stock gaining value. The filled area shows overall trend direction."),
        ("💵 $100 Sim", "If you invested $100 at the start of the period, this is what it'd be worth today — accounting for every price movement."),
        ("📈 RSI", "Relative Strength Index (0–100). Below 30 = oversold (buy signal). Above 70 = overbought (sell signal). ~50 = neutral."),
        ("⚡ Volatility", "How wildly the stock swings. High vol = bigger gains OR losses. Low vol = stable, predictable movement."),
        ("🤖 AI Verdict", "Based on trend, RSI, moving averages & scoring model. BUY/HOLD/SELL with a confidence %. Not financial advice."),
        ("⚠️ Risk", "Past performance ≠ future results. Always consult a licensed financial advisor before making investment decisions."),
    ]
    for title, body in guides:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-title">{title}</div>
            <div class="info-body">{body}</div>
        </div>""", unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────────

# Resolve which ticker to analyze
final_ticker = st.session_state.get("selected_ticker") or ticker_input
if "selected_ticker" in st.session_state:
    del st.session_state["selected_ticker"]

if not final_ticker and not analyze_btn:
    # Hero screen
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:60px;margin-bottom:16px">📡</div>
        <div style="font-size:32px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px">
            AI Stock Intelligence
        </div>
        <div style="color:#64748b;font-size:15px;max-width:500px;margin:0 auto 24px">
            Enter any stock ticker to get 5-year price charts, investment simulations,
            and AI-powered buy/sell recommendations — powered by real yfinance data.
        </div>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
            <span style="background:rgba(59,130,246,.15);color:#60a5fa;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600">📊 5-Year Charts</span>
            <span style="background:rgba(139,92,246,.15);color:#a78bfa;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600">💰 $100 Simulator</span>
            <span style="background:rgba(16,185,129,.15);color:#34d399;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600">🤖 AI Analysis</span>
            <span style="background:rgba(245,158,11,.15);color:#fbbf24;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600">⚡ Real yfinance Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Load Data ───────────────────────────────────────────────────────────────────

with st.spinner(f"Fetching real market data for **{final_ticker}**..."):
    try:
        data = get_stock_data(final_ticker)
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()

tech = compute_technicals(data["periods"]["5yr"])
rec  = compute_recommendation(tech, data)

# ─── Stock Header ─────────────────────────────────────────────────────────────────

change     = data["current_price"] - data["prev_close"]
change_pct = (change / data["prev_close"] * 100) if data["prev_close"] else 0
is_up      = change >= 0
arrow      = "▲" if is_up else "▼"
delta_color = "#10b981" if is_up else "#ef4444"

col_name, col_price = st.columns([2, 1])
with col_name:
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span style="font-size:34px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:12px;font-size:13px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:12px;color:#64748b">Sector: {data['sector']}</div>
    """, unsafe_allow_html=True)
with col_price:
    st.markdown(f"""
    <div style="text-align:right">
        <div style="font-size:34px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:14px;font-weight:600;color:{delta_color}">
            {arrow} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

# Metrics row
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8 = st.columns(8)
mets = [
    ("Market Cap",  format_large(data["market_cap"])),
    ("P/E Ratio",   f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High",    f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low",     f"${data['week52_low']:.2f}"  if data["week52_low"] else "N/A"),
    ("Avg Volume",  format_large(data["avg_volume"])),
    ("MA 50",       f"${data['ma50']:.2f}" if data["ma50"] else "N/A"),
    ("MA 200",      f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)",    str(tech["rsi"])),
]
for col, (label, val) in zip([mc1,mc2,mc3,mc4,mc5,mc6,mc7,mc8], mets):
    col.metric(label, val)

# ─── Price Charts ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Historical Price Histograms</div>', unsafe_allow_html=True)

period_tabs = st.tabs(["📅 5 Years", "📅 4 Years", "📅 3 Years", "📅 2 Years", "📅 1 Year"])
period_map  = [("5yr", "5 Year"), ("4yr", "4 Year"), ("3yr", "3 Year"), ("2yr", "2 Year"), ("1yr", "1 Year")]

for tab, (key, label) in zip(period_tabs, period_map):
    with tab:
        hist = data["periods"][key]
        if hist.empty:
            st.warning(f"Not enough data for {label} period.")
        else:
            st.plotly_chart(make_chart(hist, data["ticker"], label), use_container_width=True)

# ─── $100 Investment Simulator ────────────────────────────────────────────────────

st.markdown('<div class="section-header">💰 $100 Investment Simulator</div>', unsafe_allow_html=True)

inv_cols = st.columns(5)
for col, (key, label) in zip(inv_cols, period_map):
    inv = simulate_investment(data["periods"][key])
    pos = inv["gain"] >= 0
    color = "#10b981" if pos else "#ef4444"
    arrow = "▲" if pos else "▼"
    col.markdown(f"""
    <div class="inv-card">
        <div class="inv-period">{label}</div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Started $100</div>
        <div class="inv-value" style="color:{color}">${inv['value']:,.2f}</div>
        <div class="inv-gain" style="color:{color}">{'+' if pos else ''}${inv['gain']:,.2f}</div>
        <div class="inv-pct" style="color:{color}">{arrow} {abs(inv['pct']):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ─── AI Recommendation ────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)

verdict_class = {"BUY": "verdict-buy", "SELL": "verdict-sell", "HOLD": "verdict-hold"}[rec["rec"]]

v_left, v_right = st.columns([2, 1])

with v_left:
    conf_bar = "█" * int(rec["confidence"] / 5) + "░" * (20 - int(rec["confidence"] / 5))
    st.markdown(f"""
    <div class="{verdict_class}">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
            <div style="font-size:40px">{rec['icon']}</div>
            <div>
                <div style="font-size:11px;font-weight:700;color:{rec['color']};text-transform:uppercase;letter-spacing:1px">AI Verdict</div>
                <div style="font-size:30px;font-weight:900;color:{rec['color']}">{rec['rec']}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div style="font-size:11px;color:#64748b">Confidence</div>
                <div style="font-size:30px;font-weight:900;color:{rec['color']}">{rec['confidence']}%</div>
            </div>
        </div>
        <div style="font-size:10px;color:#64748b;font-family:monospace;letter-spacing:1px;margin-bottom:4px">
            {conf_bar}
        </div>
        <div style="font-size:11px;color:#64748b">Signal confidence based on RSI, trend, moving averages & probability model</div>
    </div>
    """, unsafe_allow_html=True)

with v_right:
    signals = [
        ("Trend",        tech["trend"],               "#e2e8f0"),
        ("Momentum",     "Strong" if tech["rsi"] > 55 else "Weak" if tech["rsi"] < 45 else "Neutral", "#e2e8f0"),
        ("RSI",          str(tech["rsi"]),             "#10b981" if tech["rsi"] < 30 else "#ef4444" if tech["rsi"] > 70 else "#e2e8f0"),
        ("Volatility",   tech["volatility"],           "#e2e8f0"),
        ("Support",      f"${rec['support']:,.2f}",   "#10b981"),
        ("Resistance",   f"${rec['resistance']:,.2f}", "#ef4444"),
    ]
    for label, val, color in signals:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;
            padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{label}</span>
            <span style="font-size:13px;font-weight:700;color:{color}">{val}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── AI Analysis ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📝 AI-Generated Analysis</div>', unsafe_allow_html=True)

with st.spinner("Generating AI analysis..."):
    analysis = get_ai_analysis(data, tech, rec)

st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:20px">
    <div style="color:#94a3b8;font-size:14px;line-height:1.8">{analysis}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This app is for educational purposes only. 
    Data sourced from Yahoo Finance via yfinance. AI analysis is generated by Claude. 
    Nothing here constitutes financial advice. Always consult a licensed financial advisor before investing.
    Past performance does not guarantee future results.
</div>
""", unsafe_allow_html=True)
