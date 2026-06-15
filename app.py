"""
StockLens - AI-Powered Fintech Stock Analyzer
=============================================
Requirements (requirements.txt):
    streamlit
    yfinance
    plotly
    pandas
    numpy
    anthropic

Run:
    streamlit run stocklens_app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockLens — AI Market Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .main .block-container { padding: 1.2rem 1.8rem; max-width: 1400px; }

    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1e2d45; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }

    /* Input */
    .stTextInput > div > div > input {
        background: #111827 !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        width: 100%;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 10px !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 16px !important; }

    /* Section headers */
    .sec-hdr {
        font-size: 11px; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 18px 0 10px; border-bottom: 1px solid #1e2d45; padding-bottom: 6px;
    }

    /* Cards */
    .card {
        background: #111827; border: 1px solid #1e2d45;
        border-radius: 12px; padding: 16px;
    }
    .inv-card { text-align: center; }
    .inv-period { font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }
    .inv-value  { font-size: 20px; font-weight: 800; margin-top: 6px; }
    .inv-gain   { font-size: 12px; font-weight: 600; margin-top: 3px; }
    .inv-pct    { font-size: 11px; margin-top: 2px; }

    /* Signal row */
    .sig-row {
        display: flex; justify-content: space-between; align-items: center;
        background: #0a0e1a; border: 1px solid #1e2d45;
        border-radius: 8px; padding: 9px 14px; margin-bottom: 7px;
    }
    .sig-lbl { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }
    .sig-val { font-size: 13px; font-weight: 700; }

    /* Disclaimer */
    .disc {
        background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.25);
        border-radius: 8px; padding: 10px 14px;
        font-size: 11px; color: #b45309; margin-top: 10px;
    }

    /* Guide cards */
    .guide {
        background: #0a0e1a; border: 1px solid #1e2d45;
        border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    }
    .guide-title { font-size: 12px; font-weight: 700; color: #3b82f6; margin-bottom: 3px; }
    .guide-body  { font-size: 11px; color: #64748b; line-height: 1.55; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 8px; padding: 4px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #64748b;
        border-radius: 6px; font-weight: 600; font-size: 13px;
    }
    .stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    tk   = yf.Ticker(ticker)
    info = tk.info
    end  = datetime.today()

    hist = tk.history(start=end - timedelta(days=5*365), end=end, interval="1mo")
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")

    hist = hist[["Close"]].dropna()
    periods = {
        "5yr": hist,
        "4yr": hist[hist.index >= end - timedelta(days=4*365)],
        "3yr": hist[hist.index >= end - timedelta(days=3*365)],
        "2yr": hist[hist.index >= end - timedelta(days=2*365)],
        "1yr": hist[hist.index >= end - timedelta(days=365)],
    }

    cp = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
    pc = info.get("previousClose") or info.get("regularMarketPreviousClose") or float(hist["Close"].iloc[-2])

    return {
        "ticker":       ticker.upper(),
        "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(cp), 2),
        "prev_close":    round(float(pc), 2),
        "market_cap":   info.get("marketCap"),
        "pe_ratio":     info.get("trailingPE") or info.get("forwardPE"),
        "week52_high":  info.get("fiftyTwoWeekHigh"),
        "week52_low":   info.get("fiftyTwoWeekLow"),
        "avg_volume":   info.get("averageVolume"),
        "dividend":     info.get("dividendYield"),
        "sector":       info.get("sector", "N/A"),
        "ma50":         info.get("fiftyDayAverage"),
        "ma200":        info.get("twoHundredDayAverage"),
        "periods":      periods,
    }


def compute_technicals(hist: pd.DataFrame) -> dict:
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi": 50, "trend": "Neutral", "trend_pct": 0, "volatility": "Unknown", "vol_pct": 0}

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[-14:])  if len(gains)  >= 14 else np.mean(gains)
    al = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi = round(100 - 100 / (1 + ag / al), 1) if al else 50.0

    trend_pct = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
    trend = ("Strong Uptrend"   if trend_pct > 20 else
             "Uptrend"          if trend_pct > 5  else
             "Strong Downtrend" if trend_pct < -20 else
             "Downtrend"        if trend_pct < -5  else "Sideways")

    returns = np.diff(closes) / closes[:-1]
    vol_pct = round(float(np.std(returns) * np.sqrt(12) * 100), 1)
    volatility = ("Very High" if vol_pct > 60 else
                  "High"      if vol_pct > 35 else
                  "Medium"    if vol_pct > 15 else "Low")

    return {"rsi": rsi, "trend": trend, "trend_pct": trend_pct,
            "volatility": volatility, "vol_pct": vol_pct}


def compute_rec(tech: dict, data: dict) -> dict:
    score = 50
    rsi, trend = tech["rsi"], tech["trend"]
    score += 20 if rsi < 30 else 10 if rsi < 45 else -20 if rsi > 70 else -5 if rsi > 55 else 0
    score += (20 if "Strong Uptrend" in trend else 10 if "Uptrend" in trend else
              -20 if "Strong Downtrend" in trend else -10 if "Downtrend" in trend else 0)
    cp, m50, m200 = data["current_price"], data.get("ma50") or data["current_price"], data.get("ma200") or data["current_price"]
    score += 8 if cp > m50  else -8
    score += 8 if cp > m200 else -8
    score = max(0, min(100, score))

    if score >= 62:   rec, color, icon = "BUY",  "#10b981", "🚀"
    elif score <= 38: rec, color, icon = "SELL", "#ef4444", "⛔"
    else:             rec, color, icon = "HOLD", "#f59e0b", "⏸️"

    h = data["periods"]["5yr"]["Close"].values
    return {"rec": rec, "confidence": score, "color": color, "icon": icon,
            "support": round(float(np.percentile(h, 15)), 2),
            "resistance": round(float(np.percentile(h, 85)), 2)}


def simulate(hist: pd.DataFrame, amount: float = 100) -> dict:
    c = hist["Close"].dropna().values.astype(float)
    if len(c) < 2:
        return {"value": amount, "gain": 0, "pct": 0}
    value = round(amount / c[0] * c[-1], 2)
    return {"value": value, "gain": round(value - amount, 2),
            "pct": round((c[-1] - c[0]) / c[0] * 100, 2)}


def fmt_large(n) -> str:
    if n is None: return "N/A"
    n = float(n)
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def make_chart(hist: pd.DataFrame, ticker: str, label: str) -> go.Figure:
    closes = hist["Close"].dropna()
    is_up  = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color  = "#10b981" if is_up else "#ef4444"
    fill   = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closes.index, y=closes,
        fill="tozeroy", fillcolor=fill,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>",
        name=ticker,
    ))
    if len(closes) >= 6:
        fig.add_trace(go.Scatter(
            x=closes.index, y=closes.rolling(3).mean(),
            line=dict(color="rgba(139,92,246,0.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip",
        ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {label} Price History", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False,
                   tickfont=dict(size=10), showline=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False,
                   tickprefix="$", tickfont=dict(size=10), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")),
        height=300,
    )
    return fig


def get_ai_analysis(data, tech, rec) -> str:
    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        return "💡 Enter your Anthropic API key in the sidebar to unlock AI-generated analysis."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        inv5 = simulate(data["periods"]["5yr"])
        inv1 = simulate(data["periods"]["1yr"])
        msg  = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=300,
            messages=[{"role": "user", "content":
                f"Write a concise 3-sentence professional equity analysis for {data['ticker']} ({data['company_name']}). "
                f"Key data: price=${data['current_price']}, RSI={tech['rsi']}, trend={tech['trend']} ({tech['trend_pct']}% 5yr), "
                f"volatility={tech['volatility']} ({tech['vol_pct']}% ann.), recommendation={rec['rec']} ({rec['confidence']}% confidence), "
                f"$100 over 5yr = ${inv5['value']} ({inv5['pct']}%), over 1yr = ${inv1['value']} ({inv1['pct']}%). "
                f"End with a one-line risk reminder. Be factual and data-driven."}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI analysis unavailable: {e}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 18px">
        <div style="font-size:30px">📡</div>
        <div style="font-size:20px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">StockLens</div>
        <div style="font-size:11px;color:#64748b">AI Market Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**🔑 Anthropic API Key** *(for AI analysis)*")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...",
                             label_visibility="collapsed")
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown("---")
    st.markdown("**📚 Beginner's Guide**")
    guides = [
        ("📊 Histogram", "Line chart of stock price over time. Rising = value growing."),
        ("💵 $100 Sim",  "What $100 invested at period start is worth at today's price."),
        ("📈 RSI",       "0–100 scale. <30 = oversold (buy signal). >70 = overbought (sell signal)."),
        ("⚡ Volatility","How wildly the stock swings. High vol = bigger gains OR losses."),
        ("🤖 Verdict",   "Scored on RSI, trend & moving averages. Not financial advice."),
        ("⚠️ Risk",      "Past performance ≠ future results. Consult a licensed advisor."),
    ]
    for title, body in guides:
        st.markdown(f"""
        <div class="guide">
            <div class="guide-title">{title}</div>
            <div class="guide-body">{body}</div>
        </div>""", unsafe_allow_html=True)


# ── Main Search Bar ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:10px 0 4px">
    <span style="font-size:28px;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">📡 StockLens</span>
    <div style="font-size:13px;color:#64748b;margin-top:2px">Enter any stock ticker to analyse</div>
</div>
""", unsafe_allow_html=True)

col_inp, col_btn = st.columns([4, 1])
with col_inp:
    ticker_input = st.text_input("Ticker", placeholder="e.g. AAPL, TSLA, NVDA, MSFT...",
                                  label_visibility="collapsed", key="ticker_field").upper().strip()
with col_btn:
    go_btn = st.button("🔍 Analyse", use_container_width=True)

# Quick picks
st.markdown("<div style='margin:8px 0 4px;font-size:11px;color:#64748b;text-align:center'>⚡ Quick picks</div>", unsafe_allow_html=True)
qcols = st.columns(8)
quick = ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META","JPM"]
for col, qt in zip(qcols, quick):
    with col:
        if st.button(qt, key=f"qp_{qt}"):
            st.session_state["run_ticker"] = qt
            st.rerun()

st.markdown("<hr style='border-color:#1e2d45;margin:12px 0'>", unsafe_allow_html=True)

# Resolve ticker
if "run_ticker" in st.session_state:
    final_ticker = st.session_state.pop("run_ticker")
elif go_btn and ticker_input:
    final_ticker = ticker_input
elif ticker_input and not go_btn:
    final_ticker = None   # waiting for button press
else:
    final_ticker = None

if not final_ticker:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#64748b">
        <div style="font-size:48px;margin-bottom:12px">📊</div>
        <div style="font-size:16px;font-weight:600;color:#94a3b8;margin-bottom:6px">Search any stock above to get started</div>
        <div style="font-size:13px">Type a ticker like <b>AAPL</b>, <b>TSLA</b>, <b>NVDA</b> and click Analyse — or pick a quick pick</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load & Display ────────────────────────────────────────────────────────────
with st.spinner(f"Fetching real market data for **{final_ticker}**..."):
    try:
        data = get_stock_data(final_ticker)
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()

tech = compute_technicals(data["periods"]["5yr"])
rec  = compute_rec(tech, data)

# ── Header ────────────────────────────────────────────────────────────────────
change     = data["current_price"] - data["prev_close"]
change_pct = change / data["prev_close"] * 100 if data["prev_close"] else 0
is_up      = change >= 0
dcolor     = "#10b981" if is_up else "#ef4444"
arrow      = "▲" if is_up else "▼"

h1, h2 = st.columns([2, 1])
with h1:
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span style="font-size:32px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:10px;font-size:13px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:12px;color:#64748b">Sector: {data['sector']}</div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div style="text-align:right">
        <div style="font-size:32px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:14px;font-weight:700;color:{dcolor}">
            {arrow} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)
        </div>
    </div>""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 Key Metrics</div>', unsafe_allow_html=True)
mcols = st.columns(8)
mets  = [
    ("Market Cap",  fmt_large(data["market_cap"])),
    ("P/E Ratio",   f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High",    f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low",     f"${data['week52_low']:.2f}"  if data["week52_low"]  else "N/A"),
    ("Avg Volume",  fmt_large(data["avg_volume"])),
    ("MA 50",       f"${data['ma50']:.2f}"  if data["ma50"]  else "N/A"),
    ("MA 200",      f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)",    str(tech["rsi"])),
]
for col, (lbl, val) in zip(mcols, mets):
    col.metric(lbl, val)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Historical Price Histograms</div>', unsafe_allow_html=True)
period_map = [("5yr","5 Year"),("4yr","4 Year"),("3yr","3 Year"),("2yr","2 Year"),("1yr","1 Year")]
tabs = st.tabs([f"📅 {label}" for _, label in period_map])
for tab, (key, label) in zip(tabs, period_map):
    with tab:
        h = data["periods"][key]
        if h.empty:
            st.warning(f"Not enough data for {label}.")
        else:
            st.plotly_chart(make_chart(h, data["ticker"], label), use_container_width=True)

# ── Investment Simulator ──────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">💰 $100 Investment Simulator</div>', unsafe_allow_html=True)
icols = st.columns(5)
for col, (key, label) in zip(icols, period_map):
    inv = simulate(data["periods"][key])
    pos = inv["gain"] >= 0
    c   = "#10b981" if pos else "#ef4444"
    ar  = "▲" if pos else "▼"
    col.markdown(f"""
    <div class="card inv-card">
        <div class="inv-period">{label}</div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Started $100</div>
        <div class="inv-value" style="color:{c}">${inv['value']:,.2f}</div>
        <div class="inv-gain"  style="color:{c}">{'+' if pos else ''}${inv['gain']:,.2f}</div>
        <div class="inv-pct"   style="color:{c}">{ar} {abs(inv['pct']):.1f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)
v1, v2 = st.columns([3, 2])

with v1:
    vc = {"BUY":"rgba(16,185,129,.08)","SELL":"rgba(239,68,68,.08)","HOLD":"rgba(245,158,11,.08)"}[rec["rec"]]
    bc = {"BUY":"rgba(16,185,129,.3)", "SELL":"rgba(239,68,68,.3)", "HOLD":"rgba(245,158,11,.3)"}[rec["rec"]]
    bar = "█" * int(rec["confidence"] / 5) + "░" * (20 - int(rec["confidence"] / 5))
    st.markdown(f"""
    <div style="background:{vc};border:1px solid {bc};border-radius:12px;padding:20px">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
            <div style="font-size:38px">{rec['icon']}</div>
            <div>
                <div style="font-size:10px;font-weight:700;color:{rec['color']};text-transform:uppercase;letter-spacing:1px">AI Verdict</div>
                <div style="font-size:28px;font-weight:900;color:{rec['color']}">{rec['rec']}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div style="font-size:10px;color:#64748b">Confidence</div>
                <div style="font-size:28px;font-weight:900;color:{rec['color']}">{rec['confidence']}%</div>
            </div>
        </div>
        <div style="font-size:10px;color:#64748b;font-family:monospace;letter-spacing:1px;margin-bottom:4px">{bar}</div>
        <div style="font-size:11px;color:#64748b">Based on RSI, trend direction, and moving average crossovers</div>
    </div>""", unsafe_allow_html=True)

with v2:
    signals = [
        ("Trend",      tech["trend"],      "#e2e8f0"),
        ("RSI",        str(tech["rsi"]),   "#10b981" if tech["rsi"]<30 else "#ef4444" if tech["rsi"]>70 else "#e2e8f0"),
        ("Volatility", tech["volatility"], "#e2e8f0"),
        ("Support",    f"${rec['support']:,.2f}",    "#10b981"),
        ("Resistance", f"${rec['resistance']:,.2f}", "#ef4444"),
        ("Momentum",   "Strong" if tech["rsi"]>55 else "Weak" if tech["rsi"]<45 else "Neutral", "#e2e8f0"),
    ]
    for lbl, val, c in signals:
        st.markdown(f"""
        <div class="sig-row">
            <span class="sig-lbl">{lbl}</span>
            <span class="sig-val" style="color:{c}">{val}</span>
        </div>""", unsafe_allow_html=True)

# ── AI Analysis ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📝 AI-Generated Analysis</div>', unsafe_allow_html=True)
with st.spinner("Generating analysis..."):
    analysis = get_ai_analysis(data, tech, rec)
st.markdown(f"""
<div class="card">
    <div style="color:#94a3b8;font-size:14px;line-height:1.8">{analysis}</div>
</div>
<div class="disc">
    ⚠️ <b>Disclaimer:</b> Educational purposes only. Data from Yahoo Finance via yfinance.
    Not financial advice. Consult a licensed financial advisor. Past performance ≠ future results.
</div>""", unsafe_allow_html=True)
