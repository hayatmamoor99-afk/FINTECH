"""
StockLens Pro - AI + Geopolitical Intelligence + Monte Carlo
requirements.txt:
    streamlit
    yfinance
    plotly
    pandas
    numpy
    anthropic
    requests
    feedparser
Run: streamlit run stocklens_app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import feedparser

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockLens Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.stApp{background:#070b14;color:#e2e8f0}
.main .block-container{padding:1rem 1.6rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2d45}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:#94a3b8!important}
.stTextInput>div>div>input{background:#0d1117!important;border:1px solid #3b82f6!important;border-radius:8px!important;color:#fff!important;font-size:17px!important;font-weight:700!important;padding:12px 16px!important}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#8b5cf6)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:700!important;font-size:13px!important;padding:10px 18px!important;width:100%}
.stButton>button:hover{opacity:.85!important}
[data-testid="stMetric"]{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:10px 14px}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:10px!important}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-size:15px!important}
.sec{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:18px 0 10px;border-bottom:1px solid #1e2d45;padding-bottom:6px}
.card{background:#0d1117;border:1px solid #1e2d45;border-radius:12px;padding:16px}
.news-card{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:12px 14px;margin-bottom:8px}
.news-title{font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px;line-height:1.4}
.news-meta{font-size:10px;color:#64748b}
.news-sent{font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;display:inline-block;margin-left:6px}
.sent-pos{background:rgba(16,185,129,.15);color:#10b981}
.sent-neg{background:rgba(239,68,68,.15);color:#ef4444}
.sent-neu{background:rgba(100,116,139,.15);color:#94a3b8}
.geo-card{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:14px}
.risk-bar{height:6px;border-radius:3px;margin-top:6px;overflow:hidden;background:#1e2d45}
.risk-fill{height:100%;border-radius:3px}
.macro-tile{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:12px 14px;text-align:center}
.macro-lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
.macro-val{font-size:18px;font-weight:800;margin-top:4px}
.macro-chg{font-size:11px;margin-top:2px}
.sig-row{display:flex;justify-content:space-between;align-items:center;background:#070b14;border:1px solid #1e2d45;border-radius:8px;padding:9px 14px;margin-bottom:7px}
.inv-card{text-align:center;background:#0d1117;border:1px solid #1e2d45;border-radius:12px;padding:14px}
.inv-period{font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase}
.inv-value{font-size:19px;font-weight:800;margin-top:6px}
.inv-gain{font-size:12px;font-weight:600;margin-top:3px}
.inv-pct{font-size:11px;margin-top:2px}
.mc-stat{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:14px;text-align:center}
.mc-lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.mc-val{font-size:20px;font-weight:800}
.mc-sub{font-size:11px;color:#64748b;margin-top:3px}
.guide{background:#070b14;border:1px solid #1e2d45;border-radius:8px;padding:10px 12px;margin-bottom:8px}
.guide-title{font-size:12px;font-weight:700;color:#3b82f6;margin-bottom:3px}
.guide-body{font-size:11px;color:#64748b;line-height:1.55}
.disc{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:10px 14px;font-size:11px;color:#b45309;margin-top:10px}
.stTabs [data-baseweb="tab-list"]{background:#0d1117;border-radius:8px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:6px;font-weight:600;font-size:13px}
.stTabs [aria-selected="true"]{background:#3b82f6!important;color:#fff!important}
#MainMenu,footer,header{visibility:hidden}
</style>""", unsafe_allow_html=True)


# ── Cached Data Functions ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min
def get_stock_data(ticker):
    tk   = yf.Ticker(ticker)
    info = tk.info
    end  = datetime.today()
    hist = tk.history(start=end - timedelta(days=5*365), end=end, interval="1mo")
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
    hist = hist[["Close"]].dropna()
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
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
        "ticker":        ticker.upper(),
        "company_name":  info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(cp), 2),
        "prev_close":    round(float(pc), 2),
        "market_cap":    info.get("marketCap"),
        "pe_ratio":      info.get("trailingPE") or info.get("forwardPE"),
        "week52_high":   info.get("fiftyTwoWeekHigh"),
        "week52_low":    info.get("fiftyTwoWeekLow"),
        "avg_volume":    info.get("averageVolume"),
        "dividend":      info.get("dividendYield"),
        "sector":        info.get("sector", "N/A"),
        "country":       info.get("country", "N/A"),
        "ma50":          info.get("fiftyDayAverage"),
        "ma200":         info.get("twoHundredDayAverage"),
        "beta":          info.get("beta"),
        "periods":       periods,
    }


@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min
def get_macro_data():
    syms = {"VIX":"^VIX","SP500":"^GSPC","DXY":"DX-Y.NYB","OIL":"CL=F","GOLD":"GC=F","TNX":"^TNX"}
    result = {}
    for name, sym in syms.items():
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                prev = float(h["Close"].iloc[-2]) if len(h) >= 2 else float(h["Close"].iloc[-1])
                curr = float(h["Close"].iloc[-1])
                result[name] = {"value": round(curr,2), "change": round(curr-prev,2),
                                "pct": round((curr-prev)/prev*100, 2)}
        except:
            result[name] = {"value": None, "change": 0, "pct": 0}
    return result


@st.cache_data(ttl=600, show_spinner=False)   # cache 10 min
def fetch_news(ticker, company, sector):
    news, seen = [], set()
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={company.split()[0]}+stock&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=geopolitics+trade+sanctions+economy&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    ]
    for url in feeds:
        try:
            for e in feedparser.parse(url).entries[:6]:
                t = e.get("title","")
                if t and t not in seen:
                    seen.add(t)
                    news.append({"title":t,"link":e.get("link","#"),"published":e.get("published","")})
        except:
            pass
        if len(news) >= 20:
            break
    return news[:20]


# ── Pure Computation (no caching needed — fast) ───────────────────────────────

def score_sentiment(title):
    t   = title.lower()
    pos = ["surge","rally","gain","rise","beat","record","growth","strong","up","bull",
           "profit","upgrade","boom","recovery","rebound","high"]
    neg = ["fall","drop","crash","loss","miss","weak","down","bear","sell","downgrade",
           "risk","war","sanction","tariff","inflation","recession","crisis","fear",
           "cut","tension","conflict","ban","restrict","default","debt"]
    ps  = sum(1 for w in pos if w in t)
    ns  = sum(1 for w in neg if w in t)
    return "positive" if ps > ns else "negative" if ns > ps else "neutral"


def compute_technicals(hist):
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi":50,"trend":"Neutral","trend_pct":0,"volatility":"Unknown","vol_pct":0}
    deltas = np.diff(closes)
    gains  = np.where(deltas>0, deltas, 0.0)
    losses = np.where(deltas<0, -deltas, 0.0)
    ag = np.mean(gains[-14:])  if len(gains)>=14  else np.mean(gains)
    al = np.mean(losses[-14:]) if len(losses)>=14 else np.mean(losses)
    rsi = round(100 - 100/(1+ag/al), 1) if al else 50.0
    trend_pct = round((closes[-1]-closes[0])/closes[0]*100, 2)
    trend = ("Strong Uptrend" if trend_pct>20 else "Uptrend" if trend_pct>5 else
             "Strong Downtrend" if trend_pct<-20 else "Downtrend" if trend_pct<-5 else "Sideways")
    returns    = np.diff(closes)/closes[:-1]
    vol_pct    = round(float(np.std(returns)*np.sqrt(12)*100), 1)
    volatility = "Very High" if vol_pct>60 else "High" if vol_pct>35 else "Medium" if vol_pct>15 else "Low"
    return {"rsi":rsi,"trend":trend,"trend_pct":trend_pct,"volatility":volatility,"vol_pct":vol_pct}


def compute_geo_risk(news, macro):
    neg_count = sum(1 for n in news if score_sentiment(n["title"])=="negative")
    pos_count = sum(1 for n in news if score_sentiment(n["title"])=="positive")
    total     = len(news) or 1
    news_risk = round(neg_count/total*100)
    vix_val   = (macro.get("VIX") or {}).get("value") or 20
    vix_risk  = min(100, int(vix_val/50*100))
    geo_score = round(news_risk*0.6 + vix_risk*0.4)
    level = ("🔴 Critical" if geo_score>70 else "🟠 High" if geo_score>50
             else "🟡 Elevated" if geo_score>30 else "🟢 Low")
    return {"score":geo_score,"level":level,"news_risk":news_risk,
            "neg_count":neg_count,"pos_count":pos_count,"vix_risk":vix_risk}


def compute_rec(tech, data, geo, macro):
    score = 50
    rsi, trend = tech["rsi"], tech["trend"]
    score += 15 if rsi<30 else 8 if rsi<45 else -15 if rsi>70 else -4 if rsi>55 else 0
    score += (15 if "Strong Uptrend" in trend else 8 if "Uptrend" in trend else
              -15 if "Strong Downtrend" in trend else -8 if "Downtrend" in trend else 0)
    cp   = data["current_price"]
    m50  = data.get("ma50") or cp
    m200 = data.get("ma200") or cp
    score += 8 if cp>m50 else -8
    score += 8 if cp>m200 else -8
    geo_penalty = round(geo["score"]/100*25)
    score -= geo_penalty
    vix = (macro.get("VIX") or {}).get("value") or 20
    score -= 10 if vix>35 else 5 if vix>25 else 0
    score += 5  if vix<15 else 0
    oil_pct = (macro.get("OIL") or {}).get("pct") or 0
    if data.get("sector") in ("Energy","Materials"):
        score += 8 if oil_pct>2 else -8 if oil_pct<-2 else 0
    score -= 5 if ((macro.get("DXY") or {}).get("pct") or 0) > 1 else 0
    score  = max(0, min(100, score))
    if score>=60:   rec, color, icon = "BUY",  "#10b981", "🚀"
    elif score<=38: rec, color, icon = "SELL", "#ef4444", "⛔"
    else:           rec, color, icon = "HOLD", "#f59e0b", "⏸️"
    h = data["periods"]["5yr"]["Close"].values
    return {"rec":rec,"confidence":score,"color":color,"icon":icon,
            "support":round(float(np.percentile(h,15)),2),
            "resistance":round(float(np.percentile(h,85)),2),
            "geo_penalty":geo_penalty}


def simulate(hist, amount=100):
    c = hist["Close"].dropna().values.astype(float)
    if len(c)<2: return {"value":amount,"gain":0,"pct":0}
    value = round(amount/c[0]*c[-1], 2)
    return {"value":value,"gain":round(value-amount,2),"pct":round((c[-1]-c[0])/c[0]*100,2)}


def run_monte_carlo(hist, days=252, simulations=500, investment=1000):
    """
    Monte Carlo simulation using Geometric Brownian Motion.
    Returns simulation paths and key probability statistics.
    """
    closes  = hist["Close"].dropna().values.astype(float)
    returns = np.diff(closes) / closes[:-1]
    mu      = float(np.mean(returns))       # daily mean return (monthly data)
    sigma   = float(np.std(returns))        # daily std dev
    S0      = closes[-1]                    # starting price = current price

    np.random.seed(42)
    dt   = 1
    paths = np.zeros((simulations, days))
    paths[:, 0] = S0

    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    final_prices = paths[:, -1]
    shares       = investment / S0

    final_values = shares * final_prices
    gains        = final_values - investment
    pct_returns  = (final_prices - S0) / S0 * 100

    prob_profit  = round(float(np.mean(final_prices > S0) * 100), 1)
    prob_loss    = round(100 - prob_profit, 1)
    prob_gt20    = round(float(np.mean(pct_returns > 20) * 100), 1)
    prob_lt20    = round(float(np.mean(pct_returns < -20) * 100), 1)

    p5   = round(float(np.percentile(final_prices, 5)),  2)
    p25  = round(float(np.percentile(final_prices, 25)), 2)
    p50  = round(float(np.percentile(final_prices, 50)), 2)
    p75  = round(float(np.percentile(final_prices, 75)), 2)
    p95  = round(float(np.percentile(final_prices, 95)), 2)
    mean = round(float(np.mean(final_prices)), 2)

    return {
        "paths":        paths,
        "final_prices": final_prices,
        "final_values": final_values,
        "S0":           S0,
        "prob_profit":  prob_profit,
        "prob_loss":    prob_loss,
        "prob_gt20":    prob_gt20,
        "prob_lt20":    prob_lt20,
        "p5":  p5,  "p25": p25, "p50": p50,
        "p75": p75, "p95": p95, "mean": mean,
        "investment":   investment,
        "shares":       shares,
        "days":         days,
        "simulations":  simulations,
        "mu":           round(mu*100,3),
        "sigma":        round(sigma*100,3),
    }


def make_mc_chart(mc, ticker):
    paths = mc["paths"]
    days  = mc["days"]
    x     = list(range(days))

    fig = go.Figure()

    # Plot sample paths (thin, semi-transparent)
    step = max(1, len(paths)//80)
    for i in range(0, len(paths), step):
        final = paths[i, -1]
        col   = "rgba(16,185,129,0.08)" if final > mc["S0"] else "rgba(239,68,68,0.08)"
        fig.add_trace(go.Scatter(x=x, y=paths[i], mode="lines",
            line=dict(color=col, width=1), showlegend=False, hoverinfo="skip"))

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    med = np.percentile(paths, 50, axis=0)

    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p95)+list(p5[::-1]),
        fill="toself", fillcolor="rgba(59,130,246,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th pct", showlegend=True))
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p75)+list(p25[::-1]),
        fill="toself", fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct", showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=med, mode="lines",
        line=dict(color="#60a5fa", width=2.5), name="Median path"))
    fig.add_trace(go.Scatter(x=x, y=[mc["S0"]]*days, mode="lines",
        line=dict(color="#64748b", width=1.5, dash="dash"), name="Current price"))

    fig.update_layout(
        paper_bgcolor="#070b14", plot_bgcolor="#0d1117",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(
            text=f"{ticker} — Monte Carlo Simulation ({mc['simulations']} paths, {mc['days']} trading days)",
            font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Trading Days", gridcolor="#1e2d45", showgrid=True,
                   zeroline=False, tickfont=dict(size=10), showline=False),
        yaxis=dict(title="Price ($)", gridcolor="#1e2d45", showgrid=True,
                   zeroline=False, tickprefix="$", tickfont=dict(size=10), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=8,r=8,t=50,b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")),
        height=360,
    )
    return fig


def make_dist_chart(mc, ticker):
    finals = mc["final_prices"]
    S0     = mc["S0"]
    colors = ["#10b981" if p > S0 else "#ef4444" for p in finals]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=finals,
        nbinsx=50,
        marker=dict(
            color=["#10b981" if p > S0 else "#ef4444" for p in finals],
            line=dict(width=0),
        ),
        opacity=0.85,
        name="Final price distribution",
    ))
    # Vertical lines for percentiles
    for val, lbl, col in [
        (mc["p5"],  "5th pct",  "#ef4444"),
        (mc["p25"], "25th pct", "#f59e0b"),
        (mc["p50"], "Median",   "#60a5fa"),
        (mc["p75"], "75th pct", "#a78bfa"),
        (mc["p95"], "95th pct", "#10b981"),
        (S0,        "Current",  "#ffffff"),
    ]:
        fig.add_vline(x=val, line=dict(color=col, width=1.5, dash="dot"),
                      annotation_text=lbl, annotation_font=dict(size=9, color=col))

    fig.update_layout(
        paper_bgcolor="#070b14", plot_bgcolor="#0d1117",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — Final Price Distribution after {mc['days']} Trading Days",
                   font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Final Price ($)", gridcolor="#1e2d45", showgrid=True,
                   zeroline=False, tickprefix="$", tickfont=dict(size=10)),
        yaxis=dict(title="Frequency", gridcolor="#1e2d45", showgrid=True,
                   zeroline=False, tickfont=dict(size=10)),
        bargap=0.02,
        margin=dict(l=8,r=8,t=50,b=8),
        height=320,
        showlegend=False,
    )
    return fig


def make_chart(hist, ticker, label):
    closes = hist["Close"].dropna()
    is_up  = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color  = "#10b981" if is_up else "#ef4444"
    fill   = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=closes.index, y=closes, fill="tozeroy", fillcolor=fill,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>", name=ticker))
    if len(closes) >= 6:
        fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(3).mean(),
            line=dict(color="rgba(139,92,246,.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="#070b14", plot_bgcolor="#0d1117",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {label}", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2d45",showgrid=True,zeroline=False,tickfont=dict(size=10),showline=False),
        yaxis=dict(gridcolor="#1e2d45",showgrid=True,zeroline=False,tickprefix="$",tickfont=dict(size=10),showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=8,r=8,t=38,b=8), hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")), height=290)
    return fig


def fmt_large(n):
    if n is None: return "N/A"
    n = float(n)
    if n>=1e12: return f"${n/1e12:.2f}T"
    if n>=1e9:  return f"${n/1e9:.2f}B"
    if n>=1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def get_ai_analysis(data, tech, rec, geo, macro, news):
    api_key = st.session_state.get("api_key","").strip()
    if not api_key:
        return "💡 Enter your Anthropic API key in the sidebar to unlock AI-generated analysis."
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        inv5     = simulate(data["periods"]["5yr"])
        inv1     = simulate(data["periods"]["1yr"])
        top_news = "\n".join(f"- {n['title']}" for n in news[:8])
        vix      = (macro.get("VIX") or {}).get("value","N/A")
        oil_val  = (macro.get("OIL") or {}).get("value","N/A")
        oil_pct  = (macro.get("OIL") or {}).get("pct",0)
        dxy      = (macro.get("DXY") or {}).get("value","N/A")
        sp500_p  = (macro.get("SP500") or {}).get("pct",0)
        tnx      = (macro.get("TNX") or {}).get("value","N/A")
        prompt = (
            f"You are a senior equity research analyst specialising in geopolitical risk and macro analysis.\n"
            f"Write a professional 5-sentence analysis for {data['ticker']} ({data['company_name']}, "
            f"sector: {data['sector']}, country: {data['country']}).\n\n"
            f"TECHNICAL: Price=${data['current_price']}, RSI={tech['rsi']}, Trend={tech['trend']} "
            f"({tech['trend_pct']}% 5yr), Volatility={tech['volatility']} ({tech['vol_pct']}% ann.), "
            f"Beta={data.get('beta','N/A')}, MA50={data.get('ma50','N/A')}, MA200={data.get('ma200','N/A')}\n"
            f"$100 over 5yr=${inv5['value']} ({inv5['pct']}%), over 1yr=${inv1['value']} ({inv1['pct']}%)\n\n"
            f"MACRO: VIX={vix}, S&P500 daily={sp500_p:+.2f}%, Oil=${oil_val} ({oil_pct:+.2f}%), "
            f"DXY={dxy}, 10yr Yield={tnx}%\n\n"
            f"GEO RISK: Score={geo['score']}/100 ({geo['level']}), "
            f"Neg headlines={geo['neg_count']}/{len(news)}, Geo penalty=-{rec['geo_penalty']} pts\n\n"
            f"HEADLINES:\n{top_news}\n\n"
            f"SIGNAL: {rec['rec']} | Confidence={rec['confidence']}% | "
            f"Support=${rec['support']} | Resistance=${rec['resistance']}\n\n"
            f"Write exactly 5 sentences: (1) overall 5yr performance with numbers, "
            f"(2) current technical picture, (3) macro environment impact on this sector, "
            f"(4) specific geopolitical risks from headlines, (5) forward outlook + risk reminder. "
            f"Be data-specific and cite actual figures."
        )
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=500,
                                     messages=[{"role":"user","content":prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"AI analysis unavailable: {e}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:10px 0 16px">
        <div style="font-size:28px">🌐</div>
        <div style="font-size:19px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">StockLens Pro</div>
        <div style="font-size:10px;color:#64748b">AI + Geopolitical Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**🔑 Anthropic API Key** *(for AI analysis)*")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...",
                             label_visibility="collapsed")
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown("---")
    st.markdown("**📚 Beginner's Guide**")
    for title, body in [
        ("📊 Charts",      "Price history. Rising line = growing value. Purple dot = 3-month MA."),
        ("💵 $100 Sim",    "What $100 invested at period start would be worth at today's price."),
        ("🌐 Geo Risk",    "Live score from news sentiment + VIX. Higher = more danger to price."),
        ("😨 VIX",         "Fear gauge. >30=fear, >40=panic, <15=calm bull market."),
        ("🎲 Monte Carlo", "Runs 500 simulated futures using your stock's own volatility & drift. Shows probable price range — not a guarantee."),
        ("📊 Percentiles", "p5 = only 5% of simulations end below this. p95 = only 5% end above. The shaded band is the most likely range."),
        ("🤖 AI Score",    "Technicals + geo risk + macro → BUY/HOLD/SELL with confidence %."),
        ("⚠️ Risk",        "Not financial advice. Consult a licensed advisor."),
    ]:
        st.markdown(f'<div class="guide"><div class="guide-title">{title}</div>'
                    f'<div class="guide-body">{body}</div></div>', unsafe_allow_html=True)


# ── Search Bar ────────────────────────────────────────────────────────────────
st.markdown("""<div style="text-align:center;padding:8px 0 4px">
    <span style="font-size:26px;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">🌐 StockLens Pro</span>
    <div style="font-size:12px;color:#64748b;margin-top:2px">
        AI · Geopolitical Risk · Live Macro · News Sentiment · Monte Carlo
    </div>
</div>""", unsafe_allow_html=True)

ci, cb = st.columns([4,1])
with ci:
    ticker_input = st.text_input("Ticker", placeholder="Enter any ticker: AAPL, TSLA, NVDA, RELIANCE.NS ...",
                                  label_visibility="collapsed", key="tf").upper().strip()
with cb:
    go_btn = st.button("🔍 Analyse", use_container_width=True)

qcols = st.columns(8)
for col, qt in zip(qcols, ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META","JPM"]):
    with col:
        if st.button(qt, key=f"q_{qt}"):
            st.session_state["run_ticker"] = qt
            st.rerun()

st.markdown("<hr style='border-color:#1e2d45;margin:10px 0'>", unsafe_allow_html=True)

final_ticker = st.session_state.pop("run_ticker", None) or (ticker_input if go_btn else None)

if not final_ticker:
    st.markdown("""<div style="text-align:center;padding:50px 20px;color:#64748b">
        <div style="font-size:50px;margin-bottom:12px">🌐</div>
        <div style="font-size:16px;font-weight:600;color:#94a3b8;margin-bottom:6px">Search any stock to begin</div>
        <div style="font-size:13px">Type a ticker and click Analyse — or use a quick pick above</div>
        <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:16px">
            <span style="background:rgba(59,130,246,.12);color:#60a5fa;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">📊 5-Year Charts</span>
            <span style="background:rgba(139,92,246,.12);color:#a78bfa;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">💰 $100 Simulator</span>
            <span style="background:rgba(16,185,129,.12);color:#34d399;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">🌐 Geo Risk Score</span>
            <span style="background:rgba(245,158,11,.12);color:#fbbf24;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">📰 Live News Sentiment</span>
            <span style="background:rgba(239,68,68,.12);color:#f87171;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">🎲 Monte Carlo Simulation</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load Data (parallel-ish via cached functions) ─────────────────────────────
prog = st.progress(0, text="📡 Fetching stock data...")
try:
    data = get_stock_data(final_ticker)
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

prog.progress(40, text="🌍 Loading macro indicators...")
macro = get_macro_data()

prog.progress(70, text="📰 Fetching live news...")
news  = fetch_news(final_ticker, data["company_name"], data["sector"])

prog.progress(90, text="🔢 Computing signals...")
tech  = compute_technicals(data["periods"]["5yr"])
geo   = compute_geo_risk(news, macro)
rec   = compute_rec(tech, data, geo, macro)
prog.progress(100, text="✅ Done!")
prog.empty()

# ── Header ────────────────────────────────────────────────────────────────────
change     = data["current_price"] - data["prev_close"]
change_pct = change/data["prev_close"]*100 if data["prev_close"] else 0
is_up      = change >= 0
dcolor     = "#10b981" if is_up else "#ef4444"

h1, h2 = st.columns([2,1])
with h1:
    st.markdown(f"""<div>
        <span style="font-size:30px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:10px;font-size:13px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">
        Sector: {data['sector']} &nbsp;|&nbsp; Country: {data['country']} &nbsp;|&nbsp; Beta: {data.get('beta','N/A')}
    </div>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""<div style="text-align:right">
        <div style="font-size:30px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:13px;font-weight:700;color:{dcolor}">
            {'▲' if is_up else '▼'} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)
        </div>
    </div>""", unsafe_allow_html=True)

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Key Metrics</div>', unsafe_allow_html=True)
for col, (lbl, val) in zip(st.columns(8), [
    ("Market Cap",  fmt_large(data["market_cap"])),
    ("P/E Ratio",   f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High",    f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low",     f"${data['week52_low']:.2f}"  if data["week52_low"]  else "N/A"),
    ("Avg Volume",  fmt_large(data["avg_volume"])),
    ("MA 50",       f"${data['ma50']:.2f}"  if data["ma50"]  else "N/A"),
    ("MA 200",      f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)",    str(tech["rsi"])),
]):
    col.metric(lbl, val)

# ── Macro ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌍 Live Macro Environment</div>', unsafe_allow_html=True)
for col, (lbl, key, color) in zip(st.columns(6), [
    ("😨 VIX Fear","VIX","#ef4444"),("📈 S&P 500","SP500","#10b981"),
    ("🛢️ Crude Oil","OIL","#f59e0b"),("💵 US Dollar","DXY","#3b82f6"),
    ("🏅 Gold","GOLD","#fbbf24"),("📉 10yr Yield","TNX","#a78bfa"),
]):
    m   = macro.get(key) or {}
    val = m.get("value")
    pct = m.get("pct",0)
    col.markdown(f"""<div class="macro-tile">
        <div class="macro-lbl">{lbl}</div>
        <div class="macro-val" style="color:{color}">{f'{val:,.2f}' if val else 'N/A'}</div>
        <div class="macro-chg" style="color:{'#10b981' if pct>=0 else '#ef4444'}">
            {'▲' if pct>=0 else '▼'} {abs(pct):.2f}%
        </div>
    </div>""", unsafe_allow_html=True)

# ── Geo Risk ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌐 Geopolitical Risk Intelligence</div>', unsafe_allow_html=True)
geo_color = "#ef4444" if geo["score"]>70 else "#f59e0b" if geo["score"]>40 else "#10b981"
gc1, gc2, gc3 = st.columns([1,1,2])

with gc1:
    st.markdown(f"""<div class="geo-card" style="border-color:{geo_color}44">
        <div style="font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:6px">🌐 Geo-Risk Score</div>
        <div style="font-size:38px;font-weight:900;color:{geo_color};margin:6px 0">
            {geo['score']}<span style="font-size:16px">/100</span>
        </div>
        <div style="font-size:14px;font-weight:700;color:{geo_color}">{geo['level']}</div>
        <div class="risk-bar"><div class="risk-fill" style="width:{geo['score']}%;background:{geo_color}"></div></div>
        <div style="font-size:11px;color:#64748b;margin-top:8px">Composite: news sentiment + VIX</div>
    </div>""", unsafe_allow_html=True)

with gc2:
    st.markdown(f"""<div class="geo-card">
        <div style="font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:8px">📰 News Sentiment</div>
        <div style="display:flex;gap:12px;margin:10px 0">
            <div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#10b981">{geo['pos_count']}</div><div style="font-size:10px;color:#64748b">Positive</div></div>
            <div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#ef4444">{geo['neg_count']}</div><div style="font-size:10px;color:#64748b">Negative</div></div>
            <div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#94a3b8">{len(news)-geo['pos_count']-geo['neg_count']}</div><div style="font-size:10px;color:#64748b">Neutral</div></div>
        </div>
        <div style="font-size:11px;color:#64748b">From {len(news)} live headlines</div>
        <div style="font-size:11px;color:#64748b;margin-top:3px">VIX risk: {geo['vix_risk']}%</div>
        <div style="font-size:11px;color:#64748b">Geo penalty on signal: <span style="color:#ef4444">-{rec['geo_penalty']} pts</span></div>
    </div>""", unsafe_allow_html=True)

with gc3:
    st.markdown('<div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">📡 Live Headlines</div>', unsafe_allow_html=True)
    for n in news[:6]:
        sent = score_sentiment(n["title"])
        sc   = "sent-pos" if sent=="positive" else "sent-neg" if sent=="negative" else "sent-neu"
        sl   = "↑ Positive" if sent=="positive" else "↓ Negative" if sent=="negative" else "→ Neutral"
        st.markdown(f"""<div class="news-card">
            <div class="news-title">{n['title'][:110]}{'...' if len(n['title'])>110 else ''}</div>
            <div class="news-meta">{n.get('published','')[:25]}
                <span class="news-sent {sc}">{sl}</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📈 Historical Price Histograms</div>', unsafe_allow_html=True)
period_map = [("5yr","5 Year"),("4yr","4 Year"),("3yr","3 Year"),("2yr","2 Year"),("1yr","1 Year")]
for tab, (key, label) in zip(st.tabs([f"📅 {l}" for _,l in period_map]), period_map):
    with tab:
        h = data["periods"][key]
        if h.empty: st.warning(f"Not enough data for {label}.")
        else:       st.plotly_chart(make_chart(h, data["ticker"], label), use_container_width=True)

# ── Investment Simulator ──────────────────────────────────────────────────────
st.markdown('<div class="sec">💰 $100 Investment Simulator</div>', unsafe_allow_html=True)
for col, (key, label) in zip(st.columns(5), period_map):
    inv = simulate(data["periods"][key])
    pos = inv["gain"] >= 0
    c   = "#10b981" if pos else "#ef4444"
    col.markdown(f"""<div class="inv-card">
        <div class="inv-period">{label}</div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Started $100</div>
        <div class="inv-value" style="color:{c}">${inv['value']:,.2f}</div>
        <div class="inv-gain"  style="color:{c}">{'+' if pos else ''}${inv['gain']:,.2f}</div>
        <div class="inv-pct"   style="color:{c}">{'▲' if pos else '▼'} {abs(inv['pct']):.1f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Monte Carlo ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🎲 Monte Carlo Price Simulation</div>', unsafe_allow_html=True)

mc_col1, mc_col2, mc_col3 = st.columns([1,1,1])
with mc_col1:
    mc_days = st.selectbox("Forecast horizon", [63,126,252,504], index=2,
                            format_func=lambda x: {63:"3 Months",126:"6 Months",252:"1 Year",504:"2 Years"}[x])
with mc_col2:
    mc_sims = st.selectbox("Simulations", [200,500,1000], index=1)
with mc_col3:
    mc_invest = st.number_input("Investment ($)", min_value=100, max_value=1000000,
                                 value=1000, step=500)

mc = run_monte_carlo(data["periods"]["5yr"], days=mc_days, simulations=mc_sims, investment=mc_invest)

# Stats row
stat_cols = st.columns(6)
expected_val = round(mc["shares"] * mc["mean"], 2)
expected_gain = round(expected_val - mc_invest, 2)

for col, (lbl, val, sub, col_color) in zip(stat_cols, [
    ("Prob. Profit",    f"{mc['prob_profit']}%",    f"{mc['prob_loss']}% loss",    "#10b981"),
    ("Prob. >+20%",     f"{mc['prob_gt20']}%",      "gain over 20%",               "#3b82f6"),
    ("Prob. <-20%",     f"{mc['prob_lt20']}%",      "loss over 20%",               "#ef4444"),
    ("Median Price",    f"${mc['p50']:,.2f}",        f"from ${mc['S0']:,.2f}",       "#60a5fa"),
    ("Expected Value",  f"${expected_val:,.2f}",     f"on ${mc_invest:,} invested",  "#10b981" if expected_gain>=0 else "#ef4444"),
    ("Price Range",     f"${mc['p5']:,.0f}–${mc['p95']:,.0f}", "5th–95th pct",      "#a78bfa"),
]):
    col.markdown(f"""<div class="mc-stat">
        <div class="mc-lbl">{lbl}</div>
        <div class="mc-val" style="color:{col_color}">{val}</div>
        <div class="mc-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# Model parameters info
st.markdown(f"""<div style="font-size:11px;color:#64748b;margin:8px 0 4px;padding:8px 12px;
    background:#0d1117;border:1px solid #1e2d45;border-radius:8px">
    📐 <b style="color:#94a3b8">Model:</b> Geometric Brownian Motion &nbsp;|&nbsp;
    μ (monthly drift) = <span style="color:#60a5fa">{mc['mu']}%</span> &nbsp;|&nbsp;
    σ (monthly volatility) = <span style="color:#a78bfa">{mc['sigma']}%</span> &nbsp;|&nbsp;
    {mc['simulations']} simulated paths &nbsp;|&nbsp; {mc['days']} trading days
</div>""", unsafe_allow_html=True)

# Charts
mc_tab1, mc_tab2 = st.tabs(["📈 Simulation Paths", "📊 Final Price Distribution"])
with mc_tab1:
    st.plotly_chart(make_mc_chart(mc, data["ticker"]), use_container_width=True)
with mc_tab2:
    st.plotly_chart(make_dist_chart(mc, data["ticker"]), use_container_width=True)

# Percentile table
st.markdown('<div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin:10px 0 6px">📋 Percentile Outcomes</div>', unsafe_allow_html=True)
ptable_cols = st.columns(5)
for col, (pct_lbl, price, meaning) in zip(ptable_cols, [
    ("Worst 5%",   mc["p5"],  "Bear case"),
    ("25th pct",   mc["p25"], "Below median"),
    ("Median",     mc["p50"], "Base case"),
    ("75th pct",   mc["p75"], "Above median"),
    ("Best 5%",    mc["p95"], "Bull case"),
]):
    pnl  = round((price - mc["S0"])/mc["S0"]*100, 1)
    pos  = pnl >= 0
    cval = round(mc["shares"] * price, 2)
    col.markdown(f"""<div class="mc-stat">
        <div class="mc-lbl">{pct_lbl}</div>
        <div class="mc-val" style="color:{'#10b981' if pos else '#ef4444'}">${price:,.2f}</div>
        <div class="mc-sub" style="color:{'#10b981' if pos else '#ef4444'}">{'+' if pos else ''}{pnl}%</div>
        <div style="font-size:10px;color:#64748b;margin-top:3px">${cval:,.0f} on ${mc_invest:,}</div>
        <div style="font-size:10px;color:#64748b">{meaning}</div>
    </div>""", unsafe_allow_html=True)

# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)
v1, v2 = st.columns([3,2])
with v1:
    vc  = {"BUY":"rgba(16,185,129,.08)","SELL":"rgba(239,68,68,.08)","HOLD":"rgba(245,158,11,.08)"}[rec["rec"]]
    bc  = {"BUY":"rgba(16,185,129,.3)","SELL":"rgba(239,68,68,.3)","HOLD":"rgba(245,158,11,.3)"}[rec["rec"]]
    bar = "█"*int(rec["confidence"]/5) + "░"*(20-int(rec["confidence"]/5))
    st.markdown(f"""<div style="background:{vc};border:1px solid {bc};border-radius:12px;padding:20px">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
            <div style="font-size:36px">{rec['icon']}</div>
            <div>
                <div style="font-size:10px;font-weight:700;color:{rec['color']};text-transform:uppercase;letter-spacing:1px">Composite AI Verdict</div>
                <div style="font-size:28px;font-weight:900;color:{rec['color']}">{rec['rec']}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div style="font-size:10px;color:#64748b">Confidence</div>
                <div style="font-size:28px;font-weight:900;color:{rec['color']}">{rec['confidence']}%</div>
            </div>
        </div>
        <div style="font-size:10px;color:#64748b;font-family:monospace;letter-spacing:1px;margin-bottom:4px">{bar}</div>
        <div style="font-size:11px;color:#64748b">
            Technicals (RSI, MA, trend) + Geo risk (-{rec['geo_penalty']} pts) + Macro (VIX, Oil, DXY)
        </div>
    </div>""", unsafe_allow_html=True)
with v2:
    for lbl, val, c in [
        ("Trend",      tech["trend"],      "#e2e8f0"),
        ("RSI",        str(tech["rsi"]),   "#10b981" if tech["rsi"]<30 else "#ef4444" if tech["rsi"]>70 else "#e2e8f0"),
        ("Volatility", tech["volatility"], "#e2e8f0"),
        ("Geo Risk",   geo["level"],        geo_color),
        ("Support",    f"${rec['support']:,.2f}",    "#10b981"),
        ("Resistance", f"${rec['resistance']:,.2f}", "#ef4444"),
    ]:
        st.markdown(f"""<div class="sig-row">
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{lbl}</span>
            <span style="font-size:13px;font-weight:700;color:{c}">{val}</span>
        </div>""", unsafe_allow_html=True)

# ── Full News Feed ────────────────────────────────────────────────────────────
with st.expander("📰 Full News Feed — All Headlines"):
    for n in news:
        sent = score_sentiment(n["title"])
        sc   = "sent-pos" if sent=="positive" else "sent-neg" if sent=="negative" else "sent-neu"
        sl   = "↑ Positive" if sent=="positive" else "↓ Negative" if sent=="negative" else "→ Neutral"
        st.markdown(f"""<div class="news-card">
            <div class="news-title">{n['title']}</div>
            <div class="news-meta">{n.get('published','')[:30]}
                <span class="news-sent {sc}">{sl}</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ── AI Analysis ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📝 AI Deep Analysis — Technicals + Geopolitics + Macro</div>', unsafe_allow_html=True)
with st.spinner("Generating AI analysis..."):
    analysis = get_ai_analysis(data, tech, rec, geo, macro, news)
st.markdown(f"""<div class="card">
    <div style="color:#94a3b8;font-size:14px;line-height:1.85">{analysis}</div>
</div>
<div class="disc">
    ⚠️ <b>Disclaimer:</b> Educational only. Data from Yahoo Finance (yfinance) and Google/Yahoo News RSS.
    Monte Carlo uses Geometric Brownian Motion — a statistical model, not a guarantee.
    AI analysis by Claude. Not financial advice. Past performance does not equal future results.
    Consult a licensed financial advisor before any investment decision.
</div>""", unsafe_allow_html=True)
