"""
StockLens Pro - AI + Geopolitical Intelligence
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

st.set_page_config(page_title="StockLens Pro", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
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
.news-card:hover{border-color:#3b82f6}
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
.guide{background:#070b14;border:1px solid #1e2d45;border-radius:8px;padding:10px 12px;margin-bottom:8px}
.guide-title{font-size:12px;font-weight:700;color:#3b82f6;margin-bottom:3px}
.guide-body{font-size:11px;color:#64748b;line-height:1.55}
.disc{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:10px 14px;font-size:11px;color:#b45309;margin-top:10px}
.stTabs [data-baseweb="tab-list"]{background:#0d1117;border-radius:8px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:6px;font-weight:600;font-size:13px}
.stTabs [aria-selected="true"]{background:#3b82f6!important;color:#fff!important}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_stock_data(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info
    end = datetime.today()
    hist = tk.history(start=end - timedelta(days=5*365), end=end, interval="1mo")
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
    hist = hist[["Close"]].dropna()

    # Make index timezone-naive so datetime comparisons always work
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
        "ticker": ticker.upper(),
        "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(cp), 2),
        "prev_close": round(float(pc), 2),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "dividend": info.get("dividendYield"),
        "sector": info.get("sector", "N/A"),
        "country": info.get("country", "N/A"),
        "ma50": info.get("fiftyDayAverage"),
        "ma200": info.get("twoHundredDayAverage"),
        "beta": info.get("beta"),
        "periods": periods,
    }


def get_macro_data():
    tickers = {"VIX": "^VIX", "SP500": "^GSPC", "DXY": "DX-Y.NYB", "OIL": "CL=F", "GOLD": "GC=F", "TNX": "^TNX"}
    result = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if not hist.empty:
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Close"].iloc[-1])
                curr = float(hist["Close"].iloc[-1])
                result[name] = {"value": round(curr,2), "change": round(curr-prev,2), "pct": round((curr-prev)/prev*100,2)}
        except:
            result[name] = {"value": None, "change": 0, "pct": 0}
    return result


def fetch_news(ticker, company, sector):
    news = []
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={company.split()[0]}+stock&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=geopolitics+trade+sanctions+economy&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    ]
    seen = set()
    for url in feeds:
        try:
            for entry in feedparser.parse(url).entries[:6]:
                title = entry.get("title","")
                if title and title not in seen:
                    seen.add(title)
                    news.append({"title": title, "link": entry.get("link","#"), "published": entry.get("published","")})
        except:
            pass
        if len(news) >= 20:
            break
    return news[:20]


def score_sentiment(title):
    t = title.lower()
    pos = ["surge","rally","gain","rise","beat","record","growth","strong","up","bull","profit","upgrade","boom","recovery","rebound","high"]
    neg = ["fall","drop","crash","loss","miss","weak","down","bear","sell","downgrade","risk","war","sanction","tariff","inflation","recession","crisis","fear","cut","tension","conflict","ban","restrict","default","debt"]
    ps = sum(1 for w in pos if w in t)
    ns = sum(1 for w in neg if w in t)
    return "positive" if ps > ns else "negative" if ns > ps else "neutral"


def compute_technicals(hist):
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi": 50, "trend": "Neutral", "trend_pct": 0, "volatility": "Unknown", "vol_pct": 0}
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[-14:])  if len(gains)  >= 14 else np.mean(gains)
    al = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi = round(100 - 100/(1+ag/al), 1) if al else 50.0
    trend_pct = round((closes[-1]-closes[0])/closes[0]*100, 2)
    trend = ("Strong Uptrend" if trend_pct>20 else "Uptrend" if trend_pct>5 else
             "Strong Downtrend" if trend_pct<-20 else "Downtrend" if trend_pct<-5 else "Sideways")
    returns = np.diff(closes)/closes[:-1]
    vol_pct = round(float(np.std(returns)*np.sqrt(12)*100),1)
    volatility = "Very High" if vol_pct>60 else "High" if vol_pct>35 else "Medium" if vol_pct>15 else "Low"
    return {"rsi": rsi, "trend": trend, "trend_pct": trend_pct, "volatility": volatility, "vol_pct": vol_pct}


def compute_geo_risk(news, macro):
    neg_count = sum(1 for n in news if score_sentiment(n["title"]) == "negative")
    pos_count = sum(1 for n in news if score_sentiment(n["title"]) == "positive")
    total = len(news) or 1
    news_risk = round(neg_count/total*100)
    vix_val = (macro.get("VIX") or {}).get("value") or 20
    vix_risk = min(100, int(vix_val/50*100))
    geo_score = round(news_risk*0.6 + vix_risk*0.4)
    level = ("🔴 Critical" if geo_score>70 else "🟠 High" if geo_score>50 else "🟡 Elevated" if geo_score>30 else "🟢 Low")
    return {"score": geo_score, "level": level, "news_risk": news_risk, "neg_count": neg_count, "pos_count": pos_count, "vix_risk": vix_risk}


def compute_rec(tech, data, geo, macro):
    score = 50
    rsi, trend = tech["rsi"], tech["trend"]
    score += 15 if rsi<30 else 8 if rsi<45 else -15 if rsi>70 else -4 if rsi>55 else 0
    score += (15 if "Strong Uptrend" in trend else 8 if "Uptrend" in trend else
              -15 if "Strong Downtrend" in trend else -8 if "Downtrend" in trend else 0)
    cp = data["current_price"]
    m50 = data.get("ma50") or cp
    m200 = data.get("ma200") or cp
    score += 8 if cp>m50 else -8
    score += 8 if cp>m200 else -8
    geo_penalty = round(geo["score"]/100*25)
    score -= geo_penalty
    vix = (macro.get("VIX") or {}).get("value") or 20
    score -= 10 if vix>35 else 5 if vix>25 else 0
    score += 5 if vix<15 else 0
    oil_pct = (macro.get("OIL") or {}).get("pct") or 0
    if data.get("sector") in ("Energy","Materials"):
        score += 8 if oil_pct>2 else -8 if oil_pct<-2 else 0
    dxy_pct = (macro.get("DXY") or {}).get("pct") or 0
    score -= 5 if dxy_pct>1 else 0
    score = max(0, min(100, score))
    if score>=60:    rec, color, icon = "BUY",  "#10b981", "🚀"
    elif score<=38:  rec, color, icon = "SELL", "#ef4444", "⛔"
    else:            rec, color, icon = "HOLD", "#f59e0b", "⏸️"
    h = data["periods"]["5yr"]["Close"].values
    return {"rec": rec, "confidence": score, "color": color, "icon": icon,
            "support": round(float(np.percentile(h,15)),2),
            "resistance": round(float(np.percentile(h,85)),2),
            "geo_penalty": geo_penalty}


def simulate(hist, amount=100):
    c = hist["Close"].dropna().values.astype(float)
    if len(c)<2: return {"value": amount, "gain": 0, "pct": 0}
    value = round(amount/c[0]*c[-1], 2)
    return {"value": value, "gain": round(value-amount,2), "pct": round((c[-1]-c[0])/c[0]*100,2)}


def fmt_large(n):
    if n is None: return "N/A"
    n = float(n)
    if n>=1e12: return f"${n/1e12:.2f}T"
    if n>=1e9:  return f"${n/1e9:.2f}B"
    if n>=1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def make_chart(hist, ticker, label):
    closes = hist["Close"].dropna()
    is_up  = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color  = "#10b981" if is_up else "#ef4444"
    fill   = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=closes.index, y=closes, fill="tozeroy", fillcolor=fill,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>", name=ticker))
    if len(closes)>=6:
        fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(3).mean(),
            line=dict(color="rgba(139,92,246,.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="#070b14", plot_bgcolor="#0d1117",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {label}", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickfont=dict(size=10), showline=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickprefix="$", tickfont=dict(size=10), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=8,r=8,t=38,b=8), hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")), height=290)
    return fig


def get_ai_analysis(data, tech, rec, geo, macro, news):
    api_key = st.session_state.get("api_key","").strip()
    if not api_key:
        return "💡 Enter your Anthropic API key in the sidebar to unlock AI-generated analysis."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        inv5 = simulate(data["periods"]["5yr"])
        inv1 = simulate(data["periods"]["1yr"])
        top_news = "\n".join(f"- {n['title']}" for n in news[:8])
        vix = (macro.get("VIX") or {}).get("value","N/A")
        oil_val = (macro.get("OIL") or {}).get("value","N/A")
        oil_pct = (macro.get("OIL") or {}).get("pct",0)
        dxy = (macro.get("DXY") or {}).get("value","N/A")
        sp500_p = (macro.get("SP500") or {}).get("pct",0)
        tnx = (macro.get("TNX") or {}).get("value","N/A")
        prompt = f"""You are a senior equity research analyst specialising in geopolitical risk and macro analysis.
Write a professional 5-sentence analysis for {data['ticker']} ({data['company_name']}, sector: {data['sector']}, country: {data['country']}).

=== TECHNICAL DATA ===
Price: ${data['current_price']} | RSI: {tech['rsi']} | Trend: {tech['trend']} ({tech['trend_pct']}% 5yr)
Volatility: {tech['volatility']} ({tech['vol_pct']}% ann.) | Beta: {data.get('beta','N/A')}
MA50: {data.get('ma50','N/A')} | MA200: {data.get('ma200','N/A')}
$100 over 5yr = ${inv5['value']} ({inv5['pct']}%) | $100 over 1yr = ${inv1['value']} ({inv1['pct']}%)

=== MACRO ENVIRONMENT ===
VIX: {vix} | S&P 500 daily: {sp500_p:+.2f}% | Crude Oil: ${oil_val} ({oil_pct:+.2f}%) | DXY: {dxy} | 10yr Yield: {tnx}%

=== GEOPOLITICAL RISK ===
Geo-Risk Score: {geo['score']}/100 ({geo['level']}) | Geo penalty on score: -{rec['geo_penalty']} pts
Negative headlines: {geo['neg_count']}/{len(news)} | Positive: {geo['pos_count']}/{len(news)}

=== LATEST HEADLINES ===
{top_news}

=== SIGNAL ===
{rec['rec']} | Confidence: {rec['confidence']}% | Support: ${rec['support']} | Resistance: ${rec['resistance']}

Write exactly 5 sentences:
1. Overall 5-year performance with key numbers
2. Current technical picture (RSI, MAs, trend)
3. How the macro environment (VIX, oil, DXY, yields) specifically affects this stock/sector
4. Specific geopolitical risks from the headlines that could move this stock
5. Forward outlook with one-line risk reminder

Be data-specific, cite actual figures, connect macro/geopolitical events to this company's business model."""
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=600,
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
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...", label_visibility="collapsed")
    if api_key:
        st.session_state["api_key"] = api_key
    st.markdown("---")
    st.markdown("**📚 Beginner's Guide**")
    for title, body in [
        ("📊 Charts", "Price history. Rising line = stock gaining value. Purple dotted = 3-month moving average."),
        ("💵 $100 Sim", "What $100 invested at period start is worth at today's price."),
        ("🌐 Geo Risk", "Live score from news sentiment + VIX fear index. Higher = more danger to price."),
        ("😨 VIX", "Wall Street fear gauge. >30 = fear. >40 = panic. <15 = calm bull market."),
        ("🛢️ Oil/DXY", "Oil affects energy costs. Strong USD hurts multinational revenue abroad."),
        ("🤖 AI Score", "Combines technicals + geo risk + macro into BUY/HOLD/SELL with confidence %."),
        ("⚠️ Risk", "Not financial advice. Consult a licensed advisor before investing."),
    ]:
        st.markdown(f'<div class="guide"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)


# ── Search Bar ────────────────────────────────────────────────────────────────
st.markdown("""<div style="text-align:center;padding:8px 0 4px">
    <span style="font-size:26px;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">🌐 StockLens Pro</span>
    <div style="font-size:12px;color:#64748b;margin-top:2px">AI · Geopolitical Risk · Live Macro · News Sentiment</div>
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
            <span style="background:rgba(239,68,68,.12);color:#f87171;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">😨 VIX · Oil · DXY · Yields</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading market data, macro indicators & live news for **{final_ticker}**..."):
    try:
        data = get_stock_data(final_ticker)
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()
    macro = get_macro_data()
    news  = fetch_news(final_ticker, data["company_name"], data["sector"])
    tech  = compute_technicals(data["periods"]["5yr"])
    geo   = compute_geo_risk(news, macro)
    rec   = compute_rec(tech, data, geo, macro)

# ── Header ────────────────────────────────────────────────────────────────────
change = data["current_price"] - data["prev_close"]
change_pct = change/data["prev_close"]*100 if data["prev_close"] else 0
is_up = change >= 0
dcolor = "#10b981" if is_up else "#ef4444"

h1, h2 = st.columns([2,1])
with h1:
    st.markdown(f"""<div>
        <span style="font-size:30px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:10px;font-size:13px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">Sector: {data['sector']} &nbsp;|&nbsp; Country: {data['country']} &nbsp;|&nbsp; Beta: {data.get('beta','N/A')}</div>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""<div style="text-align:right">
        <div style="font-size:30px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:13px;font-weight:700;color:{dcolor}">{'▲' if is_up else '▼'} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)</div>
    </div>""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Key Metrics</div>', unsafe_allow_html=True)
mcols = st.columns(8)
for col, (lbl, val) in zip(mcols, [
    ("Market Cap", fmt_large(data["market_cap"])),
    ("P/E Ratio", f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High", f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low", f"${data['week52_low']:.2f}" if data["week52_low"] else "N/A"),
    ("Avg Volume", fmt_large(data["avg_volume"])),
    ("MA 50", f"${data['ma50']:.2f}" if data["ma50"] else "N/A"),
    ("MA 200", f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)", str(tech["rsi"])),
]):
    col.metric(lbl, val)

# ── Macro ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌍 Live Macro Environment</div>', unsafe_allow_html=True)
mcol = st.columns(6)
for col, (lbl, key, color) in zip(mcol, [
    ("😨 VIX Fear","VIX","#ef4444"), ("📈 S&P 500","SP500","#10b981"),
    ("🛢️ Crude Oil","OIL","#f59e0b"), ("💵 US Dollar","DXY","#3b82f6"),
    ("🏅 Gold","GOLD","#fbbf24"), ("📉 10yr Yield","TNX","#a78bfa"),
]):
    m = macro.get(key) or {}
    val = m.get("value")
    pct = m.get("pct",0)
    pos = pct >= 0
    col.markdown(f"""<div class="macro-tile">
        <div class="macro-lbl">{lbl}</div>
        <div class="macro-val" style="color:{color}">{f'{val:,.2f}' if val else 'N/A'}</div>
        <div class="macro-chg" style="color:{'#10b981' if pos else '#ef4444'}">{'▲' if pos else '▼'} {abs(pct):.2f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Geo Risk ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌐 Geopolitical Risk Intelligence</div>', unsafe_allow_html=True)
geo_color = "#ef4444" if geo["score"]>70 else "#f59e0b" if geo["score"]>40 else "#10b981"
gc1, gc2, gc3 = st.columns([1,1,2])

with gc1:
    st.markdown(f"""<div class="geo-card" style="border-color:{geo_color}33">
        <div style="font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:6px">🌐 Geo-Risk Score</div>
        <div style="font-size:38px;font-weight:900;color:{geo_color};margin:6px 0">{geo['score']}<span style="font-size:16px">/100</span></div>
        <div style="font-size:14px;font-weight:700;color:{geo_color}">{geo['level']}</div>
        <div class="risk-bar"><div class="risk-fill" style="width:{geo['score']}%;background:{geo_color}"></div></div>
        <div style="font-size:11px;color:#64748b;margin-top:8px">Composite: news sentiment + VIX index</div>
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
        <div style="font-size:11px;color:#64748b;margin-top:4px">VIX risk component: {geo['vix_risk']}%</div>
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
            <div class="news-meta">{n.get('published','')[:25]} <span class="news-sent {sc}">{sl}</span></div>
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

# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)
v1, v2 = st.columns([3,2])
with v1:
    vc = {"BUY":"rgba(16,185,129,.08)","SELL":"rgba(239,68,68,.08)","HOLD":"rgba(245,158,11,.08)"}[rec["rec"]]
    bc = {"BUY":"rgba(16,185,129,.3)","SELL":"rgba(239,68,68,.3)","HOLD":"rgba(245,158,11,.3)"}[rec["rec"]]
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
        <div style="font-size:11px;color:#64748b">Technicals (RSI, MA, trend) + Geo risk (-{rec['geo_penalty']} pts) + Macro (VIX, Oil, DXY)</div>
    </div>""", unsafe_allow_html=True)
with v2:
    for lbl, val, c in [
        ("Trend", tech["trend"], "#e2e8f0"),
        ("RSI", str(tech["rsi"]), "#10b981" if tech["rsi"]<30 else "#ef4444" if tech["rsi"]>70 else "#e2e8f0"),
        ("Volatility", tech["volatility"], "#e2e8f0"),
        ("Geo Risk", geo["level"], geo_color),
        ("Support", f"${rec['support']:,.2f}", "#10b981"),
        ("Resistance", f"${rec['resistance']:,.2f}", "#ef4444"),
    ]:
        st.markdown(f"""<div class="sig-row">
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{lbl}</span>
            <span style="font-size:13px;font-weight:700;color:{c}">{val}</span>
        </div>""", unsafe_allow_html=True)

# ── Full News ─────────────────────────────────────────────────────────────────
with st.expander("📰 Full News Feed — All Headlines"):
    for n in news:
        sent = score_sentiment(n["title"])
        sc   = "sent-pos" if sent=="positive" else "sent-neg" if sent=="negative" else "sent-neu"
        sl   = "↑ Positive" if sent=="positive" else "↓ Negative" if sent=="negative" else "→ Neutral"
        st.markdown(f"""<div class="news-card">
            <div class="news-title">{n['title']}</div>
            <div class="news-meta">{n.get('published','')[:30]} <span class="news-sent {sc}">{sl}</span></div>
        </div>""", unsafe_allow_html=True)

# ── AI Analysis ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📝 AI Deep Analysis — Technicals + Geopolitics + Macro</div>', unsafe_allow_html=True)
with st.spinner("Generating geopolitical & macro AI analysis..."):
    analysis = get_ai_analysis(data, tech, rec, geo, macro, news)
st.markdown(f"""<div class="card">
    <div style="color:#94a3b8;font-size:14px;line-height:1.85">{analysis}</div>
</div>
<div class="disc">⚠️ <b>Disclaimer:</b> Educational only. Data from Yahoo Finance (yfinance) & Google/Yahoo News RSS.
AI analysis by Claude. Not financial advice. Past performance ≠ future results. Consult a licensed financial advisor.</div>
""", unsafe_allow_html=True).sec{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:18px 0 10px;border-bottom:1px solid #1e2d45;padding-bottom:6px}
.card{background:#0d1117;border:1px solid #1e2d45;border-radius:12px;padding:16px}
.news-card{background:#0d1117;border:1px solid #1e2d45;border-radius:10px;padding:12px 14px;margin-bottom:8px}
.news-card:hover{border-color:#3b82f6}
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
.guide{background:#070b14;border:1px solid #1e2d45;border-radius:8px;padding:10px 12px;margin-bottom:8px}
.guide-title{font-size:12px;font-weight:700;color:#3b82f6;margin-bottom:3px}
.guide-body{font-size:11px;color:#64748b;line-height:1.55}
.disc{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:10px 14px;font-size:11px;color:#b45309;margin-top:10px}
.stTabs [data-baseweb="tab-list"]{background:#0d1117;border-radius:8px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:6px;font-weight:600;font-size:13px}
.stTabs [aria-selected="true"]{background:#3b82f6!important;color:#fff!important}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_stock_data(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info
    end = datetime.today()
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
        "ticker": ticker.upper(),
        "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(cp), 2),
        "prev_close": round(float(pc), 2),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "dividend": info.get("dividendYield"),
        "sector": info.get("sector", "N/A"),
        "country": info.get("country", "N/A"),
        "ma50": info.get("fiftyDayAverage"),
        "ma200": info.get("twoHundredDayAverage"),
        "beta": info.get("beta"),
        "periods": periods,
    }


def get_macro_data():
    tickers = {"VIX": "^VIX", "SP500": "^GSPC", "DXY": "DX-Y.NYB", "OIL": "CL=F", "GOLD": "GC=F", "TNX": "^TNX"}
    result = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if not hist.empty:
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Close"].iloc[-1])
                curr = float(hist["Close"].iloc[-1])
                result[name] = {"value": round(curr,2), "change": round(curr-prev,2), "pct": round((curr-prev)/prev*100,2)}
        except:
            result[name] = {"value": None, "change": 0, "pct": 0}
    return result


def fetch_news(ticker, company, sector):
    news = []
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={company.split()[0]}+stock&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=geopolitics+trade+sanctions+economy&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    ]
    seen = set()
    for url in feeds:
        try:
            for entry in feedparser.parse(url).entries[:6]:
                title = entry.get("title","")
                if title and title not in seen:
                    seen.add(title)
                    news.append({"title": title, "link": entry.get("link","#"), "published": entry.get("published","")})
        except:
            pass
        if len(news) >= 20:
            break
    return news[:20]


def score_sentiment(title):
    t = title.lower()
    pos = ["surge","rally","gain","rise","beat","record","growth","strong","up","bull","profit","upgrade","boom","recovery","rebound","high"]
    neg = ["fall","drop","crash","loss","miss","weak","down","bear","sell","downgrade","risk","war","sanction","tariff","inflation","recession","crisis","fear","cut","tension","conflict","ban","restrict","default","debt"]
    ps = sum(1 for w in pos if w in t)
    ns = sum(1 for w in neg if w in t)
    return "positive" if ps > ns else "negative" if ns > ps else "neutral"


def compute_technicals(hist):
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi": 50, "trend": "Neutral", "trend_pct": 0, "volatility": "Unknown", "vol_pct": 0}
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[-14:])  if len(gains)  >= 14 else np.mean(gains)
    al = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi = round(100 - 100/(1+ag/al), 1) if al else 50.0
    trend_pct = round((closes[-1]-closes[0])/closes[0]*100, 2)
    trend = ("Strong Uptrend" if trend_pct>20 else "Uptrend" if trend_pct>5 else
             "Strong Downtrend" if trend_pct<-20 else "Downtrend" if trend_pct<-5 else "Sideways")
    returns = np.diff(closes)/closes[:-1]
    vol_pct = round(float(np.std(returns)*np.sqrt(12)*100),1)
    volatility = "Very High" if vol_pct>60 else "High" if vol_pct>35 else "Medium" if vol_pct>15 else "Low"
    return {"rsi": rsi, "trend": trend, "trend_pct": trend_pct, "volatility": volatility, "vol_pct": vol_pct}


def compute_geo_risk(news, macro):
    neg_count = sum(1 for n in news if score_sentiment(n["title"]) == "negative")
    pos_count = sum(1 for n in news if score_sentiment(n["title"]) == "positive")
    total = len(news) or 1
    news_risk = round(neg_count/total*100)
    vix_val = (macro.get("VIX") or {}).get("value") or 20
    vix_risk = min(100, int(vix_val/50*100))
    geo_score = round(news_risk*0.6 + vix_risk*0.4)
    level = ("🔴 Critical" if geo_score>70 else "🟠 High" if geo_score>50 else "🟡 Elevated" if geo_score>30 else "🟢 Low")
    return {"score": geo_score, "level": level, "news_risk": news_risk, "neg_count": neg_count, "pos_count": pos_count, "vix_risk": vix_risk}


def compute_rec(tech, data, geo, macro):
    score = 50
    rsi, trend = tech["rsi"], tech["trend"]
    score += 15 if rsi<30 else 8 if rsi<45 else -15 if rsi>70 else -4 if rsi>55 else 0
    score += (15 if "Strong Uptrend" in trend else 8 if "Uptrend" in trend else
              -15 if "Strong Downtrend" in trend else -8 if "Downtrend" in trend else 0)
    cp = data["current_price"]
    m50 = data.get("ma50") or cp
    m200 = data.get("ma200") or cp
    score += 8 if cp>m50 else -8
    score += 8 if cp>m200 else -8
    geo_penalty = round(geo["score"]/100*25)
    score -= geo_penalty
    vix = (macro.get("VIX") or {}).get("value") or 20
    score -= 10 if vix>35 else 5 if vix>25 else 0
    score += 5 if vix<15 else 0
    oil_pct = (macro.get("OIL") or {}).get("pct") or 0
    if data.get("sector") in ("Energy","Materials"):
        score += 8 if oil_pct>2 else -8 if oil_pct<-2 else 0
    dxy_pct = (macro.get("DXY") or {}).get("pct") or 0
    score -= 5 if dxy_pct>1 else 0
    score = max(0, min(100, score))
    if score>=60:    rec, color, icon = "BUY",  "#10b981", "🚀"
    elif score<=38:  rec, color, icon = "SELL", "#ef4444", "⛔"
    else:            rec, color, icon = "HOLD", "#f59e0b", "⏸️"
    h = data["periods"]["5yr"]["Close"].values
    return {"rec": rec, "confidence": score, "color": color, "icon": icon,
            "support": round(float(np.percentile(h,15)),2),
            "resistance": round(float(np.percentile(h,85)),2),
            "geo_penalty": geo_penalty}


def simulate(hist, amount=100):
    c = hist["Close"].dropna().values.astype(float)
    if len(c)<2: return {"value": amount, "gain": 0, "pct": 0}
    value = round(amount/c[0]*c[-1], 2)
    return {"value": value, "gain": round(value-amount,2), "pct": round((c[-1]-c[0])/c[0]*100,2)}


def fmt_large(n):
    if n is None: return "N/A"
    n = float(n)
    if n>=1e12: return f"${n/1e12:.2f}T"
    if n>=1e9:  return f"${n/1e9:.2f}B"
    if n>=1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def make_chart(hist, ticker, label):
    closes = hist["Close"].dropna()
    is_up  = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color  = "#10b981" if is_up else "#ef4444"
    fill   = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=closes.index, y=closes, fill="tozeroy", fillcolor=fill,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>", name=ticker))
    if len(closes)>=6:
        fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(3).mean(),
            line=dict(color="rgba(139,92,246,.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="#070b14", plot_bgcolor="#0d1117",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {label}", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickfont=dict(size=10), showline=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickprefix="$", tickfont=dict(size=10), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=8,r=8,t=38,b=8), hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")), height=290)
    return fig


def get_ai_analysis(data, tech, rec, geo, macro, news):
    api_key = st.session_state.get("api_key","").strip()
    if not api_key:
        return "💡 Enter your Anthropic API key in the sidebar to unlock AI-generated analysis."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        inv5 = simulate(data["periods"]["5yr"])
        inv1 = simulate(data["periods"]["1yr"])
        top_news = "\n".join(f"- {n['title']}" for n in news[:8])
        vix = (macro.get("VIX") or {}).get("value","N/A")
        oil_val = (macro.get("OIL") or {}).get("value","N/A")
        oil_pct = (macro.get("OIL") or {}).get("pct",0)
        dxy = (macro.get("DXY") or {}).get("value","N/A")
        sp500_p = (macro.get("SP500") or {}).get("pct",0)
        tnx = (macro.get("TNX") or {}).get("value","N/A")
        prompt = f"""You are a senior equity research analyst specialising in geopolitical risk and macro analysis.
Write a professional 5-sentence analysis for {data['ticker']} ({data['company_name']}, sector: {data['sector']}, country: {data['country']}).

=== TECHNICAL DATA ===
Price: ${data['current_price']} | RSI: {tech['rsi']} | Trend: {tech['trend']} ({tech['trend_pct']}% 5yr)
Volatility: {tech['volatility']} ({tech['vol_pct']}% ann.) | Beta: {data.get('beta','N/A')}
MA50: {data.get('ma50','N/A')} | MA200: {data.get('ma200','N/A')}
$100 over 5yr = ${inv5['value']} ({inv5['pct']}%) | $100 over 1yr = ${inv1['value']} ({inv1['pct']}%)

=== MACRO ENVIRONMENT ===
VIX: {vix} | S&P 500 daily: {sp500_p:+.2f}% | Crude Oil: ${oil_val} ({oil_pct:+.2f}%) | DXY: {dxy} | 10yr Yield: {tnx}%

=== GEOPOLITICAL RISK ===
Geo-Risk Score: {geo['score']}/100 ({geo['level']}) | Geo penalty on score: -{rec['geo_penalty']} pts
Negative headlines: {geo['neg_count']}/{len(news)} | Positive: {geo['pos_count']}/{len(news)}

=== LATEST HEADLINES ===
{top_news}

=== SIGNAL ===
{rec['rec']} | Confidence: {rec['confidence']}% | Support: ${rec['support']} | Resistance: ${rec['resistance']}

Write exactly 5 sentences:
1. Overall 5-year performance with key numbers
2. Current technical picture (RSI, MAs, trend)
3. How the macro environment (VIX, oil, DXY, yields) specifically affects this stock/sector
4. Specific geopolitical risks from the headlines that could move this stock
5. Forward outlook with one-line risk reminder

Be data-specific, cite actual figures, connect macro/geopolitical events to this company's business model."""
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=600,
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
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...", label_visibility="collapsed")
    if api_key:
        st.session_state["api_key"] = api_key
    st.markdown("---")
    st.markdown("**📚 Beginner's Guide**")
    for title, body in [
        ("📊 Charts", "Price history. Rising line = stock gaining value. Purple dotted = 3-month moving average."),
        ("💵 $100 Sim", "What $100 invested at period start is worth at today's price."),
        ("🌐 Geo Risk", "Live score from news sentiment + VIX fear index. Higher = more danger to price."),
        ("😨 VIX", "Wall Street fear gauge. >30 = fear. >40 = panic. <15 = calm bull market."),
        ("🛢️ Oil/DXY", "Oil affects energy costs. Strong USD hurts multinational revenue abroad."),
        ("🤖 AI Score", "Combines technicals + geo risk + macro into BUY/HOLD/SELL with confidence %."),
        ("⚠️ Risk", "Not financial advice. Consult a licensed advisor before investing."),
    ]:
        st.markdown(f'<div class="guide"><div class="guide-title">{title}</div><div class="guide-body">{body}</div></div>', unsafe_allow_html=True)


# ── Search Bar ────────────────────────────────────────────────────────────────
st.markdown("""<div style="text-align:center;padding:8px 0 4px">
    <span style="font-size:26px;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">🌐 StockLens Pro</span>
    <div style="font-size:12px;color:#64748b;margin-top:2px">AI · Geopolitical Risk · Live Macro · News Sentiment</div>
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
            <span style="background:rgba(239,68,68,.12);color:#f87171;font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px">😨 VIX · Oil · DXY · Yields</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading market data, macro indicators & live news for **{final_ticker}**..."):
    try:
        data = get_stock_data(final_ticker)
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()
    macro = get_macro_data()
    news  = fetch_news(final_ticker, data["company_name"], data["sector"])
    tech  = compute_technicals(data["periods"]["5yr"])
    geo   = compute_geo_risk(news, macro)
    rec   = compute_rec(tech, data, geo, macro)

# ── Header ────────────────────────────────────────────────────────────────────
change = data["current_price"] - data["prev_close"]
change_pct = change/data["prev_close"]*100 if data["prev_close"] else 0
is_up = change >= 0
dcolor = "#10b981" if is_up else "#ef4444"

h1, h2 = st.columns([2,1])
with h1:
    st.markdown(f"""<div>
        <span style="font-size:30px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:10px;font-size:13px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">Sector: {data['sector']} &nbsp;|&nbsp; Country: {data['country']} &nbsp;|&nbsp; Beta: {data.get('beta','N/A')}</div>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""<div style="text-align:right">
        <div style="font-size:30px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:13px;font-weight:700;color:{dcolor}">{'▲' if is_up else '▼'} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)</div>
    </div>""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Key Metrics</div>', unsafe_allow_html=True)
mcols = st.columns(8)
for col, (lbl, val) in zip(mcols, [
    ("Market Cap", fmt_large(data["market_cap"])),
    ("P/E Ratio", f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High", f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low", f"${data['week52_low']:.2f}" if data["week52_low"] else "N/A"),
    ("Avg Volume", fmt_large(data["avg_volume"])),
    ("MA 50", f"${data['ma50']:.2f}" if data["ma50"] else "N/A"),
    ("MA 200", f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)", str(tech["rsi"])),
]):
    col.metric(lbl, val)

# ── Macro ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌍 Live Macro Environment</div>', unsafe_allow_html=True)
mcol = st.columns(6)
for col, (lbl, key, color) in zip(mcol, [
    ("😨 VIX Fear","VIX","#ef4444"), ("📈 S&P 500","SP500","#10b981"),
    ("🛢️ Crude Oil","OIL","#f59e0b"), ("💵 US Dollar","DXY","#3b82f6"),
    ("🏅 Gold","GOLD","#fbbf24"), ("📉 10yr Yield","TNX","#a78bfa"),
]):
    m = macro.get(key) or {}
    val = m.get("value")
    pct = m.get("pct",0)
    pos = pct >= 0
    col.markdown(f"""<div class="macro-tile">
        <div class="macro-lbl">{lbl}</div>
        <div class="macro-val" style="color:{color}">{f'{val:,.2f}' if val else 'N/A'}</div>
        <div class="macro-chg" style="color:{'#10b981' if pos else '#ef4444'}">{'▲' if pos else '▼'} {abs(pct):.2f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Geo Risk ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌐 Geopolitical Risk Intelligence</div>', unsafe_allow_html=True)
geo_color = "#ef4444" if geo["score"]>70 else "#f59e0b" if geo["score"]>40 else "#10b981"
gc1, gc2, gc3 = st.columns([1,1,2])

with gc1:
    st.markdown(f"""<div class="geo-card" style="border-color:{geo_color}33">
        <div style="font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:6px">🌐 Geo-Risk Score</div>
        <div style="font-size:38px;font-weight:900;color:{geo_color};margin:6px 0">{geo['score']}<span style="font-size:16px">/100</span></div>
        <div style="font-size:14px;font-weight:700;color:{geo_color}">{geo['level']}</div>
        <div class="risk-bar"><div class="risk-fill" style="width:{geo['score']}%;background:{geo_color}"></div></div>
        <div style="font-size:11px;color:#64748b;margin-top:8px">Composite: news sentiment + VIX index</div>
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
        <div style="font-size:11px;color:#64748b;margin-top:4px">VIX risk component: {geo['vix_risk']}%</div>
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
            <div class="news-meta">{n.get('published','')[:25]} <span class="news-sent {sc}">{sl}</span></div>
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

# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)
v1, v2 = st.columns([3,2])
with v1:
    vc = {"BUY":"rgba(16,185,129,.08)","SELL":"rgba(239,68,68,.08)","HOLD":"rgba(245,158,11,.08)"}[rec["rec"]]
    bc = {"BUY":"rgba(16,185,129,.3)","SELL":"rgba(239,68,68,.3)","HOLD":"rgba(245,158,11,.3)"}[rec["rec"]]
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
        <div style="font-size:11px;color:#64748b">Technicals (RSI, MA, trend) + Geo risk (-{rec['geo_penalty']} pts) + Macro (VIX, Oil, DXY)</div>
    </div>""", unsafe_allow_html=True)
with v2:
    for lbl, val, c in [
        ("Trend", tech["trend"], "#e2e8f0"),
        ("RSI", str(tech["rsi"]), "#10b981" if tech["rsi"]<30 else "#ef4444" if tech["rsi"]>70 else "#e2e8f0"),
        ("Volatility", tech["volatility"], "#e2e8f0"),
        ("Geo Risk", geo["level"], geo_color),
        ("Support", f"${rec['support']:,.2f}", "#10b981"),
        ("Resistance", f"${rec['resistance']:,.2f}", "#ef4444"),
    ]:
        st.markdown(f"""<div class="sig-row">
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{lbl}</span>
            <span style="font-size:13px;font-weight:700;color:{c}">{val}</span>
        </div>""", unsafe_allow_html=True)

# ── Full News ─────────────────────────────────────────────────────────────────
with st.expander("📰 Full News Feed — All Headlines"):
    for n in news:
        sent = score_sentiment(n["title"])
        sc   = "sent-pos" if sent=="positive" else "sent-neg" if sent=="negative" else "sent-neu"
        sl   = "↑ Positive" if sent=="positive" else "↓ Negative" if sent=="negative" else "→ Neutral"
        st.markdown(f"""<div class="news-card">
            <div class="news-title">{n['title']}</div>
            <div class="news-meta">{n.get('published','')[:30]} <span class="news-sent {sc}">{sl}</span></div>
        </div>""", unsafe_allow_html=True)

# ── AI Analysis ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📝 AI Deep Analysis — Technicals + Geopolitics + Macro</div>', unsafe_allow_html=True)
with st.spinner("Generating geopolitical & macro AI analysis..."):
    analysis = get_ai_analysis(data, tech, rec, geo, macro, news)
st.markdown(f"""<div class="card">
    <div style="color:#94a3b8;font-size:14px;line-height:1.85">{analysis}</div>
</div>
<div class="disc">⚠️ <b>Disclaimer:</b> Educational only. Data from Yahoo Finance (yfinance) & Google/Yahoo News RSS.
AI analysis by Claude. Not financial advice. Past performance ≠ future results. Consult a licensed financial advisor.</div>
""", unsafe_allow_html=True)
