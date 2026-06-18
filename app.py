import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1C2333;
    padding:10px;
    border-radius:15px;
}

h1,h2,h3 {
    color:white;
}

.explainer {
    background-color: #161B22;
    border-left: 3px solid #00CC96;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.92rem;
    color: #C9D1D9;
    margin-bottom: 14px;
}

</style>
""", unsafe_allow_html=True)


def explain(text: str):
    """Render a short plain-language explainer box under a section."""
    st.markdown(f"<div class='explainer'>💡 {text}</div>", unsafe_allow_html=True)


# ============================================================
# CHART HELPERS  (fixes the histogram readability/zoom-drag issue)
# ============================================================

PALETTE = ["#00CC96", "#636EFA", "#FFA15A", "#EF553B", "#AB63FA", "#19D3F3"]


def make_histogram(values, title, color, x_as_percent=False):
    """
    Build a clean, high-contrast histogram that:
      - uses solid, clearly distinguishable bars (not Plotly's washed-out default)
      - locks the axes so dragging/zooming can't distort or 'spread' the chart
      - matches the app's dark theme so bars never blend into the background
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            marker=dict(
                color=color,
                line=dict(color="white", width=0.6),
            ),
            opacity=0.9,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        bargap=0.06,
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )

    # Lock the axes: this is what stops the chart from "fading / spreading"
    # when a user accidentally drags or zooms on it.
    fig.update_xaxes(fixedrange=True, tickformat=".1%" if x_as_percent else None)
    fig.update_yaxes(fixedrange=True, title="Frequency")

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("FinSight AI")
st.subheader("Advanced Stock Analysis Platform")
st.caption(
    " Disclaimer: Educational tool only, not financial advice. Every chart below is based on "
    "historical data and statistical assumptions, neither of which guarantee future results."
)

# ============================================================
# SIDEBAR EDUCATION PANEL
# ============================================================

with st.sidebar:

    st.header("📚 Learn")

    st.markdown("""
### Histogram
Shows how often returns of a certain size occurred. Taller bars near zero = calmer stock.

### Volatility
How much the price swings around, day to day. Higher = riskier, bumpier ride.

### Standard Deviation
The numeric version of volatility, the average distance returns stray from the mean.

### Skewness
Whether big moves tend to be more positive (right-skew) or more negative (left-skew).

### Kurtosis
How likely extreme, "fat-tail" days are, compared to a normal bell curve. Higher = more shock-prone.

### Monte Carlo
Replays the stock's historical behavior thousands of times to map out a range of plausible future prices, it's a probability spread, not a prediction.
""")

# ============================================================
# WATCHLIST / TOP PERFORMERS  (replaces the hardcoded NVDA default)
# ============================================================

WATCHLIST = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "LLY"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_watchlist_performance(tickers):
    rows = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1y")
            if len(hist) > 1:
                ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                rows.append({"Ticker": t, "Return": ret})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Return"])
    return pd.DataFrame(rows).sort_values("Return", ascending=False).reset_index(drop=True)


if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

st.header("Top Performing Stocks")
explain(
    "This ranks a curated watchlist of well-known, heavily-traded large-cap stocks by their "
    "actual 1-year price return. It's not a scan of the entire market, just a quick-start list. "
    "Click any ticker to load its full analysis below, or search for any other stock further down."
)

with st.spinner("Fetching 1-year performance for the watchlist..."):
    perf_df = get_watchlist_performance(WATCHLIST)

if not perf_df.empty:
    top_cols = st.columns(5)
    for i, row in enumerate(perf_df.head(5).itertuples(index=False)):
        with top_cols[i % 5]:
            arrow = "🟢" if row.Return >= 0 else "🔴"
            if st.button(
                f"{arrow} {row.Ticker}\n{row.Return:+.1f}%",
                key=f"top_{row.Ticker}",
                use_container_width=True,
            ):
                st.session_state.selected_ticker = row.Ticker

st.divider()

# ============================================================
# SEARCH ANY STOCK
# ============================================================

search_col, btn_col = st.columns([4, 1])

with search_col:
    typed = st.text_input(
        "Or search for any stock symbol",
        placeholder="e.g. AAPL, TSLA, INFY.NS, RELIANCE.NS",
        value=""
    )

with btn_col:
    st.write("")
    if st.button("Analyze", use_container_width=True):
        if typed.strip():
            st.session_state.selected_ticker = typed.strip().upper()

ticker = st.session_state.selected_ticker

if not ticker:
    st.info("👆 Pick a stock above or type a symbol to begin, nothing loads automatically.")
    st.stop()

top_bar_l, top_bar_r = st.columns([5, 1])
with top_bar_l:
    st.markdown(f"## 🔎 Analysis for **{ticker}**")
with top_bar_r:
    if st.button("🔄 Change stock"):
        st.session_state.selected_ticker = None
        st.rerun()

# ============================================================
# MAIN ANALYSIS
# ============================================================

try:

    @st.cache_data(ttl=900, show_spinner=False)
    def load_history(symbol, period="5y"):
        return yf.Ticker(symbol).history(period=period)

    data = load_history(ticker, "5y")

    if data.empty:
        st.error("No data found for that symbol. Double-check the ticker (e.g. add an exchange suffix like .NS or .L if it's not a US stock).")
        st.stop()

    current_price = data["Close"].iloc[-1]

    st.success(f"Current Price: ${current_price:.2f}")

    # --------------------
    # PRICE CHART
    # --------------------

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            line=dict(color="#00CC96", width=2),
            name=ticker,
        )
    )
    price_fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} Price History (5 Years)",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=400,
    )

    st.plotly_chart(price_fig, use_container_width=True)

    explain(
        f"This line tracks {ticker}'s closing price each trading day over the last 5 years. "
        "Use it to spot the overall trend (up, down, or sideways) and to see how the stock behaved "
        "through past rallies, corrections, or crashes."
    )

    # --------------------
    # INVESTMENT RETURNS
    # --------------------

    st.header("Investment Growth")

    periods_years = [1, 2, 3, 4, 5]
    growth_rows = []

    last_date = data.index[-1]

    for year in periods_years:
        cutoff = last_date - pd.DateOffset(years=year)
        hist = data[data.index >= cutoff]

        if len(hist) > 1:
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            value = 100 * (end_price / start_price)
            profit = value - 100

            growth_rows.append([f"{year} Year{'s' if year > 1 else ''}", round(value, 2), round(profit, 2)])

    growth_df = pd.DataFrame(
        growth_rows,
        columns=["Period", "Current Value of $100", "Profit/Loss"]
    )

    st.dataframe(growth_df, use_container_width=True)

    explain(
        "This shows what a hypothetical $100 invested N years ago would be worth today, based purely "
        "on price change (dividends not included). It's a fast way to compare how the stock performed "
        "across different holding periods."
    )

    # --------------------
    # HISTOGRAMS
    # --------------------

    st.header("Return Histograms")

    explain(
        "Each chart shows how daily returns were distributed over that time window. A tall bar near "
        "zero means most days had small moves; a wide spread means the stock is more volatile. "
        "**Mean** is the average daily return. **Std Dev** measures how much returns swing around that "
        "average (higher = more volatile). **Skewness** shows whether extreme days lean more positive "
        "or negative. **Kurtosis** shows how often extreme outlier days occur versus a normal bell curve "
        "(higher = fatter tails, more shock-prone)."
    )

    cols = st.columns(2)

    for i, year in enumerate(periods_years):
        cutoff = last_date - pd.DateOffset(years=year)
        hist = data[data.index >= cutoff]

        returns = hist["Close"].pct_change().dropna()

        if returns.empty:
            continue

        fig = make_histogram(
            returns,
            title=f"{year} Year Return Distribution",
            color=PALETTE[i % len(PALETTE)],
            x_as_percent=True,
        )

        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"**Mean:** {returns.mean():.4f} &nbsp;|&nbsp; "
                f"**Std Dev:** {returns.std():.4f} &nbsp;|&nbsp; "
                f"**Skewness:** {skew(returns):.2f} &nbsp;|&nbsp; "
                f"**Kurtosis:** {kurtosis(returns):.2f}"
            )

    # --------------------
    # BUY / SELL ENGINE
    # --------------------

    st.header("AI Probability Engine")

    returns_full = data["Close"].pct_change().dropna()

    positive_days = (returns_full > 0).mean()
    probability_up = positive_days * 100

    annual_return = ((current_price / data["Close"].iloc[0]) - 1) * 100

    volatility = (returns_full.std() * np.sqrt(252)) * 100

    score = 50
    if annual_return > 20:
        score += 15
    if probability_up > 55:
        score += 15
    if volatility < 40:
        score += 10
    score = min(score, 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability Up", f"{probability_up:.1f}%")
    c2.metric("Annual Return", f"{annual_return:.1f}%")
    c3.metric("AI Score", score)

    if score >= 80:
        st.success("🟢 Strong Buy Candidate")
    elif score >= 65:
        st.info("🔵 Moderate Buy Candidate")
    elif score >= 50:
        st.warning("🟡 Hold / Wait")
    else:
        st.error("🔴 Avoid Currently")

    explain(
        "This is a simple rule-of-thumb score, not a predictive model. It blends how often the stock "
        "closed up historically, its annualized return, and its volatility into one 0-100 number. "
        "Treat it as a quick gut-check to prompt further research, not as a buy/sell signal on its own."
    )

    # --------------------
    # MONTE CARLO
    # --------------------

    st.header("Monte Carlo Forecast")

    simulations = 1000
    days = 252

    mean = returns_full.mean()
    std = returns_full.std()

    # Vectorized simulation (much faster than the nested-loop version,
    # and produces the same kind of result: a spread of plausible future prices).
    shocks = np.random.normal(mean, std, size=(simulations, days))
    price_paths = current_price * np.cumprod(1 + shocks, axis=1)
    results = price_paths[:, -1]

    expected = np.mean(results)
    best = np.percentile(results, 95)
    worst = np.percentile(results, 5)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Price", f"${expected:.2f}")
    col2.metric("Best Case (95th %)", f"${best:.2f}")
    col3.metric("Worst Case (5th %)", f"${worst:.2f}")

    mc_fig = make_histogram(
        results,
        title="Simulated Price in 1 Year (1,000 runs)",
        color="#19D3F3",
    )
    st.plotly_chart(mc_fig, use_container_width=True)

    explain(
        "A Monte Carlo simulation replays the stock's historical average return and volatility forward "
        "thousands of times to map out a range of plausible prices one year from now. 'Expected' is the "
        "average outcome; 'Best' and 'Worst' case are the 95th and 5th percentile outcomes. Real markets "
        "can still land outside this range, this is a probability spread, not a guarantee."
    )

except Exception as e:
    st.error(f"Something went wrong while analyzing {ticker}: {e}")
