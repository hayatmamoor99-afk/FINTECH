"""
FinTech Analytics Pro
Designed by Mamoor Hayat
Copyright © 2026 All Rights Reserved
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Helper Functions
# ------------------------------

def calculate_rsi(data, window=14):
    """Calculate RSI indicator."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD, signal line, and histogram."""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def optimal_bins(data):
    """Freedman-Diaconis rule for optimal bin width."""
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    n = len(data)
    bin_width = 2 * iqr / (n ** (1/3)) if iqr > 0 else (data.max() - data.min()) / 30
    if bin_width == 0:
        bin_width = 1
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(5, min(bins, 50))

def format_currency(value):
    """Format currency with $ sign and commas."""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percent(value):
    """Format percentage."""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}%"

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="FinTech Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS – Updated for Readability
# ------------------------------
st.markdown("""
<style>
    /* Global */
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #1a237e !important;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(135deg, #e8eaf6, #c5cae9);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: #1a237e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Cards – now with white background and black text */
    .card {
        background: white !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border-left: 4px solid #1a237e;
        color: #000000 !important;
    }
    .card h4 {
        color: #000000 !important;
        font-weight: 700;
    }
    .card p, .card div, .card span {
        color: #000000 !important;
    }
    .card-gain {
        border-left-color: #2e7d32;
    }
    .card-loss {
        border-left-color: #c62828;
    }
    /* Recommendation boxes – high contrast */
    .rec-buy {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 1.2rem;
        border-radius: 10px;
        color: #1b5e20;
        font-weight: 500;
    }
    .rec-buy h3 {
        color: #1b5e20;
        margin: 0;
        font-size: 1.8rem;
    }
    .rec-hold {
        background: #fff3e0;
        border-left: 6px solid #f57c00;
        padding: 1.2rem;
        border-radius: 10px;
        color: #bf360c;
        font-weight: 500;
    }
    .rec-hold h3 {
        color: #bf360c;
        margin: 0;
        font-size: 1.8rem;
    }
    .rec-dont {
        background: #fce4ec;
        border-left: 6px solid #c62828;
        padding: 1.2rem;
        border-radius: 10px;
        color: #b71c1c;
        font-weight: 500;
    }
    .rec-dont h3 {
        color: #b71c1c;
        margin: 0;
        font-size: 1.8rem;
    }
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        text-align: center;
        margin: 0.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #5c6bc0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        margin: 0.2rem 0;
    }
    /* Buttons */
    .stButton > button {
        background: #1a237e;
        color: white;
        border-radius: 25px;
        padding: 0.4rem 2rem;
        font-weight: 500;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: #283593;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(26,35,126,0.2);
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #f3f4f9;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #e8eaf6, #c5cae9);
        border-radius: 15px;
        margin-top: 2rem;
        color: #1a237e;
        font-weight: 500;
    }
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a237e;
        background: #e8eaf6;
        border-radius: 8px;
    }
    /* About section in sidebar – white background, black text */
    .about-box {
        background: white !important;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        color: #000000 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .about-box b, .about-box p, .about-box span {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding: 0.5rem;">
    <h3 style="color:#1a237e;">🔍 Search</h3>
</div>
""", unsafe_allow_html=True)

symbol = st.sidebar.text_input("Stock / Crypto Symbol:", value="AAPL").upper()

st.sidebar.markdown("#### Top 5 Stocks")
cols_stocks = st.sidebar.columns(3)
top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
for i, sym in enumerate(top_stocks):
    if cols_stocks[i % 3].button(sym, key=f"stock_{sym}"):
        symbol = sym

st.sidebar.markdown("#### Top 5 Cryptocurrencies")
cols_crypto = st.sidebar.columns(3)
top_crypto = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
for i, sym in enumerate(top_crypto):
    if cols_crypto[i % 3].button(sym, key=f"crypto_{sym}"):
        symbol = sym

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="about-box">
    <b>📘 About</b><br><br>
    This app provides in‑depth analysis of any stock or cryptocurrency.<br><br>
    ✅ 5‑year histograms<br>
    ✅ Investment simulation<br>
    ✅ Buy/Hold/Don't buy recommendation<br>
    ✅ Advanced indicators (RSI, MACD, moving averages)<br><br>
    <span style="font-size:0.85rem;">
        Data from Yahoo Finance • Not financial advice
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align:center; color:#5c6bc0; font-size:0.8rem;">
    Designed by <b>Mamoor Hayat</b><br>
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Main App
# ------------------------------
st.markdown('<div class="header-title"> FinTech Analytics Pro</div>', unsafe_allow_html=True)

if not symbol:
    st.info("Enter a symbol in the sidebar to begin.")
    st.stop()

# ------------------------------
# Data Fetching
# ------------------------------
with st.spinner(f"Fetching data for {symbol}..."):
    try:
        ticker = yf.Ticker(symbol)
        # Current price
        current_data = ticker.history(period="1d")
        current_price = current_data['Close'].iloc[-1] if not current_data.empty else None

        # Historical data for periods
        end_date = datetime.now()
        periods = {
            '1 Year': 365,
            '2 Years': 730,
            '3 Years': 1095,
            '4 Years': 1460,
            '5 Years': 1825
        }
        historical = {}
        for name, days in periods.items():
            start = end_date - timedelta(days=days)
            df = ticker.history(start=start, end=end_date)
            if not df.empty:
                historical[name] = df

        # Company info
        try:
            info = ticker.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', None)
        except:
            company_name = symbol
            sector = 'N/A'
            industry = 'N/A'
            market_cap = None

    except Exception as e:
        st.error(f"Error fetching data for {symbol}. Please check the symbol.")
        st.error(f"Details: {e}")
        st.stop()

# ------------------------------
# Company Info Cards – Updated with inline black styling
# ------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="card">
        <h4 style="margin:0; color:black; font-weight:bold;">Company</h4>
        <p style="font-size:1.2rem; font-weight:700; margin:0.2rem 0; color:black;">{company_name}</p>
        <p style="color:#333; margin:0; font-weight:500;">{symbol}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    price_display = format_currency(current_price) if current_price else "N/A"
    st.markdown(f"""
    <div class="card">
        <h4 style="margin:0; color:black; font-weight:bold;"> Current Price</h4>
        <p style="font-size:2rem; font-weight:700; color:black; margin:0.2rem 0;">{price_display}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="card">
        <h4 style="margin:0; color:black; font-weight:bold;"> Sector</h4>
        <p style="font-size:1.2rem; font-weight:600; margin:0.2rem 0; color:black;">{sector if sector != 'N/A' else '—'}</p>
        <p style="color:#333; margin:0; font-weight:500;">{industry if industry != 'N/A' else '—'}</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    cap_display = format_currency(market_cap) if market_cap else "N/A"
    st.markdown(f"""
    <div class="card">
        <h4 style="margin:0; color:black; font-weight:bold;"> Market Cap</h4>
        <p style="font-size:1.2rem; font-weight:700; margin:0.2rem 0; color:black;">{cap_display}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Histograms and Analysis by Period
# ------------------------------
st.markdown("## Historical Price Distribution")
st.markdown("*Interactive histograms for each period – zoom, pan, and hover for details.*")

if not historical:
    st.warning("No historical data found for this symbol.")
    st.stop()

tabs = st.tabs(["📅 1 Year", "📅 2 Years", "📅 3 Years", "📅 4 Years", "📅 5 Years"])

for tab, (period_name, df) in zip(tabs, historical.items()):
    with tab:
        # Prepare data
        prices = df['Close']
        returns = prices.pct_change().dropna() * 100
        investment_value = 100 * (prices / prices.iloc[0])

        # Statistics
        mean_price = prices.mean()
        median_price = prices.median()
        min_price = prices.min()
        max_price = prices.max()
        std_price = prices.std()
        skew = prices.skew()
        kurt = prices.kurtosis()
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1

        # Optimal bins
        n_bins = optimal_bins(prices)

        # Create histogram
        fig = go.Figure()
        hist, bin_edges = np.histogram(prices, bins=n_bins)
        # Color bars based on price level (low->red, mid->orange, high->green)
        norm_prices = (bin_edges[:-1] - prices.min()) / (prices.max() - prices.min() + 1e-9)
        colors = []
        for val in norm_prices:
            if val < 0.4:
                colors.append('#c62828')
            elif val < 0.7:
                colors.append('#ff9800')
            else:
                colors.append('#2e7d32')

        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=hist,
            width=[bin_edges[i+1]-bin_edges[i] for i in range(len(bin_edges)-1)],
            marker_color=colors,
            hovertemplate='<b>Price range:</b> $%{x:.2f} – $%{customdata[0]:.2f}<br>' +
                          '<b>Days:</b> %{y}<extra></extra>',
            customdata=[[bin_edges[i+1]] for i in range(len(bin_edges)-1)]
        ))

        # Reference lines
        if current_price:
            fig.add_vline(x=current_price, line_dash='dash', line_color='#1a237e',
                          line_width=3, annotation_text=f"Current: {format_currency(current_price)}",
                          annotation_position="top")
        fig.add_vline(x=mean_price, line_dash='dot', line_color='#1565c0',
                      line_width=2, annotation_text=f"Mean: {format_currency(mean_price)}",
                      annotation_position="bottom")
        fig.add_vline(x=median_price, line_dash='dashdot', line_color='#6a1b9a',
                      line_width=2, annotation_text=f"Median: {format_currency(median_price)}",
                      annotation_position="bottom")

        fig.update_layout(
            title=f"{period_name} – Price Distribution",
            xaxis_title="Price ($)",
            yaxis_title="Number of Days",
            template="plotly_white",
            height=450,
            bargap=0.05,
            hovermode="x",
            xaxis=dict(rangeslider=dict(visible=True), type="linear"),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics and Investment Simulation in columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Key Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Min", "Max", "Std Dev", "Skewness", "Kurtosis", "Q1", "Q3", "IQR"],
                "Value": [
                    format_currency(mean_price),
                    format_currency(median_price),
                    format_currency(min_price),
                    format_currency(max_price),
                    format_currency(std_price),
                    f"{skew:.2f}",
                    f"{kurt:.2f}",
                    format_currency(q1),
                    format_currency(q3),
                    format_currency(iqr)
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("####$100 Investment Simulation")
            final_value = investment_value.iloc[-1]
            total_return = (final_value - 100) / 100 * 100
            annual_return = ((final_value / 100) ** (1 / (len(prices)/252)) - 1) * 100 if len(prices) > 0 else 0
            profit = final_value - 100

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Initial", "$100.00")
            col_b.metric("Final Value", format_currency(final_value),
                         delta=f"{total_return:.2f}%", delta_color="normal" if total_return >=0 else "inverse")
            col_c.metric("Profit / Loss", format_currency(profit),
                         delta="Gain" if profit >=0 else "Loss", delta_color="normal" if profit >=0 else "inverse")

            st.metric("Annualized Return", format_percent(annual_return))

        # Educational expander
        with st.expander("Learn to Read This Histogram (Click to expand)"):
            st.markdown("""
            **What does the histogram show?**  
            It groups the daily closing prices into bins (price ranges) and shows how many days the price fell into each range.

            - **Tall bars** → price spent many days in that range.
            - **Color coding**: 🔴 Low prices, 🟠 Mid prices, 🟢 High prices.
            - **Vertical lines**:
              - Solid blue → Current price.
              - Dotted blue → Mean (average) price.
              - Dash-dot purple → Median (middle) price.

            **Statistics explained:**
            - **Mean**: average price – sensitive to extremes.
            - **Median**: middle value – more robust.
            - **Skewness**: positive means more days at lower prices (bullish); negative means more days at higher prices (bearish).
            - **Kurtosis**: high values indicate heavy tails (more extreme price moves).
            - **IQR (Interquartile Range)**: range between 25th and 75th percentiles – shows typical spread.

            **Investment Simulation**:  
            If you invested $100 at the start of the period, the final value shows what you'd have now. The annualized return helps compare performance across different time frames.
            """)

        st.markdown("---")

# ------------------------------
# Advanced Analytics & Recommendation
# ------------------------------
st.markdown("##Advanced Analytics & Recommendation")

# Use 5‑year data if available, else fallback to longest period
five_year = historical.get('5 Years', None)
if five_year is None and historical:
    five_year = list(historical.values())[-1]  # longest available

if five_year is not None and not five_year.empty:
    data_5y = five_year.copy()
    close = data_5y['Close']

    # Calculate indicators
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    rsi = calculate_rsi(close).iloc[-1] if len(close) >= 14 else None
    macd, signal, hist = calculate_macd(close)
    macd_val = macd.iloc[-1] if not macd.empty else None
    signal_val = signal.iloc[-1] if not signal.empty else None
    macd_hist = hist.iloc[-1] if not hist.empty else None

    # Volatility (annualized)
    daily_ret = close.pct_change().dropna()
    volatility = daily_ret.std() * np.sqrt(252) * 100  # annualized %

    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    avg_ret = daily_ret.mean() * 252 * 100
    sharpe = avg_ret / (volatility + 1e-9)

    # Price change over period
    price_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    # Momentum (1‑month)
    if len(close) >= 30:
        momentum = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100
    else:
        momentum = None

    # ------------------------------
    # Recommendation Logic
    # ------------------------------
    buy_score = 0
    reasons = []
    risks = []

    # 1. Trend
    if price_change > 20:
        buy_score += 2
        reasons.append("Strong 5‑year upward trend (>20%)")
    elif price_change > 10:
        buy_score += 1
        reasons.append("Moderate 5‑year upward trend (>10%)")
    elif price_change < -20:
        buy_score -= 2
        risks.append("Significant 5‑year decline (>20%)")

    # 2. Moving averages
    if sma_50 and close.iloc[-1] > sma_50:
        buy_score += 1
        reasons.append("Price above 50‑day SMA (short‑term bullish)")
    if sma_200 and close.iloc[-1] > sma_200:
        buy_score += 1
        reasons.append("Price above 200‑day SMA (long‑term bullish)")

    # 3. RSI
    if rsi is not None:
        if rsi < 30:
            buy_score += 1
            reasons.append("RSI oversold (<30) – potential bounce")
        elif rsi > 70:
            buy_score -= 1
            risks.append("RSI overbought (>70) – potential pullback")

    # 4. MACD
    if macd_val is not None and signal_val is not None:
        if macd_val > signal_val:
            buy_score += 1
            reasons.append("MACD above signal line (bullish crossover)")
        else:
            buy_score -= 1
            risks.append("MACD below signal line (bearish crossover)")

    # 5. Momentum
    if momentum is not None:
        if momentum > 5:
            buy_score += 1
            reasons.append("Positive 1‑month momentum (>5%)")
        elif momentum < -5:
            buy_score -= 1
            risks.append("Negative 1‑month momentum (<-5%)")

    # 6. Volatility
    if volatility > 40:
        risks.append("High annualized volatility (>40%)")
    elif volatility < 20:
        reasons.append("Low volatility (<20%) – stable")

    # 7. Sharpe ratio
    if sharpe > 1:
        reasons.append(f"Good risk‑adjusted return (Sharpe > 1)")
    elif sharpe < 0:
        risks.append("Negative Sharpe ratio – poor risk/reward")

    # Final decision
    st.markdown("####Recommendation")

    col1, col2 = st.columns([2, 1])
    with col1:
        if buy_score >= 3:
            st.markdown("""
            <div class="rec-buy">
                <h3>BUY</h3>
                <p style="font-size:1.1rem; margin-top:0.2rem;">Strong positive indicators suggest a favorable entry point.</p>
            </div>
            """, unsafe_allow_html=True)
        elif buy_score >= 1:
            st.markdown("""
            <div class="rec-hold">
                <h3>⚖️ HOLD / NEUTRAL</h3>
                <p style="font-size:1.1rem; margin-top:0.2rem;">Mixed signals; consider waiting for clearer trends.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-dont">
                <h3>DON'T BUY</h3>
                <p style="font-size:1.1rem; margin-top:0.2rem;">Several risk factors indicate caution – consider other opportunities.</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4 style="margin:0; color:black; font-weight:bold;">📊 Scorecard</h4>
            <p style="color:black;"><b>Buy Score:</b> {buy_score} / 7</p>
            <p style="color:black;"><b>Volatility:</b> {volatility:.2f}%</p>
            <p style="color:black;"><b>Sharpe:</b> {sharpe:.2f}</p>
            <p style="color:black;"><b>5Y Change:</b> {price_change:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Detailed reasoning
    with st.expander("Detailed Reasoning & Risk Factors"):
        st.markdown("**Positive Factors:**")
        for r in reasons:
            st.markdown(f"- {r}")
        if not reasons:
            st.markdown("- No significant positive factors detected.")
        st.markdown("**Risk Factors:**")
        for r in risks:
            st.markdown(f"- {r}")
        if not risks:
            st.markdown("- No significant risks identified.")
        st.markdown("""
        **How the recommendation is built:**  
        We combine 7 independent signals:
        - Long‑term trend (5‑year return)
        - Short‑term moving average (50‑day)
        - Long‑term moving average (200‑day)
        - RSI momentum
        - MACD crossover
        - 1‑month price momentum
        - Volatility and Sharpe ratio

        Each signal adds or subtracts from the score. A score ≥3 is a strong **BUY**, 1‑2 is **HOLD**, and ≤0 is **DON'T BUY**.
        """)

    st.markdown("---")

    # ------------------------------
    # Future-Centric Analytics
    # ------------------------------
    st.markdown("##Future‑Centric Analytics")
    st.markdown("*Advanced indicators to anticipate price movements.*")

    # Create subplots: Price + MAs, RSI, MACD
    fig_adv = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Price & Moving Averages", "Relative Strength Index (RSI)", "MACD")
    )

    # Price
    fig_adv.add_trace(go.Scatter(x=data_5y.index, y=close, name="Price", line=dict(color='#1a237e', width=2)), row=1, col=1)
    if len(close) >= 50:
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=close.rolling(50).mean(), name="SMA 50",
                                     line=dict(color='#2e7d32', width=1.5, dash='dash')), row=1, col=1)
    if len(close) >= 200:
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=close.rolling(200).mean(), name="SMA 200",
                                     line=dict(color='#c62828', width=1.5, dash='dash')), row=1, col=1)

    # RSI
    rsi_vals = calculate_rsi(close)
    fig_adv.add_trace(go.Scatter(x=data_5y.index, y=rsi_vals, name="RSI", line=dict(color='#7b1fa2', width=2)), row=2, col=1)
    fig_adv.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_adv.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    macd_vals, signal_vals, hist_vals = calculate_macd(close)
    fig_adv.add_trace(go.Scatter(x=data_5y.index, y=macd_vals, name="MACD", line=dict(color='#1565c0', width=2)), row=3, col=1)
    fig_adv.add_trace(go.Scatter(x=data_5y.index, y=signal_vals, name="Signal", line=dict(color='#e65100', width=2)), row=3, col=1)
    # Histogram as bars
    colors_hist = ['#2e7d32' if v >= 0 else '#c62828' for v in hist_vals]
    fig_adv.add_trace(go.Bar(x=data_5y.index, y=hist_vals, name="MACD Hist", marker_color=colors_hist), row=3, col=1)

    fig_adv.update_layout(height=700, template="plotly_white", showlegend=True, hovermode="x unified")
    fig_adv.update_xaxes(title_text="Date", row=3, col=1)
    fig_adv.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_adv.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig_adv.update_yaxes(title_text="MACD", row=3, col=1)

    st.plotly_chart(fig_adv, use_container_width=True)

    with st.expander("How to Interpret These Indicators"):
        st.markdown("""
        - **Price & MAs**: When price > SMA 200 → long‑term uptrend. Golden cross (SMA 50 crosses above SMA 200) is a strong bullish signal.
        - **RSI**: Values >70 indicate overbought (potential drop), <30 oversold (potential rise).
        - **MACD**: When MACD line crosses above signal line → bullish. Histogram turning from negative to positive confirms momentum shift.
        """)

else:
    st.info("Not enough 5‑year data for advanced analytics.")

# ------------------------------
# Glossary / Educational Section
# ------------------------------
with st.expander("Glossary of Financial Terms (For Beginners)"):
    st.markdown("""
    - **Histogram**: A chart that shows how many times prices fell into each range.
    - **Mean**: The average price over the period.
    - **Median**: The middle price when all prices are sorted – less affected by extremes.
    - **Skewness**: Measures asymmetry. Positive skew = more days at lower prices (bullish); negative = more days at higher prices (bearish).
    - **Kurtosis**: Measures tail heaviness. High kurtosis = more extreme price movements.
    - **Standard Deviation**: Measures how spread out prices are – higher = more volatile.
    - **RSI (Relative Strength Index)**: Momentum oscillator (0‑100). Overbought >70, oversold <30.
    - **MACD**: Moving Average Convergence Divergence – trend‑following momentum indicator.
    - **SMA**: Simple Moving Average – average price over a set number of days.
    - **Sharpe Ratio**: Risk‑adjusted return. Higher is better.
    """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>FinTech Analytics Pro | Designed by <b>Mamoor Hayat</b></p>
    <p style="font-size:0.8rem; color:#5c6bc0;">© 2026 All Rights Reserved | Data from Yahoo Finance | Not financial advice</p>
</div>
""", unsafe_allow_html=True)
