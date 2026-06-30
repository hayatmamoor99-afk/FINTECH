"""
FinTech Analytics Pro – Probabilistic Edition
Designed by Mamoor Hayat
Copyright © 2024 All Rights Reserved
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

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def optimal_bins(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    n = len(data)
    bin_width = 2 * iqr / (n ** (1/3)) if iqr > 0 else (data.max() - data.min()) / 30
    if bin_width == 0:
        bin_width = 1
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(5, min(bins, 50))

def format_currency(value):
    if pd.isna(value) or value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percent(value):
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}%"

def monte_carlo_simulation(returns, current_price, days=252, n_simulations=1000):
    """Simulate future price paths using geometric Brownian motion."""
    mu = returns.mean()
    sigma = returns.std()
    dt = 1/252
    log_returns = np.random.normal(mu - 0.5 * sigma**2, sigma, size=(days, n_simulations))
    cumulative_log_returns = np.cumsum(log_returns, axis=0)
    price_paths = current_price * np.exp(cumulative_log_returns)
    final_prices = price_paths[-1, :]
    prob_profit = np.mean(final_prices > current_price)
    expected_return = np.mean(final_prices / current_price - 1) * 100
    lower_bound = np.percentile(final_prices, 5)
    upper_bound = np.percentile(final_prices, 95)
    return prob_profit, expected_return, lower_bound, upper_bound, price_paths

# ------------------------------------------------------------
# Page Configuration – Full Width, No Sidebar
# ------------------------------------------------------------
st.set_page_config(
    page_title="FinTech Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# Custom CSS – Light, Elegant, Readable
# ------------------------------------------------------------
st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none; }
    .main { background: #f8faff; padding: 0 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    .landing-header {
        text-align: center;
        padding: 3rem 0 1rem 0;
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .landing-header h1 { font-size: 3.5rem; font-weight: 700; color: #1a237e; margin: 0; }
    .landing-header p { font-size: 1.2rem; color: #5c6bc0; margin: 0.5rem 0 1.5rem 0; }
    .search-box { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
    .search-box input { padding: 0.8rem 1.5rem; font-size: 1.2rem; border: 2px solid #c5cae9; border-radius: 30px; width: 300px; outline: none; transition: 0.2s; }
    .search-box input:focus { border-color: #1a237e; box-shadow: 0 0 0 3px rgba(26,35,126,0.1); }
    .search-box button { padding: 0.8rem 2rem; font-size: 1.2rem; background: #1a237e; color: white; border: none; border-radius: 30px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .search-box button:hover { background: #283593; transform: scale(1.02); }
    .pick-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: 0.2s;
        cursor: pointer;
        border: 1px solid #e8eaf6;
    }
    .pick-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); border-color: #1a237e; }
    .pick-card h4 { font-size: 1.2rem; color: #1a237e; margin: 0.2rem 0; }
    .pick-card p { color: #5c6bc0; font-size: 0.9rem; margin: 0; }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
        border-left: 4px solid #1a237e;
    }
    .card h4 { color: #1a237e; font-weight: 600; margin-top: 0; }
    .prob-high { background: #e8f5e9; border-left: 6px solid #2e7d32; padding: 1.2rem; border-radius: 12px; }
    .prob-medium { background: #fff3e0; border-left: 6px solid #f57c00; padding: 1.2rem; border-radius: 12px; }
    .prob-low { background: #fce4ec; border-left: 6px solid #c62828; padding: 1.2rem; border-radius: 12px; }
    .footer { text-align: center; padding: 1.5rem; background: #e8eaf6; border-radius: 16px; margin-top: 2rem; color: #1a237e; font-weight: 500; }
    @media (max-width: 768px) {
        .landing-header h1 { font-size: 2.2rem; }
        .search-box input { width: 200px; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if 'symbol' not in st.session_state:
    st.session_state.symbol = None

# ------------------------------------------------------------
# Landing Page
# ------------------------------------------------------------
def show_landing():
    st.markdown("""
    <div class="landing-header">
        <h1>📊 FinTech Analytics Pro</h1>
        <p>Advanced probabilistic analysis for stocks & cryptocurrencies</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol_input = st.text_input("Enter symbol (e.g., AAPL, BTC-USD)", value="AAPL", key="landing_symbol").upper()
        if st.button("Analyze", key="landing_button"):
            st.session_state.symbol = symbol_input
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔥 Top 5 Stocks")
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    cols = st.columns(5)
    for i, sym in enumerate(top_stocks):
        with cols[i]:
            if st.button(sym, key=f"stock_{sym}"):
                st.session_state.symbol = sym
                st.rerun()

    st.markdown("### 🪙 Top 5 Cryptocurrencies")
    top_crypto = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    cols = st.columns(5)
    for i, sym in enumerate(top_crypto):
        with cols[i]:
            if st.button(sym, key=f"crypto_{sym}"):
                st.session_state.symbol = sym
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#5c6bc0; font-size:0.9rem;">
        Data sourced from Yahoo Finance • Not financial advice<br>
        Designed by <b>Mamoor Hayat</b> • © 2024
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Analysis Page
# ------------------------------------------------------------
def show_analysis():
    symbol = st.session_state.symbol
    if st.button("← Back to Home"):
        st.session_state.symbol = None
        st.rerun()

    st.markdown(f"## 📈 Analysis for **{symbol}**")

    try:
        ticker = yf.Ticker(symbol)
        current_data = ticker.history(period="1d")
        if current_data.empty:
            st.error("No data found. Please check symbol.")
            return
        current_price = current_data['Close'].iloc[-1]

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
        st.error(f"Error fetching data: {e}")
        return

    # Company Info Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>🏢 Company</h4>
            <p style="font-size:1.2rem; font-weight:600; margin:0.2rem 0;">{company_name}</p>
            <p style="color:#666;">{symbol}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>💰 Current Price</h4>
            <p style="font-size:2rem; font-weight:700; color:#1a237e;">{format_currency(current_price)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
            <h4>🏭 Sector</h4>
            <p style="font-size:1rem; font-weight:600;">{sector if sector != 'N/A' else '—'}</p>
            <p style="color:#666;">{industry if industry != 'N/A' else '—'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        cap_display = format_currency(market_cap) if market_cap else "N/A"
        st.markdown(f"""
        <div class="card">
            <h4>📊 Market Cap</h4>
            <p style="font-size:1.2rem; font-weight:600;">{cap_display}</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Probabilistic Recommendation
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🎯 Recommendation Engine (Probabilistic)")

    five_year_data = historical.get('5 Years')
    if five_year_data is not None and len(five_year_data) > 10:
        returns = five_year_data['Close'].pct_change().dropna()
        prob_profit, exp_return, lower, upper, paths = monte_carlo_simulation(
            returns, current_price, days=252, n_simulations=2000
        )

        st.markdown(f"""
        <div style="text-align:center; padding:1rem; background: #f3f4f9; border-radius:16px; margin-bottom:1.5rem;">
            <h3>Probability of Profit (1‑year horizon)</h3>
            <div style="font-size:3.5rem; font-weight:700; color:#1a237e;">{prob_profit*100:.1f}%</div>
            <p style="color:#5c6bc0;">Expected return: {exp_return:.2f}% &nbsp;|&nbsp; 90% confidence interval: {format_currency(lower)} – {format_currency(upper)}</p>
        </div>
        """, unsafe_allow_html=True)

        if prob_profit > 0.65:
            rec_class = "prob-high"
            rec_text = "✅ BUY – High probability of positive return"
            rec_detail = "The model suggests a strong upside potential with manageable risk."
        elif prob_profit > 0.45:
            rec_class = "prob-medium"
            rec_text = "⚖️ HOLD / NEUTRAL – Moderate probability"
            rec_detail = "Uncertainty is significant; consider waiting for clearer signals."
        else:
            rec_class = "prob-low"
            rec_text = "❌ DON'T BUY – Low probability of profit"
            rec_detail = "Historical patterns suggest the downside risk is greater than the upside potential."

        st.markdown(f"""
        <div class="{rec_class}" style="padding:1.2rem; border-radius:12px; margin-bottom:1.5rem;">
            <h3 style="margin:0;">{rec_text}</h3>
            <p style="margin:0.5rem 0 0 0; font-size:1rem;">{rec_detail}</p>
        </div>
        """, unsafe_allow_html=True)

        # Plot sample paths
        fig_paths = go.Figure()
        np.random.seed(42)
        indices = np.random.choice(paths.shape[1], 50, replace=False)
        for i in indices:
            fig_paths.add_trace(go.Scatter(
                x=list(range(len(paths))),
                y=paths[:, i],
                mode='lines',
                line=dict(width=0.8, color='rgba(26,35,126,0.2)'),
                showlegend=False
            ))
        mean_path = np.mean(paths, axis=1)
        fig_paths.add_trace(go.Scatter(
            x=list(range(len(paths))),
            y=mean_path,
            mode='lines',
            line=dict(color='#1a237e', width=3),
            name='Mean Path'
        ))
        fig_paths.add_hline(y=current_price, line_dash="dash", line_color="#c62828",
                            annotation_text="Current Price", annotation_position="bottom right")
        fig_paths.update_layout(
            title="Monte Carlo Simulation – 1‑Year Price Paths (sample)",
            xaxis_title="Trading Days (252 = 1 year)",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400,
            hovermode="x"
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        with st.expander("🧠 How does this work?"):
            st.markdown("""
            - We use the **historical daily returns** (volatility and drift) from the last 5 years.
            - We simulate **2,000 possible future paths** using Geometric Brownian Motion.
            - The probability of profit is the percentage of paths that end above today’s price.
            - The confidence interval shows the range where 90% of the paths end.
            - This is a **probabilistic forecast**, not a guarantee – but it’s grounded in historical data.
            """)
    else:
        st.warning("Not enough historical data for probabilistic simulation (need at least 10 days).")

    # ------------------------------------------------------------
    # Historical Price Distribution
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Historical Price Distribution")
    tabs = st.tabs(["1 Year", "2 Years", "3 Years", "4 Years", "5 Years"])
    for tab, (period_name, df) in zip(tabs, historical.items()):
        with tab:
            if not df.empty:
                prices = df['Close']
                mean_price = prices.mean()
                median_price = prices.median()
                min_price = prices.min()
                max_price = prices.max()
                std_price = prices.std()
                skew = prices.skew()
                kurt = prices.kurtosis()

                n_bins = optimal_bins(prices)
                hist, bin_edges = np.histogram(prices, bins=n_bins)
                colors = []
                norm_prices = (bin_edges[:-1] - prices.min()) / (prices.max() - prices.min() + 1e-9)
                for val in norm_prices:
                    if val < 0.4:
                        colors.append('#c62828')
                    elif val < 0.7:
                        colors.append('#ff9800')
                    else:
                        colors.append('#2e7d32')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_edges[:-1],
                    y=hist,
                    width=[bin_edges[i+1]-bin_edges[i] for i in range(len(bin_edges)-1)],
                    marker_color=colors,
                    hovertemplate='<b>Price range:</b> $%{x:.2f} – $%{customdata[0]:.2f}<br><b>Days:</b> %{y}<extra></extra>',
                    customdata=[[bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                ))
                if current_price:
                    fig.add_vline(x=current_price, line_dash='dash', line_color='#1a237e',
                                  line_width=3, annotation_text=f"Current: {format_currency(current_price)}",
                                  annotation_position="top")
                fig.add_vline(x=mean_price, line_dash='dot', line_color='#1565c0',
                              annotation_text=f"Mean: {format_currency(mean_price)}", annotation_position="bottom")
                fig.add_vline(x=median_price, line_dash='dashdot', line_color='#6a1b9a',
                              annotation_text=f"Median: {format_currency(median_price)}", annotation_position="bottom")

                fig.update_layout(
                    title=f"{period_name} – Price Distribution",
                    xaxis_title="Price ($)",
                    yaxis_title="Number of Days",
                    template="plotly_white",
                    height=400,
                    bargap=0.05,
                    xaxis=dict(rangeslider=dict(visible=True))
                )
                st.plotly_chart(fig, use_container_width=True)

                stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Min", "Max", "Std Dev", "Skewness", "Kurtosis"],
                    "Value": [
                        format_currency(mean_price),
                        format_currency(median_price),
                        format_currency(min_price),
                        format_currency(max_price),
                        format_currency(std_price),
                        f"{skew:.2f}",
                        f"{kurt:.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # Advanced Indicators
    # ------------------------------------------------------------
    if '5 Years' in historical:
        st.markdown("---")
        st.markdown("## 🚀 Future‑Centric Analytics")
        data_5y = historical['5 Years'].copy()
        close = data_5y['Close']

        fig_adv = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Price & Moving Averages", "RSI", "MACD")
        )
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=close, name="Price", line=dict(color='#1a237e', width=2)), row=1, col=1)
        if len(close) >= 50:
            sma50 = close.rolling(50).mean()
            fig_adv.add_trace(go.Scatter(x=data_5y.index, y=sma50, name="SMA 50",
                                         line=dict(color='#2e7d32', dash='dash')), row=1, col=1)
        if len(close) >= 200:
            sma200 = close.rolling(200).mean()
            fig_adv.add_trace(go.Scatter(x=data_5y.index, y=sma200, name="SMA 200",
                                         line=dict(color='#c62828', dash='dash')), row=1, col=1)

        rsi_vals = calculate_rsi(close)
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=rsi_vals, name="RSI", line=dict(color='#7b1fa2')), row=2, col=1)
        fig_adv.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_adv.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        macd, signal, hist = calculate_macd(close)
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=macd, name="MACD", line=dict(color='#1565c0')), row=3, col=1)
        fig_adv.add_trace(go.Scatter(x=data_5y.index, y=signal, name="Signal", line=dict(color='#e65100')), row=3, col=1)
        colors_hist = ['#2e7d32' if v >= 0 else '#c62828' for v in hist]
        fig_adv.add_trace(go.Bar(x=data_5y.index, y=hist, name="MACD Hist", marker_color=colors_hist), row=3, col=1)

        fig_adv.update_layout(height=700, template="plotly_white", showlegend=True, hovermode="x unified")
        fig_adv.update_xaxes(title_text="Date", row=3, col=1)
        fig_adv.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_adv.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig_adv.update_yaxes(title_text="MACD", row=3, col=1)
        st.plotly_chart(fig_adv, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>📊 FinTech Analytics Pro | Designed with ❤️ by <b>Mamoor Hayat</b></p>
        <p style="font-size:0.8rem; color:#5c6bc0;">© 2024 All Rights Reserved | Data from Yahoo Finance | Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Main Router
# ------------------------------------------------------------
if st.session_state.symbol is None:
    show_landing()
else:
    show_analysis()
