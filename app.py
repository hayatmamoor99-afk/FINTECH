"""
FinTech Analytics Pro – Final Edition
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
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="FinTech Analytics Pro | Final",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# Custom CSS – Clean, Light, High Contrast
# ------------------------------------------------------------
st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none; }
    .main { background: #f5f7fe; padding: 0 1rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    .landing-header {
        text-align: center;
        padding: 3rem 0 1rem 0;
        background: linear-gradient(135deg, #e0e7ff, #f0e6ff);
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
    }
    .landing-header h1 { font-size: 4rem; font-weight: 800; color: #0b1a3a; margin: 0; letter-spacing: -1px; }
    .landing-header p { font-size: 1.3rem; color: #4a5b7e; margin: 0.5rem 0 1.5rem 0; }
    .card {
        background: white !important;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
        border-left: 6px solid #2a4b7c;
        transition: 0.2s;
        color: #000000 !important;
    }
    .card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
    .card h4 { color: #000000 !important; font-weight: 700; margin-top: 0; }
    .card p, .card div, .card span { color: #000000 !important; }
    .metric-big { font-size: 2.2rem; font-weight: 700; color: #000000 !important; margin: 0.2rem 0; }
    .metric-label { font-size: 0.85rem; color: #000000 !important; text-transform: uppercase; letter-spacing: 0.5px; }
    .prob-high { background: #e5f6e5; border-left: 6px solid #1b7e3d; padding: 1.2rem; border-radius: 16px; color: #000000; }
    .prob-medium { background: #fff4e0; border-left: 6px solid #d98c2b; padding: 1.2rem; border-radius: 16px; color: #000000; }
    .prob-low { background: #fce8e8; border-left: 6px solid #b33c3c; padding: 1.2rem; border-radius: 16px; color: #000000; }
    .footer { text-align: center; padding: 1.5rem; background: #e8edf9; border-radius: 20px; margin-top: 2rem; color: #000000; }
    .badge { background: #2a4b7c; color: white; padding: 0.2rem 0.8rem; border-radius: 30px; font-size: 0.7rem; font-weight: 600; display: inline-block; }
    @media (max-width: 768px) {
        .landing-header h1 { font-size: 2.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if 'symbol' not in st.session_state:
    st.session_state.symbol = None

# ------------------------------------------------------------
# Helper Functions (All Core Logic)
# ------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_data(symbol, periods_days):
    """Fetch data with caching and error handling."""
    try:
        ticker = yf.Ticker(symbol)
        end = datetime.now()
        data = {}
        for name, days in periods_days.items():
            start = end - timedelta(days=days)
            df = ticker.history(start=start, end=end)
            if not df.empty:
                data[name] = df
        info = ticker.info
        return data, info
    except Exception as e:
        return None, None

def monte_carlo_simulation(returns, current_price, days=252, n_sim=10000):
    """Bayesian Monte Carlo with shrinkage on volatility."""
    mu = returns.mean()
    sigma = returns.std()
    # Shrink sigma slightly to avoid overdispersion
    sigma_shrunk = sigma * 0.9 + 0.1 * returns.median()  # simple shrinkage
    dt = 1/252
    log_returns = np.random.normal(mu - 0.5 * sigma_shrunk**2, sigma_shrunk, size=(days, n_sim))
    cum_log = np.cumsum(log_returns, axis=0)
    paths = current_price * np.exp(cum_log)
    final = paths[-1, :]
    prob_profit = np.mean(final > current_price)
    expected_return = np.mean(final / current_price - 1) * 100
    var_95 = current_price - np.percentile(final, 5)
    cvar_95 = current_price - np.mean(final[final <= np.percentile(final, 5)])
    return {
        'prob_profit': prob_profit,
        'expected_return': expected_return,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'paths': paths
    }

def regime_classification(prices, window=50):
    """Classify regime based on rolling z-score."""
    if len(prices) < window:
        return "Insufficient data"
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    latest_mean = rolling_mean.iloc[-1]
    latest_std = rolling_std.iloc[-1]
    current = prices.iloc[-1]
    z_score = (current - latest_mean) / latest_std if latest_std != 0 else 0
    if z_score > 0.5:
        return "Bullish"
    elif z_score < -0.5:
        return "Bearish"
    else:
        return "Range‑bound"

def compute_beta(asset_returns, market_returns):
    """Compute beta relative to market (SPY)."""
    common = asset_returns.index.intersection(market_returns.index)
    if len(common) < 10:
        return np.nan
    asset = asset_returns.loc[common]
    market = market_returns.loc[common]
    cov = np.cov(asset, market)[0,1]
    var = np.var(market)
    return cov / var if var != 0 else np.nan

def kelly_criterion(win_rate, avg_win, avg_loss):
    """Kelly fraction: f = (win_rate * avg_win - (1-win_rate)*avg_loss) / avg_win."""
    if avg_win == 0:
        return 0
    return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

def backtest_strategy(prices, short=50, long=200):
    """Simple moving average crossover backtest."""
    if len(prices) < long:
        return None
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices
    signals['short_ma'] = prices.rolling(short).mean()
    signals['long_ma'] = prices.rolling(long).mean()
    signals['signal'] = 0
    signals.loc[signals.index[short:], 'signal'] = np.where(
        signals['short_ma'][short:] > signals['long_ma'][short:], 1, 0
    )
    signals['position'] = signals['signal'].diff()
    returns = prices.pct_change()
    signals['strategy_returns'] = signals['signal'].shift(1) * returns
    # Metrics
    total_return = (signals['strategy_returns'] + 1).prod() - 1
    sharpe = (signals['strategy_returns'].mean() / signals['strategy_returns'].std() * np.sqrt(252)) if signals['strategy_returns'].std() != 0 else 0
    win_rate = (signals['strategy_returns'] > 0).mean()
    cum_ret = signals['strategy_returns'].cumsum()
    max_drawdown = (cum_ret.cummax() - cum_ret).max()
    wins = signals['strategy_returns'][signals['strategy_returns'] > 0]
    losses = signals['strategy_returns'][signals['strategy_returns'] < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    return {
        'total_return': total_return * 100,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'equity_curve': (1 + signals['strategy_returns']).cumprod()
    }

def compute_risk_metrics(returns):
    """Compute VaR, CVaR, Sortino, etc."""
    if len(returns) < 10:
        return {}
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = returns[returns <= var_95].mean()
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else np.nan
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else np.nan
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    drawdown = (peak - cum) / peak
    max_drawdown = drawdown.max()
    total_return = (cum.iloc[-1] - 1) * 100
    calmar = total_return / (max_drawdown * 100) if max_drawdown != 0 else np.nan
    return {
        'var_95': var_95 * 100,
        'var_99': var_99 * 100,
        'cvar_95': cvar_95 * 100,
        'sortino': sortino,
        'max_drawdown': max_drawdown * 100,
        'calmar': calmar
    }

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

def format_currency(value):
    if pd.isna(value) or value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_market_cap(value):
    """Format market cap in M, B, or T."""
    if pd.isna(value) or value is None:
        return "N/A"
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

# ------------------------------------------------------------
# UI Pages
# ------------------------------------------------------------

def show_landing():
    st.markdown("""
    <div class="landing-header">
        <h1>📊 FinTech Analytics Pro</h1>
        <p>Nobel‑Grade Quant Research for Everyone</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol_input = st.text_input("Enter symbol (e.g., AAPL, BTC-USD)", value="AAPL", key="landing_symbol").upper()
        if st.button("🚀 Analyze", key="landing_button"):
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
    <div style="text-align:center; color:#000000; font-size:0.9rem;">
        Powered by Bayesian Monte Carlo, Regime Detection, Kelly Criterion, and Factor Models.<br>
        Data from Yahoo Finance • Not financial advice<br>
        Designed by <b>Mamoor Hayat</b> • © 2024
    </div>
    """, unsafe_allow_html=True)

def show_analysis():
    symbol = st.session_state.symbol
    if st.button("← Back to Home"):
        st.session_state.symbol = None
        st.rerun()

    st.markdown(f"## 📈 Advanced Analysis for **{symbol}**")

    # Fetch data
    periods = {
        '1 Year': 365,
        '2 Years': 730,
        '3 Years': 1095,
        '4 Years': 1460,
        '5 Years': 1825
    }
    data, info = fetch_data(symbol, periods)
    if data is None or not data:
        st.error("Could not fetch data. Please check symbol.")
        return

    # Use the longest available period for main analysis
    main_df = None
    for period in ['5 Years', '4 Years', '3 Years', '2 Years', '1 Year']:
        if period in data and not data[period].empty:
            main_df = data[period]
            break
    if main_df is None:
        st.error("No data available.")
        return

    current_price = main_df['Close'].iloc[-1]
    company_name = info.get('longName', symbol)
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    market_cap = info.get('marketCap', None)

    # For crypto, sector may be missing; we can set a default
    if sector == 'N/A' and symbol.endswith('-USD'):
        sector = "Cryptocurrency"
        industry = "Digital Asset"

    # ------------------------------------------------------------
    # Summary Cards
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>🏢 Company</h4>
            <div class="metric-big">{company_name}</div>
            <div class="metric-label">{symbol}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>💰 Current Price</h4>
            <div class="metric-big">{format_currency(current_price)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
            <h4>🏭 Sector</h4>
            <div class="metric-big">{sector if sector != 'N/A' else '—'}</div>
            <div class="metric-label">{industry if industry != 'N/A' else '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        cap_display = format_market_cap(market_cap) if market_cap else "N/A"
        st.markdown(f"""
        <div class="card">
            <h4>📊 Market Cap</h4>
            <div class="metric-big">{cap_display}</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Advanced Metrics (5-year data if available, else fallback)
    # ------------------------------------------------------------
    if len(main_df) > 50:
        prices = main_df['Close']
        returns = prices.pct_change().dropna()

        # Monte Carlo
        mc = monte_carlo_simulation(returns, current_price, days=252, n_sim=10000)

        # Regime
        regime = regime_classification(prices)

        # Beta vs SPY
        try:
            spy = yf.Ticker("SPY").history(start=prices.index[0], end=prices.index[-1])
            if not spy.empty:
                spy_returns = spy['Close'].pct_change().dropna()
                beta = compute_beta(returns, spy_returns)
            else:
                beta = np.nan
        except:
            beta = np.nan

        # Risk metrics
        risk = compute_risk_metrics(returns)

        # Backtest strategy
        bt = backtest_strategy(prices)

        # Kelly
        if bt is not None:
            kelly_f = kelly_criterion(bt['win_rate'], bt['avg_win'], bt['avg_loss'])
        else:
            kelly_f = np.nan

        # ------------------------------------------------------------
        # Probabilistic Recommendation
        # ------------------------------------------------------------
        st.markdown("---")
        st.markdown("## 🎯 Probabilistic Forecast & Recommendation")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Probability of Profit (1Y)", f"{mc['prob_profit']*100:.1f}%")
        with colB:
            st.metric("Expected Return (1Y)", f"{mc['expected_return']:.2f}%")
        with colC:
            st.metric("VaR (95%) 1Y", f"{format_currency(mc['var_95'])}")

        # Recommendation box
        if mc['prob_profit'] > 0.60:
            rec_class = "prob-high"
            rec_text = "✅ BUY – High probability of positive return"
            rec_detail = "Strong upside potential with moderate risk."
        elif mc['prob_profit'] > 0.45:
            rec_class = "prob-medium"
            rec_text = "⚖️ HOLD – Moderate probability"
            rec_detail = "Uncertainty is high; wait for clearer signals."
        else:
            rec_class = "prob-low"
            rec_text = "❌ DON'T BUY – Low probability of profit"
            rec_detail = "Downside risk outweighs upside potential."

        st.markdown(f"""
        <div class="{rec_class}" style="padding:1.2rem; border-radius:16px; margin:1rem 0; color:#000000;">
            <h3 style="margin:0; color:#000000;">{rec_text}</h3>
            <p style="margin:0.5rem 0 0 0; color:#000000;">{rec_detail}</p>
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------------------------------------
        # Monte Carlo Paths Chart
        # ------------------------------------------------------------
        fig_paths = go.Figure()
        np.random.seed(42)
        indices = np.random.choice(mc['paths'].shape[1], 50, replace=False)
        for i in indices:
            fig_paths.add_trace(go.Scatter(
                x=list(range(len(mc['paths']))),
                y=mc['paths'][:, i],
                mode='lines',
                line=dict(width=0.6, color='rgba(42,75,124,0.15)'),
                showlegend=False
            ))
        mean_path = np.mean(mc['paths'], axis=1)
        fig_paths.add_trace(go.Scatter(
            x=list(range(len(mc['paths']))),
            y=mean_path,
            mode='lines',
            line=dict(color='#2a4b7c', width=3),
            name='Mean Path'
        ))
        fig_paths.add_hline(y=current_price, line_dash="dash", line_color="#b33c3c",
                            annotation_text="Current Price", annotation_position="bottom right")
        fig_paths.update_layout(
            title="Monte Carlo Simulation – 1‑Year Price Paths (sample of 50)",
            xaxis_title="Trading Days (252 = 1 year)",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400,
            hovermode="x"
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # ------------------------------------------------------------
        # Risk Dashboard
        # ------------------------------------------------------------
        st.markdown("## 📊 Risk & Performance Dashboard")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Volatility (Ann.)", f"{returns.std()*np.sqrt(252)*100:.2f}%")
        col_r2.metric("Max Drawdown", f"{risk.get('max_drawdown', np.nan):.2f}%")
        col_r3.metric("Sortino Ratio", f"{risk.get('sortino', np.nan):.2f}")
        col_r4.metric("Calmar Ratio", f"{risk.get('calmar', np.nan):.2f}")

        col_r5, col_r6, col_r7, col_r8 = st.columns(4)
        col_r5.metric("VaR (95%)", f"{risk.get('var_95', np.nan):.2f}%")
        col_r6.metric("VaR (99%)", f"{risk.get('var_99', np.nan):.2f}%")
        col_r7.metric("CVaR (95%)", f"{risk.get('cvar_95', np.nan):.2f}%")
        col_r8.metric("Beta (vs SPY)", f"{beta:.2f}" if not np.isnan(beta) else "N/A")

        # ------------------------------------------------------------
        # Regime & Kelly
        # ------------------------------------------------------------
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Current Regime", regime)
        col_k2.metric("Kelly Fraction (optimal bet)", f"{kelly_f*100:.1f}%" if not np.isnan(kelly_f) else "N/A")
        if bt is not None:
            col_k3.metric("Strategy Win Rate (MA crossover)", f"{bt['win_rate']*100:.1f}%")
        else:
            col_k3.metric("Strategy Win Rate", "N/A")

        # ------------------------------------------------------------
        # Backtest Equity Curve
        # ------------------------------------------------------------
        if bt is not None and 'equity_curve' in bt:
            st.markdown("### 📈 Strategy Backtest (50/200 MA Crossover)")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt['equity_curve'].index,
                y=bt['equity_curve'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='#2a4b7c', width=2)
            ))
            fig_eq.update_layout(
                title="Cumulative Return of Crossover Strategy",
                xaxis_title="Date",
                yaxis_title="Equity (1 = initial)",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Return", f"{bt['total_return']:.2f}%")
            col_s2.metric("Sharpe (Strategy)", f"{bt['sharpe']:.2f}")
            col_s3.metric("Max Drawdown (Strategy)", f"{bt['max_drawdown']:.2f}%")

        # ------------------------------------------------------------
        # Historical Distribution (Tabs)
        # ------------------------------------------------------------
        st.markdown("## 📊 Historical Price Distribution")
        tabs = st.tabs(["1 Year", "2 Years", "3 Years", "4 Years", "5 Years"])
        for tab, (name, df) in zip(tabs, data.items()):
            with tab:
                if not df.empty:
                    prices_t = df['Close']
                    n_bins = min(30, max(5, len(prices_t)//10+1))
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=prices_t,
                        nbinsx=n_bins,
                        marker_color='#2a4b7c',
                        opacity=0.7,
                        name='Distribution'
                    ))
                    fig_hist.add_vline(x=current_price, line_dash='dash', line_color='#b33c3c',
                                       annotation_text="Current", annotation_position="top")
                    fig_hist.update_layout(
                        title=f"{name} – Price Distribution",
                        xaxis_title="Price ($)",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=300,
                        bargap=0.05
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    stats_df = pd.DataFrame({
                        "Metric": ["Mean", "Median", "Min", "Max", "Std Dev", "Skewness", "Kurtosis"],
                        "Value": [
                            format_currency(prices_t.mean()),
                            format_currency(prices_t.median()),
                            format_currency(prices_t.min()),
                            format_currency(prices_t.max()),
                            format_currency(prices_t.std()),
                            f"{prices_t.skew():.2f}",
                            f"{prices_t.kurtosis():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ------------------------------------------------------------
        # Advanced Chart (Price, Bollinger, Volume, RSI, MACD)
        # ------------------------------------------------------------
        st.markdown("## 🚀 Comprehensive Technical Chart")
        if len(main_df) > 50:
            close = main_df['Close']
            volume = main_df['Volume']
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            bb_upper = sma20 + 2 * close.rolling(20).std()
            bb_lower = sma20 - 2 * close.rolling(20).std()
            rsi = calculate_rsi(close)
            macd, signal, hist = calculate_macd(close)

            fig_tech = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=("Price & Bollinger Bands", "Volume", "RSI", "MACD")
            )
            # Price
            fig_tech.add_trace(go.Scatter(x=close.index, y=close, name='Price', line=dict(color='#0b1a3a', width=2)), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=sma20, name='SMA 20', line=dict(color='#2a7c4b', width=1, dash='dash')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=sma50, name='SMA 50', line=dict(color='#d98c2b', width=1, dash='dash')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=sma200, name='SMA 200', line=dict(color='#b33c3c', width=1, dash='dash')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=bb_upper, name='BB Upper', line=dict(color='#6b7a99', width=1, dash='dot')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=bb_lower, name='BB Lower', line=dict(color='#6b7a99', width=1, dash='dot')), row=1, col=1)
            # Volume
            colors = ['#2a7c4b' if close.iloc[i] >= close.iloc[i-1] else '#b33c3c' for i in range(1, len(close))]
            fig_tech.add_trace(go.Bar(x=close.index[1:], y=volume[1:], name='Volume', marker_color=colors), row=2, col=1)
            # RSI
            fig_tech.add_trace(go.Scatter(x=close.index, y=rsi, name='RSI', line=dict(color='#7c3a9e')), row=3, col=1)
            fig_tech.add_hline(y=70, line_dash="dash", line_color="#b33c3c", row=3, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="#2a7c4b", row=3, col=1)
            # MACD
            fig_tech.add_trace(go.Scatter(x=close.index, y=macd, name='MACD', line=dict(color='#2a4b7c')), row=4, col=1)
            fig_tech.add_trace(go.Scatter(x=close.index, y=signal, name='Signal', line=dict(color='#d98c2b')), row=4, col=1)
            hist_colors = ['#2a7c4b' if v >= 0 else '#b33c3c' for v in hist]
            fig_tech.add_trace(go.Bar(x=close.index, y=hist, name='Histogram', marker_color=hist_colors), row=4, col=1)

            fig_tech.update_layout(height=800, template="plotly_white", showlegend=True, hovermode="x unified")
            fig_tech.update_xaxes(title_text="Date", row=4, col=1)
            fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_tech.update_yaxes(title_text="Volume", row=2, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig_tech.update_yaxes(title_text="MACD", row=4, col=1)
            st.plotly_chart(fig_tech, use_container_width=True)

        # ------------------------------------------------------------
        # Download Data
        # ------------------------------------------------------------
        st.markdown("### 📥 Export Data")
        if st.button("Download Full Analysis (CSV)"):
            # Prepare a combined dataframe
            df_export = main_df[['Close', 'Volume']].copy()
            df_export['Returns'] = returns
            # Add indicators if available
            if len(main_df) > 50:
                df_export['SMA20'] = sma20
                df_export['SMA50'] = sma50
                df_export['SMA200'] = sma200
                df_export['RSI'] = rsi
                df_export['MACD'] = macd
                df_export['Signal'] = signal
                df_export['Hist'] = hist
            csv = df_export.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{symbol}_analysis.csv",
                mime="text/csv"
            )

        # ------------------------------------------------------------
        # Theory Expander
        # ------------------------------------------------------------
        with st.expander("🧠 Methodology & Theoretical Foundation (Click to expand)"):
            st.markdown("""
            **1. Bayesian Monte Carlo** – We simulate 10,000 future paths using Geometric Brownian Motion with drift and volatility estimated from historical returns. A shrinkage prior reduces overfitting.

            **2. Regime Detection** – Using rolling z‑scores, we classify the market into Bullish (>0.5σ above mean), Bearish (<-0.5σ), or Range‑bound.

            **3. Kelly Criterion** – The optimal fraction of capital to bet is derived from the edge (win rate and payoff ratio) of a simple moving‑average crossover strategy.

            **4. Factor Model** – Beta is computed via OLS regression against S&P 500 (SPY) returns, providing market exposure.

            **5. Risk Metrics** – VaR and CVaR are non‑parametric historical estimates; Sortino ratio penalises only downside volatility; Calmar ratio measures return per max drawdown.

            **6. Strategy Backtest** – We backtest a 50/200‑day moving average crossover and report performance metrics to evaluate if price momentum has historically worked.

            All calculations are performed on‑the‑fly using only the data available from Yahoo Finance. No external or proprietary data is used.
            """)

    else:
        st.warning("Insufficient data (need at least 50 days). Please try another symbol.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>📊 FinTech Analytics Pro – Final Edition | Designed with ❤️ by <b>Mamoor Hayat</b></p>
        <p style="font-size:0.8rem; color:#000000;">© 2024 All Rights Reserved | Data from Yahoo Finance | Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Main Router
# ------------------------------------------------------------
if st.session_state.symbol is None:
    show_landing()
else:
    show_analysis()
