import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------

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

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------

st.title("🚀 FinSight AI")
st.subheader("Advanced Stock Analysis Platform")

# ---------------------------
# SIDEBAR EDUCATION PANEL
# ---------------------------

with st.sidebar:

    st.header("📚 Learn")

    st.markdown("""
### Histogram
Shows return distribution.

### Volatility
Measures price fluctuations.

### Standard Deviation
Measures risk.

### Skewness
Shows return asymmetry.

### Kurtosis
Shows extreme events.

### Monte Carlo
Forecasts possible future prices.
""")

# ---------------------------
# STOCK SEARCH
# ---------------------------

ticker = st.text_input(
    "Enter Stock Symbol",
    value="NVDA"
)

if ticker:

    try:

        stock = yf.Ticker(ticker)

        data = stock.history(period="5y")

        if data.empty:
            st.error("No data found.")
            st.stop()

        current_price = data["Close"].iloc[-1]

        st.success(f"Current Price: ${current_price:.2f}")

        # --------------------
        # PRICE CHART
        # --------------------

        fig = px.line(
            data,
            y="Close",
            title=f"{ticker} Price History"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------
        # INVESTMENT RETURNS
        # --------------------

        st.header("💰 Investment Growth")

        periods = {
            "1 Year":"1y",
            "2 Years":"2y",
            "3 Years":"3y",
            "4 Years":"4y",
            "5 Years":"5y"
        }

        growth_rows = []

        for label, period in periods.items():

            hist = stock.history(period=period)

            if len(hist) > 0:

                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]

                value = 100 * (end_price / start_price)

                profit = value - 100

                growth_rows.append([
                    label,
                    round(value,2),
                    round(profit,2)
                ])

        growth_df = pd.DataFrame(
            growth_rows,
            columns=[
                "Period",
                "Current Value of $100",
                "Profit/Loss"
            ]
        )

        st.dataframe(
            growth_df,
            use_container_width=True
        )

        # --------------------
        # HISTOGRAMS
        # --------------------

        st.header("📊 Return Histograms")

        cols = st.columns(2)

        years = [1,2,3,4,5]

        for i, year in enumerate(years):

            hist = stock.history(
                period=f"{year}y"
            )

            returns = (
                hist["Close"]
                .pct_change()
                .dropna()
            )

            fig = px.histogram(
                returns,
                nbins=50,
                title=f"{year} Year Return Distribution"
            )

            cols[i % 2].plotly_chart(
                fig,
                use_container_width=True
            )

            cols[i % 2].write(
                f"""
Mean: {returns.mean():.4f}

Std Dev: {returns.std():.4f}

Skewness: {skew(returns):.2f}

Kurtosis: {kurtosis(returns):.2f}
"""
            )

        # --------------------
        # BUY / SELL ENGINE
        # --------------------

        st.header("🧠 AI Probability Engine")

        returns = (
            data["Close"]
            .pct_change()
            .dropna()
        )

        positive_days = (
            returns > 0
        ).mean()

        probability_up = (
            positive_days * 100
        )

        annual_return = (
            (
                current_price /
                data["Close"].iloc[0]
            ) - 1
        ) * 100

        volatility = (
            returns.std() * np.sqrt(252)
        ) * 100

        score = 50

        if annual_return > 20:
            score += 15

        if probability_up > 55:
            score += 15

        if volatility < 40:
            score += 10

        score = min(score,100)

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Probability Up",
            f"{probability_up:.1f}%"
        )

        c2.metric(
            "Annual Return",
            f"{annual_return:.1f}%"
        )

        c3.metric(
            "AI Score",
            score
        )

        if score >= 80:
            st.success(
                "🟢 Strong Buy Candidate"
            )

        elif score >= 65:
            st.info(
                "🔵 Moderate Buy Candidate"
            )

        elif score >= 50:
            st.warning(
                "🟡 Hold / Wait"
            )

        else:
            st.error(
                "🔴 Avoid Currently"
            )

        # --------------------
        # MONTE CARLO
        # --------------------

        st.header("🔮 Monte Carlo Forecast")

        returns = (
            data["Close"]
            .pct_change()
            .dropna()
        )

        simulations = 1000
        days = 252

        mean = returns.mean()
        std = returns.std()

        results = []

        for _ in range(simulations):

            prices = [current_price]

            for _ in range(days):

                shock = np.random.normal(
                    mean,
                    std
                )

                prices.append(
                    prices[-1] *
                    (1 + shock)
                )

            results.append(
                prices[-1]
            )

        expected = np.mean(results)

        best = np.percentile(
            results,
            95
        )

        worst = np.percentile(
            results,
            5
        )

        col1,col2,col3 = st.columns(3)

        col1.metric(
            "Expected Price",
            f"${expected:.2f}"
        )

        col2.metric(
            "Best Case",
            f"${best:.2f}"
        )

        col3.metric(
            "Worst Case",
            f"${worst:.2f}"
        )

        fig = px.histogram(
            results,
            nbins=50,
            title="Future Price Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
