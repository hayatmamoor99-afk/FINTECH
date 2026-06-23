"""
FinTech Analytics Pro
Designed by Mamoor Hayat
Copyright © 2024 All Rights Reserved
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Helper function for RSI calculation (Moved to top)
def calculate_rsi(data, window=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Page configuration
st.set_page_config(
    page_title="FinTech Analytics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-title {
        color: #1a237e;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Profit/Loss colors */
    .profit {
        color: #2e7d32;
        font-weight: bold;
    }
    .loss {
        color: #c62828;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1a237e;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #283593;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(26,35,126,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #e8eaf6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        margin-top: 2rem;
        color: #1a237e;
        font-weight: 500;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a237e;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #283593;
    }
</style>
""", unsafe_allow_html=True)

# Copyright and designer info
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px;'>
    <p style='color: #1a237e; font-weight: 600;'>Designed by</p>
    <p style='color: #283593; font-size: 1.2rem; font-weight: 700;'>Mamoor Hayat</p>
    <p style='color: #5c6bc0; font-size: 0.8rem;'>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header-title">📊 FinTech Analytics Pro</div>', unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown("### 🔍 Search Stock/Crypto")
    
    # Input field
    symbol = st.text_input("Enter Stock or Crypto Symbol:", value="AAPL").upper()
    
    # Quick select buttons
    st.markdown("#### Quick Select:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("AAPL"):
            symbol = "AAPL"
    with col2:
        if st.button("GOOGL"):
            symbol = "GOOGL"
    with col3:
        if st.button("BTC-USD"):
            symbol = "BTC-USD"
    
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("TSLA"):
            symbol = "TSLA"
    with col5:
        if st.button("AMZN"):
            symbol = "AMZN"
    with col6:
        if st.button("ETH-USD"):
            symbol = "ETH-USD"
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This App
    This advanced FinTech application provides comprehensive analysis of stocks and cryptocurrencies using:
    - 📊 Historical data visualization with 5-year histograms
    - 💰 Investment simulation with profit/loss calculations
    - 🎯 AI-powered buy/sell recommendations
    - 📈 Future-centric predictive analytics
    """)

# Main content
if symbol:
    try:
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            ticker = yf.Ticker(symbol)
            
            # Get current price
            current_data = ticker.history(period="1d")
            if not current_data.empty:
                current_price = current_data['Close'].iloc[-1]
            else:
                current_price = None
            
            # Get historical data for different periods
            end_date = datetime.datetime.now()
            periods = {
                '1 Year': 365,
                '2 Years': 730,
                '3 Years': 1095,
                '4 Years': 1460,
                '5 Years': 1825
            }
            
            historical_data = {}
            for period_name, days in periods.items():
                start_date = end_date - timedelta(days=days)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    historical_data[period_name] = data
            
            # Get company info
            try:
                info = ticker.info
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
            except:
                company_name = symbol
                sector = 'N/A'
                industry = 'N/A'
                market_cap = 'N/A'

        # Display company info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='card'>
                <h4 style='color: #1a237e;'>🏢 Company</h4>
                <p style='font-size: 1.1rem; font-weight: 600;'>{company_name}</p>
                <p style='color: #666;'>{symbol}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Handle None case properly
            if current_price is not None:
                price_display = f"${current_price:.2f}"
            else:
                price_display = "N/A"
            st.markdown(f"""
            <div class='card'>
                <h4 style='color: #1a237e;'>💰 Current Price</h4>
                <p style='font-size: 1.5rem; font-weight: 700; color: #1a237e;'>
                    {price_display}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='card'>
                <h4 style='color: #1a237e;'>🏭 Sector</h4>
                <p style='font-size: 1rem; font-weight: 500;'>{sector}</p>
                <p style='color: #666; font-size: 0.9rem;'>{industry}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Handle market_cap display properly
            if isinstance(market_cap, (int, float)):
                market_cap_display = f"${market_cap:,.0f}"
            else:
                market_cap_display = "N/A"
            st.markdown(f"""
            <div class='card'>
                <h4 style='color: #1a237e;'>📊 Market Cap</h4>
                <p style='font-size: 1rem; font-weight: 500;'>
                    {market_cap_display}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Create histograms for each period
        st.markdown("---")
        st.markdown("### 📊 Historical Performance Analysis")
        st.markdown("*Interactive histograms showing price distribution over different time periods*")

        # Create tabs for different periods
        tabs = st.tabs(["📅 1 Year", "📅 2 Years", "📅 3 Years", "📅 4 Years", "📅 5 Years"])

        for tab, (period_name, data) in zip(tabs, historical_data.items()):
            with tab:
                if not data.empty:
                    # Calculate metrics
                    data['Daily_Return'] = data['Close'].pct_change() * 100
                    data['Investment_Value'] = 100 * (data['Close'] / data['Close'].iloc[0])
                    
                    # Create two columns for histogram and metrics
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Create histogram using plotly
                        fig = go.Figure()
                        
                        # Calculate histogram bins
                        prices = data['Close'].values
                        hist, bins = np.histogram(prices, bins=30)
                        
                        # Create color array based on price levels
                        colors = []
                        for i in range(len(bins)-1):
                            mid_price = (bins[i] + bins[i+1]) / 2
                            # Normalize price to 0-1 for color mapping
                            norm_price = (mid_price - prices.min()) / (prices.max() - prices.min()) if prices.max() != prices.min() else 0.5
                            if norm_price < 0.33:
                                colors.append('#c62828')  # Red for low prices
                            elif norm_price < 0.66:
                                colors.append('#ff9800')  # Orange for mid prices
                            else:
                                colors.append('#2e7d32')  # Green for high prices
                        
                        # Add histogram trace with custom colors
                        fig.add_trace(go.Bar(
                            x=bins[:-1],
                            y=hist,
                            width=[bins[i+1] - bins[i] for i in range(len(bins)-1)],
                            marker_color=colors,
                            name='Price Distribution',
                            hovertemplate='<b>Price Range:</b> $%{x:.2f} - $%{customdata[0]:.2f}<br>' +
                                         '<b>Frequency:</b> %{y} days<extra></extra>',
                            customdata=[[bins[i+1]] for i in range(len(bins)-1)]
                        ))
                        
                        # Add vertical line for current price
                        if current_price is not None:
                            fig.add_vline(
                                x=current_price,
                                line_dash="dash",
                                line_color="#1a237e",
                                line_width=3,
                                annotation_text=f"Current: ${current_price:.2f}",
                                annotation_position="top"
                            )
                        
                        # Add mean and median lines
                        mean_price = data['Close'].mean()
                        median_price = data['Close'].median()
                        
                        fig.add_vline(
                            x=mean_price,
                            line_dash="dot",
                            line_color="#1565c0",
                            line_width=2,
                            annotation_text=f"Mean: ${mean_price:.2f}",
                            annotation_position="bottom"
                        )
                        
                        fig.add_vline(
                            x=median_price,
                            line_dash="dashdot",
                            line_color="#6a1b9a",
                            line_width=2,
                            annotation_text=f"Median: ${median_price:.2f}",
                            annotation_position="bottom"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{period_name} Price Distribution Histogram',
                            xaxis_title='Price ($)',
                            yaxis_title='Number of Days',
                            template='plotly_white',
                            height=500,
                            bargap=0.05,
                            hovermode='x',
                            showlegend=False,
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type="linear"
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display key statistics
                        st.markdown("""
                        <div class='card'>
                            <h4 style='color: #1a237e;'>📊 Key Statistics</h4>
                        """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Mean Price",
                            f"${data['Close'].mean():.2f}"
                        )
                        st.metric(
                            "Median Price",
                            f"${data['Close'].median():.2f}"
                        )
                        st.metric(
                            "Min Price",
                            f"${data['Close'].min():.2f}"
                        )
                        st.metric(
                            "Max Price",
                            f"${data['Close'].max():.2f}"
                        )
                        st.metric(
                            "Std Deviation",
                            f"${data['Close'].std():.2f}"
                        )
                        st.metric(
                            "Total Days",
                            f"{len(data):,}"
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Investment simulation
                    st.markdown("#### 💰 Investment Simulation: $100 Investment")
                    
                    initial_investment = 100
                    final_value = data['Investment_Value'].iloc[-1]
                    total_return = ((final_value - initial_investment) / initial_investment) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Initial Investment",
                            f"${initial_investment:.2f}"
                        )
                    with col2:
                        st.metric(
                            "Final Value",
                            f"${final_value:.2f}",
                            delta=f"{total_return:.2f}%"
                        )
                    with col3:
                        profit_loss = final_value - initial_investment
                        if profit_loss >= 0:
                            st.metric(
                                "Profit/Loss",
                                f"+${profit_loss:.2f}",
                                delta="Profit 🎉",
                                delta_color="normal"
                            )
                        else:
                            st.metric(
                                "Profit/Loss",
                                f"${profit_loss:.2f}",
                                delta="Loss 📉",
                                delta_color="inverse"
                            )
                    
                    # Theoretical explanation for newbies
                    with st.expander("📖 What does this mean? (Click to expand)"):
                        st.markdown("""
                        **Understanding the Histogram:**
                        - The histogram shows how many days the stock traded at different price levels
                        - **Taller bars** indicate the price stayed in that range for more days
                        - **Color coding**: 🔴 Red (low prices) → 🟠 Orange (mid prices) → 🟢 Green (high prices)
                        - **Vertical lines**: 
                          - Solid blue: Current price
                          - Dotted blue: Mean (average) price
                          - Dash-dot purple: Median (middle) price
                        
                        **Key Statistics:**
                        - **Mean Price**: Average price over the period
                        - **Median Price**: Middle value when prices are sorted
                        - **Min/Max Price**: Lowest and highest prices reached
                        - **Std Deviation**: Measures price volatility (higher = more volatile)
                        
                        **Investment Simulation:**
                        - Shows what would happen if you invested $100 at the start of this period
                        - The final value shows your total investment worth today
                        - Profit/Loss indicates how much you would have gained or lost
                        
                        **What to Look For:**
                        - **Symmetric distribution**: Prices evenly spread (stable market)
                        - **Right-skewed**: More days at lower prices (bullish trend)
                        - **Left-skewed**: More days at higher prices (bearish trend)
                        - **Current price position**: Above mean = bullish, below mean = bearish
                        """)
                    
                    st.markdown("---")
        else:
            st.warning(f"No historical data available for {symbol}")

        # Advanced Analytics and Recommendation
        st.markdown("### 🎯 Advanced Analytics & Recommendation")
        
        # Calculate key metrics for recommendation
        if len(historical_data) > 0:
            # Get the 5-year data for comprehensive analysis
            five_year_data = historical_data.get('5 Years', None)
            
            if five_year_data is not None and not five_year_data.empty:
                # Calculate various metrics
                price_change = (five_year_data['Close'].iloc[-1] - five_year_data['Close'].iloc[0]) / five_year_data['Close'].iloc[0] * 100
                volatility = five_year_data['Close'].pct_change().std() * 100
                current_price = five_year_data['Close'].iloc[-1]
                sma_50 = five_year_data['Close'].rolling(window=50).mean().iloc[-1]
                sma_200 = five_year_data['Close'].rolling(window=200).mean().iloc[-1]
                
                # Calculate momentum
                if len(five_year_data) > 30:
                    momentum = (five_year_data['Close'].iloc[-1] - five_year_data['Close'].iloc[-30]) / five_year_data['Close'].iloc[-30] * 100
                else:
                    momentum = 0
                
                # Recommendation logic based on multiple factors
                buy_score = 0
                reasons = []
                risks = []
                
                # Price trend
                if price_change > 20:
                    buy_score += 2
                    reasons.append("✅ Strong positive 5-year trend")
                elif price_change > 10:
                    buy_score += 1
                    reasons.append("✅ Positive medium-term trend")
                elif price_change < -20:
                    buy_score -= 2
                    risks.append("⚠️ Significant price decline over 5 years")
                
                # Moving averages
                if current_price > sma_50:
                    buy_score += 1
                    reasons.append("✅ Trading above 50-day SMA (short-term bullish)")
                if current_price > sma_200:
                    buy_score += 1
                    reasons.append("✅ Trading above 200-day SMA (long-term bullish)")
                
                # Momentum
                if momentum > 5:
                    buy_score += 1
                    reasons.append("✅ Positive short-term momentum")
                elif momentum < -5:
                    buy_score -= 1
                    risks.append("⚠️ Negative short-term momentum")
                
                # Volatility assessment
                if volatility < 20:
                    reasons.append("✅ Lower volatility - relatively stable")
                elif volatility > 40:
                    risks.append("⚠️ High volatility - riskier investment")
                
                # Final recommendation
                st.markdown("#### 📈 Investment Recommendation")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if buy_score >= 2:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2e7d32;'>
                            <h3 style='color: #2e7d32; margin: 0;'>✅ BUY Signal</h3>
                            <p style='color: #1b5e20; margin-top: 0.5rem;'>
                                Based on our comprehensive analysis, this investment shows promising potential.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif buy_score >= 0:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #f57c00;'>
                            <h3 style='color: #e65100; margin: 0;'>⚖️ HOLD/NEUTRAL</h3>
                            <p style='color: #bf360c; margin-top: 0.5rem;'>
                                The stock shows mixed signals. Consider waiting for a clearer trend.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #fce4ec, #f8bbd0); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #c62828;'>
                            <h3 style='color: #c62828; margin: 0;'>❌ DON'T BUY</h3>
                            <p style='color: #b71c1c; margin-top: 0.5rem;'>
                                Current analysis suggests potential risks. Consider other opportunities.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='card'>
                        <h4 style='color: #1a237e;'>📊 Key Metrics</h4>
                        <p><strong>Buy Score:</strong> {buy_score}/4</p>
                        <p><strong>Volatility:</strong> {volatility:.2f}%</p>
                        <p><strong>5Y Change:</strong> {price_change:.2f}%</p>
                        <p><strong>Momentum:</strong> {momentum:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed analysis
                with st.expander("🔍 Detailed Analysis & Reasoning (Click to expand)"):
                    st.markdown("#### ✅ Positive Factors:")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                    
                    if not reasons:
                        st.markdown("- No significant positive factors identified")
                    
                    st.markdown("#### ⚠️ Risk Factors:")
                    for risk in risks:
                        st.markdown(f"- {risk}")
                    
                    if not risks:
                        st.markdown("- No significant risk factors identified")
                    
                    st.markdown("#### 📚 Theoretical Foundation:")
                    st.markdown("""
                    Our recommendation is based on a comprehensive analysis combining:
                    
                    1. **Trend Analysis**: Evaluating long-term price movements
                    2. **Moving Averages**: Identifying support/resistance levels
                    3. **Volatility Assessment**: Measuring investment risk
                    4. **Momentum Indicators**: Gauging short-term price direction
                    5. **Risk-Reward Ratio**: Balancing potential gains against risks
                    
                    The algorithm considers these factors using a weighted scoring system,
                    where positive indicators increase the buy score and negative indicators
                    decrease it. A score of 2+ suggests favorable conditions for investment.
                    """)

        # Advanced Future-Centric Features
        st.markdown("---")
        st.markdown("### 🚀 Future-Centric Analytics")
        
        if len(historical_data) > 0 and '5 Years' in historical_data:
            five_year_data = historical_data['5 Years'].copy()
            
            # Calculate moving averages and trends
            five_year_data['SMA_50'] = five_year_data['Close'].rolling(window=50).mean()
            five_year_data['SMA_200'] = five_year_data['Close'].rolling(window=200).mean()
            
            # Calculate RSI - using the function defined at the top
            five_year_data['RSI'] = calculate_rsi(five_year_data['Close'])
            
            # Create advanced chart
            fig_advanced = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price with Moving Averages', 'Relative Strength Index (RSI)')
            )
            
            # Price chart with MAs
            fig_advanced.add_trace(
                go.Scatter(x=five_year_data.index, y=five_year_data['Close'],
                          name='Price', line=dict(color='#1a237e', width=2)),
                row=1, col=1
            )
            fig_advanced.add_trace(
                go.Scatter(x=five_year_data.index, y=five_year_data['SMA_50'],
                          name='SMA 50', line=dict(color='#2e7d32', width=1.5, dash='dash')),
                row=1, col=1
            )
            fig_advanced.add_trace(
                go.Scatter(x=five_year_data.index, y=five_year_data['SMA_200'],
                          name='SMA 200', line=dict(color='#c62828', width=1.5, dash='dash')),
                row=1, col=1
            )
            
            # RSI chart
            fig_advanced.add_trace(
                go.Scatter(x=five_year_data.index, y=five_year_data['RSI'],
                          name='RSI', line=dict(color='#7b1fa2', width=2)),
                row=2, col=1
            )
            fig_advanced.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_advanced.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_advanced.update_layout(
                height=600,
                template='plotly_white',
                showlegend=True,
                hovermode='x unified'
            )
            
            fig_advanced.update_xaxes(title_text="Date", row=2, col=1)
            fig_advanced.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_advanced.update_yaxes(title_text="RSI", row=2, col=1)
            
            st.plotly_chart(fig_advanced, use_container_width=True)
            
            with st.expander("🧠 Understanding Advanced Analytics"):
                st.markdown("""
                **Moving Averages (SMA 50 & SMA 200):**
                - **SMA 50** (Green): Short-term trend indicator
                - **SMA 200** (Red): Long-term trend indicator
                - When price is above SMA 200 → Long-term uptrend
                - When SMA 50 crosses above SMA 200 → "Golden Cross" (Bullish signal)
                
                **Relative Strength Index (RSI):**
                - Measures momentum on a scale of 0-100
                - **Above 70**: Overbought (potential price decrease)
                - **Below 30**: Oversold (potential price increase)
                - **Around 50**: Neutral market
                
                **Future Insights:**
                - Combining these indicators helps predict potential price movements
                - RSI divergence from price often signals trend reversal
                - Moving average crossovers can indicate trend changes
                """)

    except Exception as e:
        st.error(f"Error fetching data for {symbol}. Please check the symbol and try again.")
        st.error(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>📊 FinTech Analytics Pro | Designed with ❤️ by Mamoor Hayat</p>
    <p style='font-size: 0.8rem; color: #5c6bc0;'>© 2024 All Rights Reserved | For Educational and Informational Purposes Only</p>
    <p style='font-size: 0.7rem; color: #7986cb;'>Data sourced from Yahoo Finance | Not financial advice</p>
</div>
""", unsafe_allow_html=True)
