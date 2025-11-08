#Streamlit Dashboard for Cryptocurrency Forecasting Project
# ---------------------------------------------------------------
# Author: Ritu Saxena
# Project: Bitcoin Price Forecasting (ARIMA, Prophet, LSTM)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# APP SETTINGS
st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide")
st.title("Bitcoin Price Analysis & Forecasting Dashboard")
st.markdown("By *Ritu Saxena*")
st.markdown("This dashboard visualizes Bitcoin price trends, volatility, and compares forecasting models (ARIMA, Prophet, LSTM).")

# ---------------------------------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("btcusd_1-min_data.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect timestamp column (auto)
    time_col = [c for c in df.columns if 'time' in c or 'date' in c][0]
    df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')

    # Set index and sort
    df = df.set_index(time_col).sort_index()

    # Resample to daily OHLCV
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['open'])

    # Add features
    daily['MA7'] = daily['close'].rolling(window=7).mean()
    daily['MA30'] = daily['close'].rolling(window=30).mean()
    daily['volatility'] = ((daily['high'] - daily['low']) / daily['open']) * 100
    daily = daily.dropna()

    return daily

daily = load_data()

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", daily.index.min().date())
end_date = st.sidebar.date_input("End Date", daily.index.max().date())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
filtered = daily.loc[str(start_date):str(end_date)]

# ---------------------------------------------------------------
# PRICE CHART
# ---------------------------------------------------------------
st.subheader("Bitcoin Daily Closing Prices with Moving Averages")
fig = px.line(filtered, x=filtered.index, y='close', title="Bitcoin Closing Prices")
fig.add_scatter(x=filtered.index, y=filtered['MA7'], mode='lines', name='7-day MA')
fig.add_scatter(x=filtered.index, y=filtered['MA30'], mode='lines', name='30-day MA')
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# VOLUME & VOLATILITY
# ---------------------------------------------------------------
st.subheader("Daily Volume and Volatility")
fig2 = px.line(filtered, x=filtered.index, y=['volume', 'volatility'], title="Volume vs Volatility")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# MODEL COMPARISON SECTION
# ---------------------------------------------------------------
st.subheader("Model Comparison Results")

# Replace with your actual model results
comparison_data = {
    'Model':   ['ARIMA', 'Prophet', 'LSTM'],
    'RMSE':    [1139.89,190.31,235.06],
    'MAE':     [1002.80,164.69,196.24],
    'MAPE':    [140.64,20.65,25.94]
}
comparison_df = pd.DataFrame(comparison_data)

st.write("Forecast Performance Metrics")
st.dataframe(comparison_df.style.format(precision=2).highlight_min(color='lightgreen', axis=0))

fig3 = px.bar(comparison_df, x='Model', y=['RMSE', 'MAE', 'MAPE'],
              barmode='group', title="Model Performance Comparison (Lower = Better)")
st.plotly_chart(fig3, use_container_width=True)

best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
st.success(f"Best Model Based on RMSE: **{best_model}**")

# ---------------------------------------------------------------
# INSIGHTS & CONCLUSION
# ---------------------------------------------------------------
st.subheader("Insights & Observations")
st.markdown("""
- **LSTM** achieved the lowest error metrics, making it the most accurate model for forecasting Bitcoin prices.  
- **Prophet** provided clear trend and seasonality insights — good interpretability.  
- **ARIMA** worked decently but struggled with high volatility patterns.  
- Bitcoin prices show significant **daily volatility**, influenced by trading volume and market sentiment.  
""")

st.markdown("Dashboard built successfully and ready for project submission!")
