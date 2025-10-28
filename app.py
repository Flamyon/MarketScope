import streamlit as st
from views import charts, dashboard, backtest

st.set_page_config(page_title="Market Scope", layout="wide")
st.title("Market Scope")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    daily_period = st.selectbox("Daily period", ["6mo","1y","2y","5y","max"], index=1)
    view = st.radio("View", ["Charts","Returns & Risk","Backtest"])

if not ticker:
    st.stop()

if view=="Charts":
    charts.render(ticker, daily_period)
elif view=="Returns & Risk":
    dashboard.render(ticker, daily_period)
else:
    backtest.render(ticker, daily_period)
