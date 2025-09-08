import streamlit as st
from core.data import download_prices, resample_4h
from core.plotting import candles

def render(ticker:str, daily_period:str="1y"):
    c1,c2 = st.columns(2)
    df1m  = download_prices(ticker,"1m","7d")
    df15m = download_prices(ticker,"15m","30d")
    with c1: st.subheader("1m");  st.plotly_chart(candles(df1m.tail(200), f"{ticker} — 1m"), use_container_width=True) if not df1m.empty else st.info("No 1m data.")
    with c2: st.subheader("15m"); st.plotly_chart(candles(df15m, f"{ticker} — 15m"), use_container_width=True) if not df15m.empty else st.info("No 15m data.")
    c3,c4 = st.columns(2)
    df1h  = download_prices(ticker,"1h","60d"); df4h=resample_4h(df1h)
    with c3: st.subheader("4h"); st.plotly_chart(candles(df4h, f"{ticker} — 4h"), use_container_width=True) if not df4h.empty else st.info("No 4h data.")
    df1d  = download_prices(ticker,"1d",daily_period)
    with c4: st.subheader("1d"); st.plotly_chart(candles(df1d, f"{ticker} — 1d"), use_container_width=True) if not df1d.empty else st.info("No 1d data.")
