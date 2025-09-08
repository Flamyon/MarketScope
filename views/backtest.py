import streamlit as st
from core.data import download_prices, resample_4h
from core.metrics import returns, equity, cagr, ann_vol, sharpe, max_dd
from core.strategies import sma_crossover
import plotly.graph_objects as go

def render(ticker:str, daily_period:str="1y"):
    tf = st.selectbox("Timeframe", ["1m","15m","1h","1d"], index=3)
    fast = st.number_input("Fast SMA", 5, 1000, 50)
    slow = st.number_input("Slow SMA", 6, 2000, 200)
    cost = st.number_input("Cost (bps per switch)", 0, 100, 0)
    if st.button("Run"):
        if tf=="1m":  df=download_prices(ticker,"1m","7d")
        elif tf=="15m": df=download_prices(ticker,"15m","30d")
        elif tf=="1h":  df=download_prices(ticker,"1h","60d")
        else:          df=download_prices(ticker,"1d",daily_period)
        if df.empty: st.warning("No data for selected timeframe."); return
        bh = returns(df.Close)
        strat, _ = sma_crossover(df.Close, fast, slow, cost)
        common = bh.index.intersection(strat.index)
        bh, strat = bh.loc[common], strat.loc[common]
        eq_bh, eq_st = equity(bh), equity(strat)

        mbh = {"CAGR":cagr(bh,tf),"Vol":ann_vol(bh,tf),"Sharpe":sharpe(bh,tf),"MaxDD":max_dd(eq_bh)}
        mst = {"CAGR":cagr(strat,tf),"Vol":ann_vol(strat,tf),"Sharpe":sharpe(strat,tf),"MaxDD":max_dd(eq_st)}
        c1,c2 = st.columns(2); c1.write(mbh); c2.write(mst)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh.values, name="Buy & Hold"))
        fig.add_trace(go.Scatter(x=eq_st.index, y=eq_st.values, name=f"SMA({fast}/{slow})"))
        fig.update_layout(title=f"{ticker} — {tf} strategy vs buy & hold")
        st.plotly_chart(fig, use_container_width=True)
