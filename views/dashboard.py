import streamlit as st
from core.data import download_prices
from core.metrics import returns, equity, cagr, ann_vol, sharpe, max_dd
from core.plotting import line

def render(ticker:str, period:str="1y"):
    df = download_prices(ticker,"1d",period)
    if df.empty: st.warning("No daily data."); return
    r = returns(df.Close); eq = equity(r)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CAGR", f"{cagr(r,'1d'):.2%}")
    c2.metric("Vol (ann.)", f"{ann_vol(r,'1d'):.2%}")
    c3.metric("Sharpe", f"{sharpe(r,'1d'):.2f}")
    c4.metric("Max DD", f"{max_dd(eq):.2%}")
    st.plotly_chart(line(df.Close, f"{ticker} Close (1d)"), use_container_width=True)
    st.plotly_chart(line(eq/eq.cummax()-1.0, "Drawdown"), use_container_width=True)
