import math
from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

ANNUAL_DAYS = 252

@dataclass
class Metrics:
    cagr: float
    vol_ann: float
    sharpe: float
    max_dd: float
    start: str
    end: str
    n_obs: int

def download_prices(
    ticker: str,
    interval: Literal["1m","15m","1h","1d"] = "1d",
    period: str = "1y",
    start: str|None = None,
    end: str|None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Uses yfinance. Notes/limits:
      - 1m: max 7d period (yfinance limit).
      - Intraday intervals require 'period' instead of (start,end) for reliability.
      - 4h will be derived by resampling 1h data.
    """
    if interval == "1m":
        period = "7d" if period is None else period  # yfinance hard limit
        df = yf.download(ticker, period=period, interval="1m", auto_adjust=auto_adjust, progress=False)
    elif interval == "15m":
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval="15m", auto_adjust=auto_adjust, progress=False)
        else:
            df = yf.download(ticker, period=period, interval="15m", auto_adjust=auto_adjust, progress=False)
    elif interval == "1h":
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=auto_adjust, progress=False)
        else:
            df = yf.download(ticker, period=period, interval="1h", auto_adjust=auto_adjust, progress=False)
    elif interval == "1d":
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=auto_adjust, progress=False)
        else:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=auto_adjust, progress=False)
    else:
        raise ValueError("Unsupported interval")

    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open','High','Low','Close','Volume']]
    df = df.dropna(how="any")
    return df

def resample_4h_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    # OHLCV resample; note: this collapses into 4H bars aligned to the index timezone
    if df_1h.empty:
        return df_1h
    o = df_1h['Open'].resample('4H').first()
    h = df_1h['High'].resample('4H').max()
    l = df_1h['Low'].resample('4H').min()
    c = df_1h['Close'].resample('4H').last()
    v = df_1h['Volume'].resample('4H').sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ['Open','High','Low','Close','Volume']
    out = out.dropna(how='any')
    return out

def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()

def cagr(returns: pd.Series, periods_per_year: int = ANNUAL_DAYS) -> float:
    if returns.empty: return np.nan
    total = (1+returns).prod()
    years = len(returns) / periods_per_year
    return total**(1/years) - 1 if years > 0 else np.nan

def ann_vol(returns: pd.Series, periods_per_year: int = ANNUAL_DAYS) -> float:
    return returns.std() * math.sqrt(periods_per_year)

def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = ANNUAL_DAYS) -> float:
    rf_daily = rf_annual / periods_per_year
    excess = returns - rf_daily
    vol = returns.std()
    if vol == 0 or np.isnan(vol): return np.nan
    return (excess.mean() / vol) * math.sqrt(periods_per_year)

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()

def compute_metrics(returns: pd.Series, freq: str) -> Metrics:
    if freq == "1d": ppy = ANNUAL_DAYS
    elif freq in ("1h","4h","15m","1m"):
        # Very rough day-equivalent conversion for Sharpe/vol; CAGR uses calendar span anyway.
        ppy = ANNUAL_DAYS  # keep consistent to compare; acceptable for a simple app
    else:
        ppy = ANNUAL_DAYS
    eq = equity_curve(returns)
    return Metrics(
        cagr=cagr(returns, ppy),
        vol_ann=ann_vol(returns, ppy),
        sharpe=sharpe(returns, 0.0, ppy),
        max_dd=max_drawdown(eq),
        start=str(returns.index[0]),
        end=str(returns.index[-1]),
        n_obs=len(returns)
    )

def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="OHLC"
    )])
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Price", height=380, margin=dict(l=20,r=20,t=50,b=20))
    return fig

def plot_line(series: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series.name or "series"))
    fig.update_layout(title=title, height=340, margin=dict(l=20,r=20,t=50,b=20))
    return fig

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def sma_crossover_returns(close: pd.Series, fast: int = 50, slow: int = 200, cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    s_fast, s_slow = sma(close, fast), sma(close, slow)
    signal = (s_fast > s_slow).astype(int)
    r = close.pct_change().fillna(0.0)
    sig_shift = signal.shift(1).fillna(0)
    strat = r * sig_shift
    # transaction cost on signal change (turnover = |Δsignal|)
    if cost_bps > 0:
        turn = (sig_shift.diff().abs().fillna(0))  # 1 on entry/exit
        # cost applied on the day of change
        strat = strat - turn * (cost_bps/10000.0)
    return strat.dropna(), signal

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Ticker Explorer", page_icon="📈", layout="wide")
st.title("Ticker Explorer")

with st.sidebar:
    st.subheader("Input")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    default_period = "1y"
    daily_period = st.selectbox("Daily data period", ["6mo","1y","2y","5y","max"], index=1)
    intraday_period = st.selectbox("Intraday data period (<= 60d)", ["7d","14d","30d","60d"], index=0)
    st.caption("Note: 1m data is limited to ~7 days by Yahoo Finance.")
    view = st.radio("View", ["Charts","Returns & Risk Dashboard","Backtest"], index=0)

if not ticker:
    st.stop()

# -----------------------------
# View 1: Charts (1m, 15m, 4h, 1d)
# -----------------------------
if view == "Charts":
    # 1m
    data_1m = download_prices(ticker, interval="1m", period="7d")
    # 15m
    data_15m = download_prices(ticker, interval="15m", period=intraday_period)
    # 1h -> 4h
    data_1h = download_prices(ticker, interval="1h", period=intraday_period)
    data_4h = resample_4h_from_1h(data_1h)
    # 1d
    data_1d = download_prices(ticker, interval="1d", period=daily_period)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("1 minute")
        if data_1m.empty: st.info("No 1m data available."); 
        else: st.plotly_chart(plot_candles(data_1m.tail(200), f"{ticker} — 1m"), use_container_width=True)
    with cols[1]:
        st.subheader("15 minutes")
        if data_15m.empty: st.info("No 15m data available.")
        else: st.plotly_chart(plot_candles(data_15m, f"{ticker} — 15m"), use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("4 hours")
        if data_4h.empty: st.info("No 4h data available.")
        else: st.plotly_chart(plot_candles(data_4h, f"{ticker} — 4h"), use_container_width=True)
    with cols[1]:
        st.subheader("1 day")
        if data_1d.empty: st.info("No 1d data available.")
        else: st.plotly_chart(plot_candles(data_1d, f"{ticker} — 1d"), use_container_width=True)

# -----------------------------
# View 2: Returns & Risk Dashboard
# -----------------------------
elif view == "Returns & Risk Dashboard":
    df = download_prices(ticker, interval="1d", period=daily_period)
    if df.empty:
        st.warning("No data.")
        st.stop()

    close = df["Close"]
    rets = simple_returns(close)
    eq = equity_curve(rets)

    m = compute_metrics(rets, "1d")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{m.cagr:.2%}")
    c2.metric("Vol (ann.)", f"{m.vol_ann:.2%}")
    c3.metric("Sharpe (naive)", f"{m.sharpe:.2f}")
    c4.metric("Max Drawdown", f"{m.max_dd:.2%}")

    st.plotly_chart(plot_line(close, f"{ticker} Close (1d)"), use_container_width=True)

    dd = eq / eq.cummax() - 1.0
    st.plotly_chart(plot_line(dd, "Drawdown"), use_container_width=True)

    with st.expander("Raw returns (head)"):
        st.dataframe(rets.head(10))

# -----------------------------
# View 3: Backtest (SMA crossover)
# -----------------------------
else:
    st.subheader("Backtest: SMA Crossover")
    timeframe = st.selectbox("Timeframe", ["1m","15m","4h","1d"], index=3)
    fast = st.number_input("Fast SMA", value=50, min_value=2, max_value=1000, step=1)
    slow = st.number_input("Slow SMA", value=200, min_value=3, max_value=2000, step=1)
    costs = st.number_input("Transaction cost (bps per switch)", value=0, min_value=0, max_value=100, step=1)
    run = st.button("Run backtest")

    if run:
        if timeframe == "4h":
            base = download_prices(ticker, interval="1h", period=intraday_period)
            df = resample_4h_from_1h(base)
        elif timeframe == "1m":
            df = download_prices(ticker, interval="1m", period="7d")
        elif timeframe == "15m":
            df = download_prices(ticker, interval="15m", period=intraday_period)
        else:  # 1d
            df = download_prices(ticker, interval="1d", period=daily_period)

        if df.empty:
            st.warning("No data returned for the selected timeframe/period.")
            st.stop()

        close = df["Close"]
        # Buy & Hold
        bh_rets = simple_returns(close)
        # Strategy
        strat_rets, signal = sma_crossover_returns(close, fast=fast, slow=slow, cost_bps=costs)

        # Align and compute metrics
        # (Make sure both start at the same index to compare fairly)
        common_idx = bh_rets.index.intersection(strat_rets.index)
        bh_rets = bh_rets.loc[common_idx]
        strat_rets = strat_rets.loc[common_idx]

        eq_bh = equity_curve(bh_rets)
        eq_st = equity_curve(strat_rets)

        # crude frequency tag for metrics display
        freq_tag = timeframe
        mbh = compute_metrics(bh_rets, freq_tag)
        mst = compute_metrics(strat_rets, freq_tag)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Buy & Hold**")
            st.write({
                "CAGR": f"{mbh.cagr:.2%}",
                "Vol (ann.)": f"{mbh.vol_ann:.2%}",
                "Sharpe": f"{mbh.sharpe:.2f}",
                "Max DD": f"{mbh.max_dd:.2%}",
                "Obs": mbh.n_obs
            })
        with c2:
            st.markdown("**SMA Crossover**")
            st.write({
                "CAGR": f"{mst.cagr:.2%}",
                "Vol (ann.)": f"{mst.vol_ann:.2%}",
                "Sharpe": f"{mst.sharpe:.2f}",
                "Max DD": f"{mst.max_dd:.2%}",
                "Obs": mst.n_obs
            })

        # Plot equity curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh.values, name="Buy & Hold"))
        fig.add_trace(go.Scatter(x=eq_st.index, y=eq_st.values, name=f"SMA({fast}/{slow})"))
        fig.update_layout(title=f"{ticker} — {timeframe} Strategy vs Buy & Hold", height=420, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Optional: show price with SMAs and signal
        with st.expander("Show price with SMAs and signal"):
            s_fast = sma(close, fast)
            s_slow = sma(close, slow)
            figp = go.Figure()
            figp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
            figp.add_trace(go.Scatter(x=s_fast.index, y=s_fast.values, name=f"SMA {fast}"))
            figp.add_trace(go.Scatter(x=s_slow.index, y=s_slow.values, name=f"SMA {slow}"))
            figp.update_layout(height=480, title=f"{ticker} — {timeframe} with SMAs", margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(figp, use_container_width=True)

# Footer note
st.caption("Data source: Yahoo Finance via yfinance. 1m data has strict period limits. 4h is resampled from 1h.")
