import pandas as pd, yfinance as yf

VALID_INTERVALS = ("1m","15m","1h","1d")

def _default_period(interval:str)->str:
    return {"1m":"7d","15m":"30d","1h":"60d","1d":"1y"}[interval]

def download_prices(ticker:str, interval:str, period:str|None=None)->pd.DataFrame:
    assert interval in VALID_INTERVALS
    if interval!="1d": period = period or _default_period(interval)
    elif not period:   period = "1y"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty: return pd.DataFrame()
    out = df[["Open","High","Low","Close","Volume"]].dropna()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out

def resample_4h(df_1h: pd.DataFrame)->pd.DataFrame:
    if df_1h.empty: return df_1h
    o=df_1h["Open"].resample("4H").first()
    h=df_1h["High"].resample("4H").max()
    l=df_1h["Low"].resample("4H").min()
    c=df_1h["Close"].resample("4H").last()
    v=df_1h["Volume"].resample("4H").sum()
    out = pd.concat([o,h,l,c,v], axis=1); out.columns=["Open","High","Low","Close","Volume"]
    return out.dropna()
