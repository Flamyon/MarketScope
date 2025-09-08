import numpy as np, pandas as pd

MINUTES_PER_YEAR = 252*390

def steps_per_year(tf:str)->float:
    return {"1m":MINUTES_PER_YEAR,"15m":MINUTES_PER_YEAR/15,"1h":MINUTES_PER_YEAR/60,"1d":252}[tf]

def returns(close: pd.Series)->pd.Series: return close.pct_change().dropna()

def equity(r: pd.Series)->pd.Series: return (1+r).cumprod()

def cagr(r: pd.Series, tf:str)->float:
    if r.empty: return np.nan
    years = len(r)/steps_per_year(tf); tot=(1+r).prod()
    return tot**(1/years)-1 if years>0 else np.nan

def ann_vol(r: pd.Series, tf:str)->float: return r.std()*np.sqrt(steps_per_year(tf))

def sharpe(r: pd.Series, tf:str, rf_annual:float=0.0)->float:
    ppy=steps_per_year(tf); vol=r.std(); 
    if vol==0: return np.nan
    return ((r - rf_annual/ppy).mean()/vol)*np.sqrt(ppy)

def max_dd(eq: pd.Series)->float:
    peak=eq.cummax(); dd=eq/peak-1; return float(dd.min())
