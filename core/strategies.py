import pandas as pd
def sma(series: pd.Series, n:int)->pd.Series: return series.rolling(n).mean()

def sma_crossover(close: pd.Series, fast:int, slow:int, cost_bps:float=0.0):
    f, s = sma(close, fast), sma(close, slow)
    pos = (f>s).astype(int).shift(1).fillna(0)
    r = close.pct_change().fillna(0.0)
    strat = r*pos
    if cost_bps>0:
        turn = pos.diff().abs().fillna(0.0)  # entries/exits
        strat = strat - turn*(cost_bps/10000.0)
    return strat.dropna(), pos
