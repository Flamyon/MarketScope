import plotly.graph_objects as go, pandas as pd

def candles(df: pd.DataFrame, title:str):
    fig=go.Figure([go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close)])
    fig.update_layout(title=title,height=380,margin=dict(l=10,r=10,t=40,b=20)); return fig

def line(series: pd.Series, title:str):
    fig=go.Figure([go.Scatter(x=series.index,y=series.values,mode="lines",name=series.name or "series")])
    fig.update_layout(title=title,height=340,margin=dict(l=10,r=10,t=40,b=20)); return fig
