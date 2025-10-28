import plotly.graph_objects as go, pandas as pd


def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column by trying Title-case, lowercase, then capitalized variants."""
    if name in df.columns:
        return df[name]
    if name.lower() in df.columns:
        return df[name.lower()]
    if name.capitalize() in df.columns:
        return df[name.capitalize()]
    raise KeyError(f"Column for '{name}' not found in DataFrame")


def candles(df: pd.DataFrame, title: str):
    o = _get_col(df, 'Open')
    h = _get_col(df, 'High')
    l = _get_col(df, 'Low')
    c = _get_col(df, 'Close')
    fig = go.Figure([go.Candlestick(x=df.index, open=o, high=h, low=l, close=c)])
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=20))
    return fig


def line(series: pd.Series, title: str):
    fig = go.Figure([go.Scatter(x=series.index, y=series.values, mode="lines", name=series.name or "series")])
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=20))
    return fig
