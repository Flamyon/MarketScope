#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_chart_json(txt_path: str) -> dict:
    raw = Path(txt_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if "chart" not in data:
        raise ValueError("Estructura inesperada: no aparece la clave 'chart'.")
    return data


def extract_ohlcv(chart_json: dict) -> pd.DataFrame:
    res = chart_json["chart"]["result"]
    if not res:
        raise ValueError("chart.result vacío.")
    r0 = res[0]

    ts = r0.get("timestamp", [])
    ind = r0.get("indicators", {})
    quote_list = ind.get("quote", [])
    if not quote_list:
        raise ValueError("indicators.quote vacío.")

    q0 = quote_list[0]
    opens = q0.get("open", [])
    highs = q0.get("high", [])
    lows  = q0.get("low", [])
    closes = q0.get("close", [])
    vols = q0.get("volume", [])

    n = len(ts)
    for name, arr in {
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols
    }.items():
        if len(arr) != n:
            raise ValueError(f"Longitud desigual en '{name}': {len(arr)} vs {n} timestamps.")

    # Construye DataFrame
    df = pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })

    # Convierte timestamp (segundos UNIX) a fecha UTC YYYY-MM-DD
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["date"] = df["date"].dt.tz_convert("UTC").dt.normalize()  # 00:00:00 UTC
    df = df.drop(columns=["timestamp"])

    return df


def clean_ohlcv(df: pd.DataFrame, drop_partial_last: bool = True) -> pd.DataFrame:
    # Elimina filas con nulos en OHLC o date/close
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()

    # Orden ascendente por fecha
    df = df.sort_values("date").reset_index(drop=True)

    # De-dup por fecha (si hubiera)
    df = df[~df["date"].duplicated(keep="first")].copy()

    # Integridad geométrica OHLC
    bad = ~(
        (df["low"] <= df[["open", "close"]].min(axis=1)) &
        (df[["open", "close"]].max(axis=1) <= df["high"])
    )
    if bad.any():
        # Opcional: eliminar las filas incoherentes
        df = df[~bad].copy()

    if drop_partial_last:
        today_utc = pd.Timestamp.now(tz=timezone.utc).normalize()
        if len(df) and df["date"].iloc[-1] >= today_utc:
            df = df.iloc[:-1, :].copy()

    # Reordena columnas y tipa
    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[cols]
    # Asegura tipos numéricos
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Pásalo a fecha (sin hora) en string YYYY-MM-DD para el CSV final
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def save_csv(df: pd.DataFrame, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")


# --- Compatibility helpers for the Streamlit app (views expect these) ---
def _title_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names to Title-case (Open, High, Low, Close, Volume)."""
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("open", "high", "low", "close", "volume") and c != lc.capitalize():
            rename[c] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    return df


def download_prices(ticker: str, interval: str, period: str | None = None) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [project_root / f"data/{ticker}_daily.csv", project_root / f"data/{ticker}.csv", project_root / f"data/{ticker.upper()}_daily.csv"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            df = _title_cols(df)
            if interval != '1d':
                return pd.DataFrame()
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def main():
    input_txt ="btc_data.txt"
    out="data/BTCUSD_daily.csv"
    keeplast ="store_true"

    chart_json = load_chart_json(input_txt)
    df = extract_ohlcv(chart_json)
    df = clean_ohlcv(df, drop_partial_last=(not keeplast))

    # Asserts básicos
    assert set(df.columns) == {"date", "open", "high", "low", "close", "volume"}
    assert df["date"].is_monotonic_increasing
    assert df["close"].notna().all()

    save_csv(df, out)

    # Resumen en terminal
    print(f"OK → {out}")
    print(f"Filas: {len(df)}  |  Rango: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
