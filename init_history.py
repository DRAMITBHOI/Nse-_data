import os
import json
import yfinance as yf
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Fetches 5-year OHLC and initializes base OBV
def backfill_history(symbol):
    clean = symbol.strip().upper()
    file_path = f"{DATA_DIR}/{clean}.json"
    if os.path.exists(file_path):
        return
    try:
        t = yf.Ticker(f"{clean}.NS")
        df = t.history(period="5y")
        if df.empty:
            return
        records = []
        obv = 0
        prev_close = df["Close"].iloc[0]
        for dt, r in df.iterrows():
            d_vol = int(r["Volume"] * 0.45) # Default 45% delivery baseline for historical approximation
            direction = 1 if r["Close"] > prev_close else (-1 if r["Close"] < prev_close else 0)
            obv += direction * d_vol
            prev_close = r["Close"]
            records.append({
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(float(r["Open"]), 2),
                "high": round(float(r["High"]), 2),
                "low": round(float(r["Low"]), 2),
                "close": round(float(r["Close"]), 2),
                "delivery_vol": d_vol,
                "deliv_pct": 45.0,
                "deliv_obv": obv
            })
        with open(file_path, "w") as f:
            json.dump(records, f)
    except Exception:
        pass

if __name__ == "__main__":
    # Add your initial broad universe or read from an NSE symbols file
    pass
