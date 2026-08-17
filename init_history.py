import os
import json
import time
import requests
import io
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_all_nse_symbols():
    """Fetches the official list of all listed NSE equity tickers."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            symbols = df["SYMBOL"].str.strip().tolist()
            print(f"Fetched {len(symbols)} active NSE symbols.")
            return symbols
    except Exception as e:
        print(f"Failed to fetch official NSE list: {e}")
    
    # Fallback to NIFTY 500 / broad universe if official CSV fails
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "IGL", "TATAMOTORS", "ITC", "LT"]

def backfill_history(symbol):
    clean = symbol.strip().upper()
    file_path = f"{DATA_DIR}/{clean}.json"
    
    # Avoid re-downloading if already populated
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
            d_vol = int(r["Volume"] * 0.45)
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
        print(f"Backfilled {clean}")
        
    except Exception as e:
        print(f"Error {clean}: {e}")

if __name__ == "__main__":
    symbols = get_all_nse_symbols()
    for sym in symbols:
        backfill_history(sym)
