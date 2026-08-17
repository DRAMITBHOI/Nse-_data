import os
import io
import json
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*"
}

def get_trading_days(years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    # Generate business days excluding weekends
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    return date_range

def fetch_mto_day(dt):
    d_str = dt.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d_str}.DAT"
    req = urllib.request.Request(url, headers=HEADERS)
    res = {}
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            lines = r.read().decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7 and parts[3] == "EQ":
                    res[parts[2]] = {
                        "deliv_vol": int(parts[5]),
                        "deliv_pct": float(parts[6])
                    }
    except Exception:
        pass
    return res

def fetch_bhavcopy_day(dt):
    d_str = dt.strftime("%d%b%Y").upper()
    year_str = dt.strftime("%Y")
    month_str = dt.strftime("%b").upper()
    url = f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/cm{d_str}bhav.csv.zip"
    req = urllib.request.Request(url, headers=HEADERS)
    res = {}
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f)
                eq_df = df[df["SERIES"] == "EQ"]
                for _, row in eq_df.iterrows():
                    res[row["SYMBOL"].strip()] = {
                        "open": float(row["OPEN"]),
                        "high": float(row["HIGH"]),
                        "low": float(row["LOW"]),
                        "close": float(row["CLOSE"]),
                        "volume": int(row["TOTTRDQTY"])
                    }
    except Exception:
        pass
    return res

def build_5y_database():
    dates = get_trading_days(5)
    all_stocks = {} # {symbol: [daily_records]}

    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        bhav = fetch_bhavcopy_day(dt)
        mto = fetch_mto_day(dt)
        
        if not bhav or not mto:
            continue
            
        common = set(bhav.keys()).intersection(set(mto.keys()))
        for sym in common:
            if sym not in all_stocks:
                all_stocks[sym] = []
                
            p = bhav[sym]
            d = mto[sym]
            all_stocks[sym].append({
                "time": date_str,
                "open": round(p["open"], 2),
                "high": round(p["high"], 2),
                "low": round(p["low"], 2),
                "close": round(p["close"], 2),
                "delivery_vol": d["deliv_vol"],
                "deliv_pct": d["deliv_pct"]
            })

    # Calculate True Delivery OBV for each stock
    for sym, records in all_stocks.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df["Price_Diff"] = df["Close"].diff() if "Close" in df else df["close"].diff()
        df["Direction"] = np.where(df["Price_Diff"] > 0, 1, np.where(df["Price_Diff"] < 0, -1, 0))
        df["deliv_obv"] = (df["Direction"] * df["delivery_vol"]).cumsum()
        
        with open(f"{DATA_DIR}/{sym}.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f)
        print(f"Saved True Delivery OBV for {sym}")

if __name__ == "__main__":
    build_5y_database()
