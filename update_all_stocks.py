import os
import io
import json
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

def get_latest_trading_date():
    return datetime.now()

def fetch_nse_delivery_mto(dt):
    """Fetches full market delivery data for all stocks in one request"""
    d_str = dt.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d_str}.DAT"
    req = urllib.request.Request(url, headers=HEADERS)
    delivery_map = {}
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            lines = resp.read().decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                # Record format: Type, SrNo, Symbol, Series, TradedQty, DeliverableQty, DeliveryPct
                if len(parts) >= 7 and parts[3] == "EQ":
                    delivery_map[parts[2]] = {
                        "deliv_vol": int(parts[5]),
                        "deliv_pct": float(parts[6])
                    }
    except Exception as e:
        print(f"MTO fetch warning for {d_str}: {e}")
    return delivery_map

def fetch_nse_bhavcopy_ohlc(dt):
    """Fetches official NSE daily OHLC price summary for all stocks"""
    d_str = dt.strftime("%d%b%Y").upper() # e.g. 18AUG2026
    year_str = dt.strftime("%Y")
    month_str = dt.strftime("%b").upper()
    url = f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/cm{d_str}bhav.csv.zip"
    
    req = urllib.request.Request(url, headers=HEADERS)
    price_map = {}
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = zip_file.namelist()[0]
            with zip_file.open(csv_name) as f:
                df = pd.read_csv(f)
                eq_df = df[df["SERIES"] == "EQ"]
                for _, r in eq_df.iterrows():
                    price_map[r["SYMBOL"].strip()] = {
                        "open": float(r["OPEN"]),
                        "high": float(r["HIGH"]),
                        "low": float(r["LOW"]),
                        "close": float(r["CLOSE"]),
                        "volume": int(r["TOTTRDQTY"])
                    }
    except Exception as e:
        print(f"Bhavcopy fetch warning for {d_str}: {e}")
    return price_map

def update_all_nse_stocks():
    today = get_latest_trading_date()
    date_str = today.strftime("%Y-%m-%d")
    
    print(f"📡 Downloading full NSE Market Bhavcopy & MTO Delivery files for {date_str}...")
    delivery_data = fetch_nse_delivery_mto(today)
    price_data = fetch_nse_bhavcopy_ohlc(today)

    if not delivery_data or not price_data:
        print("⚠️ No data available today (Market holiday or files not yet published).")
        return

    common_symbols = set(delivery_data.keys()).intersection(set(price_data.keys()))
    print(f"Updating data files for {len(common_symbols)} NSE listed stocks...")

    for sym in common_symbols:
        file_path = os.path.join(DATA_DIR, f"{sym}.json")
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Avoid duplicate daily entry
        if history and history[-1].get("time") == date_str:
            continue

        p = price_data[sym]
        d = delivery_data[sym]
        
        # Calculate True Delivery OBV
        prev_obv = history[-1]["deliv_obv"] if history else 0
        prev_close = history[-1]["close"] if history else p["open"]
        
        direction = 1 if p["close"] > prev_close else (-1 if p["close"] < prev_close else 0)
        current_obv = prev_obv + (direction * d["deliv_vol"])

        new_candle = {
            "time": date_str,
            "open": round(p["open"], 2),
            "high": round(p["high"], 2),
            "low": round(p["low"], 2),
            "close": round(p["close"], 2),
            "delivery_vol": d["deliv_vol"],
            "deliv_pct": d["deliv_pct"],
            "deliv_obv": current_obv
        }
        
        history.append(new_candle)

        # Retain last 5 years (~1,250 trading bars) to keep file sizes compact
        if len(history) > 1300:
            history = history[-1300:]

        with open(file_path, "w") as f:
            json.dump(history, f)

    print("✅ All stock files successfully updated.")

if __name__ == "__main__":
    update_all_nse_stocks()
