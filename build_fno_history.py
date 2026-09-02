import os
import io
import json
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
FNO_STORE = os.path.join(DATA_DIR, "fno_history.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def get_trading_days(start_date, end_date):
    """Generate potential trading dates (weekdays)."""
    cur = start_date
    days = []
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days

def fetch_nse_fno_bhavcopy(dt):
    """Download and extract daily F&O Bhavcopy from NSE archives."""
    dd = dt.strftime("%d")
    mon = dt.strftime("%b").upper()
    yyyy = dt.strftime("%Y")
    
    # Modern NSE Derivatives archive URL pattern
    url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip"
    
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
                    df.columns = df.columns.str.strip().str.upper()
                    return df
    except Exception:
        # Returns None on market holidays or network timeouts
        return None

def process_fno_bhavcopy(df, date_str):
    """Extract near-month Stock Futures, compute total OI and near-month price."""
    # Filter for Stock Futures only (excludes Index and Options strikes)
    fut = df[df["INSTRUMENT"] == "FUTSTK"].copy()
    if fut.empty:
        return {}

    fut["EXPIRY_DT"] = pd.to_datetime(fut["EXPIRY_DT"])
    fut = fut.sort_values(["SYMBOL", "EXPIRY_DT"])

    metrics_by_sym = {}
    for sym, grp in fut.groupby("SYMBOL"):
        sym = sym.strip().upper()
        # Sum OI across all active contract expiries (Near + Mid + Far)
        total_oi = float(grp["OPEN_INT"].sum())
        total_chg_oi = float(grp["CHG_IN_OI"].sum())
        
        # Near-month contract is the earliest expiry
        near_contract = grp.iloc[0]
        near_close = float(near_contract["CLOSE"])
        
        metrics_by_sym[sym] = {
            "fut_close": round(near_close, 2),
            "total_oi": total_oi,
            "chg_oi": total_chg_oi
        }
    return metrics_by_sym

def update_fno_historical_database(lookback_days=120):
    """Fetch and update F&O metrics for the last N calendar days."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    master_fno = {}
    if os.path.exists(FNO_STORE):
        try:
            with open(FNO_STORE, "r") as fp:
                master_fno = json.load(fp)
        except Exception:
            master_fno = {}

    end_d = datetime.now()
    start_d = end_d - timedelta(days=lookback_days)
    calendar_days = get_trading_days(start_d, end_d)
    
    print(f"🚀 Updating F&O Historical Data ({len(calendar_days)} sessions)...")
    
    fetched_count = 0
    for dt in calendar_days:
        d_str = dt.strftime("%Y-%m-%d")
        # Skip if already downloaded
        if d_str in master_fno and master_fno[d_str]:
            continue

        df_bhav = fetch_nse_fno_bhavcopy(dt)
        if df_bhav is not None and not df_bhav.empty:
            extracted = process_fno_bhavcopy(df_bhav, d_str)
            if extracted:
                master_fno[d_str] = extracted
                fetched_count += 1
                if fetched_count % 10 == 0:
                    print(f"  📥 Processed {d_str} ({len(extracted)} F&O stocks)")
        
        # Brief throttle to prevent rate-limiting
        time.sleep(0.4)

    with open(FNO_STORE, "w") as fp:
        json.dump(master_fno, fp, indent=2)

    print(f"✅ F&O Historical Sync complete. Saved to '{FNO_STORE}'.")

if __name__ == "__main__":
    import time
    update_fno_historical_database(lookback_days=180)
