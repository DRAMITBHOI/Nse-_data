import os
import io
import json
import datetime
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def get_latest_existing_date():
    files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]
    if not files:
        return datetime.date.today() - datetime.timedelta(days=10)
    
    dates = []
    for f in files[:35]:
        try:
            with open(os.path.join(DATA_DIR, f), "r") as fp:
                raw = json.load(fp)
                if raw:
                    dates.append(raw[-1]["time"])
        except Exception:
            continue
    if dates:
        max_d_str = max(dates)
        return datetime.datetime.strptime(max_d_str, "%Y-%m-%d").date()
    return datetime.date.today() - datetime.timedelta(days=10)

def fetch_nse_full_bhavdata(target_date):
    """Downloads official daily unified Price & Delivery Bhavcopy from NSE archives"""
    d_str = target_date.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"
    
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            df.columns = df.columns.str.strip()
            
            # Keep equity series
            df = df[df["SERIES"].isin(["EQ", "BE"])]
            
            day_records = {}
            for _, row in df.iterrows():
                sym = str(row["SYMBOL"]).strip().upper()
                try:
                    o = float(row["OPEN_PRICE"])
                    h = float(row["HIGH_PRICE"])
                    l = float(row["LOW_PRICE"])
                    c = float(row["CLOSE_PRICE"])
                    tot_vol = float(row["TTL_TRD_QNTY"])
                    
                    deliv_raw = str(row.get("DELIV_QTY", "")).strip().replace("-", "")
                    deliv_vol = float(deliv_raw) if deliv_raw else tot_vol
                    
                    pct_raw = str(row.get("DELIV_PER", "")).strip().replace("-", "")
                    deliv_pct = float(pct_raw) if pct_raw else (round((deliv_vol / tot_vol) * 100, 1) if tot_vol > 0 else 0.0)
                    
                    day_records[sym] = {
                        "time": target_date.strftime("%Y-%m-%d"),
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": tot_vol,
                        "delivery_vol": deliv_vol,
                        "deliv_pct": deliv_pct
                    }
                except Exception:
                    continue
            return day_records
    except Exception as e:
        print(f"ℹ️ {target_date} skipped / not yet uploaded on NSE: {e}")
        return None

def update_all_stock_jsons():
    last_date = get_latest_existing_date()
    today = datetime.date.today()
    
    print(f"📅 Last recorded date in repo: {last_date}")
    print(f"📅 Fetching missing trading days between {last_date + datetime.timedelta(days=1)} and {today}...")

    missing_dates = []
    curr = last_date + datetime.timedelta(days=1)
    while curr <= today:
        if curr.weekday() < 5:  # Skip Sat / Sun
            missing_dates.append(curr)
        curr += datetime.timedelta(days=1)

    if not missing_dates:
        print("✅ Data is already up to date!")
        update_fundamentals()
        return

    daily_updates = {}
    for d in missing_dates:
        print(f"📥 Downloading NSE Bhavcopy for {d}...")
        records = fetch_nse_full_bhavdata(d)
        if records:
            daily_updates[d.strftime("%Y-%m-%d")] = records
            print(f"   -> Extracted {len(records)} stocks for {d}")

    if not daily_updates:
        print("ℹ️ No new trading days fetched.")
        update_fundamentals()
        return

    stock_files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]
    
    updated_count = 0
    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)
        
        try:
            with open(json_path, "r") as fp:
                stock_data = json.load(fp)
        except Exception:
            continue

        existing_times = {r["time"] for r in stock_data}
        added = False

        for d_str, records in daily_updates.items():
            if d_str not in existing_times and sym in records:
                stock_data.append(records[sym])
                added = True

        if added:
            stock_data.sort(key=lambda x: x["time"])
            with open(json_path, "w") as fp:
                json.dump(stock_data, fp, indent=2)
            updated_count += 1

    print(f"🎉 Updated {updated_count} stock files with new records up to {today}!")
    update_fundamentals()

def update_fundamentals():
    print("📡 Updating index constituents & fundamentals.json...")
    index_urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"
    ]
    verified_symbols = {}
    for url in index_urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode('utf-8')))
                df.columns = df.columns.str.strip()
                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", "")).strip().upper()
                    industry = str(row.get("Industry", "General"))
                    if sym and sym != "NAN":
                        verified_symbols[sym] = {"industry": industry, "is_nse_tracked": True}
        except Exception as e:
            print(f"⚠️ Error reading {url}: {e}")

    fundamentals = {}
    stock_files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]
    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)
        try:
            with open(json_path, "r") as fp:
                raw_data = json.load(fp)
            if not raw_data:
                continue
            latest_close = float(raw_data[-1]["close"])
        except Exception:
            continue

        fundamentals[sym] = {
            "market_cap_status": "Verified Listed Equity" if sym in verified_symbols else "NSE Equity",
            "industry": verified_symbols.get(sym, {}).get("industry", "NSE Listed"),
            "price": latest_close,
            "is_qualified": True if (sym in verified_symbols or latest_close >= 20.0) else False
        }

    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w") as fp:
        json.dump(fundamentals, fp, indent=2)
    print(f"🎉 Saved {len(fundamentals)} records into {out_file}!")

if __name__ == "__main__":
    update_all_stock_jsons()
