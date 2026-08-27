import os
import io
import json
import zipfile
import datetime
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9"
}

def get_latest_existing_date():
    """Finds the most recent date recorded in data/ folder"""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json", "wyckoff_screener_results.json", "active_trade_plan.json", "backtest_report.json"]]
    if not files:
        return datetime.date.today() - datetime.timedelta(days=7)
    
    dates = []
    for f in files[:25]:
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
    return datetime.date.today() - datetime.timedelta(days=7)

def fetch_nse_daily_data(target_date):
    """Downloads official daily NSE Bhavcopy & Deliverable Volume (MTO)"""
    d_str_bhav = target_date.strftime("%d%b%Y").upper()
    d_str_mto = target_date.strftime("%d%m%Y")
    
    # 1. Download Price Bhavcopy
    bhav_url = f"https://archives.nseindia.com/content/historical/EQUITIES/{target_date.year}/{target_date.strftime('%b').upper()}/cm{d_str_bhav}bhav.csv.zip"
    bhav_df = None
    try:
        req = urllib.request.Request(bhav_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as zf:
                    bhav_df = pd.read_csv(zf)
                    bhav_df.columns = bhav_df.columns.str.strip()
                    bhav_df = bhav_df[bhav_df["SERIES"].isin(["EQ", "BE"])]
    except Exception as e:
        print(f"ℹ️ No Bhavcopy for {target_date} (Holiday or not yet uploaded).")
        return None

    # 2. Download Deliverable Position (MTO)
    mto_url = f"https://archives.nseindia.com/archives/equities/mto/MTO_{d_str_mto}.DAT"
    deliv_map = {}
    try:
        req = urllib.request.Request(mto_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8").splitlines()
            for line in content:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6 and parts[1] in ["EQ", "BE"]:
                    sym = parts[2].upper()
                    try:
                        d_qty = float(parts[5])
                        d_pct = float(parts[6]) if len(parts) > 6 else 0.0
                        deliv_map[sym] = {"delivery_vol": d_qty, "deliv_pct": d_pct}
                    except ValueError:
                        continue
    except Exception:
        print(f"⚠️ Deliverable MTO not found for {target_date}, using total traded volume.")

    # 3. Combine into date dictionary
    day_records = {}
    for _, row in bhav_df.iterrows():
        sym = str(row["SYMBOL"]).strip().upper()
        o = float(row["OPEN"])
        h = float(row["HIGH"])
        l = float(row["LOW"])
        c = float(row["CLOSE"])
        tot_vol = float(row["TOTTRDQTY"])
        
        deliv_info = deliv_map.get(sym, {"delivery_vol": tot_vol, "deliv_pct": 0.0})
        
        day_records[sym] = {
            "time": target_date.strftime("%Y-%m-%d"),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": tot_vol,
            "delivery_vol": deliv_info["delivery_vol"],
            "deliv_pct": deliv_info["deliv_pct"]
        }
    return day_records

def update_all_stock_jsons():
    last_date = get_latest_existing_date()
    today = datetime.date.today()
    
    print(f"📅 Last recorded date in repo: {last_date}")
    print(f"📅 Scanning for missing market sessions up to {today}...")

    missing_dates = []
    curr = last_date + datetime.timedelta(days=1)
    while curr <= today:
        if curr.weekday() < 5:  # Monday to Friday
            missing_dates.append(curr)
        curr += datetime.timedelta(days=1)

    if not missing_dates:
        print("✅ Data is already up to date!")
        update_fundamentals()
        return

    # Accumulate new days
    daily_updates = {}
    for d in missing_dates:
        print(f"📥 Fetching NSE data for {d}...")
        res = fetch_nse_daily_data(d)
        if res:
            daily_updates[d.strftime("%Y-%m-%d")] = res

    if not daily_updates:
        print("ℹ️ No new trading days to append.")
        update_fundamentals()
        return

    # Append to all stock JSON files in data/
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json", "wyckoff_screener_results.json", "active_trade_plan.json", "backtest_report.json"]]
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

    print(f"🎉 Successfully updated {updated_count} stock JSON files with latest market data!")
    update_fundamentals()

def update_fundamentals():
    print("📡 Updating index constituents and fundamentals...")
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
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json", "wyckoff_screener_results.json", "active_trade_plan.json", "backtest_report.json"]]
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

        if sym in verified_symbols:
            fundamentals[sym] = {
                "market_cap_status": "Verified Listed Equity",
                "industry": verified_symbols[sym]["industry"],
                "price": latest_close,
                "is_qualified": True
            }
        else:
            fundamentals[sym] = {
                "market_cap_status": "NSE Equity",
                "industry": "NSE Listed",
                "price": latest_close,
                "is_qualified": True if latest_close >= 20.0 else False
            }

    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w") as fp:
        json.dump(fundamentals, fp, indent=2)
    print(f"🎉 Successfully updated fundamentals in {out_file}!")

if __name__ == "__main__":
    update_all_stock_jsons()
