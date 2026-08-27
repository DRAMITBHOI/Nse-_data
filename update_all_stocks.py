import os
import io
import json
import time
import zipfile
import datetime
import requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

def get_latest_recorded_date():
    files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]
    if not files:
        return datetime.date.today() - datetime.timedelta(days=15)
    
    dates = []
    for f in files[:40]:
        try:
            with open(os.path.join(DATA_DIR, f), "r") as fp:
                raw = json.load(fp)
                if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[-1], dict):
                    t = str(raw[-1].get("time", "")).split(" ")[0].split("T")[0]
                    if t:
                        dates.append(t)
        except Exception:
            continue
    if dates:
        return datetime.datetime.strptime(max(dates), "%Y-%m-%d").date()
    return datetime.date.today() - datetime.timedelta(days=15)

def download_nse_session_data(session, target_date):
    """Downloads official daily NSE Bhavcopy & Deliverable stats."""
    d_mto = target_date.strftime("%d%m%Y")
    d_udiff = target_date.strftime("%Y%m%d")
    
    urls = [
        f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
        f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d_udiff}_F_0000.csv.zip"
    ]

    for url in urls:
        try:
            resp = session.get(url, timeout=25)
            if resp.status_code == 200 and len(resp.content) > 1500:
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        csv_name = z.namelist()[0]
                        with z.open(csv_name) as zf:
                            df = pd.read_csv(zf)
                else:
                    df = pd.read_csv(io.StringIO(resp.text))

                df.columns = df.columns.str.strip().str.upper()

                sym_col = next((c for c in ["SYMBOL", "TRADINGSYMBOL", "TCKRSYMB"] if c in df.columns), None)
                srs_col = next((c for c in ["SERIES", "SRIS", "SCTYSRS"] if c in df.columns), None)
                cls_col = next((c for c in ["CLOSE_PRICE", "CLOSE", "CLSPRIC"] if c in df.columns), None)
                opn_col = next((c for c in ["OPEN_PRICE", "OPEN", "OPNPRIC"] if c in df.columns), None)
                hgh_col = next((c for c in ["HIGH_PRICE", "HIGH", "HGHPRIC"] if c in df.columns), None)
                low_col = next((c for c in ["LOW_PRICE", "LOW", "LWPRIC"] if c in df.columns), None)
                vol_col = next((c for c in ["TTL_TRD_QNTY", "TTL_TRADG_VOL", "VOLUME", "TTLTRADEDQTY", "TTL_TRADED_QTY"] if c in df.columns), None)
                dlv_col = next((c for c in ["DELIV_QTY", "DELIVERY_QTY", "DLVRYQTY", "DLVRY_QTY"] if c in df.columns), None)
                pct_col = next((c for c in ["DELIV_PER", "DELIVERY_PCT", "DLVRYPER", "DLVRY_PER"] if c in df.columns), None)

                if not (sym_col and cls_col):
                    continue

                if srs_col:
                    df = df[df[srs_col].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST", "E1", "IL"])]

                day_records = {}
                for _, row in df.iterrows():
                    sym = str(row[sym_col]).strip().upper()
                    try:
                        c = float(row[cls_col])
                        o = float(row[opn_col]) if opn_col else c
                        h = float(row[hgh_col]) if hgh_col else c
                        l = float(row[low_col]) if low_col else c
                        tot_vol = float(row[vol_col]) if vol_col else 0.0

                        d_raw = str(row.get(dlv_col, "")).strip().replace("-", "") if dlv_col else ""
                        d_vol = float(d_raw) if (d_raw and d_raw.lower() != "nan") else tot_vol

                        p_raw = str(row.get(pct_col, "")).strip().replace("-", "") if pct_col else ""
                        d_pct = float(p_raw) if (p_raw and p_raw.lower() != "nan") else (round((d_vol / tot_vol) * 100, 1) if tot_vol > 0 else 0.0)

                        entry = {
                            "time": target_date.strftime("%Y-%m-%d"),
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": tot_vol,
                            "delivery_vol": d_vol,
                            "deliv_pct": d_pct
                        }

                        if sym not in day_records or entry["volume"] > day_records[sym]["volume"]:
                            day_records[sym] = entry
                    except Exception:
                        continue

                if len(day_records) > 0:
                    return day_records
        except Exception:
            continue
    return None

def update_all_stocks():
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
        session.get("https://www.nseindia.com/all-reports", timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Session notice: {e}")

    last_date = get_latest_recorded_date()
    today = datetime.date.today()

    print(f"📅 Last recorded date: {last_date}")
    print(f"📅 Scanning dates from {last_date + datetime.timedelta(days=1)} to {today}...")

    missing_dates = []
    curr = last_date + datetime.timedelta(days=1)
    while curr <= today:
        if curr.weekday() < 5:
            missing_dates.append(curr)
        curr += datetime.timedelta(days=1)

    daily_updates = {}
    if missing_dates:
        for d in missing_dates:
            print(f"📥 Fetching official NSE Bhavcopy for {d}...")
            records = download_nse_session_data(session, d)
            if records:
                daily_updates[d.strftime("%Y-%m-%d")] = records
                print(f"   -> ✅ SUCCESS: Extracted {len(records)} stocks for {d}")
            time.sleep(1)

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

        if not isinstance(stock_data, list):
            continue

        # Strict date deduplication & volume repair map
        date_map = {}
        for r in stock_data:
            if isinstance(r, dict) and "time" in r:
                t = str(r["time"]).split(" ")[0].split("T")[0]
                r["time"] = t
                if t not in date_map or float(r.get("volume", 0)) > float(date_map[t].get("volume", 0)):
                    date_map[t] = r

        for d_str, records in daily_updates.items():
            if sym in records:
                date_map[d_str] = records[sym]

        final_clean_list = [date_map[k] for k in sorted(date_map.keys())]

        with open(json_path, "w") as fp:
            json.dump(final_clean_list, fp, indent=2)
        updated_count += 1

    print(f"🎉 Processed and cleaned {updated_count} stock files!")

if __name__ == "__main__":
    update_all_stocks()
