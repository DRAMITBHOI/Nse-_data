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
    "Referer": "https://www.nseindia.com/"
}

# Repair window: Past 6 Months (March 1, 2026 to Today)
START_DATE = datetime.date(2026, 3, 1)
END_DATE = datetime.date.today()

def repair_all_stock_volumes():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
        session.get("https://www.nseindia.com/all-reports", timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Notice during session initialization: {e}")

    dates_to_pull = []
    curr = START_DATE
    while curr <= END_DATE:
        if curr.weekday() < 5:  # Monday to Friday
            dates_to_pull.append(curr)
        curr += datetime.timedelta(days=1)

    print(f"🔧 Repairing deliverable volume for {len(dates_to_pull)} trading days from {START_DATE} to {END_DATE}...")

    daily_master = {}
    for d in dates_to_pull:
        d_mto = d.strftime("%d%m%Y")
        d_udiff = d.strftime("%Y%m%d")
        
        urls = [
            f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d_udiff}_F_0000.csv.zip"
        ]
        
        for url in urls:
            try:
                resp = session.get(url, timeout=20)
                if resp.status_code == 200 and len(resp.content) > 2000:
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

                    # Catch all active NSE equity and surveillance series
                    if srs_col:
                        df = df[df[srs_col].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST", "E1", "IL"])]

                    day_map = {}
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
                                "time": d.strftime("%Y-%m-%d"),
                                "open": o,
                                "high": h,
                                "low": l,
                                "close": c,
                                "volume": tot_vol,
                                "delivery_vol": d_vol,
                                "deliv_pct": d_pct
                            }
                            if sym not in day_map or entry["volume"] > day_map[sym]["volume"]:
                                day_map[sym] = entry
                        except Exception:
                            continue
                    
                    if len(day_map) > 0:
                        daily_master[d.strftime("%Y-%m-%d")] = day_map
                        print(f"   -> ✅ {d}: Extracted {len(day_map)} stocks")
                        break
            except Exception:
                continue
        time.sleep(1)

    print("\n💾 Updating repository JSON files...")
    stock_files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]

    repaired_count = 0
    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)
            if not isinstance(data, list):
                continue
            
            # Map existing records by date
            date_map = {}
            for r in data:
                if isinstance(r, dict) and "time" in r:
                    t = str(r["time"]).split(" ")[0].split("T")[0]
                    r["time"] = t
                    date_map[t] = r
            
            # Overwrite zero/empty volume entries with official repair data
            for d_str, records in daily_master.items():
                if sym in records:
                    date_map[d_str] = records[sym]

            sorted_list = [date_map[k] for k in sorted(date_map.keys())]

            # Drop consecutive identical holiday ghost candles
            final_clean = []
            for item in sorted_list:
                if final_clean:
                    prev = final_clean[-1]
                    if (item.get("open") == prev.get("open") and 
                        item.get("high") == prev.get("high") and 
                        item.get("low") == prev.get("low") and 
                        item.get("close") == prev.get("close")):
                        continue
                final_clean.append(item)

            with open(json_path, "w") as fp:
                json.dump(final_clean, fp, indent=2)
            repaired_count += 1
        except Exception:
            continue

    print(f"🎉 Successfully repaired delivery volume across {repaired_count} stocks!")

if __name__ == "__main__":
    repair_all_stock_volumes()
