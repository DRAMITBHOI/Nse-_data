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

# Multi-Year Historical Range: From Jan 1, 2021 to Today
START_DATE = datetime.date(2021, 1, 1)
END_DATE = datetime.date.today()

def parse_legacy_mto(content_text, d_str):
    """Parses legacy NSE MTO_DDMMYYYY.DAT pipe/comma delimited delivery archives."""
    lines = content_text.strip().splitlines()
    day_map = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6:
            # Format: RecordType, SrNo, Symbol, Series, TradedQty, DelivQty, DelivPct
            srs = parts[3].upper() if len(parts) > 3 else "EQ"
            if srs in ["EQ", "BE", "BZ", "SM", "ST", "E1", "IL"]:
                sym = parts[2].upper()
                try:
                    tot_v = float(parts[4])
                    dlv_v = float(parts[5])
                    pct = float(parts[6]) if len(parts) > 6 and parts[6] else (round((dlv_v / tot_v) * 100, 1) if tot_v > 0 else 0.0)
                    day_map[sym] = {
                        "time": d_str,
                        "volume": tot_v,
                        "delivery_vol": dlv_v,
                        "deliv_pct": pct
                    }
                except Exception:
                    continue
    return day_map

def repair_all_stock_volumes():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
        session.get("https://www.nseindia.com/all-reports", timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Notice during session handshake: {e}")

    dates_to_pull = []
    curr = START_DATE
    while curr <= END_DATE:
        if curr.weekday() < 5:  # Monday to Friday
            dates_to_pull.append(curr)
        curr += datetime.timedelta(days=1)

    total_dates = len(dates_to_pull)
    print(f"🔧 Backfilling genuine NSE delivery volume for {total_dates} trading sessions from {START_DATE} to {END_DATE}...")

    daily_master = {}

    for idx, d in enumerate(dates_to_pull):
        d_mto = d.strftime("%d%m%Y")
        d_udiff = d.strftime("%Y%m%d")
        d_iso = d.strftime("%Y-%m-%d")
        
        urls = [
            f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://archives.nseindia.com/archives/equities/bhavcopy/pr/PR{d_mto}.zip",
            f"https://archives.nseindia.com/archives/equities/mto/MTO_{d_mto}.DAT",
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d_udiff}_F_0000.csv.zip"
        ]
        
        extracted = False
        for url in urls:
            try:
                resp = session.get(url, timeout=20)
                if resp.status_code == 200 and len(resp.content) > 500:
                    if url.endswith(".DAT"):
                        mto_map = parse_legacy_mto(resp.text, d_iso)
                        if mto_map:
                            daily_master[d_iso] = mto_map
                            extracted = True
                            print(f"[{idx+1}/{total_dates}] ✅ {d}: Extracted {len(mto_map)} stocks via MTO.DAT")
                            break

                    df = None
                    if url.endswith(".zip"):
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                            for z_name in z.namelist():
                                if "bhav" in z_name.lower() or "pr" in z_name.lower() or z_name.endswith(".csv"):
                                    with z.open(z_name) as zf:
                                        try:
                                            df = pd.read_csv(zf)
                                            break
                                        except Exception:
                                            continue
                    else:
                        df = pd.read_csv(io.StringIO(resp.text))
                    
                    if df is None or df.empty:
                        continue

                    df.columns = df.columns.str.strip().str.upper()
                    
                    sym_col = next((c for c in ["SYMBOL", "TRADINGSYMBOL", "TCKRSYMB", "SECURITY"] if c in df.columns), None)
                    srs_col = next((c for c in ["SERIES", "SRIS", "SCTYSRS"] if c in df.columns), None)
                    cls_col = next((c for c in ["CLOSE_PRICE", "CLOSE", "CLSPRIC", "CLOSEPRICE"] if c in df.columns), None)
                    opn_col = next((c for c in ["OPEN_PRICE", "OPEN", "OPNPRIC", "OPENPRICE"] if c in df.columns), None)
                    hgh_col = next((c for c in ["HIGH_PRICE", "HIGH", "HGHPRIC", "HIGHPRICE"] if c in df.columns), None)
                    low_col = next((c for c in ["LOW_PRICE", "LOW", "LWPRIC", "LOWPRICE"] if c in df.columns), None)
                    vol_col = next((c for c in ["TTL_TRD_QNTY", "TTL_TRADG_VOL", "VOLUME", "TTLTRADEDQTY", "TTL_TRADED_QTY", "TRADEDQTY"] if c in df.columns), None)
                    dlv_col = next((c for c in ["DELIV_QTY", "DELIVERY_QTY", "DLVRYQTY", "DLVRY_QTY", "DELIVERYQTY"] if c in df.columns), None)
                    pct_col = next((c for c in ["DELIV_PER", "DELIVERY_PCT", "DLVRYPER", "DLVRY_PER", "DELIVERYPER"] if c in df.columns), None)

                    if not (sym_col and cls_col):
                        continue

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
                                "time": d_iso,
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
                        daily_master[d_iso] = day_map
                        extracted = True
                        print(f"[{idx+1}/{total_dates}] ✅ {d}: Extracted {len(day_map)} stocks")
                        break
            except Exception:
                continue

        # Save progress every 40 trading days
        if (idx + 1) % 40 == 0 or (idx + 1) == total_dates:
            flush_to_disk(daily_master)

    print(f"\n🎉 2021–2026 Historical Backfill Completed Successfully across {len(daily_master)} trading sessions!")

def flush_to_disk(daily_master):
    if not daily_master:
        return

    stock_files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json", "segmented_backtest_report.json",
            "scanA_results.json", "nifty750.json"
        ]
    ]

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)
            if not isinstance(data, list):
                continue
            
            date_map = {}
            for r in data:
                if isinstance(r, dict) and "time" in r:
                    t = str(r["time"]).split(" ")[0].split("T")[0]
                    r["time"] = t
                    date_map[t] = r
            
            for d_str, records in daily_master.items():
                if sym in records:
                    rec = records[sym]
                    if t_rec := date_map.get(d_str):
                        # Preserve existing price if MTO only provided delivery
                        t_rec["volume"] = rec.get("volume", t_rec.get("volume", 0))
                        t_rec["delivery_vol"] = rec.get("delivery_vol", t_rec.get("volume", 0))
                        t_rec["deliv_pct"] = rec.get("deliv_pct", 0)
                        if "open" in rec:
                            t_rec["open"] = rec["open"]
                            t_rec["high"] = rec["high"]
                            t_rec["low"] = rec["low"]
                            t_rec["close"] = rec["close"]
                    else:
                        if "open" in rec:
                            date_map[d_str] = rec

            sorted_list = [date_map[k] for k in sorted(date_map.keys())]

            final_clean = []
            for item in sorted_list:
                if final_clean:
                    prev = final_clean[-1]
                    if (item.get("open") == prev.get("open") and 
                        item.get("high") == prev.get("high") and 
                        item.get("low") == prev.get("low") and 
                        item.get("close") == prev.get("close") and 
                        item.get("volume", 0) == 0):
                        continue
                final_clean.append(item)

            with open(json_path, "w") as fp:
                json.dump(final_clean, fp, indent=2)
        except Exception:
            continue

if __name__ == "__main__":
    repair_all_stock_volumes()
