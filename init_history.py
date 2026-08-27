import os
import io
import json
import time
import zipfile
import datetime
import requests
import pandas as pd

DATA_DIR = "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/"
}

# Target historical gap: June 1, 2025 to Sept 15, 2025
START_DATE = datetime.date(2025, 6, 1)
END_DATE = datetime.date(2025, 9, 15)

def repair_historical_gap():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception:
        pass

    curr = START_DATE
    dates_to_pull = []
    while curr <= END_DATE:
        if curr.weekday() < 5:
            dates_to_pull.append(curr)
        curr += datetime.timedelta(days=1)

    print(f"🔧 Repairing {len(dates_to_pull)} trading days in 2025 gap...")

    repaired_days = {}
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
                    vol_col = next((c for c in ["TTL_TRD_QNTY", "TTL_TRADG_VOL", "VOLUME", "TTLTRADEDQTY"] if c in df.columns), None)
                    dlv_col = next((c for c in ["DELIV_QTY", "DELIVERY_QTY", "DLVRYQTY"] if c in df.columns), None)
                    pct_col = next((c for c in ["DELIV_PER", "DELIVERY_PCT", "DLVRYPER"] if c in df.columns), None)

                    # Include all Trade-to-Trade and Surveillance Series
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

                            day_map[sym] = {
                                "time": d.strftime("%Y-%m-%d"),
                                "open": o,
                                "high": h,
                                "low": l,
                                "close": c,
                                "volume": tot_vol,
                                "delivery_vol": d_vol,
                                "deliv_pct": d_pct
                            }
                        except Exception:
                            continue
                    
                    if len(day_map) > 0:
                        repaired_days[d.strftime("%Y-%m-%d")] = day_map
                        print(f"   -> ✅ Extracted {len(day_map)} stocks for {d}")
                        break
            except Exception:
                continue
        time.sleep(1)

    # Patch local/repo stock JSON files
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json", "wyckoff_screener_results.json"]]
    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)
            if not isinstance(data, list):
                continue
            
            data_dict = {r["time"]: r for r in data if isinstance(r, dict) and "time" in r}
            
            for d_str, records in repaired_days.items():
                if sym in records:
                    data_dict[d_str] = records[sym]
            
            final_list = sorted(list(data_dict.values()), key=lambda x: str(x.get("time", "")))
            with open(json_path, "w") as fp:
                json.dump(final_list, fp, indent=2)
        except Exception:
            continue

    print("🎉 Historical gap successfully patched!")

if __name__ == "__main__":
    repair_historical_gap()
