import os
import io
import json
import time
import zipfile
import datetime
import requests
import pandas as pd

DATA_DIR = "data"
PROGRESS_FILE = os.path.join(DATA_DIR, "init_progress.json")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

START_DATE = datetime.date(2021, 1, 1)
END_DATE = datetime.date(2026, 9, 6)

def get_official_mainboard_symbols(session):
    """Fetches the official master list of active NSE mainboard equities."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = df.columns.str.strip().str.upper()
            syms = set(df["SYMBOL"].dropna().astype(str).str.strip().str.upper().tolist())
            print(f"📋 Loaded {len(syms)} Official NSE Mainboard Listed Stocks from EQUITY_L.")
            return syms
    except Exception as e:
        print(f"⚠️ Notice: Could not download live EQUITY_L.csv ({e}). Processing all available equities.")
    return set()

def load_progress_date():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as fp:
                saved = json.load(fp).get("last_completed_date")
                if saved:
                    return datetime.datetime.strptime(saved, "%Y-%m-%d").date() + datetime.timedelta(days=1)
        except Exception:
            pass
    return START_DATE

def save_progress_date(d):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fp:
        json.dump({"last_completed_date": d.strftime("%Y-%m-%d")}, fp, indent=2)

def parse_legacy_mto(content_text, d_str):
    lines = content_text.strip().splitlines()
    day_map = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6:
            srs = parts[3].upper() if len(parts) > 3 else "EQ"
            if srs in ["EQ", "BE", "BZ", "SM", "ST"]:
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

def repair_all_stock_volumes(batch_limit_days=180):
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
        session.get("https://www.nseindia.com/all-reports", timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Notice during session handshake: {e}")

    mainboard_universe = get_official_mainboard_symbols(session)

    start_date = load_progress_date()
    if start_date > END_DATE:
        print("🎉 Full range 2021 to 2026 is already completely initialized!")
        return

    dates_to_pull = []
    curr = start_date
    while curr <= END_DATE and len(dates_to_pull) < batch_limit_days:
        if curr.weekday() < 5:
            dates_to_pull.append(curr)
        curr += datetime.timedelta(days=1)

    total_dates = len(dates_to_pull)
    print(f"🔧 Backfilling True NSE Delivery & OHLC Data for {total_dates} trading sessions ({start_date} to {dates_to_pull[-1]})...")

    daily_master = {}

    for idx, d in enumerate(dates_to_pull):
        d_mto = d.strftime("%d%m%Y")
        d_udiff = d.strftime("%Y%m%d")
        d_iso = d.strftime("%Y-%m-%d")

        urls = [
            f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d_mto}.csv",
            f"https://archives.nseindia.com/archives/equities/bhavcopy/pr/PR{d_mto}.zip",
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d_udiff}_F_0000.csv.zip",
            f"https://archives.nseindia.com/archives/equities/mto/MTO_{d_mto}.DAT"
        ]

        extracted = False
        for url in urls:
            try:
                resp = session.get(url, timeout=18)
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

                    # Filter for active Equity and Trade-to-Trade mainboard series
                    if srs_col:
                        df = df[df[srs_col].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]

                    day_map = {}
                    for _, row in df.iterrows():
                        sym = str(row[sym_col]).strip().upper()
                        if mainboard_universe and sym not in mainboard_universe:
                            continue

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
                                "open": round(o, 2),
                                "high": round(h, 2),
                                "low": round(l, 2),
                                "close": round(c, 2),
                                "volume": round(tot_vol),
                                "delivery_vol": round(d_vol),
                                "deliv_pct": round(d_pct, 1)
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

        save_progress_date(d)

        # Flush to disk every 30 dates
        if (idx + 1) % 30 == 0 or (idx + 1) == total_dates:
            flush_to_disk(daily_master)
            daily_master.clear()

    print(f"\n💾 Batch complete. Progress saved up to {load_progress_date()}.")

def flush_to_disk(daily_master):
    if not daily_master:
        return

    print("  ↳ Flushing collected dates to disk...")
    # Gather all unique symbols present in the newly extracted dates
    all_extracted_symbols = set()
    for d_str, records in daily_master.items():
        all_extracted_symbols.update(records.keys())

    for sym in all_extracted_symbols:
        json_path = os.path.join(DATA_DIR, f"{sym}.json")
        data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                    if isinstance(loaded, list):
                        data = loaded
            except Exception:
                data = []

        date_map = {}
        for r in data:
            if isinstance(r, dict) and "time" in r:
                t = str(r["time"])[:10]
                date_map[t] = r

        for d_str, records in daily_master.items():
            if sym in records:
                rec = records[sym]
                if t_rec := date_map.get(d_str):
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

        try:
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(sorted_list, fp, indent=2)
        except Exception:
            continue

if __name__ == "__main__":
    repair_all_stock_volumes()
