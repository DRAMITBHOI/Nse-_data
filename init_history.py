import os
import json
import time
import urllib.request
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def backfill_5_years():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365) # 5 Years
    trading_days = pd.date_range(start=start_date, end=end_date, freq="B")
    
    print(f"📡 Backfilling 5 Years of True NSE Delivery Data (~{len(trading_days)} business sessions)...")
    
    all_stocks = {}

    for dt in trading_days:
        d_str = dt.strftime("%d%m%Y")
        date_formatted = dt.strftime("%Y-%m-%d")
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"
        req = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                df = pd.read_csv(resp)
        except Exception:
            continue  # Weekend / Exchange Holiday

        df.columns = df.columns.str.strip()
        df["SERIES"] = df["SERIES"].astype(str).str.strip()
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

        eq_df = df[df["SERIES"].isin(["EQ", "BE"])].copy()

        for _, r in eq_df.iterrows():
            sym = r["SYMBOL"]
            try:
                deliv_vol = int(float(str(r["DELIV_QTY"]).strip().replace("-", "0")))
                deliv_pct = float(str(r["DELIV_PER"]).strip().replace("-", "0"))
                o_p = float(r["OPEN_PRICE"])
                h_p = float(r["HIGH_PRICE"])
                l_p = float(r["LOW_PRICE"])
                c_p = float(r["CLOSE_PRICE"])
            except (ValueError, TypeError):
                continue

            if sym not in all_stocks:
                all_stocks[sym] = []

            all_stocks[sym].append({
                "time": date_formatted,
                "open": round(o_p, 2),
                "high": round(h_p, 2),
                "low": round(l_p, 2),
                "close": round(c_p, 2),
                "delivery_vol": deliv_vol,
                "deliv_pct": round(deliv_pct, 2)
            })

        print(f"Processed {date_formatted}")
        time.sleep(0.2)

    print(f"\n💾 Saving {len(all_stocks)} stock JSON files...")
    for sym, records in all_stocks.items():
        records.sort(key=lambda x: x["time"])
        with open(os.path.join(DATA_DIR, f"{sym}.json"), "w") as f:
            json.dump(records, f)

    print("✅ 5-Year True NSE Delivery history initialized successfully.")

if __name__ == "__main__":
    backfill_5_years()
