import os
import json
import urllib.request
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def update_all_nse_stocks():
    today = datetime.now()
    d_str = today.strftime("%d%m%Y")
    
    # Official Consolidated NSE Security-Wise Bhavdata (Price + Delivery in 1 file)
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d_str}.csv"
    req = urllib.request.Request(url, headers=HEADERS)

    print(f"📡 Downloading official NSE Sec-Bhavdata for {d_str}...")
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            df = pd.read_csv(resp)
    except Exception as e:
        print(f"⚠️ NSE Data not available for {d_str} (Market holiday or not yet uploaded): {e}")
        return

    # Clean whitespace from column headers & text fields
    df.columns = df.columns.str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    # Filter only Equity segment ('EQ' and 'BE')
    eq_df = df[df["SERIES"].isin(["EQ", "BE"])].copy()
    
    date_formatted = today.strftime("%Y-%m-%d")
    updated_count = 0

    for _, row in eq_df.iterrows():
        sym = row["SYMBOL"]
        
        # Parse numeric delivery & price fields safely
        try:
            deliv_vol = int(float(str(row["DELIV_QTY"]).strip().replace("-", "0")))
            deliv_pct = float(str(row["DELIV_PER"]).strip().replace("-", "0"))
            o_price = float(row["OPEN_PRICE"])
            h_price = float(row["HIGH_PRICE"])
            l_price = float(row["LOW_PRICE"])
            c_price = float(row["CLOSE_PRICE"])
        except (ValueError, TypeError):
            continue

        file_path = os.path.join(DATA_DIR, f"{sym}.json")
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Avoid duplicate records for the same day
        if history and history[-1].get("time") == date_formatted:
            continue

        history.append({
            "time": date_formatted,
            "open": round(o_price, 2),
            "high": round(h_price, 2),
            "low": round(l_price, 2),
            "close": round(c_price, 2),
            "delivery_vol": deliv_vol,
            "deliv_pct": round(deliv_pct, 2)
        })

        # Keep last 5 years (~1300 trading days)
        if len(history) > 1300:
            history = history[-1300:]

        with open(file_path, "w") as f:
            json.dump(history, f)
        
        updated_count += 1

    print(f"✅ Successfully updated {updated_count} NSE stocks with 100% authentic exchange delivery data.")

if __name__ == "__main__":
    update_all_nse_stocks()
