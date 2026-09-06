import os
import io
import json
import urllib.request
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def fetch_nse_marketcap_universe():
    print("📡 Downloading official NSE index constituent lists...")

    index_sources = [
        ("https://archives.nseindia.com/content/indices/ind_nifty500list.csv", "Nifty 500"),
        ("https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv", "Nifty Smallcap 250"),
        ("https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv", "Nifty Midcap 150"),
        ("https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv", "Nifty Microcap 250")
    ]

    verified_indices = {}

    for url, category in index_sources:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode('utf-8')))
                df.columns = df.columns.str.strip()

                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", "")).strip().upper()
                    industry = str(row.get("Industry", category))
                    if sym and sym != "NAN":
                        verified_indices[sym] = {
                            "category": category,
                            "industry": industry
                        }
            print(f"✅ Loaded {category} directory.")
        except Exception as e:
            print(f"⚠️ Failed to load {category}: {e}")

    # Scan entire data directory so all onboarded mainboard stocks are cataloged
    fundamentals = {}
    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "backtest_report.json", "init_progress.json", "scan_ultra_results.json",
            "scan_hp1_results.json", "scan_hp2_results.json"
        ]
    ]

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)
        latest_close = 0.0

        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[-1], dict):
                    latest_close = float(data[-1].get("close", 0.0) or 0.0)
        except Exception:
            pass

        idx_info = verified_indices.get(sym, {})
        fundamentals[sym] = {
            "category": idx_info.get("category", "NSE Equity"),
            "industry": idx_info.get("industry", "General Listed"),
            "price": latest_close,
            "qualified": True
        }

    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"\n🎉 Saved {len(fundamentals)} equities into {out_file}!")

if __name__ == "__main__":
    fetch_nse_marketcap_universe()
