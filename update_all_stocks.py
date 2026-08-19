import os
import json
import urllib.request
import io
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def fetch_nse_index_constituents():
    print("📡 Downloading official NSE Broad-Market Index constituents...")
    
    # Official NSE Index Files (Large, Mid, Small, and Micro Caps)
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
                
                # Extract Symbol & Industry
                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", "")).strip().upper()
                    industry = str(row.get("Industry", "General"))
                    if sym and sym != "NAN":
                        verified_symbols[sym] = {
                            "industry": industry,
                            "is_nse_tracked": True
                        }
            print(f"✅ Loaded index data from: {url.split('/')[-1]}")
        except Exception as e:
            print(f"⚠️ Error reading {url}: {e}")

    print(f"📊 Total verified NSE universe: {len(verified_symbols)} stocks.")

    # 2. Match with local stock JSONs and estimate valid market size
    fundamentals = {}
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]]
    
    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)
        
        try:
            with open(json_path, "r") as f:
                raw_data = json.load(f)
            if not raw_data:
                continue
            latest_close = float(raw_data[-1]["close"])
        except Exception:
            continue

        # If present in official NSE Broad-market lists, mark as verified quality equity
        if sym in verified_symbols:
            fundamentals[sym] = {
                "market_cap_status": "Verified Listed Equity",
                "industry": verified_symbols[sym]["industry"],
                "price": latest_close,
                "is_qualified": True
            }
        else:
            # Other listed stocks: include if price >= 20 (excludes non-liquid penny stocks)
            fundamentals[sym] = {
                "market_cap_status": "NSE Equity",
                "industry": "NSE Listed",
                "price": latest_close,
                "is_qualified": True if latest_close >= 20.0 else False
            }

    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"🎉 Successfully populated {len(fundamentals)} stocks into {out_file}!")

if __name__ == "__main__":
    fetch_nse_index_constituents()
