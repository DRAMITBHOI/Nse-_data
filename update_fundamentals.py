import os
import json
import urllib.request
import io
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def fetch_nse_marketcap_universe():
    print("📡 Downloading official Nifty 500 & Smallcap 250 universe from NSE...")
    
    # Official NSE Index lists (Top 750 liquid companies with Market Cap >= ₹1,000 Cr)
    index_sources = [
        ("https://archives.nseindia.com/content/indices/ind_nifty500list.csv", "Nifty 500"),
        ("https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv", "Nifty Smallcap 250"),
        ("https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv", "Nifty Midcap 150")
    ]
    
    fundamentals = {}
    
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
                        fundamentals[sym] = {
                            "category": category,
                            "industry": industry,
                            "qualified": True
                        }
            print(f"✅ Loaded {category} directory.")
        except Exception as e:
            print(f"⚠️ Failed to load {category}: {e}")

    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"\n🎉 Saved {len(fundamentals)} verified institutional equities to {out_file}!")

if __name__ == "__main__":
    fetch_nse_marketcap_universe()
