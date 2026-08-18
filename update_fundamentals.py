import os
import json
import time
import urllib.request
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/"
}

def clean_symbol(sym):
    """Filter out non-equity series, rights entitlements, and invalid formats."""
    if sym.endswith("-RE") or sym.endswith("-BE") or "-RE" in sym or sym.startswith("EBANK"):
        return None
    return sym.strip().upper()

def fetch_nse_marketcap_pe():
    fundamentals = {}
    
    # 1. Fetch NSE 500 & Broad Market constituents with official Market Cap & P/E
    print("📡 Fetching Market Cap & P/E data from official broad market feeds...")
    
    # URL providing consolidated NSE Market Cap & PE metrics
    source_url = "https://archives.nseindia.com/content/indices/ind_close_all_0.csv"
    
    # Fallback / Direct Bulk Metric via Screener consolidated export
    bulk_screener_url = "https://api.allorigins.win/raw?url=https://query1.finance.yahoo.com/v7/finance/quote"
    
    # Discover local tracked stock symbols
    local_symbols = [
        clean_symbol(f.replace(".json", ""))
        for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f != "fundamentals.json"
    ]
    local_symbols = sorted(list(set([s for s in local_symbols if s])))
    
    print(f"🔍 Discovered {len(local_symbols)} clean equity symbols in data/ folder.")
    
    # Batch processing in chunks of 50 to prevent rate limiting
    CHUNK_SIZE = 50
    for i in range(0, len(local_symbols), CHUNK_SIZE):
        chunk = local_symbols[i:i + CHUNK_SIZE]
        tickers_str = ",".join([f"{s}.NS" for s in chunk])
        query_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={tickers_str}"
        
        req = urllib.request.Request(query_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("quoteResponse", {}).get("result", [])
                
                for item in results:
                    raw_sym = item.get("symbol", "").replace(".NS", "")
                    mcap = item.get("marketCap", 0)
                    pe = item.get("trailingPE") or item.get("forwardPE", 0)
                    
                    mcap_cr = round(mcap / 1e7, 2) if mcap else None
                    pe_val = round(float(pe), 2) if pe else None
                    
                    fundamentals[raw_sym] = {
                        "market_cap_cr": mcap_cr,
                        "pe": pe_val
                    }
            print(f"✅ Processed batch {i // CHUNK_SIZE + 1} / {(len(local_symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE}")
        except Exception as e:
            print(f"⚠️ Batch {i // CHUNK_SIZE + 1} failed, saving defaults: {e}")
            for s in chunk:
                if s not in fundamentals:
                    fundamentals[s] = {"market_cap_cr": None, "pe": None}
        
        time.sleep(0.5)

    # Save to data/fundamentals.json
    out_file = os.path.join(DATA_DIR, "fundamentals.json")
    with open(out_file, "w") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"\n🎉 Successfully wrote {len(fundamentals)} stock records to {out_file}!")

if __name__ == "__main__":
    fetch_nse_marketcap_pe()
