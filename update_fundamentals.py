import os
import json
import time
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def update_fundamentals():
    # 1. Discover all tracked NSE symbols from your existing data folder
    symbols = [
        f.replace(".json", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f != "fundamentals.json"
    ]
    
    if not symbols:
        print("⚠️ No stock JSON files found in data/ folder to update fundamentals for.")
        return

    print(f"📡 Updating Fundamentals (Market Cap & P/E) for {len(symbols)} stocks...")
    
    fundamentals = {}
    
    # Load existing fundamentals if available to avoid losing data on failed calls
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            fundamentals = {}

    updated_count = 0

    for idx, sym in enumerate(symbols):
        ticker_symbol = f"{sym}.NS"
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Fast info lookup for market cap
            fast_info = getattr(ticker, "fast_info", None)
            mcap_val = getattr(fast_info, "market_cap", None) if fast_info else None
            
            # Convert to ₹ Crores (1 Cr = 10,000,000)
            mcap_cr = round(mcap_val / 1e7, 2) if mcap_val else None
            
            # Fallback to standard info for Trailing P/E
            pe_val = None
            try:
                info_dict = ticker.info
                if info_dict:
                    pe_val = info_dict.get("trailingPE") or info_dict.get("forwardPE")
                    if pe_val:
                        pe_val = round(float(pe_val), 2)
            except Exception:
                pass
            
            # Update record
            fundamentals[sym] = {
                "market_cap_cr": mcap_cr,
                "pe": pe_val
            }
            
            updated_count += 1
            if idx % 25 == 0:
                print(f"[{idx+1}/{len(symbols)}] Processed {sym}: MCap=₹{mcap_cr} Cr, P/E={pe_val}")
                
            time.sleep(0.1)  # Polite delay
            
        except Exception as e:
            print(f"⚠️ Error fetching fundamentals for {sym}: {e}")
            continue

    # 2. Write updated fundamentals file
    with open(fund_path, "w") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"\n✅ Successfully updated fundamentals for {updated_count}/{len(symbols)} stocks.")

if __name__ == "__main__":
    update_fundamentals()
