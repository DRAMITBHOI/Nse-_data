import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"

# ==========================================
# ⚙️ HIGH-CONVICTION SCREENER FILTERS
# ==========================================
MIN_PRICE = 30.0               # Exclude penny/micro stocks (< ₹30)
MIN_9D_DELIV_TURNOVER_CR = 1.5 # Min 9-Day Mean Delivery Turnover in ₹ Crores
MIN_PRICE_DROP_PCT = -5.0      # Price dropped >= 5% (<= -5.0%)
MIN_OBV_GAIN_PCT = 5.0         # True Demat OBV gained >= 5.0%

# Active Swing Lookback: 1W (5D) to 26W (130D)
LOOKBACK_STEPS = list(range(5, 135, 5))

def format_tf(days):
    weeks = round(days / 5)
    return f"{weeks}W ({days}D)"

def run_cloud_screener():
    print("🚀 Starting High-Conviction True Delivery OBV Screener...")
    
    # 1. Load Nifty 500 / Smallcap 250 universe
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
            print(f"📊 Loaded {len(fundamentals)} institutional stocks from fundamentals.json.")
        except Exception as e:
            print(f"⚠️ Error loading fundamentals.json: {e}")

    results = []

    # 2. Only scan verified Nifty 500 & Smallcap 250 stocks
    target_symbols = list(fundamentals.keys()) if fundamentals else [
        f.replace(".json", "") for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]
    ]

    for sym in target_symbols:
        f_name = f"{sym}.json"
        json_path = os.path.join(DATA_DIR, f_name)
        
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if not raw_data or len(raw_data) < 15:
            continue

        # 3. Compute True Demat OBV
        obv = 0
        closes, obvs, vols, turnovers = [], [], [], []
        
        for i, r in enumerate(raw_data):
            c = float(r["close"])
            v = float(r.get("delivery_vol", 0))
            closes.append(c)
            vols.append(v)
            turnovers.append((c * v) / 1e7) # Delivery Turnover in ₹ Crores
            
            if i > 0:
                prev_c = closes[i - 1]
                if c > prev_c:
                    obv += v
                elif c < prev_c:
                    obv -= v
            else:
                obv = v
            obvs.append(obv)

        curr_c = closes[-1]
        if curr_c < MIN_PRICE:
            continue

        # 4. Check 9-Day Average Delivery Turnover (>= ₹1.5 Cr daily)
        sma_9_turnover = np.mean(turnovers[-9:]) if len(turnovers) >= 9 else np.mean(turnovers)
        if sma_9_turnover < MIN_9D_DELIV_TURNOVER_CR:
            continue

        curr_obv = obvs[-1]
        matches = []

        # 5. Multi-Timeframe Scan (1W to 26W)
        for days in LOOKBACK_STEPS:
            if len(closes) > days:
                past_c = closes[-days - 1]
                past_obv = obvs[-days - 1]
                p_chg = ((curr_c - past_c) / past_c) * 100
                obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0.0

                # Strong divergence: Price down >= 5% while True Demat OBV up >= 5%
                if p_chg <= MIN_PRICE_DROP_PCT and obv_chg >= MIN_OBV_GAIN_PCT:
                    matches.append({
                        "label": format_tf(days),
                        "price_chg": round(p_chg, 2),
                        "obv_chg": round(obv_chg, 2)
                    })

        if matches:
            best = max(matches, key=lambda x: x["obv_chg"])
            min_tf = matches[0]["label"]
            max_tf = matches[-1]["label"]
            span = f"{min_tf} to {max_tf}" if min_tf != max_tf else min_tf
            meta = fundamentals.get(sym, {})

            results.append({
                "Symbol": sym,
                "LTP (₹)": round(curr_c, 2),
                "9D Deliv Cr/Day": f"₹{sma_9_turnover:.2f} Cr",
                "Category": meta.get("category", "NSE Top Cap"),
                "Industry": meta.get("industry", "NSE"),
                "Active Span": span,
                "Strongest Timeframe": f"{best['label']} (P: {best['price_chg']}%, OBV: +{best['obv_chg']}%)",
                "Triggered Windows": len(matches)
            })

    # Sort by strongest confluence
    results.sort(key=lambda x: x["Triggered Windows"], reverse=True)
    for r in results:
        del r["Triggered Windows"]

    out_path = os.path.join(DATA_DIR, "screener_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 Screener complete! Found {len(results)} high-conviction institutional accumulation candidates.")
    print(f"📁 Output saved to {out_path}")

if __name__ == "__main__":
    run_cloud_screener()
