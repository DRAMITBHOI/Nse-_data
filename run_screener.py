import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"

# ==========================================
# ⚙️ STRICT SCREENER FILTERS
# ==========================================
MIN_MARKET_CAP_CR = 1000.0     # Minimum Market Cap in ₹ Crores
MAX_PE_RATIO = 30.0            # Maximum Trailing/Forward P/E Ratio
MIN_9D_AVG_VOLUME = 100_000    # Minimum 9-Day Moving Average Delivery Volume
MIN_PRICE_DROP_PCT = -5.0      # Price decreased by at least 5% (<= -5.0%)
MIN_OBV_GAIN_PCT = 2.5         # True Delivery OBV increased by at least 2.5%

# Dynamic Lookback Generator: 1 Week (5D) to 52 Weeks (252D) in 1-week steps
LOOKBACK_STEPS = list(range(5, 255, 5))
if 252 not in LOOKBACK_STEPS:
    LOOKBACK_STEPS.append(252)

def format_tf(days):
    weeks = round(days / 5)
    return f"{weeks}W ({days}D)"

def run_cloud_screener():
    print("🚀 Starting 1W–52W True Delivery OBV Cloud Screener...")
    
    # 1. Load fundamentals (Market Cap & P/E)
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
            print(f"📊 Loaded fundamentals database ({len(fundamentals)} stocks found).")
        except Exception as e:
            print(f"⚠️ Error loading fundamentals.json: {e}")
            fundamentals = {}
    else:
        print("⚠️ Warning: data/fundamentals.json not found! Please run update_fundamentals.py first.")

    # 2. Get all tracked stock JSON files
    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]
    ]
    print(f"🔍 Analyzing {len(stock_files)} stocks locally on runner...")

    results = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "")
        fund = fundamentals.get(sym, {})
        mcap = fund.get("market_cap_cr", None)
        pe = fund.get("pe", None)

        # -----------------------------------------------------------
        # 🛑 STRICT FUNDAMENTAL GATES
        # Discard stock if Market Cap is missing OR < ₹1,000 Cr
        if mcap is None or float(mcap) < MIN_MARKET_CAP_CR:
            continue

        # Discard stock if P/E is missing OR <= 0 OR > 30
        if pe is None or float(pe) <= 0 or float(pe) > MAX_PE_RATIO:
            continue
        # -----------------------------------------------------------

        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if not raw_data or len(raw_data) < 10:
            continue

        # 3. Calculate True Delivery OBV in memory
        obv = 0
        closes, obvs, vols = [], [], []
        for i, r in enumerate(raw_data):
            c = float(r["close"])
            v = float(r.get("delivery_vol", 0))
            closes.append(c)
            vols.append(v)
            if i > 0:
                prev_c = closes[i - 1]
                if c > prev_c:
                    obv += v
                elif c < prev_c:
                    obv -= v
            else:
                obv = v
            obvs.append(obv)

        # 4. 9-Day Moving Average Mean Delivery Volume Gate
        sma_9 = np.mean(vols[-9:]) if len(vols) >= 9 else np.mean(vols)
        if sma_9 < MIN_9D_AVG_VOLUME:
            continue

        curr_c = closes[-1]
        curr_obv = obvs[-1]
        matches = []

        # 5. Multi-Timeframe Scan from 1W to 52W
        for days in LOOKBACK_STEPS:
            if len(closes) > days:
                past_c = closes[-days - 1]
                past_obv = obvs[-days - 1]
                p_chg = ((curr_c - past_c) / past_c) * 100
                obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0.0

                # Condition: Price decreased >= 5% AND True OBV increased >= 2.5%
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

            results.append({
                "Symbol": sym,
                "LTP (₹)": round(curr_c, 2),
                "9D Avg Deliv Vol": f"{sma_9/1e3:.1f}K",
                "Market Cap (Cr)": f"₹{float(mcap):,.0f}",
                "P/E": round(float(pe), 1),
                "Active Span": span,
                "Strongest Timeframe": f"{best['label']} (P: {best['price_chg']}%, OBV: +{best['obv_chg']}%)",
                "Triggered Windows": len(matches)
            })

    # Sort results by multi-confluence (stocks showing accumulation across most windows)
    results.sort(key=lambda x: x["Triggered Windows"], reverse=True)
    for r in results:
        del r["Triggered Windows"]

    out_path = os.path.join(DATA_DIR, "screener_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 Screener complete! Found {len(results)} high-quality accumulation setups.")
    print(f"📁 Saved to {out_path}")

if __name__ == "__main__":
    run_cloud_screener()
