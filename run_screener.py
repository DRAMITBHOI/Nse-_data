import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"

# ==========================================
# ⚙️ SCREENER FILTERS
# ==========================================
MIN_PRICE = 20.0               # Minimum Stock Price (₹20 to eliminate sub-penny stocks)
MIN_9D_AVG_VOLUME = 75_000     # Minimum 9-Day Average Delivery Volume (Demat shares)
MIN_PRICE_DROP_PCT = -3.0      # Price drop >= 3% (<= -3.0%)
MIN_OBV_GAIN_PCT = 2.0         # True Demat OBV increased >= 2.0%

# Dynamic Lookback Steps: 1W to 52W in weekly intervals
LOOKBACK_STEPS = list(range(5, 255, 5))
if 252 not in LOOKBACK_STEPS:
    LOOKBACK_STEPS.append(252)

def format_tf(days):
    weeks = round(days / 5)
    return f"{weeks}W ({days}D)"

def run_cloud_screener():
    print("🚀 Starting 1W–52W True Delivery OBV Cloud Screener...")
    
    # 1. Load fundamentals directory
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
            print(f"📊 Loaded fundamentals index ({len(fundamentals)} equities).")
        except Exception as e:
            print(f"⚠️ Error reading fundamentals.json: {e}")

    # 2. Get local stock data files
    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]
    ]
    print(f"🔍 Scanning {len(stock_files)} stocks...")

    results = []
    skipped_vol = 0
    skipped_penny = 0

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        fund = fundamentals.get(sym, {})

        # Basic verification check
        if fund and not fund.get("is_qualified", True):
            skipped_penny += 1
            continue

        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if not raw_data or len(raw_data) < 10:
            continue

        # 3. Calculate True Demat Delivery OBV
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

        curr_c = closes[-1]
        if curr_c < MIN_PRICE:
            skipped_penny += 1
            continue

        # 4. 9-Day Moving Average Mean Delivery Volume Gate
        sma_9 = np.mean(vols[-9:]) if len(vols) >= 9 else np.mean(vols)
        if sma_9 < MIN_9D_AVG_VOLUME:
            skipped_vol += 1
            continue

        curr_obv = obvs[-1]
        matches = []

        # 5. Scan 1W to 52W Lookback Spans
        for days in LOOKBACK_STEPS:
            if len(closes) > days:
                past_c = closes[-days - 1]
                past_obv = obvs[-days - 1]
                p_chg = ((curr_c - past_c) / past_c) * 100
                obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0.0

                # Price down >= 3% while True Delivery OBV up >= 2.0%
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
            industry = fund.get("industry", "NSE Listed")

            results.append({
                "Symbol": sym,
                "LTP (₹)": round(curr_c, 2),
                "9D Avg Deliv Vol": f"{sma_9/1e3:.1f}K",
                "Industry / Category": industry,
                "Active Span": span,
                "Strongest Timeframe": f"{best['label']} (P: {best['price_chg']}%, OBV: +{best['obv_chg']}%)",
                "Triggered Windows": len(matches)
            })

    # Sort results by multi-confluence accumulation setups
    results.sort(key=lambda x: x["Triggered Windows"], reverse=True)
    for r in results:
        del r["Triggered Windows"]

    out_path = os.path.join(DATA_DIR, "screener_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Filter Summary: Skipped {skipped_penny} penny/sub-₹20 stocks, {skipped_vol} illiquid stocks.")
    print(f"🎉 Screener complete! Found {len(results)} accumulation candidates.")
    print(f"📁 Output written to {out_path}")

if __name__ == "__main__":
    run_cloud_screener()
