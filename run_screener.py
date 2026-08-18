import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"

# ==========================================
# ⚙️ SCREENER FILTERS
# ==========================================
MIN_MARKET_CAP_CR = 500.0      # Min Market Cap in ₹ Cr (Lowered to 500 to catch mid/small-caps)
MAX_PE_RATIO = 45.0            # Max P/E (Raised to 45 to include high-growth compounding sectors)
MIN_9D_AVG_VOLUME = 50_000     # Min 9D Average Delivery Volume (50k shares)
MIN_PRICE_DROP_PCT = -3.0      # Price decreased by >= 3% (<= -3.0%)
MIN_OBV_GAIN_PCT = 1.5         # True Delivery OBV increased by >= 1.5%

# Dynamic Lookback Steps: 1W to 52W
LOOKBACK_STEPS = list(range(5, 255, 5))
if 252 not in LOOKBACK_STEPS:
    LOOKBACK_STEPS.append(252)

def format_tf(days):
    weeks = round(days / 5)
    return f"{weeks}W ({days}D)"

def run_cloud_screener():
    print("🚀 Starting 1W–52W True Delivery OBV Cloud Screener...")
    
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
            print(f"📊 Loaded fundamentals.json ({len(fundamentals)} records).")
        except Exception as e:
            print(f"⚠️ Error reading fundamentals.json: {e}")

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]
    ]
    print(f"🔍 Discovered {len(stock_files)} stock history files in data/.")

    results = []
    skipped_vol = 0
    skipped_fund = 0

    for f_name in stock_files:
        sym = f_name.replace(".json", "")
        fund = fundamentals.get(sym, {})
        mcap = fund.get("market_cap_cr", None)
        pe = fund.get("pe", None)

        # Filter by Market Cap if available
        if mcap is not None and float(mcap) < MIN_MARKET_CAP_CR:
            skipped_fund += 1
            continue

        # Filter by P/E if available (> 45 or <= 0)
        if pe is not None and (float(pe) <= 0 or float(pe) > MAX_PE_RATIO):
            skipped_fund += 1
            continue

        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if not raw_data or len(raw_data) < 10:
            continue

        # Calculate True Demat Delivery OBV
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

        # 9D Delivery Volume Gate
        sma_9 = np.mean(vols[-9:]) if len(vols) >= 9 else np.mean(vols)
        if sma_9 < MIN_9D_AVG_VOLUME:
            skipped_vol += 1
            continue

        curr_c = closes[-1]
        curr_obv = obvs[-1]
        matches = []

        # Multi-timeframe evaluation
        for days in LOOKBACK_STEPS:
            if len(closes) > days:
                past_c = closes[-days - 1]
                past_obv = obvs[-days - 1]
                p_chg = ((curr_c - past_c) / past_c) * 100
                obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0.0

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
                "Market Cap (Cr)": f"₹{float(mcap):,.0f}" if mcap else "N/A",
                "P/E": round(float(pe), 1) if pe else "N/A",
                "Active Span": span,
                "Strongest Timeframe": f"{best['label']} (P: {best['price_chg']}%, OBV: +{best['obv_chg']}%)",
                "Triggered Windows": len(matches)
            })

    results.sort(key=lambda x: x["Triggered Windows"], reverse=True)
    for r in results:
        del r["Triggered Windows"]

    out_path = os.path.join(DATA_DIR, "screener_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📊 Filter stats: Skipped {skipped_fund} on fundamentals, {skipped_vol} on volume.")
    print(f"🎉 Screener complete! Found {len(results)} stocks. Saved to {out_path}")

if __name__ == "__main__":
    run_cloud_screener()
