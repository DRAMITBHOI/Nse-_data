import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_MARKET_CAP_CR = 1000
MAX_PE_RATIO = 30.0
MIN_9D_AVG_VOLUME = 100_000
MIN_PRICE_DROP_PCT = -5.0
MIN_OBV_GAIN_PCT = 2.5

LOOKBACK_STEPS = list(range(5, 255, 5))
if 252 not in LOOKBACK_STEPS:
    LOOKBACK_STEPS.append(252)

def format_tf(days):
    return f"{round(days/5)}W ({days}D)"

def run_cloud_screener():
    print("🚀 Starting 1W–52W True Delivery OBV Screen on GitHub Runners...")
    
    # Load fundamentals
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            fundamentals = {}

    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json"]]
    print(f"Analyzing {len(stock_files)} stocks locally on runner...")

    results = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "")
        fund = fundamentals.get(sym, {})
        mcap = fund.get("market_cap_cr", None)
        pe = fund.get("pe", None)

        if mcap is not None and mcap < MIN_MARKET_CAP_CR:
            continue
        if pe is not None and (pe <= 0 or pe > MAX_PE_RATIO):
            continue

        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if not raw_data or len(raw_data) < 10:
            continue

        # Compute OBV in memory
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

        # 9D Vol check
        sma_9 = np.mean(vols[-9:]) if len(vols) >= 9 else np.mean(vols)
        if sma_9 < MIN_9D_AVG_VOLUME:
            continue

        curr_c = closes[-1]
        curr_obv = obvs[-1]
        matches = []

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
                "Market Cap (Cr)": f"₹{mcap:,.0f}" if mcap else "N/A",
                "P/E": round(pe, 1) if pe else "N/A",
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

    print(f"🎉 Screener complete! Found {len(results)} accumulation candidates. Saved to {out_path}")

if __name__ == "__main__":
    run_cloud_screener()
