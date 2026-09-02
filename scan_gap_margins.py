import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gap_margin_candidates.json")

MIN_GAP_PCT = 4.0        # Minimum gap magnitude: 4%
MARGIN_PROXIMITY = 1.0   # Current price within 1.0% of gap upper/lower margin
LOOKBACK_DAYS = 20       # Look for gaps created within the last 20 sessions

def clean_data_fast(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = r.get("time", "")
        if not raw_t:
            continue
        d_str = str(raw_t)[:10]
        c = float(r.get("close", 0) or 0)
        if c <= 0:
            continue
        v = float(r.get("volume", 0) or 0)
        o = float(r.get("open", c) or c)
        h = float(r.get("high", c) or c)
        l = float(r.get("low", c) or c)
        
        if d_str not in date_map or v > date_map[d_str]["volume"]:
            date_map[d_str] = {
                "time": d_str,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v
            }
    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 30:
        return []
    return [date_map[k] for k in sorted_dates]

def scan_gap_stocks():
    print("🚀 Scanning for Gap Margin (TMPV) Retest Setups...")
    
    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "gap_margin_candidates.json"
        ]
    ]

    candidates = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as fp:
                raw = json.load(fp)
        except Exception:
            continue

        clean = clean_data_fast(raw)
        if len(clean) < 30:
            continue

        closes = [r["close"] for r in clean]
        highs = [r["high"] for r in clean]
        lows = [r["low"] for r in clean]
        opens = [r["open"] for r in clean]
        times = [r["time"] for r in clean]
        N = len(closes)
        curr_price = closes[-1]

        # Scan for bullish gaps created in the last LOOKBACK_DAYS
        start_idx = max(1, N - LOOKBACK_DAYS)
        for i in range(start_idx, N):
            prior_high = highs[i - 1]
            gap_day_low = lows[i]

            # True Bullish Gap: gap_day_low > prior_high
            if gap_day_low > prior_high:
                gap_size_pct = round(((gap_day_low - prior_high) / prior_high) * 100.0, 2)
                
                if gap_size_pct >= MIN_GAP_PCT:
                    gap_upper = gap_day_low
                    gap_lower = prior_high
                    gap_date = times[i]

                    # Check if the gap was completely invalidated/violated in subsequent days
                    if i < N - 1:
                        post_min_low = min(lows[i + 1 :])
                        if post_min_low < (gap_lower * 0.985):
                            # Gap was decisively broken; skip
                            continue

                    # Proximity calculations
                    dist_to_upper_pct = round(abs(curr_price - gap_upper) / gap_upper * 100.0, 2)
                    dist_to_lower_pct = round(abs(curr_price - gap_lower) / gap_lower * 100.0, 2)

                    action_zone = None
                    proximity_val = None

                    # 1. Sitting right on the Gap Upper Margin (Support Bounce test)
                    if dist_to_upper_pct <= MARGIN_PROXIMITY:
                        action_zone = "🟢 Retesting Upper Gap Margin (Bounce Zone)"
                        proximity_val = dist_to_upper_pct

                    # 2. Sitting right at the Gap Lower Margin (Fill Completion / Deep Test)
                    elif dist_to_lower_pct <= MARGIN_PROXIMITY:
                        action_zone = "🟡 Retesting Lower Gap Margin (Base Defense Zone)"
                        proximity_val = dist_to_lower_pct

                    if action_zone:
                        candidates.append({
                            "Symbol": sym,
                            "Setup": action_zone,
                            "LTP": round(curr_price, 2),
                            "Gap Upper": round(gap_upper, 2),
                            "Gap Lower": round(gap_lower, 2),
                            "Gap Size %": f"+{gap_size_pct}%",
                            "Gap Created": gap_date,
                            "Margin Distance %": f"{proximity_val}%",
                            "Days Since Gap": N - 1 - i
                        })
                        # Found the most recent active gap for this stock
                        break

    # Sort by nearest to margin
    candidates.sort(key=lambda x: float(x["Margin Distance %"].replace("%", "")))

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Total Candidates": len(candidates),
        "Candidates": candidates
    }

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎯 Gap Margin Scan complete: Found {len(candidates)} qualifying stocks. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    scan_gap_stocks()
