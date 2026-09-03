import os
import io
import json
import time
import urllib.request
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gap_margin_candidates.json")
FNO_STORE = os.path.join(DATA_DIR, "fno_history.json")

MIN_GAP_PCT = 2.0        # Updated threshold: >= 2.0% gap
MARGIN_PROXIMITY = 1.0   # Current price within <= 1.0% of gap boundary
LOOKBACK_DAYS = 20       # Active gap lookback window

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

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

def load_fno_symbols():
    """Load active F&O constituent symbols from local store or official NSE feed."""
    # 1. From local fno_history.json
    if os.path.exists(FNO_STORE):
        try:
            with open(FNO_STORE, "r") as fp:
                fno_data = json.load(fp)
                if fno_data:
                    latest_day = sorted(fno_data.keys())[-1]
                    syms = set(fno_data[latest_day].keys())
                    if len(syms) >= 50:
                        return syms
        except Exception:
            pass

    # 2. Direct fallback to official NSE F&O list
    try:
        url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
            df.columns = df.columns.str.strip().str.upper()
            if "SYMBOL" in df.columns:
                clean_syms = set(df["SYMBOL"].dropna().astype(str).str.strip().str.upper())
                clean_syms.discard("NIFTY")
                clean_syms.discard("BANKNIFTY")
                clean_syms.discard("FINNIFTY")
                clean_syms.discard("MIDCPNIFTY")
                if clean_syms:
                    return clean_syms
    except Exception:
        pass

    return set()

def scan_gap_stocks():
    print("🚀 Scanning F&O Universe for Gap Margin Retests (≥ 2.0% Gaps)...")
    
    fno_symbols = load_fno_symbols()
    if fno_symbols:
        print(f"🎯 Target Universe Locked: {len(fno_symbols)} active F&O securities.")
    else:
        print("ℹ️ Scanning full available universe in data directory.")

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "gap_margin_candidates.json", "integrated_institutional_report.json"
        ]
    ]

    candidates = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        
        # Enforce strict F&O universe inclusion if active list exists
        if fno_symbols and sym not in fno_symbols:
            continue

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
        times = [r["time"] for r in clean]
        N = len(closes)
        curr_price = closes[-1]

        # Scan for bullish gaps created within the last LOOKBACK_DAYS
        start_idx = max(1, N - LOOKBACK_DAYS)
        for i in range(start_idx, N):
            prior_high = highs[i - 1]
            gap_day_low = lows[i]

            # Bullish Gap: gap_day_low > prior_high
            if gap_day_low > prior_high:
                gap_size_pct = round(((gap_day_low - prior_high) / prior_high) * 100.0, 2)
                
                if gap_size_pct >= MIN_GAP_PCT:
                    gap_upper = gap_day_low
                    gap_lower = prior_high
                    gap_date = times[i]

                    # Filter out gaps that collapsed decisively beneath the lower boundary
                    if i < N - 1:
                        post_min_low = min(lows[i + 1 :])
                        if post_min_low < (gap_lower * 0.985):
                            continue

                    # Proximity measurements
                    dist_to_upper_pct = round(abs(curr_price - gap_upper) / gap_upper * 100.0, 2)
                    dist_to_lower_pct = round(abs(curr_price - gap_lower) / gap_lower * 100.0, 2)

                    action_zone = None
                    proximity_val = None

                    # 1. Upper Gap Margin (Top of Gap / Support Bounce Zone)
                    if dist_to_upper_pct <= MARGIN_PROXIMITY:
                        action_zone = "🟢 Upper Margin (Bounce Retest)"
                        proximity_val = dist_to_upper_pct

                    # 2. Lower Gap Margin (Bottom of Gap / Base Defense Zone)
                    elif dist_to_lower_pct <= MARGIN_PROXIMITY:
                        action_zone = "🟡 Lower Margin (Fill Base Defense)"
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
                        break

    # Sort by nearest proximity to the margin
    candidates.sort(key=lambda x: float(x["Margin Distance %"].replace("%", "")))

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Total Candidates": len(candidates),
        "Candidates": candidates
    }

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎯 Complete: Found {len(candidates)} F&O setups near ≥2.0% gap margins. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    scan_gap_stocks()
