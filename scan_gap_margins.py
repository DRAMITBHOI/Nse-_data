import os
import json
import time

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gap_margin_candidates.json")

MIN_GAP_PCT = 2.0        # Gap magnitude >= 2.0%
MARGIN_PROXIMITY = 4.0   # Current price within <= 4.0% of the active margin
LOOKBACK_DAYS = 30       # Active gap lookback window

# OFFICIAL NSE F&O UNDERLYING STOCK UNIVERSE
ACTIVE_FNO_SYMBOLS = {
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA",
    "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON",
    "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL",
    "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB",
    "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL",
    "INDHOTEL", "INDIACEM", "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER",
    "INFY", "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL",
    "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT", "LTIM",
    "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MCX",
    "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY",
    "OFSS", "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND",
    "PIIND", "PNB", "POLYCAB", "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE",
    "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS",
    "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS",
    "WIPRO", "ZYDUSLIFE"
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

def scan_gap_stocks():
    print(f"🚀 Scanning {len(ACTIVE_FNO_SYMBOLS)} F&O stocks for >= 2% Gaps within 4% margin proximity...")
    
    candidates = []
    scanned_count = 0

    for sym in sorted(ACTIVE_FNO_SYMBOLS):
        f_name = f"{sym}.json"
        json_path = os.path.join(DATA_DIR, f_name)
        if not os.path.exists(json_path):
            continue

        scanned_count += 1
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

        # Scan for gaps in the recent LOOKBACK_DAYS
        start_idx = max(1, N - LOOKBACK_DAYS)
        for i in range(start_idx, N):
            prior_high = highs[i - 1]
            gap_day_low = lows[i]

            # 1. Bullish Void: Day's low > Prior day's high
            if gap_day_low > prior_high:
                gap_size_pct = round(((gap_day_low - prior_high) / prior_high) * 100.0, 2)
                
                if gap_size_pct >= MIN_GAP_PCT:
                    gap_upper = gap_day_low
                    gap_lower = prior_high
                    gap_date = times[i]

                    # Filter out gaps that collapsed below lower boundary by > 2.0%
                    if i < N - 1:
                        post_min_low = min(lows[i + 1 :])
                        if post_min_low < (gap_lower * 0.980):
                            continue

                    dist_to_upper_pct = round(abs(curr_price - gap_upper) / gap_upper * 100.0, 2)
                    dist_to_lower_pct = round(abs(curr_price - gap_lower) / gap_lower * 100.0, 2)

                    # Condition A: Bullish Move after gap -> test re-entry near UPPER margin (<= 4%)
                    if curr_price >= gap_upper and dist_to_upper_pct <= MARGIN_PROXIMITY:
                        candidates.append({
                            "Symbol": sym,
                            "Setup": "🟢 Bullish Extension (Upper Margin Test)",
                            "LTP": round(curr_price, 2),
                            "Target Margin": round(gap_upper, 2),
                            "Gap Upper": round(gap_upper, 2),
                            "Gap Lower": round(gap_lower, 2),
                            "Gap Size %": f"+{gap_size_pct}%",
                            "Gap Created": gap_date,
                            "Margin Distance %": f"{dist_to_upper_pct}%",
                            "Days Since Gap": N - 1 - i
                        })
                        break

                    # Condition B: Bearish Retrace after gap -> test re-entry near LOWER margin (<= 4%)
                    elif curr_price < gap_upper and curr_price >= (gap_lower * 0.96) and dist_to_lower_pct <= MARGIN_PROXIMITY:
                        candidates.append({
                            "Symbol": sym,
                            "Setup": "🟡 Bearish Retrace (Lower Margin Defense)",
                            "LTP": round(curr_price, 2),
                            "Target Margin": round(gap_lower, 2),
                            "Gap Upper": round(gap_upper, 2),
                            "Gap Lower": round(gap_lower, 2),
                            "Gap Size %": f"+{gap_size_pct}%",
                            "Gap Created": gap_date,
                            "Margin Distance %": f"{dist_to_lower_pct}%",
                            "Days Since Gap": N - 1 - i
                        })
                        break

    # Sort closest to the margin first
    candidates.sort(key=lambda x: float(x["Margin Distance %"].replace("%", "")))

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Universe": f"Official NSE F&O ({scanned_count} stocks scanned)",
        "Total Candidates": len(candidates),
        "Candidates": candidates
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎯 Complete: Found {len(candidates)} F&O setups near gap margins. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    scan_gap_stocks()
