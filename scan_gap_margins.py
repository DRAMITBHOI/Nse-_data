import json
import os
import time

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gap_margin_candidates.json")

MIN_GAP_PCT = 2.0  # Initial Close-to-Open gap >= 2.0%
MIN_UNFILLED_PCT = 1.5  # Remaining unfilled gap buffer >= 1.5%
MARGIN_PROXIMITY = 4.0  # LTP within 4.0% of the gap target
LOOKBACK_DAYS = 30  # Search window

ACTIVE_FNO_SYMBOLS = {
    "AARTIIND",
    "ABB",
    "ABBOTINDIA",
    "ABCAPITAL",
    "ABFRL",
    "ACC",
    "ADANIENT",
    "ADANIPORTS",
    "ALKEM",
    "AMBUJACEM",
    "APOLLOHOSP",
    "APOLLOTYRE",
    "ASHOKLEY",
    "ASIANPAINT",
    "ASTRAL",
    "ATUL",
    "AUBANK",
    "AUROPHARMA",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BALRAMCHIN",
    "BANDHANBNK",
    "BANKBARODA",
    "BATAINDIA",
    "BEL",
    "BERGEPAINT",
    "BHARATFORG",
    "BHARTIARTL",
    "BHEL",
    "BIOCON",
    "BOSCHLTD",
    "BPCL",
    "BRITANNIA",
    "BSOFT",
    "CANBK",
    "CANFINHOME",
    "CHAMBLFERT",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "COROMANDEL",
    "CROMPTON",
    "CUB",
    "CUMMINSIND",
    "DABUR",
    "DALBHARAT",
    "DEEPAKNTR",
    "DIVISLAB",
    "DIXON",
    "DLF",
    "DRREDDY",
    "EICHERMOT",
    "ESCORTS",
    "EXIDEIND",
    "FEDERALBNK",
    "GAIL",
    "GLENMARK",
    "GMRINFRA",
    "GNFC",
    "GODREJCP",
    "GODREJPROP",
    "GRANULES",
    "GRASIM",
    "GUJGASLTD",
    "HAL",
    "HAVELLS",
    "HCLTECH",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDCOPPER",
    "HINDPETRO",
    "HINDUNILVR",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "IDEA",
    "IDFCFIRSTB",
    "IEX",
    "IGL",
    "INDHOTEL",
    "INDIACEM",
    "INDIAMART",
    "INDIGO",
    "INDUSINDBK",
    "INDUSTOWER",
    "INFY",
    "IOC",
    "IPCALAB",
    "IRCTC",
    "ITC",
    "JINDALSTEL",
    "JKCEMENT",
    "JSWSTEEL",
    "JUBLFOOD",
    "KOTAKBANK",
    "LALPATHLAB",
    "LAURUSLABS",
    "LICHSGFIN",
    "LT",
    "LTIM",
    "LTTS",
    "LUPIN",
    "M&M",
    "M&MFIN",
    "MANAPPURAM",
    "MARICO",
    "MARUTI",
    "MCX",
    "METROPOLIS",
    "MFSL",
    "MGL",
    "MOTHERSON",
    "MPHASIS",
    "MRF",
    "MUTHOOTFIN",
    "NATIONALUM",
    "NAUKRI",
    "NAVINFLUOR",
    "NESTLEIND",
    "NMDC",
    "NTPC",
    "OBEROIRLTY",
    "OFSS",
    "ONGC",
    "PAGEIND",
    "PEL",
    "PERSISTENT",
    "PETRONET",
    "PFC",
    "PIDILITIND",
    "PIIND",
    "PNB",
    "POLYCAB",
    "PVRINOX",
    "RAMCOCEM",
    "RBLBANK",
    "RECLTD",
    "RELIANCE",
    "SAIL",
    "SBICARD",
    "SBILIFE",
    "SBIN",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SRF",
    "SUNPHARMA",
    "SUNTV",
    "SYNGENE",
    "TATACHEM",
    "TATACOMM",
    "TATACONSUM",
    "TATAMOTORS",
    "TATAPOWER",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TORNTPHARM",
    "TORNTPOWER",
    "TRENT",
    "TVSMOTOR",
    "UBL",
    "ULTRACEMCO",
    "UPL",
    "VEDL",
    "VOLTAS",
    "WIPRO",
    "ZYDUSLIFE",
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
          "volume": v,
      }
  sorted_dates = sorted(date_map.keys())
  if len(sorted_dates) < 30:
    return []
  return [date_map[k] for k in sorted_dates]


def scan_gap_stocks():
  print(
      f"🚀 Scanning {len(ACTIVE_FNO_SYMBOLS)} F&O stocks (Gap >= {MIN_GAP_PCT}%"
      f" & Remaining Unfilled >= {MIN_UNFILLED_PCT}%)..."
  )

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

    opens = [r["open"] for r in clean]
    closes = [r["close"] for r in clean]
    lows = [r["low"] for r in clean]
    times = [r["time"] for r in clean]
    N = len(closes)
    curr_price = closes[-1]

    start_idx = max(1, N - LOOKBACK_DAYS)
    # Check most recent gaps first
    for i in range(N - 1, start_idx - 1, -1):
      prior_close = closes[i - 1]
      gap_open = opens[i]

      # Bullish gap open condition
      if gap_open <= prior_close:
        continue

      initial_gap_pct = round(
          ((gap_open - prior_close) / prior_close) * 100.0, 2
      )
      if initial_gap_pct < MIN_GAP_PCT:
        continue

      gap_lower = prior_close

      # Lowest price reached across all bars from gap creation up to current day
      lowest_post_gap = min(lows[i:])

      # 1. Full fill check: price reached or breached the fill line
      if lowest_post_gap <= gap_lower:
        continue

      # 2. Dynamic unfilled calculation:
      # If previous sessions dipped into the gap, the ceiling of the unfilled area drops to that lowest wick
      unfilled_upper = min(gap_open, lowest_post_gap)
      remaining_gap_pct = round(
          ((unfilled_upper - gap_lower) / gap_lower) * 100.0, 2
      )

      # Discard if partial fills reduced tradable gap to < 1.5% (e.g. PFC case)
      if remaining_gap_pct < MIN_UNFILLED_PCT:
        continue

      # 3. Proximity to gap target (prior_close)
      dist_to_target_pct = round(
          ((curr_price - gap_lower) / gap_lower) * 100.0, 2
      )

      if 0 <= dist_to_target_pct <= MARGIN_PROXIMITY:
        candidates.append({
            "Symbol": sym,
            "Setup": "🎯 Tradeable Gap Close",
            "LTP": round(curr_price, 2),
            "Target Margin": round(gap_lower, 2),  # Target fill price
            "Gap Upper": round(unfilled_upper, 2),  # Top of current open space
            "Gap Lower": round(gap_lower, 2),  # Gap origin/close
            "Gap Size %": f"+{remaining_gap_pct}% (Init: {initial_gap_pct}%)",
            "Gap Created": times[i],
            "Margin Distance %": f"{dist_to_target_pct}%",
            "Days Since Gap": N - 1 - i,
        })
        break

  candidates.sort(key=lambda x: float(x["Margin Distance %"].replace("%", "")))

  payload = {
      "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
      "Universe": f"Official NSE F&O ({scanned_count} stocks scanned)",
      "Total Candidates": len(candidates),
      "Candidates": candidates,
  }

  os.makedirs(DATA_DIR, exist_ok=True)
  with open(OUTPUT_FILE, "w") as fp:
    json.dump(payload, fp, indent=2)

  print(
      f"🎯 Found {len(candidates)} high-potential unfilled gap setups. Saved to"
      f" {OUTPUT_FILE}."
  )


if __name__ == "__main__":
  scan_gap_stocks()
