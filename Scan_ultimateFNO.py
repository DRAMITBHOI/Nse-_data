import os
import sys
import json
import glob
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURATION & REPOSITORY PATHS
# ---------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(DATA_DIR, "fno_champion_scan_results.json")
ALT_RESULTS_JSON = "fno_champion_scan_results.json"
MIN_TURNOVER_CR = 10.0

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_ULTIMATE", os.environ.get("TELEGRAM_BOT_TOKEN", "9160227761:AAE2zmSpvqttJl3_KKl3jjyKkMg-CopgtSY")).strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_ULTIMATE", os.environ.get("TELEGRAM_CHAT_ID", "7860290981")).strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    for sp in ["local_secrets.json", "../local_secrets.json", os.path.join(DATA_DIR, "local_secrets.json")]:
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as fp:
                    cfg = json.load(fp)
                    TELEGRAM_BOT_TOKEN = cfg.get("TELEGRAM_BOT_TOKEN_ULTIMATE", cfg.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN))
                    TELEGRAM_CHAT_ID = cfg.get("TELEGRAM_CHAT_ID_ULTIMATE", cfg.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID))
            except Exception:
                pass

EXCLUDED_KEYWORDS = ["BEES", "ETF", "LIQUID", "GOLD", "SILVER", "NIFTY", "SENSEX", "AXIS", "CASE", "GRW"]

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[SIMULATION ALERT]:\n{message}\n")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram dispatch failed: {e}")

# ---------------------------------------------------------
# STRICT NSE F&O UNIVERSE WHITELIST
# ---------------------------------------------------------
def get_fno_whitelist():
    """Fetches the official active NSE F&O equities list.

    Falls back to the complete verified master universe if offline.
    """
    # 1. Try online fetch from official NSE contract files
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            symbols = set()
            for row in lines[5:]:
                parts = row.split(",")
                if len(parts) >= 2:
                    sym = parts[1].strip().upper()
                    if sym and not any(k in sym for k in EXCLUDED_KEYWORDS):
                        symbols.add(sym)
            if len(symbols) >= 150:
                return symbols
    except Exception:
        pass

    # 2. Complete Static NSE F&O Master Universe (2026 Derivative Basket)
    return {
        "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", 
        "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANANDRATHI", "ANGELONE", "APLAPOLLO", 
        "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATHERENERG", "AUBANK", 
        "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", 
        "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL", "BHARATFORG", "BHARTIARTL", 
        "BHEL", "BIOCON", "BLUESTARCO", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE", "CAMS", 
        "CANBK", "CANFINHOME", "CDSL", "CGPOWER", "CHOLAFIN", "CIPLA", "COALINDIA", 
        "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "DABUR", 
        "DALBHARAT", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY", "EICHERMOT", 
        "ENRIN", "ESCORTS", "ETERNAL", "EXIDEIND", "FEDERALBNK", "FORCEMOT", "FORTIS", 
        "GAIL", "GLENMARK", "GMRAIRPORT", "GODFRYPHLP", "GODREJCP", "GODREJPROP", "GRANULES", 
        "GRASIM", "GUJGASLTD", "GVT&D", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", 
        "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HUDCO", 
        "HYUNDAI", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", 
        "IGL", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "INOXWIND", 
        "IOC", "IPCALAB", "IREDA", "IRFC", "IRCTC", "ITC", "JINDALSTEL", "JIOFIN", "JSWENERGY", 
        "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI", "KFINTECH", "KOTAKBANK", 
        "KPITTECH", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LICI", "LODHA", "LT", "LTF", 
        "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MAXHEALTH", 
        "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", 
        "NATIONALUM", "NAUKRI", "NBCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NUVAMA", 
        "NYKAA", "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PERSISTENT", 
        "PETRONET", "PFC", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "POLICYBZR", 
        "POLYCAB", "POONAWALLA", "POWERGRID", "PRESTIGE", "PVRINOX", "RAMCOCEM", "RBLBANK", 
        "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", 
        "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA", "SUNTV", "SUPREMEIND", "SUZLON", 
        "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", 
        "TATATECH", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", 
        "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UPL", "VBL", "VEDL", "VOLTAS", "WIPRO", 
        "YESBANK", "ZYDUSLIFE"
    }

# ---------------------------------------------------------
# DATA NORMALIZATION & STOCK SPLIT HANDLING
# ---------------------------------------------------------
def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict): continue
        t = str(r.get("time", ""))[:10]
        c = float(r.get("close", 0) or 0)
        if not t or c <= 0: continue
        v = float(r.get("volume", 0) or 0)
        dv = float(r.get("delivery_vol", 0) or 0)
        dp = float(r.get("deliv_pct", 0) or 0)
        if dp > 0 and dv == 0: dv = v * (dp / 100.0)
        if v > 0 and dv > v: dv = v
        date_map[t] = {
            "time": t, "open": float(r.get("open", c) or c),
            "high": float(r.get("high", c) or c), "low": float(r.get("low", c) or c),
            "close": c, "volume": v, "delivery_vol": dv,
            "deliv_pct": (dv / v * 100.0) if v > 0 else 0.0
        }
    clean = [date_map[k] for k in sorted(date_map.keys())]
    
    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj = k; break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
            if adj:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj
                    clean[j]["volume"] = clean[j]["volume"] * adj
    return clean

def load_nifty_benchmark(all_files):
    for p in all_files:
        bname = os.path.basename(p).lower()
        if bname in ["nifty.json", "nifty50.json"]:
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    clean = clean_and_prepare(json.load(fp))
                if len(clean) > 40:
                    df_bm = pd.DataFrame(clean)
                    df_bm["p60"] = df_bm["close"].pct_change(60).fillna(0)
                    return {str(r["time"]): float(r["p60"]) for _, r in df_bm.iterrows()}
            except Exception:
                pass
    return {}

# ---------------------------------------------------------
# PRODUCTION SCREENER CORE
# ---------------------------------------------------------
def run_scanner():
    fno_universe = get_fno_whitelist()
    all_json = glob.glob("**/*.json", recursive=True)
    nifty_map = load_nifty_benchmark(all_json)

    stock_map = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if any(k in bname.lower() for k in ["nifty", "audit", "result", "trade", "secret", "package", "checkpoint", "live"]):
            continue
        if any(k in sym for k in EXCLUDED_KEYWORDS):
            continue
        # STRICT WHITELIST GATE: Only allow genuine F&O symbols
        if sym not in fno_universe:
            continue

        if sym not in stock_map or os.path.getsize(p) > os.path.getsize(stock_map[sym]):
            stock_map[sym] = p

    fresh_triggers = []
    pre_breakout_coils = []

    print(f"🔍 Scan_ultimateFNO: Processing {len(stock_map)} Verified F&O Stocks...")

    for sym, filepath in sorted(stock_map.items()):
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue

        if len(clean) < 70: continue

        df = pd.DataFrame(clean)
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(200, min_periods=40).mean()
        df["gross_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
        df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
        df["perf_60d"] = df["close"].pct_change(60).fillna(0)

        # Demat OBV
        diff = df["close"].diff()
        direction = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
        df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()
        df["range_closeness"] = np.where(df["high"] > df["low"], (df["close"] - df["low"]) / (df["high"] - df["low"]), 0.5)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        delivs = df["delivery_vol"].values
        deliv_sma20s = df["deliv_sma20"].values
        demat_obvs = df["deliv_obv"].values
        dobv_maxes = df["dobv_20max"].values
        rcs = df["range_closeness"].values
        perf60s = df["perf_60d"].values
        to50s = df["turnover_50d"].values
        sma200s = df["sma_200"].values
        times = df["time"].values

        i = len(df) - 1
        t0 = times[i]
        c0, h0, l0 = closes[i], highs[i], lows[i]

        # 1. Turnover & Trend Filters
        if to50s[i] < MIN_TURNOVER_CR: continue
        is_stage2 = bool(c0 >= sma200s[i]) if pd.notnull(sma200s[i]) else True
        n_perf = nifty_map.get(t0, 0.001)
        is_rs = bool(perf60s[i] >= n_perf)
        if not (is_stage2 and is_rs): continue

        # 2. Base Shelf Search (Tightness <= 8.5%)
        shelf_found = False
        lp_h, lp_l, best_len, best_spread = 0.0, 0.0, 15, 999.0
        for s_len in range(10, 23):
            s_h = highs[max(0, i - s_len):i].max()
            s_l = lows[max(0, i - s_len):i].min()
            if s_l > 0:
                spread = ((s_h - s_l) / s_l) * 100.0
                if spread <= 8.5 and spread < best_spread:
                    best_spread = spread
                    lp_h = s_h
                    lp_l = s_l
                    best_len = s_len
                    shelf_found = True

        if not shelf_found: continue

        # 3. Institutional Accumulation Gate (>= 1.50)
        base_up = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] >= closes[k - 1])
        base_dn = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] < closes[k - 1])
        acc_ratio = round((base_up / base_dn), 2) if base_dn > 0 else 2.50
        if acc_ratio < 1.50: continue

        dv_surge = round(delivs[i] / deliv_sma20s[i - 1], 2) if deliv_sma20s[i - 1] > 0 else 1.0
        rc = round(rcs[i], 2)
        obv_ok = bool(demat_obvs[i] >= dobv_maxes[i - 1] * 0.98)
        cleared_pivot = bool(c0 >= lp_h)

        # Trigger Category 1: Active Breakout Confirmation
        if cleared_pivot and (dv_surge >= 1.60) and (rc >= 0.65) and obv_ok:
            stop_loss = round(min(l0, lp_l), 2)
            risk_pct = round(((c0 - stop_loss) / c0) * 100.0, 2)
            ext_pct = round(((c0 - lp_h) / lp_h) * 100.0, 2)
            fresh_triggers.append({
                "symbol": sym,
                "pivot_ceiling": f"₹{lp_h:.2f}",
                "close": f"₹{c0:.2f}",
                "extended_%": f"+{ext_pct:.2f}%",
                "stop_loss": f"₹{stop_loss:.2f}",
                "risk_%": f"{risk_pct:.2f}%",
                "accumulation_ratio": acc_ratio,
                "delivery_surge": f"{dv_surge:.2f}x",
                "range_closeness": rc,
                "shelf_spread_%": f"{best_spread:.1f}%",
                "action": "BUY (EXIT 2.2 TRAIL)"
            })
        # Trigger Category 2: Institutional Pre-Breakout Coil Watchlist
        elif not cleared_pivot and (c0 >= lp_h * 0.965):
            dist_to_p = round(((lp_h - c0) / c0) * 100.0, 2)
            pre_breakout_coils.append({
                "symbol": sym,
                "close": f"₹{c0:.2f}",
                "pivot_target": f"₹{lp_h:.2f}",
                "dist_to_pivot": f"{dist_to_p:.2f}%",
                "accumulation_ratio": acc_ratio,
                "delivery_surge_today": f"{dv_surge:.2f}x",
                "shelf_spread_%": f"{best_spread:.1f}%",
                "action": "COILING WATCHLIST"
            })

    fresh_triggers.sort(key=lambda x: float(x["delivery_surge"].replace("x","")), reverse=True)
    pre_breakout_coils.sort(key=lambda x: x["accumulation_ratio"], reverse=True)

    scan_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    payload = {
        "scan_time": scan_time_str,
        "fresh_triggers": fresh_triggers,
        "coiling_watchlist": pre_breakout_coils
    }

    # Write both to data/ and root to guarantee presence across all scripts
    for out_path in [RESULTS_JSON, ALT_RESULTS_JSON]:
        try:
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
        except Exception:
            pass

    print("\n" + "=" * 90)
    print(f"🎯 1. CONFIRMED F&O BREAKOUTS ({len(fresh_triggers)}):")
    if fresh_triggers:
        print(pd.DataFrame(fresh_triggers).to_string(index=False))
        tg_msg = f"🚀 <b>ULTIMATE F&O EOD CONFIRMATION</b>\n🕒 <i>{scan_time_str}</i>\n\n"
        for t in fresh_triggers:
            tg_msg += (
                f"🔥 <b>#{t['symbol']}</b> (Clearance: {t['extended_%']})\n"
                f"• Close: <b>{t['close']}</b> (Pivot: {t['pivot_ceiling']})\n"
                f"• Stop Loss: <b>{t['stop_loss']}</b> (Risk: {t['risk_%']})\n"
                f"• Deliv Surge: <b>{t['delivery_surge']}</b> | Acc: <b>{t['accumulation_ratio']}x</b>\n"
                f"• Shelf Spread: {t['shelf_spread_%']} | Exit: <b>Exit 2.2 Trail</b>\n\n"
            )
        send_telegram_alert(tg_msg)
    else:
        print("No F&O breakouts confirmed today.")

    print("\n" + "=" * 90)
    print(f"👀 2. TOP COILING PRE-BREAKOUT WATCHLIST ({len(pre_breakout_coils)}):")
    if pre_breakout_coils:
        print(pd.DataFrame(pre_breakout_coils[:15]).to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    run_scanner()
