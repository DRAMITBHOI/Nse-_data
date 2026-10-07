import os, json, glob, time, datetime, urllib.request, urllib.parse
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_JSON = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0
os.makedirs(DATA_DIR, exist_ok=True)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/"
}

def clean_data(raw):
    if not raw or not isinstance(raw, list): return []
    dmap = {}
    for r in raw:
        if not isinstance(r, dict): continue
        t = str(r.get("time", ""))[:10]
        c = float(r.get("close", 0) or 0)
        if not t or c <= 0: continue
        dmap[t] = {
            "time": t, "open": float(r.get("open", c) or c),
            "high": float(r.get("high", c) or c), "low": float(r.get("low", c) or c),
            "close": c, "volume": float(r.get("volume", 0) or 0)
        }
    clean = [dmap[k] for k in sorted(dmap.keys())]
    for i in range(len(clean)-1, 0, -1):
        p_c, c_o = clean[i-1]["close"], clean[i]["open"]
        if p_c > 0 and c_o > 0 and (p_c / c_o) >= 1.35:
            ratio = p_c / c_o
            k = 2.0 if 1.7 <= ratio <= 2.3 else (5.0 if 4.3 <= ratio <= 5.5 else (10.0 if 8.5 <= ratio <= 11.5 else None))
            if k:
                for j in range(i):
                    for f in ["open", "high", "low", "close"]: clean[j][f] = round(clean[j][f] / k, 2)
                    clean[j]["volume"] = clean[j]["volume"] * k
    return clean

def load_benchmarks():
    n_perf = {}
    for f in ["nifty.json", "nifty50.json", "NIFTY.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    c = clean_data(json.load(fp))
                if len(c) > 50:
                    df = pd.DataFrame(c)
                    df["p60"] = df["close"].pct_change(60).fillna(0)
                    n_perf = {r["time"]: float(r["p60"]) for _, r in df.iterrows()}
                    break
            except Exception: pass
    n750 = set()
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            n750 = {str(x).strip().upper() for x in (d if isinstance(d, list) else d.keys())}
        except Exception: pass
    return n_perf, n750

def extract_target(clean, n_perf, is_n750):
    if len(clean) < 60: return None
    df = pd.DataFrame(clean)
    df["vol_sma"] = df["volume"].rolling(20, min_periods=5).mean()
    df["turnover"] = (df["close"] * df["volume"]) / 1e7
    df["to_50d"] = df["turnover"].rolling(50, min_periods=10).mean().fillna(0)
    df["sma200"] = df["close"].rolling(200, min_periods=50).mean()
    df["p60"] = df["close"].pct_change(60).fillna(0)
    
    last = len(df) - 1
    c = df["close"].values[last]
    if df["to_50d"].values[last] < MIN_TURNOVER_CR: return None
    if pd.notnull(df["sma200"].values[last]) and c < df["sma200"].values[last]: return None
    
    t = df["time"].values[last]
    if df["p60"].values[last] <= n_perf.get(t, 0.001): return None

    highs, lows = df["high"].values, df["low"].values
    v_sma = float(df["vol_sma"].values[last] or 50000.0)

    # 1. V-Reversal Level (Flush >= 18%)
    r20_h = highs[last-20:last+1].max()
    if (r20_h - lows[last]) / r20_h >= 0.18:
        p_ceil = round(float(highs[last-3:last+1].max()), 2)
        sl = round(float(lows[last-4:last+1].min() * 0.995), 2)
        if -2.0 <= ((p_ceil - c) / c) * 100 <= 6.5:
            return {"setup": "V-REVERSAL", "universe": "NIFTY 750" if is_n750 else "NON-N750",
                    "pivot": p_ceil, "sl": sl, "v_sma": v_sma}

    # 2. Launchpad Level (Spread <= 8% / 13%)
    max_sp = 0.08 if is_n750 else 0.13
    for s_len in range(10, 19):
        sh, slow = highs[last-s_len:last+1].max(), lows[last-s_len:last+1].min()
        if slow > 0 and ((sh - slow) / slow) <= max_sp:
            p_ceil, sl = round(float(sh), 2), round(float(slow), 2)
            if -2.0 <= ((p_ceil - c) / c) * 100 <= 6.5:
                return {"setup": "LAUNCHPAD", "universe": "NIFTY 750" if is_n750 else "NON-N750",
                        "pivot": p_ceil, "sl": sl, "v_sma": v_sma}
            break
    return None

def get_session():
    try:
        r = urllib.request.urlopen(urllib.request.Request("https://www.nseindia.com", headers=NSE_HEADERS), timeout=8)
        return r.headers.get("Set-Cookie", "")
    except Exception: return ""

def get_quote(symbol, cookies):
    sym = urllib.parse.quote(symbol.strip().upper())
    h = dict(NSE_HEADERS)
    if cookies: h["Cookie"] = cookies
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://www.nseindia.com/api/quote-equity?symbol={sym}", headers=h), timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8", errors="ignore"))
            p = d.get("priceInfo", {})
            v = float(d.get("preOpenMarket", {}).get("totalTradedVolume", 0) or d.get("securityDetails", {}).get("volumeTraded", 0) or 0)
            return {"ltp": float(p.get("lastPrice", 0) or 0), "high": float(p.get("intraDayHighLow", {}).get("max", 0) or 0), "vol": v}
    except Exception: return None

def run():
    n_perf, n750 = load_benchmarks()
    files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.json")) if not any(x in f for x in ["nifty", "fundamentals", "backtest", "swing3", "breakout"])]
    print(f"Filtering {len(files)} stocks for pivot targets...")
    
    targets = {}
    for p in files:
        sym = os.path.splitext(os.path.basename(p))[0].upper()
        try:
            with open(p, "r", encoding="utf-8") as fp: raw = json.load(fp)
            c = clean_data(raw)
            if c:
                t = extract_target(c, n_perf, sym in n750)
                if t: targets[sym] = t
        except Exception: pass

    now = datetime.datetime.now()
    m_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    mins = 375.0 if now > now.replace(hour=15, minute=30, second=0, microsecond=0) else max(1.0, (now - m_open).total_seconds() / 60.0)
    pace_factor = 375.0 / mins

    print(f"Polling {len(targets)} targets (Elapsed: {int(mins)}m)...")
    cookies = get_session()
    inst_bo, light_bo, coils = [], [], []

    for sym, info in targets.items():
        q = get_quote(sym, cookies)
        time.sleep(0.08)
        if not q or q["ltp"] <= 0: continue

        ltp, high, vol = q["ltp"], q["high"], q["vol"]
        pivot, sl, vsma = info["pivot"], info["sl"], info["v_sma"]
        surge = round((vol * pace_factor) / vsma, 2) if vsma > 0 else 1.0
        diff = round(((ltp - pivot) / pivot) * 100.0, 2)
        is_bo = (ltp >= pivot) or (high >= pivot)

        row = {
            "symbol": sym, "universe": info["universe"], "setup": info["setup"],
            "pivot_ceiling": pivot, "ltp": ltp, "day_high": high, "stop_loss": sl,
            "risk_%": round(((pivot - sl) / pivot) * 100, 2), "dist_pivot_%": diff,
            "proj_vol_pace": surge, "action": "INSTITUTIONAL BREAKOUT" if surge >= 1.4 else "LIGHT-VOL CROSS"
        }

        if is_bo and surge >= 1.4: inst_bo.append(row)
        elif is_bo: light_bo.append(row)
        elif -2.5 <= diff < 0:
            row["action"] = f"COILING ({abs(diff):.1f}% below)"
            coils.append(row)

    inst_bo.sort(key=lambda x: x["proj_vol_pace"], reverse=True)
    light_bo.sort(key=lambda x: x["dist_pivot_%"], reverse=True)
    coils.sort(key=lambda x: x["dist_pivot_%"], reverse=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump({
            "Scan_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "Elapsed_Minutes": int(mins),
            "Institutional_Breakouts": inst_bo,
            "Light_Vol_Crosses": light_bo,
            "Coiling_Approaching": coils[:25]
        }, fp, indent=2)
    print(f"Saved: {len(inst_bo)} Institutional Breakouts, {len(light_bo)} Light Crosses.")

if __name__ == "__main__":
    run()
