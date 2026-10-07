import os
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd

DATA_DIR = "data"
LEVELS_JSON = os.path.join(DATA_DIR, "all_stock_breakout_levels.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "live_intraday_breakouts.json")

os.makedirs(DATA_DIR, exist_ok=True)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def get_nse_session():
    cookie_req = urllib.request.Request("https://www.nseindia.com", headers=NSE_HEADERS)
    try:
        resp = urllib.request.urlopen(cookie_req, timeout=10)
        return resp.headers.get("Set-Cookie", "")
    except Exception:
        return ""


def fetch_live_quote(symbol, cookies):
    sym_encoded = urllib.parse.quote(symbol.strip().upper())
    url = f"https://www.nseindia.com/api/quote-equity?symbol={sym_encoded}"
    headers = dict(NSE_HEADERS)
    if cookies:
        headers["Cookie"] = cookies

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            price_info = data.get("priceInfo", {})
            pre_open = data.get("preOpenMarket", {})
            sec_details = data.get("securityDetails", {})

            ltp = float(price_info.get("lastPrice", 0.0) or 0.0)
            day_high = float(price_info.get("intraDayHighLow", {}).get("max", 0.0) or 0.0)

            traded_vol = float(pre_open.get("totalTradedVolume", 0.0) or 0.0)
            if traded_vol == 0:
                traded_vol = float(sec_details.get("volumeTraded", 0.0) or 0.0)

            return {"ltp": ltp, "high": day_high, "traded_vol": traded_vol}
    except Exception:
        return None


def calculate_market_minutes():
    now = datetime.datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < market_open:
        return 1.0
    if now > market_close:
        return 375.0
    return max(1.0, (now - market_open).total_seconds() / 60.0)


def execute_intraday_scan():
    if not os.path.exists(LEVELS_JSON):
        print(f"⚠️ {LEVELS_JSON} not found. Running level generation fallback...")
        try:
            import generate_all_breakout_levels
            generate_all_breakout_levels.generate_all_breakout_levels()
        except Exception as e:
            print(f"Fallback generation error: {e}")

    targets = {}
    if os.path.exists(LEVELS_JSON):
        try:
            with open(LEVELS_JSON, "r", encoding="utf-8") as fp:
                targets = json.load(fp).get("Targets", {})
        except Exception:
            pass

    elapsed_mins = calculate_market_minutes()
    pace_factor = 375.0 / elapsed_mins

    print(f"📡 Scanning {len(targets)} stocks across NSE (Elapsed: {int(elapsed_mins)}m / 375m)...")
    cookies = get_nse_session()

    confirmed_institutional = []
    light_vol_crosses = []
    approaching = []

    for sym, info in targets.items():
        quote = fetch_live_quote(sym, cookies)
        time.sleep(0.08)

        if not quote or quote["ltp"] <= 0:
            continue

        ltp = quote["ltp"]
        day_high = quote["high"]
        traded_vol = quote["traded_vol"]
        pivot = info["pivot_ceiling"]
        sl = info["stop_loss"]
        v_sma = info.get("vol_sma20", 50000.0)
        setup = info["setup"]
        univ = info["universe"]

        projected_day_vol = traded_vol * pace_factor
        vol_surge_mult = round(projected_day_vol / v_sma, 2) if v_sma > 0 else 1.0

        is_crossing = (ltp >= pivot) or (day_high >= pivot)
        diff_pct = round(((ltp - pivot) / pivot) * 100.0, 2)

        row = {
            "symbol": sym,
            "universe": univ,
            "setup": setup,
            "pivot_ceiling": pivot,
            "ltp": ltp,
            "day_high": day_high,
            "stop_loss": sl,
            "risk_%": round(((pivot - sl) / pivot) * 100, 2),
            "dist_pivot_%": diff_pct,
            "proj_vol_pace": vol_surge_mult,
            "action": "INSTITUTIONAL BREAKOUT" if vol_surge_mult >= 1.40 else "LIGHT-VOL CROSS",
        }

        if is_crossing and vol_surge_mult >= 1.40:
            confirmed_institutional.append(row)
        elif is_crossing:
            light_vol_crosses.append(row)
        elif -2.5 <= diff_pct < 0:
            row["action"] = f"COILING ({abs(diff_pct):.1f}% below)"
            approaching.append(row)

    confirmed_institutional.sort(key=lambda x: x["proj_vol_pace"], reverse=True)
    light_vol_crosses.sort(key=lambda x: x["dist_pivot_%"], reverse=True)
    approaching.sort(key=lambda x: x["dist_pivot_%"], reverse=True)

    payload = {
        "Scan_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Elapsed_Minutes": int(elapsed_mins),
        "Institutional_Breakouts_Count": len(confirmed_institutional),
        "Light_Vol_Crosses_Count": len(light_vol_crosses),
        "Coiling_Approaching_Count": len(approaching),
        "Institutional_Breakouts": confirmed_institutional,
        "Light_Vol_Crosses": light_vol_crosses,
        "Coiling_Approaching": approaching[:25],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Saved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    execute_intraday_scan()
