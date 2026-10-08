import os
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd

# ==============================================================================
# 1. TIMEZONE & PATH CONFIGURATION
# ==============================================================================
# Force Indian Standard Time (IST = UTC + 5:30) on any cloud runner
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

DATA_DIR = "data"
OUTPUT_JSON = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
S3_RESULTS = os.path.join(DATA_DIR, "swing3_scanner_results.json")
R2_RESULTS = os.path.join(DATA_DIR, "swing3_run2_non_nifty750_scanner_results.json")

os.makedirs(DATA_DIR, exist_ok=True)

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
if TG_TOKEN.lower().startswith("bot"):
    TG_TOKEN = TG_TOKEN[3:]

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# ==============================================================================
# 2. TELEGRAM ALERT DISPATCHER
# ==============================================================================
def send_telegram_alert(breakouts, scan_time_str, elapsed_mins):
    if not TG_TOKEN or not TG_CHAT_ID or not breakouts:
        return

    lines = [
        "🚨 <b>NSE LIVE INTRADAY BREAKOUT ALERT</b>",
        f"🕒 <i>Time: {scan_time_str} (Session: {int(elapsed_mins)}m/375m)</i>",
        "────────────────────────",
    ]

    for b in breakouts:
        tag = "🔥 INSTITUTIONAL SURGE" if b["proj_vol_pace"] >= 1.20 else "⚡ PIVOT BREACH"
        lines.append(
            f"• <b>{b['symbol']}</b> ({b['universe']} | {b['setup']}) - <b>{tag}</b>\n"
            f"  LTP: <b>₹{b['ltp']:.2f}</b> (Pivot: ₹{b['pivot_ceiling']:.2f}, {b['dist_pivot_%']:+.2f}%)\n"
            f"  SL: ₹{b['stop_loss']:.2f} (Risk: {b['risk_%']:.1f}%)\n"
            f"  Vol Pace: <b>{b['proj_vol_pace']:.2f}x SMA20</b>\n"
        )

    msg_text = "\n".join(lines)
    tg_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": TG_CHAT_ID,
        "text": msg_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            tg_url, data=payload, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("📲 Telegram alert delivered successfully.")
    except Exception as e:
        print(f"❌ Telegram delivery failed: {e}")

# ==============================================================================
# 3. LIVE NSE QUOTE FETCHING
# ==============================================================================
def get_nse_session():
    cookie_req = urllib.request.Request("https://www.nseindia.com", headers=NSE_HEADERS)
    try:
        resp = urllib.request.urlopen(cookie_req, timeout=8)
        return resp.headers.get("Set-Cookie", "")
    except Exception:
        return ""

def get_quote(symbol, cookies):
    sym = urllib.parse.quote(symbol.strip().upper())
    h = dict(NSE_HEADERS)
    if cookies:
        h["Cookie"] = cookies
    try:
        req = urllib.request.Request(
            f"https://www.nseindia.com/api/quote-equity?symbol={sym}", headers=h
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            d = json.loads(resp.read().decode("utf-8", errors="ignore"))
            p = d.get("priceInfo", {})
            v = float(
                d.get("preOpenMarket", {}).get("totalTradedVolume", 0)
                or d.get("securityDetails", {}).get("volumeTraded", 0)
                or 0
            )
            return {
                "ltp": float(p.get("lastPrice", 0) or 0),
                "high": float(p.get("intraDayHighLow", {}).get("max", 0) or 0),
                "vol": v,
            }
    except Exception:
        return None

# ==============================================================================
# 4. PRE-BREAKOUT WATCHLIST LOADER
# ==============================================================================
def load_verified_watchlists():
    targets = {}

    # Universe 1: Nifty 750 Pre-breakouts
    if os.path.exists(S3_RESULTS):
        try:
            with open(S3_RESULTS, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            for item in d.get("Watchlist", []):
                sym = item.get("symbol", "").strip().upper()
                pivot = float(item.get("pivot_target", 0.0) or item.get("shelf_high", 0.0))
                sl = float(item.get("shelf_low", 0.0) or (pivot * 0.93))
                vsma = float(item.get("vol_sma20", 50000.0) or 50000.0)
                if sym and pivot > 0:
                    targets[sym] = {
                        "universe": "NIFTY 750",
                        "setup": item.get("setup_class", "LAUNCHPAD COIL"),
                        "pivot": pivot,
                        "sl": sl,
                        "v_sma": vsma,
                    }
        except Exception as e:
            print(f"⚠️ Error reading Nifty 750 watchlist: {e}")

    # Universe 2: Non-Nifty 750 Pre-breakouts
    if os.path.exists(R2_RESULTS):
        try:
            with open(R2_RESULTS, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            for item in d.get("Watchlist", []):
                sym = item.get("symbol", "").strip().upper()
                pivot = float(item.get("pivot_target", 0.0) or item.get("shelf_high", 0.0))
                sl = float(item.get("shelf_low", 0.0) or (pivot * 0.92))
                raw_setup = item.get("setup_class", "LAUNCHPAD COIL")
                setup = "V-REVERSAL" if "V-REVERSAL" in raw_setup else "LAUNCHPAD"
                vsma = float(item.get("vol_sma20", 50000.0) or 50000.0)
                if sym and pivot > 0:
                    targets[sym] = {
                        "universe": "NON-N750",
                        "setup": setup,
                        "pivot": pivot,
                        "sl": sl,
                        "v_sma": vsma,
                    }
        except Exception as e:
            print(f"⚠️ Error reading Non-N750 watchlist: {e}")

    return targets

# ==============================================================================
# 5. SCANNER RUNNER WITH TRUE IST ELAPSED TIME
# ==============================================================================
def run():
    targets = load_verified_watchlists()
    print(f"📊 Monitoring {len(targets)} verified pre-breakout targets from daily scanners...")

    # Accurate IST Time Calculation
    now_ist = datetime.datetime.now(IST)
    m_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    m_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if now_ist < m_open:
        mins = 1.0
    elif now_ist > m_close:
        mins = 375.0
    else:
        mins = max(1.0, (now_ist - m_open).total_seconds() / 60.0)

    pace_factor = 375.0 / mins
    cookies = get_nse_session()

    breakouts = []
    coiling = []

    for sym, info in targets.items():
        q = get_quote(sym, cookies)
        time.sleep(0.08)
        if not q or q["ltp"] <= 0:
            continue

        ltp, high, vol = q["ltp"], q["high"], q["vol"]
        pivot, sl, vsma = info["pivot"], info["sl"], info["v_sma"]

        # Full-day projected volume surge
        surge = round((vol * pace_factor) / vsma, 2) if vsma > 0 else 1.0
        diff = round(((ltp - pivot) / pivot) * 100.0, 2)
        is_bo = (ltp >= pivot) or (high >= pivot)

        row = {
            "symbol": sym,
            "universe": info["universe"],
            "setup": info["setup"],
            "pivot_ceiling": pivot,
            "ltp": ltp,
            "day_high": high,
            "stop_loss": sl,
            "risk_%": round(((pivot - sl) / pivot) * 100, 2) if pivot > 0 else 0.0,
            "dist_pivot_%": diff,
            "proj_vol_pace": surge,
            "action": "BREAKOUT" if is_bo else "COILING",
        }

        if is_bo:
            breakouts.append(row)
        else:
            coiling.append(row)

    breakouts.sort(key=lambda x: x["dist_pivot_%"], reverse=True)
    coiling.sort(key=lambda x: x["dist_pivot_%"], reverse=True)

    scan_ts = now_ist.strftime("%Y-%m-%d %I:%M:%S %p IST")

    output_payload = {
        "Scan_Timestamp": scan_ts,
        "Elapsed_Minutes": int(mins),
        "Breakouts": breakouts,
        "Coiling": coiling[:25],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)

    print(f"✅ [{scan_ts}] Scanned {len(targets)} targets: {len(breakouts)} Breakouts, {len(coiling)} Coiling.")

    if breakouts:
        send_telegram_alert(breakouts, scan_ts, mins)

if __name__ == "__main__":
    run()
