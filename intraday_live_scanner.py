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
# 2. TELEGRAM BROADCASTER
# ==============================================================================
def send_telegram_msg(msg_text):
    if not TG_TOKEN or not TG_CHAT_ID or not msg_text:
        return
    tg_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": TG_CHAT_ID,
        "text": msg_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(tg_url, data=payload, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        print(f"❌ Telegram delivery failed: {e}")

def notify_telegram(new_triggers, all_breakouts, scan_ts, mins):
    # 1. Immediate Alert for Freshly Triggered Stocks
    if new_triggers:
        lines = [
            "🚨 <b>FRESH INTRADAY BREAKOUT TRIGGERED!</b>",
            f"🕒 <i>Time: {scan_ts} (Elapsed: {int(mins)}m/375m)</i>",
            "────────────────────────",
        ]
        for b in new_triggers:
            lines.append(
                f"• <b>{b['symbol']}</b> ({b['universe']} | {b['setup']})\n"
                f"  LTP: <b>₹{b['ltp']:.2f}</b> (Pivot: ₹{b['pivot_ceiling']:.2f}, {b['dist_pivot_%']:+.2f}%)\n"
                f"  SL: ₹{b['stop_loss']:.2f} (Risk: {b['risk_%']:.1f}%)\n"
                f"  Vol Pace: <b>{b['proj_vol_pace']:.2f}x SMA20</b>\n"
            )
        send_telegram_msg("\n".join(lines))

    # 2. Cumulative Progress Update (Sent every run if breakouts exist)
    if all_breakouts:
        summary_lines = [
            "📊 <b>ALL-DAY BREAKOUT PROGRESS MONITOR</b>",
            f"🕒 <i>Time: {scan_ts} | Active Breakouts: {len(all_breakouts)}</i>",
            "────────────────────────",
        ]
        for b in all_breakouts:
            status_icon = "🟢" if b["dist_pivot_%"] >= 0 else "🔴"
            summary_lines.append(
                f"{status_icon} <b>{b['symbol']}</b>: ₹{b['ltp']:.2f} ({b['dist_pivot_%']:+.2f}% vs Pivot) "
                f"| High: ₹{b['day_high']:.2f} | Pace: {b['proj_vol_pace']:.1f}x"
            )
        send_telegram_msg("\n".join(summary_lines))

# ==============================================================================
# 3. NSE QUOTES & COOKIE SESSION
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
        req = urllib.request.Request(f"https://www.nseindia.com/api/quote-equity?symbol={sym}", headers=h)
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
# 4. WATCHLIST & PERSISTENT HISTORY LOADER
# ==============================================================================
def load_verified_watchlists():
    targets = {}
    for univ, path in [("NIFTY 750", S3_RESULTS), ("NON-N750", R2_RESULTS)]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                for item in d.get("Watchlist", []):
                    sym = item.get("symbol", "").strip().upper()
                    pivot = float(item.get("pivot_target", 0.0) or item.get("shelf_high", 0.0))
                    sl = float(item.get("shelf_low", 0.0) or (pivot * 0.92))
                    vsma = float(item.get("vol_sma20", 50000.0) or 50000.0)
                    if sym and pivot > 0:
                        targets[sym] = {
                            "universe": univ,
                            "setup": item.get("setup_class", "LAUNCHPAD COIL"),
                            "pivot": pivot,
                            "sl": sl,
                            "v_sma": vsma,
                        }
            except Exception:
                pass
    return targets

def load_previous_day_state(today_date_str):
    """Retains breakouts that triggered earlier today."""
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            # Only keep history if the file was created today
            if d.get("Scan_Date") == today_date_str:
                return {item["symbol"]: item for item in d.get("Breakouts_All_Day", [])}
        except Exception:
            pass
    return {}

# ==============================================================================
# 5. EXECUTION & PROGRESS ACCUMULATOR
# ==============================================================================
def run():
    now_ist = datetime.datetime.now(IST)
    today_date = now_ist.strftime("%Y-%m-%d")
    m_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    m_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if now_ist < m_open:
        mins = 1.0
    elif now_ist > m_close:
        mins = 375.0
    else:
        mins = max(1.0, (now_ist - m_open).total_seconds() / 60.0)

    pace_factor = 375.0 / mins
    targets = load_verified_watchlists()
    existing_breakouts = load_previous_day_state(today_date)
    cookies = get_nse_session()

    all_breakouts_map = dict(existing_breakouts)
    new_triggers = []
    coiling = []

    for sym, info in targets.items():
        q = get_quote(sym, cookies)
        time.sleep(0.08)
        if not q or q["ltp"] <= 0:
            continue

        ltp, high, vol = q["ltp"], q["high"], q["vol"]
        pivot, sl, vsma = info["pivot"], info["sl"], info["v_sma"]

        surge = round((vol * pace_factor) / vsma, 2) if vsma > 0 else 1.0
        diff = round(((ltp - pivot) / pivot) * 100.0, 2)
        has_broken_out = (ltp >= pivot) or (high >= pivot) or (sym in existing_breakouts)

        row = {
            "symbol": sym,
            "universe": info["universe"],
            "setup": info["setup"],
            "pivot_ceiling": pivot,
            "ltp": ltp,
            "day_high": max(high, existing_breakouts.get(sym, {}).get("day_high", high)),
            "stop_loss": sl,
            "risk_%": round(((pivot - sl) / pivot) * 100, 2) if pivot > 0 else 0.0,
            "dist_pivot_%": diff,
            "proj_vol_pace": surge,
            "action": "BREAKOUT" if has_broken_out else "COILING",
            "first_trigger_time": existing_breakouts.get(sym, {}).get("first_trigger_time", now_ist.strftime("%I:%M %p")),
        }

        if has_broken_out:
            if sym not in existing_breakouts:
                new_triggers.append(row)
            all_breakouts_map[sym] = row
        else:
            coiling.append(row)

    breakout_list = sorted(list(all_breakouts_map.values()), key=lambda x: x["dist_pivot_%"], reverse=True)
    coiling_list = sorted(coiling, key=lambda x: x["dist_pivot_%"], reverse=True)

    scan_ts = now_ist.strftime("%Y-%m-%d %I:%M:%S %p IST")

    output_payload = {
        "Scan_Date": today_date,
        "Scan_Timestamp": scan_ts,
        "Elapsed_Minutes": int(mins),
        "Total_Breakouts_Today": len(breakout_list),
        "Breakouts_All_Day": breakout_list,
        "Coiling": coiling_list[:25],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)

    print(f"✅ [{scan_ts}] Total Breakouts: {len(breakout_list)} ({len(new_triggers)} new) | Coiling: {len(coiling_list)}")
    notify_telegram(new_triggers, breakout_list, scan_ts, mins)

if __name__ == "__main__":
    run()
