import os
import sys
import re
import json
import time
import datetime
import base64
import glob
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------
# DIRECTORY & FILE TARGETS
# ---------------------------------------------------------
DATA_DIR = "Codes/data" if os.path.exists("Codes/data") else "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_fno_ultimate_breakouts.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
EOD_RESULTS_FILE = "fno_champion_scan_results.json"
REPO_NAME = "DRAMITBHOI/Nse-_data"
MIN_TURNOVER_CR = 10.0
WATCHLIST_MAX_AGE_DAYS = 5

# ---------------------------------------------------------
# CREDENTIALS (TELEGRAM & GITHUB API)
# ---------------------------------------------------------
DEFAULT_BOT_TOKEN = "9160227761:AAE2zmSpvqttJl3_KKl3jjyKkMg-CopgtSY"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN).strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_ULTIMATE", "7860290981").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

# Read local_secrets.json fallback
for sec_p in ["local_secrets.json", "../local_secrets.json"]:
    if os.path.exists(sec_p):
        try:
            with open(sec_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if not GITHUB_TOKEN:
                    GITHUB_TOKEN = cfg.get("GITHUB_TOKEN", "")
                if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "7860290981":
                    TELEGRAM_CHAT_ID = cfg.get("TELEGRAM_CHAT_ID_ULTIMATE", cfg.get("TELEGRAM_CHAT_ID", "7860290981"))
        except Exception:
            pass

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}

def get_ist_now():
    return datetime.datetime.now(IST)

def is_market_open():
    now = get_ist_now()
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

def get_market_session_elapsed():
    now = get_ist_now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 1
    elif now > market_close:
        return 375
    else:
        return max(1, min(375, int((now - market_open).total_seconds() / 60)))

def get_u_curve_expected_volume_fraction(elapsed_mins):
    if elapsed_mins <= 0:
        return 0.02
    elif elapsed_mins <= 75:
        return 0.05 + (elapsed_mins / 75.0) * 0.27
    elif elapsed_mins <= 225:
        return 0.32 + ((elapsed_mins - 75.0) / 150.0) * 0.28
    elif elapsed_mins <= 330:
        return 0.60 + ((elapsed_mins - 225.0) / 105.0) * 0.20
    else:
        return 0.80 + ((elapsed_mins - 330.0) / 45.0) * 0.20

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
        r = requests.post(url, json=payload, timeout=10)
        if not r.json().get("ok"):
            print(f"❌ Telegram API Error: {r.text}")
    except Exception as e:
        print(f"❌ Telegram Dispatch Error: {e}")

# ---------------------------------------------------------
# DIRECT GITHUB PERSISTENCE FOR ZERO-SPAM STATE MEMORY
# ---------------------------------------------------------
def fetch_remote_state():
    """Fetches live_fno_ultimate_breakouts.json directly from GitHub API."""
    if not GITHUB_TOKEN:
        return None, None
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/data/live_fno_ultimate_breakouts.json"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            res = r.json()
            content = base64.b64decode(res["content"]).decode("utf-8")
            return json.loads(content), res["sha"]
    except Exception:
        pass
    return None, None

def push_remote_state(state_data, sha=None):
    """Pushes the updated state directly to GitHub to lock in alert memory."""
    if not GITHUB_TOKEN:
        return
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/data/live_fno_ultimate_breakouts.json"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    content_str = json.dumps(state_data, indent=2)
    b64_content = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
    payload = {
        "message": "Auto-update: Lock FNO alert state [skip ci]",
        "content": b64_content
    }
    if sha:
        payload["sha"] = sha
    try:
        requests.put(url, headers=headers, json=payload, timeout=6)
    except Exception:
        pass

# ---------------------------------------------------------
# PRICE CLIENT & NORMALIZATION
# ---------------------------------------------------------
class LivePriceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def fetch_live_quote(self, symbol):
        try:
            url = f"https://www.google.com/finance/quote/{symbol}:NSE"
            r = self.session.get(url, timeout=4)
            if r.status_code == 200:
                html = r.text
                p_m = re.search(r'data-last-price="([\d\.]+)"', html)
                if p_m:
                    ltp = float(p_m.group(1))
                    d_high, d_low = ltp, ltp
                    rng_m = re.search(r'Day range.*?₹([\d\.,]+)\s*-\s*₹([\d\.,]+)', html, re.DOTALL)
                    if rng_m:
                        d_low = float(rng_m.group(1).replace(",", ""))
                        d_high = float(rng_m.group(2).replace(",", ""))

                    d_vol = 0.0
                    vol_m = re.search(r'Volume</div><div class="P6K39c">([\d\.,MKmk]+)</div>', html)
                    if vol_m:
                        v_str = vol_m.group(1).replace(",", "").upper()
                        if "M" in v_str: d_vol = float(v_str.replace("M", "")) * 1_000_000
                        elif "K" in v_str: d_vol = float(v_str.replace("K", "")) * 1_000
                        else: d_vol = float(v_str)

                    if ltp > 0:
                        return {"ltp": ltp, "day_high": max(d_high, ltp), "day_low": min(d_low, ltp), "day_vol": d_vol}
        except Exception:
            pass

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
            r = self.session.get(url, timeout=4)
            if r.status_code == 200:
                meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                ltp = float(meta.get("regularMarketPrice", 0) or 0)
                d_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
                d_low = float(meta.get("regularMarketDayLow", ltp) or ltp)
                d_vol = float(meta.get("regularMarketVolume", 0) or 0)
                if ltp > 0:
                    return {"ltp": ltp, "day_high": max(d_high, ltp), "day_low": min(d_low, ltp), "day_vol": d_vol}
        except Exception:
            pass
        return None

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

# ---------------------------------------------------------
# TARGET BUILDER (EOD SCANNER COILS + 5-SESSION LOOKBACK)
# ---------------------------------------------------------
def load_all_target_candidates():
    """Combines:

    1. Today's fresh coiling candidates from Scan_ultimateFNO
    2. Historical coiling candidates from the past 5 trading sessions
    """
    targets = {}
    today_dt = get_ist_now().date()

    # Source 1: Check fno_champion_scan_results.json
    eod_files = glob.glob(f"**/{EOD_RESULTS_FILE}", recursive=True)
    if eod_files:
        try:
            with open(eod_files[0], "r", encoding="utf-8") as fp:
                eod_data = json.load(fp)
            coils = eod_data.get("coiling_watchlist", [])
            for c in coils:
                sym = str(c.get("symbol", "")).upper().strip()
                if not sym or sym in EXCLUDED_SYMBOLS: continue
                p_str = str(c.get("pivot_target", c.get("pivot_ceiling", "0"))).replace("₹", "").replace(",", "")
                pivot = float(p_str)
                if pivot <= 0: continue
                targets[sym] = {
                    "symbol": sym,
                    "pivot_ceiling": pivot,
                    "stop_loss": pivot * 0.935,
                    "accumulation_ratio": float(c.get("accumulation_ratio", 1.8)),
                    "shelf_spread_%": float(str(c.get("shelf_spread_%", "6.5")).replace("%", "")),
                    "added_date": str(today_dt)
                }
        except Exception:
            pass

    # Source 2: Ingest Multi-day Watchlist from live_fno_ultimate_breakouts.json (Up to 5 Days)
    prior_state, _ = fetch_remote_state()
    if not prior_state and os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prior_state = json.load(f)
        except Exception:
            pass

    if prior_state:
        hist_watchlist = prior_state.get("Active_MultiDay_Watchlist", {})
        for sym, item in hist_watchlist.items():
            if sym in targets or sym in EXCLUDED_SYMBOLS: continue
            try:
                entry_dt = datetime.datetime.strptime(item.get("added_date", str(today_dt)), "%Y-%m-%d").date()
                age = (today_dt - entry_dt).days
            except Exception:
                age = 99
            if age <= WATCHLIST_MAX_AGE_DAYS:
                targets[sym] = item

    # Source 3: Direct Scan of local files for active FNO shelves
    all_json = glob.glob("**/*.json", recursive=True)
    stock_map = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if any(k in bname.lower() for k in ["nifty", "audit", "result", "trade", "secret", "package", "checkpoint", "live"]):
            continue
        if sym not in stock_map or os.path.getsize(p) > os.path.getsize(stock_map[sym]):
            stock_map[sym] = p

    for sym, filepath in stock_map.items():
        if sym in targets or sym in EXCLUDED_SYMBOLS: continue
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue
        if len(clean) < 70: continue

        df = pd.DataFrame(clean)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        delivs = df["delivery_vol"].values
        volumes = df["volume"].values
        i = len(df) - 1

        # Base search
        best_spread, lp_h, lp_l, best_len = 999.0, 0.0, 0.0, 15
        shelf_found = False
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
        base_up = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] >= closes[k - 1])
        base_dn = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] < closes[k - 1])
        acc_ratio = round((base_up / base_dn), 2) if base_dn > 0 else 2.50
        if acc_ratio < 1.50: continue

        sma20_v = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 50000.0
        targets[sym] = {
            "symbol": sym,
            "pivot_ceiling": float(lp_h),
            "stop_loss": float(min(lows[i], lp_l)),
            "accumulation_ratio": acc_ratio,
            "shelf_spread_%": round(best_spread, 2),
            "sma20_vol": sma20_v,
            "added_date": str(today_dt)
        }

    return targets

# ---------------------------------------------------------
# LIVE SCANNER CORE
# ---------------------------------------------------------
def run_live_fno_scan():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()
    market_active = is_market_open()

    print(f"\n🚀 Running Live F&O Ultimate Scanner at {now_str}")
    print(f"⏱️ Session Elapsed: {elapsed_mins}/375 mins | Market Active: {market_active}")

    # Load persistent state from GitHub or local disk
    remote_state, sha = fetch_remote_state()
    state = remote_state or {}
    if not state and os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}

    alert1_sent = set(state.get("Alert1_Sent", [])) if state.get("Date") == today_date else set()
    alert2_sent = set(state.get("Alert2_Sent", [])) if state.get("Date") == today_date else set()
    benchmark_highs = state.get("Benchmark_Highs", {}) if state.get("Date") == today_date else {}
    master_confirmed = state.get("Master_Confirmed_Breakouts", {}) if state.get("Date") == today_date else {}

    targets = load_all_target_candidates()
    print(f"🎯 Monitored Target Pool: {len(targets)} stocks (including past 5 trading sessions).")

    live_client = LivePriceClient()
    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    active_multiday_watchlist = {}
    coiling_list = []
    state_changed = False

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        stop_loss = float(meta["stop_loss"])
        acc_ratio = float(meta["accumulation_ratio"])
        shelf_spread = float(meta["shelf_spread_%"])
        sma20_vol = float(meta.get("sma20_vol", 50000.0))

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.03)

        if not quote: continue
        ltp = quote["ltp"]
        day_high = quote["day_high"]
        day_low = quote["day_low"]
        day_vol = quote["day_vol"]

        # Prune if structurally invalidated below shelf floor
        if ltp < stop_loss * 0.98:
            continue
        active_multiday_watchlist[sym] = meta

        bar_rng = day_high - day_low
        range_closeness = round((ltp - day_low) / bar_rng, 2) if bar_rng > 0 else 0.50
        proj_vol = day_vol / expected_fraction
        vol_pace = round(proj_vol / max(1.0, sma20_vol), 2)

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((ltp - stop_loss) / ltp) * 100.0, 2) if ltp > stop_loss else 6.5
        cleared_pivot = bool(clearance_pct >= 0.15)
        wick_touched = bool(day_high >= pivot)

        if wick_touched or cleared_pivot:
            if sym not in benchmark_highs:
                benchmark_highs[sym] = round(day_high, 2)
                state_changed = True
            else:
                benchmark_highs[sym] = round(max(benchmark_highs[sym], day_high), 2)

        record = {
            "symbol": sym, "pivot_ceiling": round(pivot, 2), "ltp": round(ltp, 2),
            "day_high": round(day_high, 2), "day_low": round(day_low, 2), "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct, "clearance_%": clearance_pct, "accumulation_ratio": acc_ratio,
            "shelf_spread_%": shelf_spread, "range_closeness": range_closeness, "proj_vol_pace": vol_pace
        }

        # Breakout qualification (Pace >= 1.60x & Range Close >= 0.65)
        is_breakout = cleared_pivot and (vol_pace >= 1.60) and (range_closeness >= 0.65)

        # -------------------------------------------------------------
        # ALERT 1: TICK 1 CEILING CLEARANCE (STRICTLY ONCE PER DAY)
        # -------------------------------------------------------------
        if is_breakout and market_active:
            master_confirmed[sym] = record
            if sym not in alert1_sent:
                msg1 = (
                    f"🚀 <b>ULTIMATE F&O BREAKOUT CONFIRMED</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                    f"• <b>Live LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}%</b>)\n"
                    f"• <b>Pivot Ceiling</b>: ₹{pivot:,.2f} (Cleared!)\n"
                    f"• <b>Projected Vol Pace</b>: <b>{vol_pace:.2f}x SMA20</b>\n"
                    f"• <b>Accumulation Ratio</b>: <b>{acc_ratio}x</b>\n"
                    f"• <b>Range Closeness</b>: {range_closeness} (Clean Bar)\n"
                    f"• <b>Hard Stop Loss</b>: ₹{stop_loss:,.2f} (Risk: {risk_pct}%)\n"
                    f"• <b>Shelf Spread</b>: {shelf_spread}%\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🛡️ <i>Exit Trailing Rule: Exit 2.2 Model</i>"
                )
                send_telegram_alert(msg1)
                alert1_sent.add(sym)
                state_changed = True
                print(f"🔔 ALERT 1 SENT: #{sym}")

        # -------------------------------------------------------------
        # ALERT 2: DAY HIGH EXPANSION (STRICTLY ONCE PER DAY)
        # -------------------------------------------------------------
        if sym in benchmark_highs and sym not in alert2_sent and market_active:
            prior_ref_h = float(benchmark_highs[sym])
            if ltp >= (prior_ref_h * 1.0015) and ltp > pivot and is_breakout:
                msg2 = (
                    f"⚡ <b>ULTIMATE F&O 2ND ALERT: HIGH EXPANSION</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                    f"• <b>Prior Day High Cleared</b>: ₹{prior_ref_h:,.2f}\n"
                    f"• <b>New LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}%</b>)\n"
                    f"• <b>Projected Vol Pace</b>: <b>{vol_pace:.2f}x SMA20</b>\n"
                    f"• <b>Accumulation Ratio</b>: <b>{acc_ratio}x</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔥 <i>Momentum accelerating into high-conviction runner!</i>"
                )
                send_telegram_alert(msg2)
                alert2_sent.add(sym)
                benchmark_highs[sym] = round(day_high, 2)
                state_changed = True
                print(f"🔔 ALERT 2 SENT: #{sym}")

        if not cleared_pivot:
            record["action"] = "WICK REJECTED" if (wick_touched and range_closeness < 0.50) else "COILING"
            coiling_list.append(record)

    # Save to disk
    output_payload = {
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "Alert1_Sent": list(alert1_sent),
        "Alert2_Sent": list(alert2_sent),
        "Benchmark_Highs": benchmark_highs,
        "Master_Confirmed_Breakouts": master_confirmed,
        "Active_MultiDay_Watchlist": active_multiday_watchlist,
        "Coiling_Watchlist": coiling_list[:30]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Lock state to GitHub API to guarantee zero spam across subsequent runs
    if state_changed or not remote_state:
        push_remote_state(output_payload, sha)

    print(f"✅ Live FNO Scan Complete. Alerted Today: Alert1={list(alert1_sent)} | Alert2={list(alert2_sent)}")

if __name__ == "__main__":
    run_live_fno_scan()
