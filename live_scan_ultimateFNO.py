import os
import sys
import re
import json
import time
import datetime
import base64
import glob
import pandas as pd
import requests

# --------------------------------------------------------------------------
# PATHS & CONFIGURATION (STRICT UNIFIED REPO PATHS)
# --------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_fno_ultimate_breakouts.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
EOD_RESULTS_FILE = "fno_champion_scan_results.json"
REPO_NAME = "DRAMITBHOI/Nse-_data"
WATCHLIST_MAX_AGE_DAYS = 5

# Telegram Credentials
DEFAULT_BOT_TOKEN = "9160227761:AAE2zmSpvqttJl3_KKl3jjyKkMg-CopgtSY"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN).strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_ULTIMATE", "7860290981").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

# Check local_secrets.json fallback
for sec_p in ["local_secrets.json", "../local_secrets.json", os.path.join(DATA_DIR, "local_secrets.json")]:
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
EXCLUDED_KEYWORDS = ["BEES", "ETF", "LIQUID", "GOLD", "SILVER", "NIFTY", "SENSEX", "AXIS", "CASE", "GRW"]

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

# --------------------------------------------------------------------------
# FAILSAFE GITHUB REMOTE STATE (ZERO-SPAM MEMORY)
# --------------------------------------------------------------------------
def fetch_remote_state():
    """Fetches state directly from GitHub API to guarantee continuity across runners."""
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
    """Commits state immediately back to GitHub repo to lock alert sets."""
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

# --------------------------------------------------------------------------
# LIVE CLIENT
# --------------------------------------------------------------------------
class LivePriceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def fetch_live_quote(self, symbol):
        # 1. Google Finance
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

        # 2. Yahoo Finance
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
    return clean

# --------------------------------------------------------------------------
# TARGET BUILDER: STRICT PREVIOUS 1-WEEK COILING CANDIDATES ONLY
# --------------------------------------------------------------------------
def load_fno_1week_coiling_targets():
    """STRICT TARGET FILTER:

    Only stocks that have qualified for the institutional coiling watchlist
    in fno_champion_scan_results.json during the past 5 trading sessions.
    No un-coiled stocks are permitted.
    """
    targets = {}
    today_dt = get_ist_now().date()

    # 1. Ingest fresh targets from EOD fno_champion_scan_results.json
    eod_matches = glob.glob(f"**/{EOD_RESULTS_FILE}", recursive=True)
    if eod_matches:
        try:
            with open(eod_matches[0], "r", encoding="utf-8") as fp:
                eod_data = json.load(fp)
            # Combine both watchlist and fresh breakouts from EOD
            coils = list(eod_data.get("coiling_watchlist", [])) + list(eod_data.get("confirmed_breakouts", []))
            for c in coils:
                sym = str(c.get("symbol", "")).upper().strip()
                if not sym or any(k in sym for k in EXCLUDED_KEYWORDS):
                    continue
                p_str = str(c.get("pivot_ceiling", c.get("pivot_target", "0"))).replace("₹", "").replace(",", "")
                pivot = float(p_str)
                if pivot <= 0: continue
                sl_str = str(c.get("stop_loss", pivot * 0.935)).replace("₹", "").replace(",", "")
                stop_loss = float(sl_str)
                spread_pct = float(str(c.get("shelf_spread_%", "6.5")).replace("%", ""))
                acc_ratio = float(c.get("accumulation_ratio", 1.80))

                targets[sym] = {
                    "symbol": sym,
                    "pivot_ceiling": pivot,
                    "stop_loss": stop_loss,
                    "accumulation_ratio": acc_ratio,
                    "shelf_spread_%": spread_pct,
                    "added_date": str(today_dt)
                }
        except Exception as e:
            print(f"⚠️ Error reading EOD results: {e}")

    # 2. Ingest & Carry-Forward historical watchlists up to 5 trading sessions
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
            if sym in targets or any(k in sym for k in EXCLUDED_KEYWORDS):
                continue
            try:
                entry_dt = datetime.datetime.strptime(item.get("added_date", str(today_dt)), "%Y-%m-%d").date()
                age_days = (today_dt - entry_dt).days
            except Exception:
                age_days = 99

            if age_days <= WATCHLIST_MAX_AGE_DAYS:
                pivot = float(item.get("pivot_ceiling", 0))
                if pivot > 0:
                    targets[sym] = item

    return targets

# --------------------------------------------------------------------------
# MAIN LIVE SCANNER ENGINE
# --------------------------------------------------------------------------
def run_live_fno_scan():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()
    market_active = is_market_open()

    print(f"\n🚀 Running Ultimate FNO Live Scanner at {now_str}")
    print(f"⏱️ Session Elapsed: {elapsed_mins}/375 mins | Active Market Session: {market_active}")

    # Step 1: Load Persistent State (From GitHub API or Disk)
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
    eod_summary_sent = state.get("EOD_Summary_Sent", False) if state.get("Date") == today_date else False

    # Step 2: Strictly load 1-week pre-breakout targets
    targets = load_fno_1week_coiling_targets()
    if not targets:
        print("❌ No 1-week coiling candidates found in EOD results.")
        return

    print(f"🎯 Monitoring {len(targets)} Certified Pre-Breakout Setups from the past 1 week:")
    print("   ", list(targets.keys()))

    live_client = LivePriceClient()
    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    active_multiday_watchlist = {}
    coiling_list = []
    state_modified = False

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        stop_loss = float(meta["stop_loss"])
        acc_ratio = float(meta.get("accumulation_ratio", 1.80))
        shelf_spread = float(meta.get("shelf_spread_%", 6.5))

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.04)

        ltp = None
        day_high = None
        day_low = None
        day_vol = 0.0
        sma20_vol = 50000.0

        if quote:
            ltp = quote["ltp"]
            day_high = quote["day_high"]
            day_low = quote["day_low"]
            day_vol = quote["day_vol"]

        # Read historical 20 SMA volume
        local_files = glob.glob(f"**/{sym}.json", recursive=True)
        if local_files:
            try:
                with open(local_files[0], "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) >= 20:
                    vols = [float(x.get("volume", 0) or 0) for x in clean[-20:]]
                    if vols: sma20_vol = sum(vols) / len(vols)
                    if ltp is None or ltp <= 0:
                        ltp = float(clean[-1].get("close", pivot))
                        day_high = float(clean[-1].get("high", ltp))
                        day_low = float(clean[-1].get("low", ltp))
            except Exception:
                pass

        if ltp is None or ltp <= 0:
            ltp = pivot
            day_high = pivot
            day_low = pivot

        day_high = max(day_high or ltp, ltp)
        day_low = min(day_low or ltp, ltp)

        # Disqualification: Drop if price closes below the structural base shelf floor
        if ltp < stop_loss:
            print(f"🗑️ Purged {sym}: Broken below base stop floor ₹{stop_loss:,.2f}")
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
                state_modified = True
            else:
                benchmark_highs[sym] = round(max(benchmark_highs[sym], day_high), 2)

        record = {
            "symbol": sym,
            "pivot_ceiling": round(pivot, 2),
            "ltp": round(ltp, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct,
            "clearance_%": clearance_pct,
            "accumulation_ratio": acc_ratio,
            "shelf_spread_%": shelf_spread,
            "range_closeness": range_closeness,
            "proj_vol_pace": vol_pace
        }

        # Breakout Rule: Pace >= 1.60x AND Range Closeness >= 0.65
        is_breakout = cleared_pivot and (vol_pace >= 1.60) and (range_closeness >= 0.65)

        # ------------------------------------------------------------------
        # ALERT 1: TICK 1 CEILING BREAKOUT (FIRES STRICTLY ONCE PER DAY)
        # ------------------------------------------------------------------
        if is_breakout and market_active:
            record["action"] = "CONFIRMED BREAKOUT"
            master_confirmed[sym] = record

            if sym not in alert1_sent:
                ref_h = benchmark_highs.get(sym, day_high)
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
                state_modified = True
                print(f"📢 ALERT 1 DISPATCHED FOR {sym}")

        # ------------------------------------------------------------------
        # ALERT 2: DAY HIGH EXPANSION (FIRES STRICTLY ONCE PER DAY)
        # ------------------------------------------------------------------
        if sym in benchmark_highs and sym not in alert2_sent and market_active:
            prior_ref_h = float(benchmark_highs[sym])
            if ltp >= (prior_ref_h * 1.0015) and ltp > pivot and is_breakout:
                record["action"] = "CONFIRMED BREAKOUT"
                master_confirmed[sym] = record

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
                state_modified = True
                print(f"📢 ALERT 2 DISPATCHED FOR {sym}")

        if not cleared_pivot:
            record["action"] = "WICK REJECTED" if (wick_touched and range_closeness < 0.50) else "COILING"
            coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    final_confirmed = list(master_confirmed.values())

    # --------------------------------------------------------------------------
    # 03:15 PM EOD CLOSING SUMMARY (FIRES EXACTLY ONCE PER DAY)
    # --------------------------------------------------------------------------
    if elapsed_mins >= 360 and not eod_summary_sent:
        summary_msg = (
            f"🎯 <b>ULTIMATE F&O CLOSING CONFIRMATIONS (3:15 PM)</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
        )
        if final_confirmed:
            summary_msg += f"<b>Actionable Entries ({len(final_confirmed)}):</b>\n"
            for b in final_confirmed:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: LTP ₹{b['ltp']:,.2f} (+{b['clearance_%']}%) | "
                    f"Acc: {b['accumulation_ratio']}x | Pace: {b['proj_vol_pace']}x | SL: ₹{b['stop_loss']:,.2f}\n"
                )
            summary_msg += "\n✅ <i>Place CNC delivery / Hedged Futures orders before 03:28 PM IST.</i>\n"
        else:
            summary_msg += "<i>No F&O stocks held breakout ceiling with verified institutional volume today.</i>\n"

        wicked = [c for c in coiling_list if c["action"] == "WICK REJECTED"][:5]
        if wicked:
            summary_msg += "\n<b>⚠️ Upper Wick Rejections (Avoid):</b>\n"
            for w in wicked:
                summary_msg += f"🔴 {w['symbol']}: High ₹{w['day_high']:,.2f} ➔ LTP ₹{w['ltp']:,.2f} ({w['clearance_%']:+.2f}%)\n"

        send_telegram_alert(summary_msg)
        eod_summary_sent = True
        state_modified = True

    # Step 3: Save Output & Persist Directly to GitHub
    output_payload = {
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "EOD_Summary_Sent": eod_summary_sent,
        "Alert1_Sent": list(alert1_sent),
        "Alert2_Sent": list(alert2_sent),
        "Benchmark_Highs": benchmark_highs,
        "Master_Confirmed_Breakouts": master_confirmed,
        "Breakouts_All_Day": final_confirmed,
        "Active_MultiDay_Watchlist": active_multiday_watchlist,
        "Coiling_Watchlist": coiling_list[:30]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Lock state remotely to eliminate any runner wipe spam
    if state_modified or not remote_state:
        push_remote_state(output_payload, sha)

    print(f"✅ Scan Complete. Alert1_Sent={list(alert1_sent)} | Alert2_Sent={list(alert2_sent)}")

if __name__ == "__main__":
    run_live_fno_scan()
