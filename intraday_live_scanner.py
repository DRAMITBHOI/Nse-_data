import os
import sys
import re
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts.json")

# Configured Telegram credentials with direct fallback
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or "8796649927:AAHOx-1Fc_IyIttdQboQc0ul9qsM0g35O9c"
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or "7860290981"

# Explicit IST Timezone (UTC + 05:30)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}

def get_ist_now():
    return datetime.datetime.now(IST)

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID missing. Alert skipped.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=12)
        if r.status_code == 200:
            print("📡 Telegram alert dispatched successfully (HTTP 200)")
        else:
            print(f"⚠️ Telegram returned status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Telegram API Error: {e}")

def get_market_session_elapsed():
    now = get_ist_now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 1
    elif now > market_close:
        return 375
    else:
        elapsed = int((now - market_open).total_seconds() / 60)
        return max(1, min(375, elapsed))

class LivePriceClient:
    """
    Fetches real-time market data reliably inside GitHub Actions (Azure Datacenter).
    Primary: Google Finance (Zero delay, non-blocking)
    Fallback: Yahoo Finance JSON quote endpoint (fast query, no yfinance package hang)
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def fetch_live_quote(self, symbol):
        # 1. Primary: Google Finance NSE
        try:
            url = f"https://www.google.com/finance/quote/{symbol}:NSE"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                html = r.text
                price_match = re.search(r'data-last-price="([\d\.]+)"', html)
                if price_match:
                    ltp = float(price_match.group(1))

                    day_high = ltp
                    high_match = re.search(r'Day range.*?₹[\d\.,]+\s*-\s*₹([\d\.,]+)', html, re.DOTALL)
                    if high_match:
                        day_high = float(high_match.group(1).replace(",", ""))

                    day_vol = 0.0
                    vol_match = re.search(r'Volume</div><div class="P6K39c">([\d\.,MKmk]+)</div>', html)
                    if vol_match:
                        v_str = vol_match.group(1).replace(",", "").upper()
                        if "M" in v_str:
                            day_vol = float(v_str.replace("M", "")) * 1_000_000
                        elif "K" in v_str:
                            day_vol = float(v_str.replace("K", "")) * 1_000
                        else:
                            day_vol = float(v_str)

                    if ltp > 0:
                        return {"ltp": ltp, "day_high": max(day_high, ltp), "day_vol": day_vol}
        except Exception:
            pass

        # 2. Secondary Fallback: Direct Yahoo Finance quote query
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                ltp = float(meta.get("regularMarketPrice", 0) or 0)
                day_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
                day_vol = float(meta.get("regularMarketVolume", 0) or 0)
                if ltp > 0:
                    return {"ltp": ltp, "day_high": max(day_high, ltp), "day_vol": day_vol}
        except Exception:
            pass

        return None

def load_comprehensive_targets():
    targets = {}
    sources = [
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultra750_results.json", "NIFTY 750"),
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultraN750_results.json", "NON-N750")
    ]
    for url, default_univ in sources:
        try:
            cache_buster = f"?_ts={int(time.time() * 1000)}"
            r = requests.get(url + cache_buster, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}, timeout=12)
            if r.status_code == 200:
                d = r.json()
                items = list(d.get("Watchlist", []))
                fresh = d.get("Fresh_Triggers", [])
                if isinstance(fresh, list):
                    items.extend(fresh)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    sym = str(item.get("symbol", "")).strip().upper()
                    if not sym or sym.endswith("BEES") or sym in EXCLUDED_SYMBOLS:
                        continue
                    pivot = float(item.get("pivot_ceiling") or item.get("pivot_target") or item.get("pivot") or 0.0)
                    if pivot <= 0:
                        continue
                    stop_loss = float(item.get("stop_loss", pivot * 0.92))
                    setup = item.get("setup_class") or item.get("type") or "LAUNCHPAD COIL"
                    universe = item.get("universe", default_univ)
                    targets[sym] = {"symbol": sym, "universe": universe, "setup": setup, "pivot_ceiling": pivot, "stop_loss": stop_loss}
                print(f"✅ Loaded {len(items)} setups from {url.split('/')[-1]} ({default_univ})")
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prior_state = json.load(f)
            for item in prior_state.get("Coiling", []):
                sym = str(item.get("symbol", "")).strip().upper()
                if sym and sym not in targets and (not sym.endswith("BEES")) and (sym not in EXCLUDED_SYMBOLS):
                    pivot = float(item.get("pivot_ceiling", 0.0))
                    stop_loss = float(item.get("stop_loss", pivot * 0.92))
                    ltp = float(item.get("ltp", pivot)) if pd.notnull(item.get("ltp")) else pivot
                    if ltp >= stop_loss and pivot > 0:
                        targets[sym] = {"symbol": sym, "universe": item.get("universe", "NON-N750"), "setup": item.get("setup", "LAUNCHPAD COIL"), "pivot_ceiling": pivot, "stop_loss": stop_loss}
        except Exception:
            pass

    print(f"📋 Total Active Live Watchlist: {len(targets)} setups across N750 & Non-N750.")
    return targets

def run_live_intraday_scan():
    targets = load_comprehensive_targets()
    if not targets:
        print("❌ No watchlisted setups found to scan.")
        return

    prior_breakouts_map = {}
    prior_hits = {}
    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for b in prev_data.get("Breakouts_All_Day", []):
                    prior_breakouts_map[b["symbol"]] = b
                prior_hits = prev_data.get("Consecutive_Hits", {})
        except Exception:
            pass

    elapsed_mins = get_market_session_elapsed()
    now_dt = get_ist_now()
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")

    print(f"⚡ Streaming Real-Time Quotes (Elapsed: {elapsed_mins}m / 375m IST)...")
    live_client = LivePriceClient()

    confirmed_breakouts = []
    radar_candidates = []
    coiling_list = []
    new_alerts_to_send = []
    current_hits = {}

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        stop_loss = float(meta["stop_loss"])

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.06)

        ltp = None
        day_high = None
        day_vol = 0.0
        sma20_vol = 50000.0

        if quote:
            ltp = quote["ltp"]
            day_high = quote["day_high"]
            day_vol = quote["day_vol"]

        # Calculate historical 20-day volume baseline from local EOD JSON cache
        local_json = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(local_json):
            try:
                with open(local_json, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                if raw and isinstance(raw, list):
                    vols = [float(x.get("volume", 0) or 0) for x in raw[-20:]]
                    if vols:
                        sma20_vol = sum(vols) / len(vols)
                    if ltp is None or ltp <= 0:
                        last_c = float(raw[-1].get("close", pivot))
                        ltp = last_c
                        day_high = float(raw[-1].get("high", last_c))
            except Exception:
                pass

        if ltp is None or pd.isna(ltp) or ltp <= 0:
            ltp = pivot
            day_high = pivot

        if day_high is None or pd.isna(day_high) or day_high < ltp:
            day_high = ltp

        sma20_vol = max(1.0, sma20_vol)
        expected_fraction = max(0.01, min(1.0, elapsed_mins / 375.0))
        projected_vol = day_vol / expected_fraction
        vol_pace = round(projected_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 8.0

        cleared_pivot = bool(clearance_pct >= 0.25)
        wick_touched_pivot = bool(day_high >= pivot)
        pace_threshold = 1.25 if meta["universe"] == "NIFTY 750" else 0.75
        is_volume_confirmed = bool(vol_pace >= pace_threshold)

        record = {
            "symbol": sym,
            "universe": meta["universe"],
            "setup": meta["setup"],
            "pivot_ceiling": round(pivot, 2),
            "ltp": round(ltp, 2),
            "day_high": round(day_high, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct,
            "clearance_%": clearance_pct,
            "dist_pivot_%": clearance_pct,
            "proj_vol_pace": vol_pace
        }

        if cleared_pivot:
            current_hits[sym] = prior_hits.get(sym, 0) + 1
        else:
            current_hits[sym] = 0

        # Time-Gated Trade Windows
        if elapsed_mins >= 330:  # 02:45 PM – 03:30 PM IST: Closing Execution Window
            if cleared_pivot and is_volume_confirmed:
                record["action"] = "CONFIRMED BREAKOUT"
                confirmed_breakouts.append(record)
                if sym not in prior_breakouts_map:
                    new_alerts_to_send.append(("EXECUTION", record))
                    prior_breakouts_map[sym] = record
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK REJECTED"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)
        else:  # 09:15 AM – 02:45 PM IST: Radar Clearance Window
            if cleared_pivot and is_volume_confirmed:
                record["action"] = "RADAR PIVOT CLEARANCE"
                radar_candidates.append(record)
                if sym not in prior_breakouts_map and current_hits[sym] >= 1:
                    new_alerts_to_send.append(("RADAR", record))
                    prior_breakouts_map[sym] = record
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK TESTING"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    confirmed_breakouts.sort(key=lambda x: x["proj_vol_pace"], reverse=True)
    radar_candidates.sort(key=lambda x: x["proj_vol_pace"], reverse=True)

    output_payload = {
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "Breakouts_All_Day": confirmed_breakouts if elapsed_mins >= 330 else radar_candidates,
        "Coiling": coiling_list[:50],
        "Consecutive_Hits": current_hits
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ Saved live scan: {len(confirmed_breakouts)} Executions, {len(radar_candidates)} Radar, {len(coiling_list)} Coiling.")

    # 1. Real-time individual alerts
    for alert_type, alert in new_alerts_to_send:
        if alert_type == "RADAR":
            msg = (
                f"👀 <b>RADAR: PIVOT CEILING CLEARED (ZERO DELAY)</b>\n"
                f"• <b>Symbol</b>: <code>{alert['symbol']}</code> ({alert['universe']} | {alert['setup']})\n"
                f"• <b>Pivot Ceiling</b>: ₹{alert['pivot_ceiling']:,.2f}\n"
                f"• <b>Live NSE LTP</b>: ₹{alert['ltp']:,.2f} (<b>{alert['clearance_%']:+.2f}%</b>)\n"
                f"• <b>Day High</b>: ₹{alert['day_high']:,.2f}\n"
                f"• <b>Volume Pace</b>: <b>{alert['proj_vol_pace']}x SMA20</b>\n"
                f"• <b>Stop Loss</b>: ₹{alert['stop_loss']:,.2f} (Risk: {alert['risk_%']}%)\n"
                f"📌 <i>Holding +0.25%. Track volume into 02:45–03:15 PM execution window.</i>"
            )
        else:
            msg = (
                f"🎯 <b>CONFIRMED CLOSING BREAKOUT TRIGGER</b>\n"
                f"• <b>Symbol</b>: <code>{alert['symbol']}</code> ({alert['universe']} | {alert['setup']})\n"
                f"• <b>Pivot Ceiling</b>: ₹{alert['pivot_ceiling']:,.2f}\n"
                f"• <b>Live NSE LTP</b>: ₹{alert['ltp']:,.2f} (<b>{alert['clearance_%']:+.2f}%</b>)\n"
                f"• <b>Volume Pace</b>: <b>{alert['proj_vol_pace']}x SMA20</b>\n"
                f"• <b>Stop Loss</b>: ₹{alert['stop_loss']:,.2f} (Risk: {alert['risk_%']}%)\n"
                f"✅ <i>Verified close above pivot. Execute delivery entry.</i>"
            )
        send_telegram_alert(msg)
        time.sleep(1)

    # 2. Daily EOD Execution Summary (03:20 PM - 03:30 PM IST)
    if elapsed_mins >= 365:
        summary_msg = (
            f"📊 <b>EOD BREAKOUT EXECUTION MONITOR (LIVE STREAM)</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
            f"<b>Active Confirmed Breakouts ({len(confirmed_breakouts)}):</b>\n"
        )
        if confirmed_breakouts:
            for b in confirmed_breakouts:
                summary_msg += f"🟢 <b>{b['symbol']}</b>: ₹{b['ltp']:,.2f} ({b['clearance_%']:+.2f}%) | Pace: {b['proj_vol_pace']}x\n"
        else:
            summary_msg += "<i>No candidates held above pivot with verified volume today.</i>\n"

        wicked = [c for c in coiling_list if c["action"] == "WICK REJECTED"][:5]
        if wicked:
            summary_msg += "\n<b>⚠️ Upper Wick Rejections:</b>\n"
            for w in wicked:
                summary_msg += f"🔴 {w['symbol']}: High ₹{w['day_high']:,.2f} ➔ LTP ₹{w['ltp']:,.2f} ({w['clearance_%']:+.2f}%)\n"

        send_telegram_alert(summary_msg)

if __name__ == "__main__":
    run_live_intraday_scan()
