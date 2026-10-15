import os
import sys
import json
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or chat ID missing. Alert skipped.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status == 200:
                print("📡 Telegram alert dispatched successfully (HTTP 200)")
            else:
                print(f"⚠️ Telegram returned status {resp.status}")
    except Exception as e:
        print(f"❌ Telegram API Error: {e}")

def get_market_session_elapsed():
    now = datetime.datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now < market_open:
        return 1
    elif now > market_close:
        return 375
    else:
        elapsed = int((now - market_open).total_seconds() / 60)
        return max(1, min(375, elapsed))

def load_comprehensive_targets():
    """
    Loads target candidates from the updated Swing Ultra results files:
    - swing_ultra750_results.json (Nifty 750 Universe)
    - swing_ultraN750_results.json (Non-Nifty 750 Universe)
    """
    targets = {}
    
    sources = [
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultra750_results.json", "NIFTY 750"),
        ("https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/swing_ultraN750_results.json", "NON-N750")
    ]
    
    for url, default_univ in sources:
        try:
            cache_buster = f"?_ts={int(time.time() * 1000)}"
            req = urllib.request.Request(
                url + cache_buster,
                headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                d = json.loads(resp.read().decode("utf-8"))

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

                pivot = item.get("pivot_ceiling") or item.get("pivot_target") or item.get("pivot") or 0.0
                pivot = float(pivot)
                if pivot <= 0:
                    continue

                stop_loss = float(item.get("stop_loss", pivot * 0.92))
                setup = item.get("setup_class") or item.get("type") or "LAUNCHPAD COIL"
                universe = item.get("universe", default_univ)

                targets[sym] = {
                    "symbol": sym,
                    "universe": universe,
                    "setup": setup,
                    "pivot_ceiling": pivot,
                    "stop_loss": stop_loss
                }
            print(f"✅ Loaded {len(items)} setups from {url.split('/')[-1]} ({default_univ})")
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    # Retain continuous multi-day setups from prior local live state (persists until SL hit)
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
                        targets[sym] = {
                            "symbol": sym,
                            "universe": item.get("universe", "NON-N750"),
                            "setup": item.get("setup", "LAUNCHPAD COIL"),
                            "pivot_ceiling": pivot,
                            "stop_loss": stop_loss
                        }
        except Exception:
            pass

    print(f"📋 Total Active Live Watchlist: {len(targets)} symbols across N750 & Non-N750.")
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
    now_dt = datetime.datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    symbols = list(targets.keys())
    tickers_list = [f"{s}.NS" for s in symbols]

    print(f"⚡ Fetching market data for {len(tickers_list)} candidates (Elapsed: {elapsed_mins}m / 375m)...")
    
    # Download data with automatic fallback
    live_data = None
    try:
        live_data = yf.download(
            tickers_list,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        print(f"⚠️ yfinance bulk download error: {e}")

    confirmed_breakouts = []
    radar_candidates = []
    coiling_list = []
    new_alerts_to_send = []
    current_hits = {}

    for sym, meta in targets.items():
        yf_sym = f"{sym}.NS"
        pivot = meta["pivot_ceiling"]
        stop_loss = meta["stop_loss"]
        ltp = None
        day_high = None
        day_vol = 0.0
        sma20_vol = 1.0

        # Primary extraction from batch yfinance data
        if live_data is not None:
            try:
                df = live_data[yf_sym] if len(tickers_list) > 1 else live_data
                df = df.dropna(how="all").copy()
                if len(df) >= 1:
                    df.columns = [c.lower() for c in df.columns]
                    today = df.iloc[-1]
                    if pd.notnull(today.get("close")):
                        ltp = float(today["close"])
                        day_high = float(today["high"])
                        day_vol = float(today.get("volume", 0.0))
                    if len(df) > 1:
                        sma20_vol = float(df["volume"].iloc[:-1].tail(20).mean())
            except Exception:
                pass

        # Resilient fallback: Query individual ticker if batch extraction returned NaN
        if ltp is None or pd.isna(ltp):
            try:
                t = yf.Ticker(yf_sym)
                fast_info = getattr(t, "fast_info", None)
                if fast_info and getattr(fast_info, "last_price", None) is not None:
                    ltp = float(fast_info.last_price)
                    day_high = float(getattr(fast_info, "day_high", ltp))
                    day_vol = float(getattr(fast_info, "last_volume", 0.0))
            except Exception:
                pass

        # Secondary fallback: Read local EOD JSON file if available
        if ltp is None or pd.isna(ltp):
            local_json = os.path.join(DATA_DIR, f"{sym}.json")
            if os.path.exists(local_json):
                try:
                    with open(local_json, "r", encoding="utf-8") as fp:
                        raw = json.load(fp)
                    if raw and isinstance(raw, list):
                        last_bar = raw[-1]
                        ltp = float(last_bar.get("close", pivot))
                        day_high = float(last_bar.get("high", ltp))
                        day_vol = float(last_bar.get("volume", 0.0))
                except Exception:
                    pass

        # Final safeguard to eliminate NaN values
        if ltp is None or pd.isna(ltp) or ltp <= 0:
            ltp = pivot
            day_high = pivot

        if day_high is None or pd.isna(day_high):
            day_high = ltp

        sma20_vol = max(1.0, sma20_vol if (pd.notnull(sma20_vol) and sma20_vol > 0) else 50000.0)

        # Volume projection calculation
        expected_fraction = max(0.01, min(1.0, elapsed_mins / 375.0))
        projected_vol = day_vol / expected_fraction
        vol_pace = round(projected_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 8.0

        cleared_pivot = clearance_pct >= 0.25
        wick_touched_pivot = day_high >= pivot

        pace_threshold = 1.25 if meta["universe"] == "NIFTY 750" else 0.75
        is_volume_confirmed = vol_pace >= pace_threshold

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

        # Time-gated classification
        if elapsed_mins >= 330:  # 02:45 PM – 03:30 PM: Closing Execution Window
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

        else:  # 09:15 AM – 02:45 PM: Radar Monitoring Window
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

    for alert_type, alert in new_alerts_to_send:
        if alert_type == "RADAR":
            msg = (
                f"👀 <b>RADAR: PIVOT CEILING CLEARED</b>\n"
                f"• <b>Symbol</b>: <code>{alert['symbol']}</code> ({alert['universe']} | {alert['setup']})\n"
                f"• <b>Pivot Ceiling</b>: ₹{alert['pivot_ceiling']:,.2f}\n"
                f"• <b>Live LTP</b>: ₹{alert['ltp']:,.2f} (<b>{alert['clearance_%']:+.2f}%</b>)\n"
                f"• <b>Day High</b>: ₹{alert['day_high']:,.2f}\n"
                f"• <b>Volume Pace</b>: <b>{alert['proj_vol_pace']}x SMA20</b>\n"
                f"• <b>Stop Loss</b>: ₹{alert['stop_loss']:,.2f} (Risk: {alert['risk_%']}%)\n"
                f"📌 <i>Early clearance (+0.25%). Monitor depth into the 02:45–03:15 PM execution window.</i>"
            )
        else:
            msg = (
                f"🎯 <b>CONFIRMED CLOSING BREAKOUT TRIGGER</b>\n"
                f"• <b>Symbol</b>: <code>{alert['symbol']}</code> ({alert['universe']} | {alert['setup']})\n"
                f"• <b>Pivot Ceiling</b>: ₹{alert['pivot_ceiling']:,.2f}\n"
                f"• <b>Live LTP</b>: ₹{alert['ltp']:,.2f} (<b>{alert['clearance_%']:+.2f}%</b>)\n"
                f"• <b>Volume Pace</b>: <b>{alert['proj_vol_pace']}x SMA20</b>\n"
                f"• <b>Stop Loss</b>: ₹{alert['stop_loss']:,.2f} (Risk: {alert['risk_%']}%)\n"
                f"✅ <i>Holding +0.25% above pivot during closing window. Verified execution signal.</i>"
            )
        send_telegram_alert(msg)
        time.sleep(1)

    if elapsed_mins >= 365:
        summary_msg = (
            f"📊 <b>EOD BREAKOUT EXECUTION MONITOR</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
            f"<b>Active Confirmed Breakouts ({len(confirmed_breakouts)}):</b>\n"
        )
        if confirmed_breakouts:
            for b in confirmed_breakouts:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: ₹{b['ltp']:,.2f} ({b['clearance_%']:+.2f}%) | "
                    f"Pace: {b['proj_vol_pace']}x\n"
                )
        else:
            summary_msg += "<i>No candidates held above pivot with verified volume today.</i>\n"

        wicked_stocks = [c for c in coiling_list if c["action"] == "WICK REJECTED"][:5]
        if wicked_stocks:
            summary_msg += "\n<b>⚠️ Upper Wick Rejections (Failed to Hold +0.25%):</b>\n"
            for w in wicked_stocks:
                summary_msg += f"🔴 {w['symbol']}: High ₹{w['day_high']:,.2f} ➔ LTP ₹{w['ltp']:,.2f} ({w['clearance_%']:+.2f}%)\n"

        send_telegram_alert(summary_msg)

if __name__ == "__main__":
    run_live_intraday_scan()
