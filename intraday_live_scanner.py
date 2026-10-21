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
ACTIVE_PORTFOLIO_FILE = os.path.join(DATA_DIR, "active_portfolio_positions.json")

# Hardcoded Telegram credentials
TELEGRAM_BOT_TOKEN = "8796649927:AAHOx-1Fc_IyIttdQboQc0ul9qsM0g35O9c"
TELEGRAM_CHAT_ID = "7860290981"

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


def get_u_curve_expected_volume_fraction(elapsed_mins):
    """
    Intraday Volume U-Curve in Indian Equity Markets:
    - 0-75m   (09:15 - 10:30): Morning drive (~32% cumulative volume)
    - 75-225m (10:30 - 01:00): Midday lull (+28% -> 60% cumulative)
    - 225-330m(01:00 - 02:45): Pre-positioning (+20% -> 80% cumulative)
    - 330-375m(02:45 - 03:30): Closing institutional auction (+20% -> 100%)
    """
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


class LivePriceClient:
    """
    Fetches real-time market data reliably inside cloud runners.
    Primary: Google Finance (Zero delay, non-blocking)
    Fallback: Yahoo Finance JSON chart endpoint
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
                    day_low = ltp

                    range_match = re.search(r'Day range.*?₹([\d\.,]+)\s*-\s*₹([\d\.,]+)', html, re.DOTALL)
                    if range_match:
                        day_low = float(range_match.group(1).replace(",", ""))
                        day_high = float(range_match.group(2).replace(",", ""))

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
                        return {
                            "ltp": ltp,
                            "day_high": max(day_high, ltp),
                            "day_low": min(day_low, ltp),
                            "day_vol": day_vol
                        }
        except Exception:
            pass

        # 2. Fallback: Yahoo Finance Chart Endpoint
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                ltp = float(meta.get("regularMarketPrice", 0) or 0)
                day_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
                day_low = float(meta.get("regularMarketDayLow", ltp) or ltp)
                day_vol = float(meta.get("regularMarketVolume", 0) or 0)
                if ltp > 0:
                    return {
                        "ltp": ltp,
                        "day_high": max(day_high, ltp),
                        "day_low": min(day_low, ltp),
                        "day_vol": day_vol
                    }
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
                    spread_pct = float(item.get("spread_%", item.get("shelf_spread_%", 6.0)) or 6.0)

                    targets[sym] = {
                        "symbol": sym,
                        "universe": universe,
                        "setup": setup,
                        "pivot_ceiling": pivot,
                        "stop_loss": stop_loss,
                        "shelf_spread_%": spread_pct
                    }
                print(f"✅ Loaded {len(items)} setups from {url.split('/')[-1]} ({default_univ})")
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    # Fallback to locally cached watchlists
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
                            "stop_loss": stop_loss,
                            "shelf_spread_%": float(item.get("shelf_spread_%", 6.0))
                        }
        except Exception:
            pass

    print(f"📋 Total Active Live Watchlist: {len(targets)} setups across N750 & Non-N750.")
    return targets


def monitor_exit2_2_active_positions(live_client: LivePriceClient, elapsed_mins: int):
    """
    EXIT 2.2 ENGINE: Real-Time Intraday Risk Management for Active Positions
    1. Catastrophe SL
    2. Climactic Churn Harvest (Peak gain >= 20%, Vol Pace >= 2.0x, Range Closeness <= 0.35)
    3. Distribution Cluster (< 20 EMA & >= 3D Distribution)
    4. Mega-Runner 15% Cushion Lock (Peak gain >= 50%)
    5. 20 EMA Breach Engine (Heavy volume or VDU buffer failure)
    """
    if not os.path.exists(ACTIVE_PORTFOLIO_FILE):
        return

    try:
        with open(ACTIVE_PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            positions_data = json.load(f)
    except Exception:
        return

    active_positions = positions_data.get("Active_Positions", [])
    if not active_positions:
        return

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)
    updated_positions = []

    for pos in active_positions:
        sym = pos.get("symbol", "").upper()
        entry_price = float(pos.get("entry_price", 0.0))
        catastrophe_sl = float(pos.get("catastrophe_sl", entry_price * 0.92))
        ema20 = float(pos.get("ema_20", entry_price * 0.95))
        dist_count_12d = int(pos.get("dist_count_12d", 0))
        peak_gain_pct = float(pos.get("peak_gain_%", 0.0))
        peak_close = float(pos.get("peak_close", entry_price))
        vdu_under_count = int(pos.get("vdu_under_count", 0))
        sma20_vol = float(pos.get("sma20_vol", 50000.0))
        atr_14 = float(pos.get("atr_14", entry_price * 0.02))

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.05)

        if not quote:
            updated_positions.append(pos)
            continue

        ltp = quote["ltp"]
        day_high = quote["day_high"]
        day_low = quote["day_low"]
        day_vol = quote["day_vol"]

        # Track high water mark
        curr_gain_pct = ((ltp - entry_price) / entry_price) * 100.0
        peak_gain_pct = max(peak_gain_pct, curr_gain_pct)
        peak_close = max(peak_close, ltp)

        vol_pace = (day_vol / expected_fraction) / max(1.0, sma20_vol)
        bar_range = day_high - day_low
        range_closeness = (ltp - day_low) / bar_range if bar_range > 0 else 0.50

        exit_triggered = False
        exit_rule = ""

        # Rule 1: Catastrophe Stop
        if ltp < catastrophe_sl:
            exit_triggered = True
            exit_rule = f"1. Catastrophe SL Breach (LTP ₹{ltp:,.2f} < ₹{catastrophe_sl:,.2f})"

        # Rule 2: Climactic Churn Harvest
        elif peak_gain_pct >= 20.0 and vol_pace >= 2.0 and range_closeness <= 0.35:
            exit_triggered = True
            exit_rule = f"2. Climactic Churn Harvest (Peak +{peak_gain_pct:.1f}%, Vol Pace {vol_pace:.1f}x, Closeness {range_closeness:.2f})"

        # Rule 3: Distribution Cluster (>= 3D & Close < 20 EMA)
        elif dist_count_12d >= 3 and ltp < ema20:
            exit_triggered = True
            exit_rule = f"3. Distribution Cluster Breakdown ({dist_count_12d} Dist Days & LTP < 20 EMA)"

        # Rule 4: Mega-Runner 15% Cushion Lock
        elif peak_gain_pct >= 50.0:
            cushion_stop = max(ema20, peak_close * 0.85)
            if ltp < cushion_stop:
                exit_triggered = True
                exit_rule = f"4. Mega-Runner 15% Cushion Lock (LTP ₹{ltp:,.2f} < ₹{cushion_stop:,.2f})"

        # Rule 5: 20 EMA Breach Engine
        elif ltp < ema20:
            if vol_pace >= 1.25:
                exit_triggered = True
                exit_rule = f"5A. Heavy Volume 20 EMA Breach (Pace: {vol_pace:.1f}x SMA20)"
            else:
                vdu_under_count += 1
                atr_floor = ema20 - (1.5 * atr_14)
                pct_floor = ema20 * 0.95
                dynamic_leeway = min(pct_floor, atr_floor)
                guaranteed_be = entry_price * 1.015
                active_vdu_stop = max(guaranteed_be, dynamic_leeway)

                if ltp < active_vdu_stop:
                    exit_triggered = True
                    exit_rule = f"5B. Dynamic VDU Floor Breach (LTP ₹{ltp:,.2f} < Floor ₹{active_vdu_stop:,.2f})"
                elif vdu_under_count > 5:
                    exit_triggered = True
                    exit_rule = "5B. 5-Session VDU Reclaim Failure"
        else:
            vdu_under_count = 0

        if exit_triggered and not pos.get("exit_alert_sent", False):
            pnl_pct = round(curr_gain_pct, 2)
            exit_msg = (
                f"🚨 <b>EXIT 2.2 TRIGGER: CLOSE POSITION</b>\n"
                f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                f"• <b>Reason</b>: <b>{exit_rule}</b>\n"
                f"• <b>Live LTP</b>: ₹{ltp:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
                f"• <b>Entry Price</b>: ₹{entry_price:,.2f}\n"
                f"• <b>High Water Mark</b>: +{peak_gain_pct:.2f}%\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🛑 <i>Action: Execute square-off order immediately.</i>"
            )
            send_telegram_alert(exit_msg)
            pos["exit_alert_sent"] = True
            pos["exit_price"] = ltp
            pos["exit_reason"] = exit_rule

        pos["peak_gain_%"] = round(peak_gain_pct, 2)
        pos["peak_close"] = round(peak_close, 2)
        pos["vdu_under_count"] = vdu_under_count
        updated_positions.append(pos)

    positions_data["Active_Positions"] = updated_positions
    try:
        with open(ACTIVE_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(positions_data, f, indent=2)
    except Exception:
        pass


def run_live_intraday_scan():
    targets = load_comprehensive_targets()
    if not targets:
        print("❌ No watchlisted setups found to scan.")
        return

    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    # Load persistent state
    alert1_sent = set()
    alert2_sent = set()
    consecutive_hits = {}
    prior_alert_highs = {}
    eod_summary_sent = False

    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if prev.get("Date") == today_date:
                    alert1_sent = set(prev.get("Alert1_Sent", []))
                    alert2_sent = set(prev.get("Alert2_Sent", []))
                    consecutive_hits = prev.get("Consecutive_Hits", {})
                    prior_alert_highs = prev.get("Prior_Alert_Highs", {})
                    eod_summary_sent = prev.get("EOD_Summary_Sent", False)
        except Exception:
            pass

    print(f"⚡ Streaming Real-Time Quotes (Elapsed: {elapsed_mins}m / 375m IST)...")
    live_client = LivePriceClient()

    # Run Exit 2.2 risk management on active portfolio positions
    monitor_exit2_2_active_positions(live_client, elapsed_mins)

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    confirmed_breakouts = []
    radar_candidates = []
    coiling_list = []
    current_consecutive_hits = {}

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        stop_loss = float(meta["stop_loss"])
        universe = meta["universe"]
        setup = meta["setup"]
        shelf_spread = float(meta.get("shelf_spread_%", 6.0))

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.06)

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
                        day_low = float(raw[-1].get("low", last_c))
            except Exception:
                pass

        if ltp is None or pd.isna(ltp) or ltp <= 0:
            ltp = pivot
            day_high = pivot
            day_low = pivot

        if day_high is None or pd.isna(day_high) or day_high < ltp:
            day_high = ltp
        if day_low is None or pd.isna(day_low) or day_low > ltp:
            day_low = ltp

        sma20_vol = max(1.0, sma20_vol)
        projected_vol = day_vol / expected_fraction
        vol_pace = round(projected_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 8.0

        # Required volume pace
        is_v_reversal = "V-REVERSAL" in setup
        if is_v_reversal:
            pace_threshold = 1.75 if universe == "NIFTY 750" else 1.50
        else:
            pace_threshold = 1.25 if universe == "NIFTY 750" else 0.85
            if shelf_spread > 7.0:
                pace_threshold += 0.30

        is_vol_confirmed = bool(vol_pace >= pace_threshold)
        cleared_pivot = bool(clearance_pct >= 0.25)
        wick_touched_pivot = bool(day_high >= pivot)

        # Persistence Tracker
        hits = (consecutive_hits.get(sym, 0) + 1) if cleared_pivot else 0
        current_consecutive_hits[sym] = hits

        # Dynamic Persistence: 1 tick with surge volume (>= 2.5x), else 2 ticks
        if vol_pace >= 2.50:
            is_persistent = bool(hits >= 1)
        else:
            is_persistent = bool(hits >= 2)

        record = {
            "symbol": sym,
            "universe": universe,
            "setup": setup,
            "pivot_ceiling": round(pivot, 2),
            "ltp": round(ltp, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct,
            "clearance_%": clearance_pct,
            "proj_vol_pace": vol_pace,
            "consecutive_ticks": hits
        }

        # ----------------------------------------------------------------------
        # ALERT 1: Initial Pivot Clearance (Zero Delay, Fires Strictly ONCE)
        # ----------------------------------------------------------------------
        if cleared_pivot and is_vol_confirmed and is_persistent:
            record["action"] = "RADAR PIVOT CLEARANCE"
            radar_candidates.append(record)

            if sym not in alert1_sent:
                msg1 = (
                    f"👀 <b>RADAR: PIVOT CEILING CLEARED (ZERO DELAY)</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code> ({universe} | {setup})\n"
                    f"• <b>Pivot Ceiling</b>: ₹{pivot:,.2f}\n"
                    f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}%</b>)\n"
                    f"• <b>Day High</b>: ₹{day_high:,.2f}\n"
                    f"• <b>Volume Pace</b>: <b>{vol_pace}x SMA20</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f} (Risk: {risk_pct}%)\n"
                    f"📌 <i>Alert 1 dispatched. Watching for day-high continuation into 03:15 PM window.</i>"
                )
                send_telegram_alert(msg1)
                alert1_sent.add(sym)
                prior_alert_highs[sym] = day_high
                time.sleep(1)

        # ----------------------------------------------------------------------
        # ALERT 2: Day High Break Continuation (Fires ONLY when Day High is Crossed)
        # ----------------------------------------------------------------------
        if sym in alert1_sent and sym not in alert2_sent:
            reference_high = float(prior_alert_highs.get(sym, day_high))
            # Must cross the prior high watermark by at least +0.20%
            if ltp > (reference_high * 1.002):
                msg2 = (
                    f"🚀 <b>ALERT 2: DAY HIGH BREAK (MOMENTUM EXPANSION)</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code> ({universe} | {setup})\n"
                    f"• <b>Prior Day High Broken</b>: ₹{reference_high:,.2f}\n"
                    f"• <b>New Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}% above pivot</b>)\n"
                    f"• <b>New Day High</b>: ₹{day_high:,.2f}\n"
                    f"• <b>Volume Pace</b>: <b>{vol_pace}x SMA20</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f}\n"
                    f"⚡ <i>Momentum continuation confirmed: Buyers absorbing overhead resistance.</i>"
                )
                send_telegram_alert(msg2)
                alert2_sent.add(sym)
                prior_alert_highs[sym] = day_high
                time.sleep(1)

        # Closing Hour Evaluation (02:45 PM - 03:30 PM IST)
        if elapsed_mins >= 330:
            if cleared_pivot and is_vol_confirmed:
                record["action"] = "CONFIRMED BREAKOUT"
                confirmed_breakouts.append(record)
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK REJECTED"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)
        else:
            if not cleared_pivot:
                record["action"] = "WICK TESTING" if wick_touched_pivot else "COILING"
                coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    confirmed_breakouts.sort(key=lambda x: x["proj_vol_pace"], reverse=True)
    radar_candidates.sort(key=lambda x: x["proj_vol_pace"], reverse=True)

    # --------------------------------------------------------------------------
    # 03:15 PM EOD Execution Window Summary (Dispatched strictly ONCE)
    # --------------------------------------------------------------------------
    if elapsed_mins >= 360 and not eod_summary_sent:
        summary_msg = (
            f"🎯 <b>CONFIRMED CLOSING BREAKOUTS (3:15 PM EXECUTION)</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
        )
        if confirmed_breakouts:
            summary_msg += f"<b>Actionable Entries ({len(confirmed_breakouts)}):</b>\n"
            for b in confirmed_breakouts:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: LTP ₹{b['ltp']:,.2f} (+{b['clearance_%']}%) | "
                    f"Pace: {b['proj_vol_pace']}x | SL: ₹{b['stop_loss']:,.2f}\n"
                )
            summary_msg += "\n✅ <i>Place CNC delivery orders before 03:28 PM IST.</i>\n"
        else:
            summary_msg += "<i>No candidates held above pivot with verified volume today.</i>\n"

        wicked = [c for c in coiling_list if c["action"] == "WICK REJECTED"][:5]
        if wicked:
            summary_msg += "\n<b>⚠️ Upper Wick Rejections (Avoid):</b>\n"
            for w in wicked:
                summary_msg += f"🔴 {w['symbol']}: High ₹{w['day_high']:,.2f} ➔ LTP ₹{w['ltp']:,.2f} ({w['clearance_%']:+.2f}%)\n"

        send_telegram_alert(summary_msg)
        eod_summary_sent = True

    output_payload = {
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "EOD_Summary_Sent": eod_summary_sent,
        "Alert1_Sent": list(alert1_sent),
        "Alert2_Sent": list(alert2_sent),
        "Prior_Alert_Highs": prior_alert_highs,
        "Consecutive_Hits": current_consecutive_hits,
        "Breakouts_All_Day": confirmed_breakouts if elapsed_mins >= 330 else radar_candidates,
        "Coiling": coiling_list[:50]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ Scan completed: {len(confirmed_breakouts)} Confirmed, {len(radar_candidates)} Radar, {len(coiling_list)} Coiling.")


if __name__ == "__main__":
    run_live_intraday_scan()
