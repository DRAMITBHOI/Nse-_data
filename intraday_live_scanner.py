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
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
ACTIVE_WATCHLIST_FILE = os.path.join(DATA_DIR, "active_watchlist.json")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    secrets_path = "local_secrets.json"
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                TELEGRAM_BOT_TOKEN = cfg.get("TELEGRAM_BOT_TOKEN", cfg.get("8659551805:AAGc436Ii25qrFvcrxh2Iu4kDbVRxKu344k", TELEGRAM_BOT_TOKEN))
                TELEGRAM_CHAT_ID = cfg.get("TELEGRAM_CHAT_ID", cfg.get("7860290981", TELEGRAM_CHAT_ID))
        except Exception:
            pass

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}


def get_ist_now():
    return datetime.datetime.now(IST)


def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[SIMULATION ALERT]:\n{message}\n")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=12)
        if not r.json().get("ok"):
            print(f"❌ Telegram API Error: {r.text}")
    except Exception as e:
        print(f"❌ Telegram Dispatch Error: {e}")


def get_market_session_elapsed():
    now = get_ist_now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 0
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


class LivePriceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def fetch_live_quote(self, symbol):
        # 1. Google Finance Primary
        try:
            url = f"https://www.google.com/finance/quote/{symbol}:NSE"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                html = r.text
                p_m = re.search(r'data-last-price="([\d\.]+)"', html)
                if p_m:
                    ltp = float(p_m.group(1))
                    d_high, d_low, d_open = ltp, ltp, ltp
                    rng_m = re.search(r'Day range.*?₹([\d\.,]+)\s*-\s*₹([\d\.,]+)', html, re.DOTALL)
                    if rng_m:
                        d_low = float(rng_m.group(1).replace(",", ""))
                        d_high = float(rng_m.group(2).replace(",", ""))

                    d_vol = 0.0
                    vol_m = re.search(r'Volume</div><div class="P6K39c">([\d\.,MKmk]+)</div>', html)
                    if vol_m:
                        v_str = vol_m.group(1).replace(",", "").upper()
                        if "M" in v_str:
                            d_vol = float(v_str.replace("M", "")) * 1_000_000
                        elif "K" in v_str:
                            d_vol = float(v_str.replace("K", "")) * 1_000
                        else:
                            d_vol = float(v_str)

                    if ltp > 0:
                        return {
                            "ltp": ltp,
                            "day_open": d_open,
                            "day_high": max(d_high, ltp),
                            "day_low": min(d_low, ltp),
                            "day_vol": d_vol
                        }
        except Exception:
            pass

        # 2. Yahoo Finance Fallback
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
            r = self.session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                ltp = float(meta.get("regularMarketPrice", 0) or 0)
                d_open = float(meta.get("regularMarketDayOpen", ltp) or ltp)
                d_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
                d_low = float(meta.get("regularMarketDayLow", ltp) or ltp)
                d_vol = float(meta.get("regularMarketVolume", 0) or 0)
                if ltp > 0:
                    return {
                        "ltp": ltp,
                        "day_open": d_open,
                        "day_high": max(d_high, ltp),
                        "day_low": min(d_low, ltp),
                        "day_vol": d_vol
                    }
        except Exception:
            pass
        return None


def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()
        if not raw_t:
            continue
        d_str = raw_t[:10]
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0:
                continue
            entry = {
                "time": d_str,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0),
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]
    return clean


def monitor_combination7_active_positions(live_client: LivePriceClient, elapsed_mins: int):
    """
    Monitors open positions in data/my_trades.json according to Combination 7 rules:
    - Phase 1 (Pre-+1.5R): Stop = Initial Hard SL (Cheat Low / V-Reversal Low - 0.5%)
    - Milestone (+1.5R): Book 20% partial profit, move Stop to Entry + 0.5%
    - Phase 2 (Post-+1.5R): 80% runner trails Daily 20 EMA
    - Integrated Bifurcated Gap-Down Protocol (9:15 AM severe cut vs 9:30 AM leeway)
    """
    if not os.path.exists(MY_TRADES_FILE):
        return
    try:
        with open(MY_TRADES_FILE, "r", encoding="utf-8") as f:
            trades = json.load(f)
    except Exception:
        return

    if not isinstance(trades, list) or not trades:
        return

    updated_trades = []

    for t in trades:
        if str(t.get("status", "OPEN")).upper() != "OPEN":
            updated_trades.append(t)
            continue

        sym = str(t.get("symbol", "")).upper().strip()
        entry_price = float(t.get("entry_price", 0.0))
        initial_sl = float(t.get("hard_sl", t.get("stop_loss", entry_price * 0.955)))
        hit_milestone = bool(t.get("hit_milestone", False))
        active_stop = float(t.get("active_stop", initial_sl))
        peak_gain_pct = float(t.get("peak_gain_%", 0.0))

        # Milestone: +1.5R calculation
        risk_pct = ((entry_price - initial_sl) / entry_price) * 100.0 if entry_price > initial_sl else 3.5
        t1_target_pct = min(1.5 * risk_pct, 6.0)
        t1_target_price = round(entry_price * (1.0 + t1_target_pct / 100.0), 2)

        # Indicator extraction from daily history
        ema20 = entry_price * 0.95
        ema50 = entry_price * 0.90
        sym_json = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(sym_json):
            try:
                with open(sym_json, "r", encoding="utf-8") as fp:
                    raw_data = json.load(fp)
                clean = clean_and_prepare(raw_data)
                if len(clean) >= 20:
                    df_hist = pd.DataFrame(clean)
                    ema20 = float(df_hist["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    if len(clean) >= 50:
                        ema50 = float(df_hist["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            except Exception:
                pass

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.04)

        if not quote:
            updated_trades.append(t)
            continue

        ltp = quote["ltp"]
        day_open = quote.get("day_open", ltp)
        day_high = quote["day_high"]

        curr_gain_pct = ((ltp - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        peak_gain_pct = max(peak_gain_pct, curr_gain_pct)

        # Milestone Check (+1.5R)
        if not hit_milestone and (day_high >= t1_target_price or ltp >= t1_target_price):
            hit_milestone = True
            active_stop = max(active_stop, round(entry_price * 1.005, 2))
            t["hit_milestone"] = True
            t["active_stop"] = active_stop
            if not t.get("t1_alert_sent", False):
                t1_msg = (
                    f"🎯 <b>COMBINATION 7: BOOK 20% PROFIT (+1.5R HIT)</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                    f"• <b>Target Price</b>: ₹{t1_target_price:,.2f} (+{t1_target_pct:.2f}%)\n"
                    f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{curr_gain_pct:+.2f}%</b>)\n"
                    f"• <b>Action</b>: Harvest 20% partial profit. Trail 80% runner on 20 EMA.\n"
                    f"• <b>Ratcheted Stop</b>: ₹{active_stop:,.2f} (Entry + 0.5% Guaranteed Safe)"
                )
                send_telegram_alert(t1_msg)
                t["t1_alert_sent"] = True

        if hit_milestone:
            active_stop = max(active_stop, ema20)

        # ======================================================================
        # BIFURCATED GAP-DOWN DEFENSE & INTRADAY STOP CHECKS
        # ======================================================================
        exit_triggered = False
        exit_rule = ""

        # Check Overnight Gap-Down if checked in first 20 minutes
        if elapsed_mins <= 20 and day_open < active_stop:
            gap_depth_pct = ((active_stop - day_open) / active_stop) * 100.0
            is_severe = (gap_depth_pct > 2.0) or (day_open < ema50)

            if is_severe:
                exit_triggered = True
                exit_rule = f"Gap-Down Severe Breach (>2% or <50 EMA) @ ₹{day_open:,.2f}"
            else:
                # Shallow gap holding 50 EMA: 15-minute leeway
                if elapsed_mins >= 15:
                    if ltp < active_stop:
                        exit_triggered = True
                        exit_rule = f"Gap-Down 15-Min Leeway Reclaim Failed @ ₹{ltp:,.2f}"
                else:
                    t["status_note"] = "Under 15-min gap-down observation (holding above 50 EMA)"

        # Normal Stop Loss Check (if not exited on gap)
        if not exit_triggered:
            if not hit_milestone and ltp <= active_stop:
                exit_triggered = True
                exit_rule = f"Hard Stop Breached (LTP ₹{ltp:,.2f} <= ₹{active_stop:,.2f})"
            elif hit_milestone and ltp < ema20:
                # For runner, check 20 EMA close / breakdown
                exit_triggered = True
                exit_rule = f"Runner 20 EMA Breakdown (LTP ₹{ltp:,.2f} < 20 EMA ₹{ema20:,.2f})"

        if exit_triggered and not t.get("exit_alert_sent", False):
            pnl_pct = round(curr_gain_pct, 2)
            exit_msg = (
                f"🚨 <b>COMBINATION 7 EXIT TRIGGER: SQUARE OFF</b>\n"
                f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                f"• <b>Reason</b>: <b>{exit_rule}</b>\n"
                f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
                f"• <b>Entry Price</b>: ₹{entry_price:,.2f}\n"
                f"• <b>Active Stop</b>: ₹{active_stop:,.2f}\n"
                f"• <b>20 EMA Floor</b>: ₹{ema20:,.2f}\n"
                f"• <b>Peak Unrealized</b>: +{peak_gain_pct:.2f}%\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🛑 <i>Action: Execute square-off order immediately.</i>"
            )
            send_telegram_alert(exit_msg)
            t["exit_alert_sent"] = True
            t["exit_price"] = ltp
            t["exit_reason"] = exit_rule
            t["status"] = "CLOSED"

        t["peak_gain_%"] = round(peak_gain_pct, 2)
        t["active_stop"] = round(active_stop, 2)
        updated_trades.append(t)

    try:
        with open(MY_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_trades, f, indent=2)
    except Exception:
        pass


def load_active_prebreakout_targets():
    """
    Loads candidates directly from active_watchlist.json (and fallback to swing_ultra results).
    Persists candidates indefinitely unless structural floor is violated.
    """
    targets = {}

    # Primary: active_watchlist.json
    if os.path.exists(ACTIVE_WATCHLIST_FILE):
        try:
            with open(ACTIVE_WATCHLIST_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                for sym, item in data.items():
                    targets[sym.upper()] = item
            elif isinstance(data, list):
                for item in data:
                    sym = str(item.get("symbol", "")).upper().strip()
                    if sym:
                        targets[sym] = item
        except Exception:
            pass

    # Secondary: fallback to swing_ultra750_results.json and swing_ultraN750_results.json
    for fpath in ["swing_ultra750_results.json", "swing_ultraN750_results.json"]:
        full_p = os.path.join(DATA_DIR, fpath)
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                for item in d.get("Watchlist", []):
                    sym = str(item.get("symbol", "")).upper().strip()
                    if sym and sym not in targets:
                        targets[sym] = item
            except Exception:
                pass

    return {k: v for k, v in targets.items() if k not in EXCLUDED_SYMBOLS}


def run_live_intraday_scan():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    live_client = LivePriceClient()

    # Step 1: Monitor Active Positions for Combination 7 SL / Milestone / 20 EMA Runner Exits
    monitor_combination7_active_positions(live_client, elapsed_mins)

    # Step 2: Load Active Pre-Breakout Targets
    targets = load_active_prebreakout_targets()
    if not targets:
        print("❌ No active pre-breakout candidates found in watchlist.")
        return

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    alert1_sent = set()
    alert2_sent = set()
    benchmark_highs = {}
    master_confirmed_breakouts = {}
    eod_summary_sent = False

    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("Date") == today_date:
                alert1_sent = set(prev.get("Alert1_Sent", []))
                alert2_sent = set(prev.get("Alert2_Sent", []))
                benchmark_highs = prev.get("Benchmark_Highs", {})
                master_confirmed_breakouts = prev.get("Master_Confirmed_Breakouts", {})
                eod_summary_sent = prev.get("EOD_Summary_Sent", False)
        except Exception:
            pass

    radar_candidates = []
    coiling_list = []
    persisted_watchlist = {}

    for sym, meta in targets.items():
        pivot = float(meta.get("pivot_ceiling", 0.0))
        stop_loss = float(meta.get("stop_loss", pivot * 0.955))
        universe = meta.get("universe", "NIFTY 750")
        setup = meta.get("setup", "LAUNCHPAD CHEAT")
        max_allowed_open = float(meta.get("max_allowed_open", round(pivot * 1.02, 2)))

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.04)

        ltp = None
        day_open = None
        day_high = None
        day_low = None
        day_vol = 0.0
        sma20_vol = 50000.0

        if quote:
            ltp = quote["ltp"]
            day_open = quote.get("day_open", ltp)
            day_high = quote["day_high"]
            day_low = quote["day_low"]
            day_vol = quote["day_vol"]

        local_json = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(local_json):
            try:
                with open(local_json, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) >= 20:
                    vols = [float(x.get("volume", 0) or 0) for x in clean[-20:]]
                    if vols:
                        sma20_vol = sum(vols) / len(vols)
                    if ltp is None or ltp <= 0:
                        last_c = float(clean[-1].get("close", pivot))
                        ltp = last_c
                        day_open = float(clean[-1].get("open", last_c))
                        day_high = float(clean[-1].get("high", last_c))
                        day_low = float(clean[-1].get("low", last_c))
            except Exception:
                pass

        if ltp is None or ltp <= 0:
            ltp = pivot
            day_open = pivot
            day_high = pivot
            day_low = pivot

        day_high = max(day_high or ltp, ltp)
        day_low = min(day_low or ltp, ltp)
        day_open = day_open or ltp

        # PERSISTENCE & PRUNING: Only prune if structural base stop floor is breached
        if ltp < stop_loss:
            continue

        persisted_watchlist[sym] = meta

        sma20_vol = max(1.0, sma20_vol)
        proj_vol = day_vol / expected_fraction
        vol_pace = round(proj_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 3.5

        # Volume Pace Threshold
        pace_threshold = 1.20 if universe == "NIFTY 750" else 0.90
        is_vol_confirmed = bool(vol_pace >= pace_threshold)

        # ======================================================================
        # EMPIRICAL GAP-UP & BREAKOUT ACTION PLAN
        # ======================================================================
        opening_gap_pct = round(((day_open - pivot) / pivot) * 100.0, 2) if pivot > 0 else 0.0

        is_breakout_valid = False
        disqualification_reason = ""

        # Tier 3: Severe Extension (> +2.0% Gap-Up) -> AVOID / DISQUALIFIED
        if day_open > max_allowed_open or opening_gap_pct > 2.0:
            disqualification_reason = f"Extended Gap-Up (+{opening_gap_pct:.1f}% > +2.0% Cap - 60% Fade Risk)"
            is_breakout_valid = False

        # Tier 2: Modest Gap (0.0% < Gap <= +2.0%) -> 9:20 AM Confirmation
        elif opening_gap_pct > 0.0:
            if elapsed_mins < 5:
                disqualification_reason = f"Modest Gap (+{opening_gap_pct:.1f}%): Awaiting 9:20 AM Opening Range High"
                is_breakout_valid = False
            else:
                # Requires LTP pushing above the opening range
                if ltp > day_open and ltp >= pivot and is_vol_confirmed:
                    is_breakout_valid = True

        # Tier 1: Normal Open (Open <= Pivot) -> Standard Pivot Clearance
        else:
            if ltp >= pivot and is_vol_confirmed:
                is_breakout_valid = True

        wick_touched = bool(day_high >= pivot)

        if wick_touched or is_breakout_valid:
            if sym not in benchmark_highs:
                benchmark_highs[sym] = round(day_high, 2)
            else:
                benchmark_highs[sym] = round(max(benchmark_highs[sym], day_high), 2)

        record = {
            "symbol": sym,
            "universe": universe,
            "setup": setup,
            "pivot_ceiling": round(pivot, 2),
            "ltp": round(ltp, 2),
            "day_open": round(day_open, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct,
            "opening_gap_%": opening_gap_pct,
            "clearance_%": clearance_pct,
            "proj_vol_pace": vol_pace,
            "disqualification_reason": disqualification_reason
        }

        # ALERT 1: CONFIRMED BREAKOUT TRIGGER
        if is_breakout_valid:
            record["action"] = "CONFIRMED BREAKOUT"
            radar_candidates.append(record)
            master_confirmed_breakouts[sym] = record

            if sym not in alert1_sent:
                ref_h = benchmark_highs.get(sym, day_high)
                msg1 = (
                    f"👀 <b>RADAR: COMBINATION 7 BREAKOUT TRIGGERED</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code> ({universe} | {setup})\n"
                    f"• <b>Pivot Ceiling</b>: ₹{pivot:,.2f}\n"
                    f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}%</b>)\n"
                    f"• <b>Day Open</b>: ₹{day_open:,.2f} (Gap: {opening_gap_pct:+.2f}%)\n"
                    f"• <b>Volume Pace</b>: <b>{vol_pace}x SMA20</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f} (Risk: {risk_pct}%)\n"
                    f"• <b>T1 (+1.5R) Target</b>: ₹{round(ltp * (1.0 + min(1.5*risk_pct, 6.0)/100.0), 2):,.2f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <i>Alert 1 dispatched. Tracking for continuation above ₹{ref_h:,.2f}.</i>"
                )
                send_telegram_alert(msg1)
                alert1_sent.add(sym)
                time.sleep(1)

        # ALERT 2: DAY-HIGH EXPANSION
        if sym not in alert2_sent and sym in benchmark_highs:
            prior_ref_high = float(benchmark_highs[sym])
            if ltp >= (prior_ref_high * 1.0015) and ltp > pivot and is_vol_confirmed and not disqualification_reason:
                record["action"] = "CONFIRMED BREAKOUT"
                master_confirmed_breakouts[sym] = record

                msg2 = (
                    f"🚀 <b>2ND ALERT: DAY HIGH BREAK (EXPANSION)</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code> ({universe} | {setup})\n"
                    f"• <b>Prior Day High Broken</b>: ₹{prior_ref_high:,.2f}\n"
                    f"• <b>New Live NSE LTP</b>: ₹{ltp:,.2f} (<b>+{round(((ltp - pivot)/pivot)*100, 2):.2f}%</b>)\n"
                    f"• <b>New Day High</b>: ₹{day_high:,.2f}\n"
                    f"• <b>Volume Pace</b>: <b>{vol_pace}x SMA20</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f}\n"
                    f"⚡ <i>Momentum expansion confirmed: Resistance cleared!</i>"
                )
                send_telegram_alert(msg2)
                alert2_sent.add(sym)
                benchmark_highs[sym] = round(day_high, 2)
                time.sleep(1)

        if not is_breakout_valid:
            if disqualification_reason:
                record["action"] = "DISQUALIFIED / EXTENDED"
            elif wick_touched:
                record["action"] = "WICK REJECTED"
            else:
                record["action"] = "COILING"
            coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    final_confirmed = list(master_confirmed_breakouts.values())

    # 03:15 PM EOD SUMMARY
    if elapsed_mins >= 360 and not eod_summary_sent:
        summary_msg = (
            f"🎯 <b>CONFIRMED CLOSING BREAKOUTS (3:15 PM EXECUTION)</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
        )
        if final_confirmed:
            summary_msg += f"<b>Actionable Entries ({len(final_confirmed)}):</b>\n"
            for b in final_confirmed:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: LTP ₹{b['ltp']:,.2f} (+{b['clearance_%']}%) | "
                    f"Pace: {b['proj_vol_pace']}x | SL: ₹{b['stop_loss']:,.2f}\n"
                )
            summary_msg += "\n✅ <i>Place CNC delivery orders before 03:28 PM IST.</i>\n"
        else:
            summary_msg += "<i>No candidates held above pivot with verified volume today.</i>\n"

        send_telegram_alert(summary_msg)
        eod_summary_sent = True

    output_payload = {
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "EOD_Summary_Sent": eod_summary_sent,
        "Alert1_Sent": list(alert1_sent),
        "Alert2_Sent": list(alert2_sent),
        "Benchmark_Highs": benchmark_highs,
        "Master_Confirmed_Breakouts": master_confirmed_breakouts,
        "Breakouts_All_Day": final_confirmed,
        "Active_MultiDay_Watchlist": persisted_watchlist,
        "Coiling": coiling_list[:50]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Save persisted active watchlist back
    with open(ACTIVE_WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(persisted_watchlist, f, indent=2)

    print(f"✅ Combination 7 Live Scan Complete: {len(final_confirmed)} Confirmed, {len(coiling_list)} Coils/Observed.")


if __name__ == "__main__":
    run_live_intraday_scan()
