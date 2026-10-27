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
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")

# Secure Token Retrieval: Env Secrets first -> local_secrets.json fallback
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    secrets_path = "local_secrets.json"
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                TELEGRAM_BOT_TOKEN = cfg.get("8659551805:AAGc436Ii25qrFvcrxh2Iu4kDbVRxKu344k", TELEGRAM_BOT_TOKEN)
                TELEGRAM_CHAT_ID = cfg.get("7860290981", TELEGRAM_CHAT_ID)
        except Exception:
            pass

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}
WATCHLIST_MAX_AGE_DAYS = 5


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
                    d_high, d_low = ltp, ltp
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
                        return {"ltp": ltp, "day_high": max(d_high, ltp), "day_low": min(d_low, ltp), "day_vol": d_vol}
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
        raw_t = str(r.get("time", "")).strip()
        if not raw_t: continue
        d_str = raw_t[:10]
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0: continue
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
    running_vol = 50000.0
    for i in range(len(clean)):
        v = clean[i]["volume"]
        dv = clean[i]["delivery_vol"]
        pct = clean[i]["deliv_pct"]
        if v > 0:
            running_vol = 0.9 * running_vol + 0.1 * v
        else:
            clean[i]["volume"] = running_vol
            v = running_vol
        if dv <= 0:
            clean[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            clean[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            clean[i]["delivery_vol"] = v
            clean[i]["deliv_pct"] = 100.0
    return clean


def monitor_exit2_2_active_positions(live_client: LivePriceClient, elapsed_mins: int):
    """Monitors open positions in data/my_trades.json against the 5 Exit 2.2 rules."""
    if not os.path.exists(MY_TRADES_FILE):
        return
    try:
        with open(MY_TRADES_FILE, "r", encoding="utf-8") as f:
            trades = json.load(f)
    except Exception:
        return

    if not isinstance(trades, list) or not trades:
        return

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)
    updated_trades = []

    for t in trades:
        if str(t.get("status", "OPEN")).upper() != "OPEN":
            updated_trades.append(t)
            continue

        sym = str(t.get("symbol", "")).upper().strip()
        entry_price = float(t.get("entry_price", 0.0))
        catastrophe_sl = float(t.get("hard_sl", entry_price * 0.92))
        peak_gain_pct = float(t.get("peak_gain_%", 0.0))
        peak_close = float(t.get("peak_close", entry_price))
        vdu_under_count = int(t.get("vdu_under_count", 0))

        ema20 = entry_price * 0.95
        sma20_vol = 50000.0
        atr_14 = entry_price * 0.02
        dist_count_12d = 0

        sym_json = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(sym_json):
            try:
                with open(sym_json, "r", encoding="utf-8") as fp:
                    raw_data = json.load(fp)
                clean = clean_and_prepare(raw_data)
                if len(clean) >= 20:
                    df_hist = pd.DataFrame(clean)
                    ema20 = float(df_hist["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    vols = df_hist["volume"].iloc[-20:].tolist()
                    sma20_vol = float(sum(vols) / len(vols)) if vols else 50000.0
                    prev_c = df_hist["close"].shift(1)
                    tr = pd.concat([df_hist["high"] - df_hist["low"], (df_hist["high"] - prev_c).abs(), (df_hist["low"] - prev_c).abs()], axis=1).max(axis=1)
                    atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
                    dv_sma = df_hist["delivery_vol"].rolling(20, min_periods=1).mean().iloc[-1]
                    dp_avg = df_hist["deliv_pct"].rolling(50, min_periods=5).mean().iloc[-1]
                    c_rng = df_hist["high"] - df_hist["low"]
                    rc = np.where(c_rng > 0, (df_hist["close"] - df_hist["low"]) / c_rng, 0.5)
                    is_dist = (
                        (df_hist["close"] < df_hist["close"].shift(1)) &
                        ((df_hist["delivery_vol"] >= 1.35 * dv_sma) | (df_hist["deliv_pct"] >= 1.15 * dp_avg)) &
                        (rc <= 0.45)
                    ).astype(int)
                    dist_count_12d = int(is_dist.rolling(12, min_periods=1).sum().iloc[-1])
            except Exception:
                pass

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.04)

        if not quote:
            updated_trades.append(t)
            continue

        ltp = quote["ltp"]
        day_high = quote["day_high"]
        day_low = quote["day_low"]
        day_vol = quote["day_vol"]

        curr_gain_pct = ((ltp - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        peak_gain_pct = max(peak_gain_pct, curr_gain_pct)
        peak_close = max(peak_close, ltp)

        vol_pace = (day_vol / expected_fraction) / max(1.0, sma20_vol)
        bar_range = day_high - day_low
        range_closeness = (ltp - day_low) / bar_range if bar_range > 0 else 0.50

        exit_triggered = False
        exit_rule = ""

        # 1. Catastrophe Stop
        if ltp < catastrophe_sl:
            exit_triggered = True
            exit_rule = f"1. Catastrophe SL Breach (LTP ₹{ltp:,.2f} < Hard SL ₹{catastrophe_sl:,.2f})"

        # 2. Climactic Churn Harvest
        elif peak_gain_pct >= 20.0 and vol_pace >= 2.0 and range_closeness <= 0.35:
            exit_triggered = True
            exit_rule = f"2. Climactic Churn Harvest (Peak +{peak_gain_pct:.1f}%, Vol Pace {vol_pace:.1f}x)"

        # 3. Distribution Cluster Breakdown
        elif dist_count_12d >= 3 and ltp < ema20:
            exit_triggered = True
            exit_rule = f"3. Distribution Cluster Breakdown ({dist_count_12d} Dist Days & LTP < 20 EMA ₹{ema20:,.2f})"

        # 4. Mega-Runner 15% Cushion Lock
        elif peak_gain_pct >= 50.0:
            cushion_stop = max(ema20, peak_close * 0.85)
            if ltp < cushion_stop:
                exit_triggered = True
                exit_rule = f"4. Mega-Runner 15% Cushion Lock (LTP ₹{ltp:,.2f} < Cushion ₹{cushion_stop:,.2f})"

        # 5. 20 EMA Breakdown Engine
        elif ltp < ema20:
            if vol_pace >= 1.25:
                exit_triggered = True
                exit_rule = f"5A. Heavy Volume 20 EMA Breach (LTP ₹{ltp:,.2f} < ₹{ema20:,.2f} | Pace {vol_pace:.1f}x)"
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

        if exit_triggered and not t.get("exit_alert_sent", False):
            pnl_pct = round(curr_gain_pct, 2)
            exit_msg = (
                f"🚨 <b>EXIT 2.2 TRIGGER: CLOSE POSITION</b>\n"
                f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                f"• <b>Reason</b>: <b>{exit_rule}</b>\n"
                f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
                f"• <b>Entry Price</b>: ₹{entry_price:,.2f}\n"
                f"• <b>20 EMA Floor</b>: ₹{ema20:,.2f}\n"
                f"• <b>Hard SL</b>: ₹{catastrophe_sl:,.2f}\n"
                f"• <b>Peak Unrealized</b>: +{peak_gain_pct:.2f}%\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🛑 <i>Action: Execute square-off order immediately.</i>"
            )
            send_telegram_alert(exit_msg)
            t["exit_alert_sent"] = True
            t["exit_price"] = ltp
            t["exit_reason"] = exit_rule

        t["peak_gain_%"] = round(peak_gain_pct, 2)
        t["peak_close"] = round(peak_close, 2)
        t["vdu_under_count"] = vdu_under_count
        updated_trades.append(t)

    try:
        with open(MY_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_trades, f, indent=2)
    except Exception:
        pass


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def load_comprehensive_targets(n750_set):
    """
    Loads candidates strictly generated by Swing Ultra 750 & Swing Ultra Non-750.
    Retains candidates up to 5 days, pruning setups whose structural floor is broken.
    """
    targets = {}
    today_dt = get_ist_now().date()
    sources = [
        (os.path.join(DATA_DIR, "swing_ultra750_results.json"), "NIFTY 750"),
        (os.path.join(DATA_DIR, "swing_ultraN750_results.json"), "NON-N750")
    ]

    for fpath, default_univ in sources:
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            items = list(d.get("Watchlist", [])) + list(d.get("Fresh_Triggers", []))
            for item in items:
                sym = str(item.get("symbol", "")).strip().upper()
                if not sym or sym.endswith("BEES") or sym in EXCLUDED_SYMBOLS:
                    continue
                p_str = str(item.get("pivot_ceiling") or item.get("pivot_target") or item.get("pivot") or "0").replace("₹", "").replace(",", "").strip()
                pivot = float(p_str)
                if pivot <= 0: continue
                sl_str = str(item.get("stop_loss", pivot * 0.92)).replace("₹", "").replace(",", "").strip()
                stop_loss = float(sl_str)
                spread_pct = float(str(item.get("spread_%", "6.0")).replace("%", "") or 6.0)
                setup = item.get("setup_class") or item.get("type") or "LAUNCHPAD COIL"
                univ = "NIFTY 750" if (n750_set and sym in n750_set) else default_univ

                targets[sym] = {
                    "symbol": sym,
                    "universe": univ,
                    "setup": setup,
                    "pivot_ceiling": pivot,
                    "stop_loss": stop_loss,
                    "shelf_spread_%": spread_pct,
                    "added_date": str(today_dt)
                }
        except Exception:
            pass

    # Carry forward historical watchlists up to 5 sessions
    if os.path.exists(LIVE_BREAKOUTS_FILE):
        try:
            with open(LIVE_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
            historical_watchlist = prev.get("Active_MultiDay_Watchlist", {})
            for sym, item in historical_watchlist.items():
                if sym in targets or sym in EXCLUDED_SYMBOLS:
                    continue
                try:
                    entry_dt = datetime.datetime.strptime(item.get("added_date", str(today_dt)), "%Y-%m-%d").date()
                    age_days = (today_dt - entry_dt).days
                except Exception:
                    age_days = 99

                # Check if still within 5 session window
                if age_days <= WATCHLIST_MAX_AGE_DAYS:
                    pivot = float(item.get("pivot_ceiling", 0))
                    sl = float(item.get("stop_loss", pivot * 0.92))
                    if pivot > 0:
                        targets[sym] = item
        except Exception:
            pass

    return targets


def run_live_intraday_scan():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    n750_set = load_nifty750_symbols()
    targets = load_comprehensive_targets(n750_set)

    # State variables
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

    live_client = LivePriceClient()

    # Step 1: Monitor Active Trades for Exit 2.2
    monitor_exit2_2_active_positions(live_client, elapsed_mins)

    if not targets:
        print("❌ No institutional candidates currently eligible for scanning.")
        return

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    radar_candidates = []
    coiling_list = []
    multi_day_watchlist = {}

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        stop_loss = float(meta["stop_loss"])
        universe = meta["universe"]
        setup = meta["setup"]
        shelf_spread = float(meta.get("shelf_spread_%", 6.0))

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
                        day_high = float(clean[-1].get("high", last_c))
                        day_low = float(clean[-1].get("low", last_c))
            except Exception:
                pass

        if ltp is None or ltp <= 0:
            ltp = pivot
            day_high = pivot
            day_low = pivot

        day_high = max(day_high or ltp, ltp)
        day_low = min(day_low or ltp, ltp)

        # PRUNE INVALIDATED SETUPS: If price breaks below base stop floor, purge it from multi-day tracking
        if ltp < stop_loss:
            continue

        multi_day_watchlist[sym] = meta

        sma20_vol = max(1.0, sma20_vol)
        proj_vol = day_vol / expected_fraction
        vol_pace = round(proj_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((pivot - stop_loss) / pivot) * 100.0, 2) if pivot > stop_loss else 8.0

        # Universe Aligned Volume Hurdle
        is_v_reversal = "V-REVERSAL" in setup
        if is_v_reversal:
            pace_threshold = 1.30 if universe == "NIFTY 750" else 1.20
        else:
            if elapsed_mins <= 60:
                pace_threshold = 1.05 if universe == "NIFTY 750" else 0.80
            else:
                pace_threshold = 1.20 if universe == "NIFTY 750" else 0.85
                if shelf_spread > 7.0:
                    pace_threshold += 0.20

        is_vol_confirmed = bool(vol_pace >= pace_threshold)
        cleared_pivot = bool(clearance_pct >= 0.15)
        wick_touched = bool(day_high >= pivot)

        if wick_touched or cleared_pivot:
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
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_%": risk_pct,
            "clearance_%": clearance_pct,
            "proj_vol_pace": vol_pace
        }

        # ----------------------------------------------------------------------
        # ALERT 1: ZERO-DELAY SINGLE-TICK CEILING CLEARANCE
        # ----------------------------------------------------------------------
        if cleared_pivot and is_vol_confirmed:
            record["action"] = "CONFIRMED BREAKOUT"
            radar_candidates.append(record)
            master_confirmed_breakouts[sym] = record

            if sym not in alert1_sent:
                ref_h = benchmark_highs.get(sym, day_high)
                msg1 = (
                    f"👀 <b>RADAR: INSTITUTIONAL BREAKOUT CONFIRMED</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code> ({universe} | {setup})\n"
                    f"• <b>Pivot Ceiling</b>: ₹{pivot:,.2f}\n"
                    f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{clearance_pct:+.2f}%</b>)\n"
                    f"• <b>Session Day High</b>: ₹{day_high:,.2f}\n"
                    f"• <b>Volume Pace</b>: <b>{vol_pace}x SMA20</b>\n"
                    f"• <b>Stop Loss</b>: ₹{stop_loss:,.2f} (Risk: {risk_pct}%)\n"
                    f"📌 <i>Alert 1 dispatched on tick 1. Tracking for continuation above ₹{ref_h:,.2f}.</i>"
                )
                send_telegram_alert(msg1)
                alert1_sent.add(sym)
                time.sleep(1)

        # ----------------------------------------------------------------------
        # ALERT 2: AUTONOMOUS DAY-HIGH BREAK
        # ----------------------------------------------------------------------
        if sym not in alert2_sent and sym in benchmark_highs:
            prior_ref_high = float(benchmark_highs[sym])
            if ltp >= (prior_ref_high * 1.0015) and ltp > pivot and is_vol_confirmed:
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

        # Watchlist Coiling & Rejection State
        if not cleared_pivot:
            record["action"] = "WICK REJECTED" if wick_touched else "COILING"
            coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    final_confirmed = list(master_confirmed_breakouts.values())

    # --------------------------------------------------------------------------
    # 03:15 PM EOD CLOSING SUMMARY (CNC DELIVERY EXECUTION)
    # --------------------------------------------------------------------------
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
        "Benchmark_Highs": benchmark_highs,
        "Master_Confirmed_Breakouts": master_confirmed_breakouts,
        "Breakouts_All_Day": final_confirmed,
        "Active_MultiDay_Watchlist": multi_day_watchlist,
        "Coiling": coiling_list[:50]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ V1 Scan Complete: {len(final_confirmed)} Master Confirmed, {len(coiling_list)} Coils.")


if __name__ == "__main__":
    run_live_intraday_scan()
