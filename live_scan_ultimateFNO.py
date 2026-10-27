import os
import sys
import re
import json
import time
import datetime
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import requests

DATA_DIR = "Codes/data" if os.path.exists("Codes/data") else "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_fno_ultimate_breakouts.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
MIN_TURNOVER_CR = 10.0

# ---------------------------------------------------------
# TELEGRAM CREDENTIALS
# ---------------------------------------------------------
DEFAULT_BOT_TOKEN = "9160227761:AAE2zmSpvqttJl3_KKl3jjyKkMg-CopgtSY"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN).strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_ULTIMATE", "").strip()

if not TELEGRAM_CHAT_ID:
    for sec_path in ["local_secrets.json", "../local_secrets.json"]:
        if os.path.exists(sec_path):
            try:
                with open(sec_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    TELEGRAM_CHAT_ID = cfg.get("TELEGRAM_CHAT_ID_ULTIMATE", cfg.get("TELEGRAM_CHAT_ID", cfg.get("7860290981", "")))
                    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == DEFAULT_BOT_TOKEN:
                        TELEGRAM_BOT_TOKEN = cfg.get("TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN)
                    if TELEGRAM_CHAT_ID:
                        break
            except Exception:
                pass

if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = "7860290981"

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}

def get_ist_now():
    return datetime.datetime.now(IST)

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

def load_nifty_benchmark():
    import glob
    all_json = glob.glob("**/*.json", recursive=True)
    for p in all_json:
        bname = os.path.basename(p).lower()
        if bname in ["nifty.json", "nifty50.json"]:
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    clean = clean_and_prepare(json.load(fp))
                if len(clean) > 40:
                    df_bm = pd.DataFrame(clean)
                    df_bm["p60"] = df_bm["close"].pct_change(60).fillna(0)
                    return {str(r["time"]): float(r["p60"]) for _, r in df_bm.iterrows()}
            except Exception:
                pass
    return {}

def monitor_exit2_2_active_positions(live_client: LivePriceClient, elapsed_mins: int):
    import glob
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
        dist_count_10d = 0

        sym_matches = glob.glob(f"**/{sym}.json", recursive=True)
        if sym_matches:
            try:
                with open(sym_matches[0], "r", encoding="utf-8") as fp:
                    clean = clean_and_prepare(json.load(fp))
                if len(clean) >= 20:
                    df_h = pd.DataFrame(clean)
                    ema20 = float(df_h["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    vols = df_h["volume"].iloc[-20:].tolist()
                    sma20_vol = float(sum(vols) / len(vols)) if vols else 50000.0
                    prev_c = df_h["close"].shift(1)
                    tr = pd.concat([df_h["high"] - df_h["low"], (df_h["high"] - prev_c).abs(), (df_h["low"] - prev_c).abs()], axis=1).max(axis=1)
                    atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
                    dv_sma = df_h["delivery_vol"].rolling(20, min_periods=1).mean().iloc[-1]
                    dp_avg = df_h["deliv_pct"].rolling(50, min_periods=5).mean().iloc[-1]
                    c_rng = df_h["high"] - df_h["low"]
                    rc = np.where(c_rng > 0, (df_h["close"] - df_h["low"]) / c_rng, 0.5)
                    is_dist = (
                        (df_h["close"] < df_h["close"].shift(1)) &
                        ((df_h["delivery_vol"] >= 1.30 * dv_sma) | (df_h["deliv_pct"] >= 1.15 * dp_avg)) &
                        (rc <= 0.45)
                    ).astype(int)
                    dist_count_10d = int(is_dist.rolling(10, min_periods=1).sum().iloc[-1])
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

        if ltp < catastrophe_sl:
            exit_triggered = True
            exit_rule = f"1. Catastrophe SL Breach (LTP ₹{ltp:,.2f} < Hard SL ₹{catastrophe_sl:,.2f})"
        elif peak_gain_pct >= 20.0 and vol_pace >= 2.0 and range_closeness <= 0.35:
            exit_triggered = True
            exit_rule = f"2. Climactic Churn Harvest (Peak +{peak_gain_pct:.1f}%, Vol Pace {vol_pace:.1f}x)"
        elif dist_count_10d >= 3 and ltp < ema20:
            exit_triggered = True
            exit_rule = f"3. Distribution Cluster ({dist_count_10d} Dist Days & LTP < 20 EMA ₹{ema20:,.2f})"
        elif peak_gain_pct >= 50.0:
            cushion_stop = max(ema20, peak_close * 0.85)
            if ltp < cushion_stop:
                exit_triggered = True
                exit_rule = f"4. Mega-Runner 15% Cushion Lock (LTP ₹{ltp:,.2f} < Cushion ₹{cushion_stop:,.2f})"
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
                f"• <b>Rule Triggered</b>: <b>{exit_rule}</b>\n"
                f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
                f"• <b>Entry Price</b>: ₹{entry_price:,.2f}\n"
                f"• <b>20 EMA Floor</b>: ₹{ema20:,.2f}\n"
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

def build_live_targets():
    import glob
    all_json = glob.glob("**/*.json", recursive=True)
    nifty_map = load_nifty_benchmark()

    stock_map = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if any(k in bname.lower() for k in ["nifty", "audit", "result", "trade", "secret", "package", "checkpoint"]):
            continue
        if sym not in stock_map or os.path.getsize(p) > os.path.getsize(stock_map[sym]):
            stock_map[sym] = p

    targets = {}
    for sym, filepath in stock_map.items():
        if sym in EXCLUDED_SYMBOLS: continue
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue

        if len(clean) < 70:
            continue

        df = pd.DataFrame(clean)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["sma_200"] = df["close"].rolling(200, min_periods=40).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
        df["perf_60d"] = df["close"].pct_change(60).fillna(0)
        df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean().fillna(10000)
        df["gross_sma20"] = df["volume"].rolling(20, min_periods=1).mean().fillna(20000)

        diff = df["close"].diff()
        direction = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
        df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()

        i = len(df) - 1
        t0 = df["time"].iloc[i]
        c0 = df["close"].iloc[i]

        if df["turnover_50d"].iloc[i] < MIN_TURNOVER_CR: continue
        if pd.notnull(df["sma_200"].iloc[i]) and c0 < df["sma_200"].iloc[i]: continue
        n_perf = nifty_map.get(str(t0), 0.001)
        if df["perf_60d"].iloc[i] < n_perf: continue

        highs = df["high"].values
        lows = df["low"].values
        delivs = df["delivery_vol"].values
        closes = df["close"].values

        shelf_found = False
        lp_h, lp_l, best_len, best_spread = 0.0, 0.0, 15, 999.0
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

        targets[sym] = {
            "symbol": sym,
            "pivot_ceiling": float(lp_h),
            "stop_loss": float(min(lows[i], lp_l)),
            "accumulation_ratio": acc_ratio,
            "shelf_spread_%": round(best_spread, 2),
            "sma20_vol": float(df["gross_sma20"].iloc[i]),
            "deliv_sma20": float(df["deliv_sma20"].iloc[i]),
            "dobv_20max": float(df["dobv_20max"].iloc[i]),
            "last_close": float(c0)
        }

    return targets

def run_live_fno_scan():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    print(f"\n🚀 Running Live F&O Ultimate Scanner at {now_str}...")
    print(f"⏱️ Session Elapsed: {elapsed_mins}/375 minutes.")

    live_client = LivePriceClient()
    monitor_exit2_2_active_positions(live_client, elapsed_mins)

    targets = build_live_targets()
    if not targets:
        print("❌ No stocks currently meet the Accumulation Ratio ≥ 1.50 & Spread criteria.")
        return

    print(f"🎯 Radar Universe: {len(targets)} institutionally accumulated F&O stocks monitored.")

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

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    radar_candidates = []
    coiling_list = []

    for sym, meta in targets.items():
        pivot = meta["pivot_ceiling"]
        stop_loss = meta["stop_loss"]
        acc_ratio = meta["accumulation_ratio"]
        shelf_spread = meta["shelf_spread_%"]
        sma20_vol = max(1.0, meta["sma20_vol"])

        quote = live_client.fetch_live_quote(sym)
        time.sleep(0.04)

        if quote:
            ltp = quote["ltp"]
            day_high = quote["day_high"]
            day_low = quote["day_low"]
            day_vol = quote["day_vol"]
        else:
            ltp = meta["last_close"]
            day_high = ltp
            day_low = ltp
            day_vol = 0.0

        bar_rng = day_high - day_low
        range_closeness = round((ltp - day_low) / bar_rng, 2) if bar_rng > 0 else 0.50

        proj_vol = day_vol / expected_fraction
        vol_pace = round(proj_vol / sma20_vol, 2) if sma20_vol > 0 else 1.0

        clearance_pct = round(((ltp - pivot) / pivot) * 100.0, 2)
        risk_pct = round(((ltp - stop_loss) / ltp) * 100.0, 2) if ltp > stop_loss else 6.5

        cleared_pivot = bool(clearance_pct >= 0.15)
        wick_touched = bool(day_high >= pivot)

        if wick_touched or cleared_pivot:
            if sym not in benchmark_highs:
                benchmark_highs[sym] = round(day_high, 2)
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

        is_confirmed_breakout = cleared_pivot and (vol_pace >= 1.60) and (range_closeness >= 0.65)

        # ALERT 1: TICK 1 LIVE INSTITUTIONAL CLEARANCE
        if is_confirmed_breakout:
            record["action"] = "BUY (CONFIRMED BREAKOUT)"
            radar_candidates.append(record)
            master_confirmed_breakouts[sym] = record

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
                time.sleep(1)

        # ALERT 2: DAY-HIGH EXPANSION BREAK
        if sym not in alert2_sent and sym in benchmark_highs:
            prior_ref_high = float(benchmark_highs[sym])
            if ltp >= (prior_ref_high * 1.0015) and ltp > pivot and (vol_pace >= 1.60) and (range_closeness >= 0.65):
                record["action"] = "BUY (EXPANSION CONTINUATION)"
                master_confirmed_breakouts[sym] = record

                msg2 = (
                    f"⚡ <b>ULTIMATE F&O 2ND ALERT: HIGH EXPANSION</b>\n"
                    f"• <b>Symbol</b>: <code>#{sym}</code>\n"
                    f"• <b>Prior Day High Cleared</b>: ₹{prior_ref_high:,.2f}\n"
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
                time.sleep(1)

        if not cleared_pivot:
            record["action"] = "WICK REJECTED" if (wick_touched and range_closeness < 0.50) else "COILING"
            coiling_list.append(record)

    coiling_list.sort(key=lambda x: x["clearance_%"], reverse=True)
    final_confirmed = list(master_confirmed_breakouts.values())

    # 03:15 PM EOD CLOSING SUMMARY (CNC DELIVERY EXECUTION)
    if elapsed_mins >= 360 and not eod_summary_sent:
        summary_msg = (
            f"🎯 <b>ULTIMATE F&O CLOSING CONFIRMATIONS (3:15 PM)</b>\n"
            f"🕒 <i>Time: {now_str}</i>\n\n"
        )
        if final_confirmed:
            summary_msg += f"<b>Verified Institutional Entries ({len(final_confirmed)}):</b>\n"
            for b in final_confirmed:
                summary_msg += (
                    f"🟢 <b>{b['symbol']}</b>: LTP ₹{b['ltp']:,.2f} (Clearance +{b['clearance_%']}%) | "
                    f"Acc: {b['accumulation_ratio']}x | Pace: {b['proj_vol_pace']}x | SL: ₹{b['stop_loss']:,.2f}\n"
                )
            summary_msg += "\n✅ <i>Place CNC delivery orders before 03:28 PM IST.</i>\n"
        else:
            summary_msg += "<i>No F&O stocks held breakout ceiling with verified institutional volume today.</i>\n"

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
        "Coiling_Watchlist": coiling_list[:30]
    }

    with open(LIVE_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ Live F&O Scan Complete: {len(final_confirmed)} Confirmed Breakouts, {len(coiling_list)} Coils.")

if __name__ == "__main__":
    run_live_fno_scan()
