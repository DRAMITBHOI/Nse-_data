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

LIVE_BREAKOUTS_V2_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts_v2.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {"LIQUIDCASE", "LIQGRWBEES", "LIQUIDBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES", "SILVERBEES"}


def get_ist_now():
    return datetime.datetime.now(IST)


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
        # 1. Primary: Google Finance
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
                        return {"ltp": ltp, "day_high": max(day_high, ltp), "day_low": min(day_low, ltp), "day_vol": day_vol}
        except Exception:
            pass

        # 2. Secondary: Yahoo Chart endpoint
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
                    return {"ltp": ltp, "day_high": max(day_high, ltp), "day_low": min(day_low, ltp), "day_vol": day_vol}
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

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj_factor = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj_factor = k
                        break
                if not adj_factor and 1.70 <= ratio <= 2.30:
                    adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50:
                    adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50:
                    adj_factor = 10.0
            if adj_factor:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj_factor, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj_factor, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj_factor, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj_factor, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj_factor
                    clean[j]["volume"] = clean[j]["volume"] * adj_factor

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


def monitor_exit2_2_active_positions_v2(live_client: LivePriceClient, elapsed_mins: int):
    """
    EXIT 2.2 ENGINE (V2 Synchronization):
    Monitors data/my_trades.json, computes indicators, and writes tracking updates.
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
                    tr = pd.concat([
                        df_hist["high"] - df_hist["low"],
                        (df_hist["high"] - prev_c).abs(),
                        (df_hist["low"] - prev_c).abs()
                    ], axis=1).max(axis=1)
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
        time.sleep(0.05)

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

        t["ltp"] = round(ltp, 2)
        t["current_pnl_%"] = round(curr_gain_pct, 2)
        t["peak_gain_%"] = round(peak_gain_pct, 2)
        t["peak_close"] = round(peak_close, 2)
        t["ema_20"] = round(ema20, 2)
        t["vdu_under_count"] = vdu_under_count
        if exit_triggered:
            t["exit_pending"] = True
            t["exit_reason"] = exit_rule
            t["exit_price"] = ltp

        updated_trades.append(t)

    try:
        with open(MY_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_trades, f, indent=2)
    except Exception:
        pass


def get_nifty50_market_sentiment(client: LivePriceClient):
    try:
        url = "https://www.google.com/finance/quote/NIFTY_50:INDEXNSE"
        r = client.session.get(url, timeout=5)
        if r.status_code == 200:
            match = re.search(r'data-last-price="([\d\.]+)"', r.text)
            prev_match = re.search(r'Previous close.*?₹?([\d\.,]+)', r.text, re.DOTALL)
            if match and prev_match:
                curr = float(match.group(1))
                prev = float(prev_match.group(1).replace(",", ""))
                return round(((curr - prev) / prev) * 100.0, 2)
    except Exception:
        pass
    return 0.0


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
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    if os.path.exists(LIVE_BREAKOUTS_V2_FILE):
        try:
            with open(LIVE_BREAKOUTS_V2_FILE, "r", encoding="utf-8") as f:
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

    return targets


def run_live_intraday_scan_v2():
    targets = load_comprehensive_targets()
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")
    now_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST")
    elapsed_mins = get_market_session_elapsed()

    live_client = LivePriceClient()

    # Step 1: Monitor Active Open Trades for Exit 2.2 Rules
    monitor_exit2_2_active_positions_v2(live_client, elapsed_mins)

    if not targets:
        print("❌ No watchlisted setups found to scan.")
        return

    prior_consecutive_hits = {}
    if os.path.exists(LIVE_BREAKOUTS_V2_FILE):
        try:
            with open(LIVE_BREAKOUTS_V2_FILE, "r", encoding="utf-8") as f:
                prev_file = json.load(f)
                if prev_file.get("Date") == today_date:
                    prior_consecutive_hits = prev_file.get("Consecutive_Hits", {})
        except Exception:
            pass

    nifty_chg = get_nifty50_market_sentiment(live_client)
    is_nifty_weak = nifty_chg < -0.50
    print(f"📊 Market Condition: Nifty 50 {nifty_chg:+.2f}% | Weak Market Filter: {'ON' if is_nifty_weak else 'OFF'}")

    expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)

    confirmed_breakouts = []
    radar_candidates = []
    coiling_list = []
    current_consecutive_hits = {}

    for sym, meta in targets.items():
        pivot = float(meta["pivot_ceiling"])
        structural_sl = float(meta["stop_loss"])
        setup = meta["setup"]
        universe = meta["universe"]
        shelf_spread = float(meta.get("shelf_spread_%", 6.0))
        is_v_reversal = "V-REVERSAL" in setup

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
        drop_from_high_pct = round(((day_high - ltp) / day_high) * 100.0, 2) if day_high > 0 else 0.0

        effective_sl = max(structural_sl, round(day_low * 0.995, 2)) if is_v_reversal else structural_sl
        risk_pct = round(((ltp - effective_sl) / ltp) * 100.0, 2) if ltp > effective_sl else 8.0

        if is_v_reversal:
            base_pace = 1.75 if universe == "NIFTY 750" else 1.50
            min_clearance = 0.25
            max_clearance = 4.50
        else:
            base_pace = 1.25 if universe == "NIFTY 750" else 0.85
            if shelf_spread > 7.0:
                base_pace += 0.30
            min_clearance = 0.25
            max_clearance = 6.00

        if is_nifty_weak:
            base_pace += 0.30

        is_volume_confirmed = bool(vol_pace >= base_pace)
        cleared_pivot = bool(min_clearance <= clearance_pct <= max_clearance)

        max_allowed_drop = 1.50 if elapsed_mins >= 330 else 2.50
        no_closing_fade = bool(drop_from_high_pct <= max_allowed_drop)
        wick_touched_pivot = bool(day_high >= pivot)

        # Dynamic Persistence: Option B
        hits = (prior_consecutive_hits.get(sym, 0) + 1) if cleared_pivot else 0
        current_consecutive_hits[sym] = hits

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
            "stop_loss": round(effective_sl, 2),
            "risk_%": risk_pct,
            "shelf_spread_%": shelf_spread,
            "clearance_%": clearance_pct,
            "drop_from_high_%": drop_from_high_pct,
            "proj_vol_pace": vol_pace,
            "consecutive_ticks": hits
        }

        if elapsed_mins >= 330:
            if cleared_pivot and is_volume_confirmed and no_closing_fade:
                record["action"] = "CONFIRMED BREAKOUT"
                confirmed_breakouts.append(record)
            elif wick_touched_pivot and not cleared_pivot:
                record["action"] = "WICK REJECTED"
                coiling_list.append(record)
            else:
                record["action"] = "COILING"
                coiling_list.append(record)
        else:
            if cleared_pivot and is_volume_confirmed and no_closing_fade and is_persistent:
                record["action"] = "RADAR PIVOT CLEARANCE"
                radar_candidates.append(record)
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
        "Date": today_date,
        "Scan_Timestamp": now_str,
        "Elapsed_Minutes": elapsed_mins,
        "Nifty50_Intraday_%": nifty_chg,
        "Consecutive_Hits": current_consecutive_hits,
        "Breakouts_All_Day": confirmed_breakouts if elapsed_mins >= 330 else radar_candidates,
        "Coiling": coiling_list[:50]
    }

    with open(LIVE_BREAKOUTS_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ [Scanner V2 Synchronized] {len(confirmed_breakouts)} Confirmed Breakouts, {len(radar_candidates)} Radar, {len(coiling_list)} Coils.")


if __name__ == "__main__":
    run_live_intraday_scan_v2()
