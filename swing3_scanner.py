import os
import json
import glob
import datetime
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_scanner_results.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0


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


def load_nifty_benchmark():
    for f in ["nifty.json", "nifty50.json", "NIFTY.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) > 50:
                    df = pd.DataFrame(clean)
                    df["perf_60d"] = df["close"].pct_change(60).fillna(0)
                    return {r["time"]: float(r["perf_60d"]) for _, r in df.iterrows()}
            except Exception:
                continue
    return {}


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


def scan_stock_swing3(symbol, clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return None, None, None

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

    in_trade = False
    entry_p = 0.0
    initial_sl = 0.0
    current_sl = 0.0
    entry_d = ""
    entry_idx = 0

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_avgs = df["gross_vol_sma20"].values
    deliv_vols = df["delivery_vol"].values
    deliv_avgs = df["deliv_sma20"].values
    deliv_pcts = df["deliv_pct"].values
    deliv_pct_avgs = df["deliv_pct_50d"].values
    to_50d = df["turnover_50d"].values
    range_closes = df["range_closeness"].values
    ema20s = df["ema_20"].values
    sma200s = df["sma_200"].values
    perf_60s = df["perf_60d"].values
    demat_obvs = df["deliv_obv"].values
    demat_obv_maxes = df["dobv_20max"].values
    times = df["time"].values

    for i in range(40, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        turnover = to_50d[i]
        dp = deliv_pcts[i]
        dp_avg = deliv_pct_avgs[i]
        rc = range_closes[i]
        ema = ema20s[i]
        sma200 = sma200s[i]
        stock_p60 = perf_60s[i]
        t = times[i]

        if in_trade:
            current_sl = max(initial_sl, round(float(ema), 2))
            if c < current_sl:
                in_trade = False
                continue
            continue

        if turnover < MIN_TURNOVER_CR:
            continue

        is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
        n_perf = nifty_perf_map.get(t)
        if n_perf is None:
            prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
            n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
        is_rs = bool(stock_p60 > n_perf)

        if not (is_stage2 and is_rs):
            continue

        entry_triggered = False
        calc_entry = 0.0
        calc_sl = 0.0

        # ENGINE 1: V-REVERSAL
        recent_20_high = highs[i - 20:i].max()
        if (recent_20_high - l) / recent_20_high >= 0.18:
            recent_trough = lows[i - 5:i].min()
            prior_3d_high = highs[i - 4:i].max()
            c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
            c_vol_rev = v >= (1.3 * v_avg)
            c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg)
            c_candle_rev = rc >= 0.55

            if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                entry_triggered = True
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)

        # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT (SPREAD <= 8%)
        if not entry_triggered:
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= 0.08:
                    valid_launchpad = True
                    launchpad_high = round(float(s_high), 2)
                    launchpad_low = round(float(s_low), 2)
                    break

            if valid_launchpad:
                base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5

                c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg)
                c_candle_bo = rc >= 0.65
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if i > 0 else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.50) and obv_high:
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                conv_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                if conv_score >= 2.5:
                    in_trade = True
                    entry_p = calc_entry
                    initial_sl = calc_sl
                    current_sl = initial_sl
                    entry_d = t
                    entry_idx = i

    active_position = None
    if in_trade:
        cur_c = closes[-1]
        cur_ema = round(float(ema20s[-1]), 2)
        trailing_sl = max(initial_sl, cur_ema)
        pnl_pct = round(((cur_c - entry_p) / entry_p) * 100.0, 2)
        active_position = {
            "symbol": symbol,
            "entry_date": entry_d,
            "entry_price": entry_p,
            "current_close": round(cur_c, 2),
            "trailing_stop_20ema": trailing_sl,
            "pnl_%": pnl_pct,
            "holding_days": len(df) - 1 - entry_idx,
            "status": "HOLD" if cur_c >= trailing_sl else "EXIT (CLOSE < 20 EMA)"
        }

    fresh_trigger = None
    last_i = len(df) - 1
    c_today = closes[last_i]
    to_today = to_50d[last_i]
    dv_today = deliv_vols[last_i]
    dv_avg_today = deliv_avgs[last_i]
    stock_p60_today = perf_60s[last_i]
    t_today = times[last_i]

    if in_trade and entry_d == t_today:
        deliv_surge_ratio = dv_today / dv_avg_today if dv_avg_today > 0 else 1.0
        n_p = nifty_perf_map.get(t_today, 0.001)
        rs_ratio = (stock_p60_today / n_p) if n_p > 0 else (1.0 + abs(stock_p60_today))
        conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

        ext_pct = round(((c_today - entry_p) / entry_p) * 100.0, 2)
        fresh_trigger = {
            "symbol": symbol,
            "date": t_today,
            "type": "V-REVERSAL" if (entry_p != initial_sl) else "LAUNCHPAD-BO",
            "pivot_ceiling": entry_p,
            "close": round(c_today, 2),
            "extended_pct": ext_pct,
            "stop_loss": initial_sl,
            "risk_%": round(((entry_p - initial_sl) / entry_p) * 100, 2),
            "conviction_score": conviction_score,
            "turnover_cr": round(to_today, 1),
            "action": "BUY AT OPEN" if ext_pct <= 3.5 else "LIMIT RETEST"
        }

    watchlist_item = None
    if not in_trade and to_today >= MIN_TURNOVER_CR:
        is_st2 = bool(c_today >= sma200s[last_i]) if pd.notnull(sma200s[last_i]) else False
        n_p = nifty_perf_map.get(t_today, 0.001)
        is_rs = bool(stock_p60_today > n_p)

        if is_st2 and is_rs:
            deliv_surge_ratio = dv_today / dv_avg_today if dv_avg_today > 0 else 1.0
            rs_ratio = (stock_p60_today / n_p) if n_p > 0 else (1.0 + abs(stock_p60_today))
            conv_sc = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

            # Option A: Pre-V-Reversal Reclaim
            r20_h = highs[last_i - 20:last_i + 1].max()
            pullback_depth = (r20_h - lows[last_i]) / r20_h
            if pullback_depth >= 0.18:
                prior_3d_high = highs[last_i - 3:last_i].max()
                dist_to_reclaim = round(((float(prior_3d_high) - c_today) / c_today) * 100.0, 2)
                if 0 <= dist_to_reclaim <= 3.0:
                    watchlist_item = {
                        "symbol": symbol,
                        "setup_class": "V-REVERSAL PRE-RECLAIM",
                        "close": round(c_today, 2),
                        "pivot_target": round(float(prior_3d_high), 2),
                        "dist_to_pivot_pct": dist_to_reclaim,
                        "spread_%": round(pullback_depth * 100, 2),
                        "conviction_score": conv_sc,
                        "turnover_cr": round(to_today, 1)
                    }

            # Option B: Launchpad Tight Shelf Squeeze (Spread <= 8%)
            if not watchlist_item:
                for shelf_len in range(10, 19):
                    s_h = highs[last_i - shelf_len:last_i + 1].max()
                    s_l = lows[last_i - shelf_len:last_i + 1].min()
                    if s_l > 0:
                        spread = (s_h - s_l) / s_l
                        if spread <= 0.08:
                            dist_pct = round(((float(s_h) - c_today) / c_today) * 100.0, 2)
                            if 0 <= dist_pct <= 3.0:
                                watchlist_item = {
                                    "symbol": symbol,
                                    "setup_class": "LAUNCHPAD COIL",
                                    "close": round(c_today, 2),
                                    "pivot_target": round(float(s_h), 2),
                                    "dist_to_pivot_pct": dist_pct,
                                    "spread_%": round(spread * 100, 2),
                                    "conviction_score": conv_sc,
                                    "turnover_cr": round(to_today, 1)
                                }
                                break

    return fresh_trigger, active_position, watchlist_item


def execute_scanner():
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing3_results.json",
        "swing3_run2_results.json", "swing3_run2_global_results.json",
        "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
        "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json"
    }

    target_files = [
        f for f in json_files
        if os.path.basename(f) not in excluded and os.path.splitext(os.path.basename(f))[0].upper() in nifty750_set
    ]

    triggers = []
    active_positions = []
    watchlist = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                trig, active, watch = scan_stock_swing3(sym, clean, nifty_perf_map)
                if trig:
                    triggers.append(trig)
                if active:
                    active_positions.append(active)
                if watch:
                    watchlist.append(watch)
        except Exception:
            continue

    # Decremental Conviction Score sort for both lists
    triggers.sort(key=lambda x: x["conviction_score"], reverse=True)
    watchlist.sort(key=lambda x: x["conviction_score"], reverse=True)

    payload = {
        "Scan_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Target_Universe": "Nifty 750 Universe",
        "Strategy_Specs": "Swing 3.0 Production (Shelf <= 8%, Conviction >= 2.5, 20 EMA Trail)",
        "Fresh_Triggers_Count": len(triggers),
        "Active_Positions_Count": len(active_positions),
        "Watchlist_Count": len(watchlist),
        "Fresh_Triggers": triggers,
        "Active_Positions": active_positions,
        "Watchlist": watchlist[:30]
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Swing 3.0 Scan Complete. Fresh: {len(triggers)}, Active: {len(active_positions)}, Watchlist: {len(watchlist)}")


if __name__ == "__main__":
    execute_scanner()
