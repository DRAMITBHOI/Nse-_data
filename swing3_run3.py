import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_run3_results.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0


# -------------------------------------------------------------
# 1. DATA PREPARATION & CORPORATE ACTIONS
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 2. BENCHMARK & UNIVERSE HELPERS
# -------------------------------------------------------------
def load_nifty_perf_map():
    nifty_candidates = [
        os.path.join(DATA_DIR, "nifty.json"),
        os.path.join(DATA_DIR, "nifty50.json"),
        os.path.join(DATA_DIR, "NIFTY.json"),
        os.path.join(DATA_DIR, "NIFTY50.json"),
    ]
    raw = None
    for path in nifty_candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list) and len(data) > 50:
                        raw = data
                        break
            except Exception:
                continue

    if not raw:
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=5y"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                yf_data = json.loads(resp.read().decode("utf-8"))
                res = yf_data["chart"]["result"][0]
                timestamps = res["timestamp"]
                quotes = res["indicators"]["quote"][0]
                raw = []
                for ts, o, h, l, c, v in zip(timestamps, quotes["open"], quotes["high"], quotes["low"], quotes["close"], quotes["volume"]):
                    if c is not None and c > 0:
                        d_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        raw.append({"time": d_str, "open": o, "high": h, "low": l, "close": c, "volume": v or 100000})
        except Exception:
            pass

    clean = clean_and_prepare(raw)
    if not clean:
        return {}

    df = pd.DataFrame(clean)
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    nifty_map = {}
    for _, row in df.iterrows():
        nifty_map[row["time"]] = float(row["perf_60d"]) if pd.notnull(row["perf_60d"]) else 0.001
    return nifty_map


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            elif isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


# -------------------------------------------------------------
# 3. SWING 3 RUN 3 BASELINE SIGNAL EXTRACTOR
# -------------------------------------------------------------
def extract_run3_signals(symbol, clean_data, nifty_perf_map, nifty750_set):
    if len(clean_data) < 60 or symbol not in nifty750_set:
        return []

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(0)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    direction = np.where(df["close"] >= df["close"].shift(1), 1.0, -1.0)
    df["demat_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["demat_obv_20max"] = df["demat_obv"].rolling(20, min_periods=5).max()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.0)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_avgs = df["gross_vol_sma20"].values
    deliv_vols = df["delivery_vol"].values
    deliv_avgs = df["deliv_sma"].values
    to_50d = df["turnover_50d"].values
    deliv_pcts = df["deliv_pct"].values
    deliv_pct_avgs = df["deliv_pct_50d"].values
    range_closes = df["range_closeness"].values
    ema20s = df["ema20"].values
    sma200s = df["sma200"].values
    perf_60s = df["perf_60d"].values
    demat_obvs = df["demat_obv"].values
    demat_obv_maxes = df["demat_obv_20max"].values
    times = df["time"].values

    signals = []
    in_position = False
    entry_price = 0.0
    initial_stop = 0.0
    current_stop = 0.0
    r_unit = 0.0
    moved_to_be = False
    entry_idx = 0
    setup_name = ""
    tag_info = {}

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

        if not in_position:
            if turnover < MIN_TURNOVER_CR:
                continue

            entry_triggered = False
            curr_setup = ""
            calc_entry = 0.0
            calc_sl = 0.0
            base_cumul_ratio = 1.0
            actual_spread = 1.0

            # ENGINE 1: V-REVERSAL
            recent_20_high = highs[i - 20:i].max()
            if (recent_20_high - l) / recent_20_high >= 0.18:
                recent_trough = lows[i - 5:i].min()
                prior_3d_high = highs[i - 4:i].max()
                c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
                c_vol_rev = v >= (1.3 * v_avg)
                c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg if dp_avg > 0 else False)
                c_candle_rev = rc >= 0.55

                if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                    entry_triggered = True
                    curr_setup = "V-REVERSAL"
                    calc_entry = round(prior_3d_high, 2)
                    calc_sl = round(recent_trough * 0.995, 2)
                    base_cumul_ratio = 2.5
                    actual_spread = 0.07

            # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT
            if not entry_triggered:
                valid_launchpad = False
                launchpad_high = 0.0
                launchpad_low = 0.0

                for shelf_len in range(10, 19):
                    s_high = highs[i - shelf_len:i].max()
                    s_low = lows[i - shelf_len:i].min()
                    if s_low > 0:
                        spr = (s_high - s_low) / s_low
                        if spr <= 0.13:
                            valid_launchpad = True
                            launchpad_high = round(float(s_high), 2)
                            launchpad_low = round(float(s_low), 2)
                            actual_spread = spr
                            break

                if valid_launchpad:
                    base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                    base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                    base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 2.0

                    c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                    c_vol_bo = v >= (1.4 * v_avg)
                    c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
                    c_candle_bo = rc >= 0.65

                    if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.15):
                        entry_triggered = True
                        curr_setup = "LAUNCHPAD-BO"
                        calc_entry = launchpad_high
                        calc_sl = round(min(l, launchpad_low), 2)

            if entry_triggered:
                r_dist = calc_entry - calc_sl
                if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                    # Stage 2 + 60d RS Filter (Nifty 50 SMA regime completely removed)
                    is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
                    n_perf = nifty_perf_map.get(t)
                    if n_perf is None:
                        prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
                        n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
                    is_rs = bool(stock_p60 > n_perf)

                    if is_stage2 and is_rs:
                        in_position = True
                        entry_price = calc_entry
                        initial_stop = calc_sl
                        current_stop = initial_stop
                        r_unit = r_dist
                        moved_to_be = False
                        entry_idx = i
                        setup_name = curr_setup

                        deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                        rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                        conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                        tag_info = {
                            "cumul_ratio": base_cumul_ratio,
                            "is_obv_20max": bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if i > 0 else True,
                            "spread": actual_spread,
                            "conviction_score": conviction_score,
                            "deliv_surge_ratio": round(deliv_surge_ratio, 2)
                        }

        else:
            if not moved_to_be and h >= (entry_price + 1.5 * r_unit):
                current_stop = max(current_stop, entry_price)
                moved_to_be = True

            if moved_to_be:
                current_stop = max(current_stop, round(float(ema), 2))

            if c < current_stop:
                exit_price = round(current_stop, 2)
                pnl_pts = exit_price - entry_price
                pnl_pct = round((pnl_pts / entry_price) * 100.0, 2)
                r_multiple = round(pnl_pts / r_unit, 2)

                signals.append({
                    "Symbol": symbol,
                    "Setup": setup_name,
                    "Entry Date": times[entry_idx],
                    "Exit Date": times[i],
                    "Duration (Days)": i - entry_idx,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL %": pnl_pct,
                    "R Multiple": r_multiple,
                    "Outcome": "WIN" if pnl_pts > 0 else "LOSS",
                    "cumul_ratio": tag_info["cumul_ratio"],
                    "is_obv_20max": tag_info["is_obv_20max"],
                    "spread": tag_info["spread"],
                    "conviction_score": tag_info["conviction_score"],
                    "deliv_surge_ratio": tag_info["deliv_surge_ratio"]
                })

                in_position = False
                entry_price = 0.0
                current_stop = 0.0
                moved_to_be = False

    return signals


# -------------------------------------------------------------
# 4. METRICS & DAILY CONCURRENCY FILTER
# -------------------------------------------------------------
def compute_metrics(trades_list, label):
    if not trades_list:
        return {
            "Configuration": label,
            "Trades": 0, "Win Rate %": 0.0, "Profit Factor": 0.0,
            "Avg Gain %": 0.0, "Avg Loss %": 0.0, "Max Profit %": 0.0,
            "Avg Hold (Days)": 0.0, "Distance to 400": 400
        }

    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["Outcome"] == "WIN"]
    losses = df[df["Outcome"] == "LOSS"]

    win_rate = round((len(wins) / total) * 100.0, 2)
    avg_gain = round(wins["PnL %"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["PnL %"].mean(), 2) if not losses.empty else 0.0
    max_gain = round(df["PnL %"].max(), 2) if not df.empty else 0.0
    avg_duration = round(df["Duration (Days)"].mean(), 1)

    gross_win = wins["PnL %"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["PnL %"].sum()) if not losses.empty else 1.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "Configuration": label,
        "Trades": total,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,
        "Avg Gain %": avg_gain,
        "Avg Loss %": avg_loss,
        "Max Profit %": max_gain,
        "Avg Hold (Days)": avg_duration,
        "Distance to 400": abs(total - 400)
    }


def filter_daily(trades, max_per_day=1, sort_key="conviction_score"):
    trades_by_date = {}
    for t in trades:
        trades_by_date.setdefault(t["Entry Date"], []).append(t)
    out = []
    for d, d_trades in sorted(trades_by_date.items()):
        d_trades.sort(key=lambda x: x[sort_key], reverse=True)
        out.extend(d_trades[:max_per_day])
    return out


# -------------------------------------------------------------
# 5. TARGET ~400 TRADES COMBINATION ENGINE
# -------------------------------------------------------------
def run_target400_ablation():
    nifty_perf_map = load_nifty_perf_map()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "gap_margin_candidates.json", "swing3_results.json",
        "backtest_results.json", "swing3_compare.json", "swing3_run2_results.json",
        "swing3_run2_compare_results.json", "swing3_run3_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"🚀 Scanning {len(target_files)} tickers in Nifty 750 (Swing 3 Run 3 Base)...")
    raw_signals = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym not in nifty750_set:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                t = extract_run3_signals(sym, clean, nifty_perf_map, nifty750_set)
                raw_signals.extend(t)
        except Exception:
            continue

    if not raw_signals:
        print("⚠️ No qualifying signals found.")
        return

    # Predicates for the 4 Levers:
    # A: Delivery Ratio >= 2.0x
    # B: Demat OBV at 20d High
    # C: Spread Squeeze <= 8% (or 7.5%)
    # D1: Score >= 2.5 (Max 1/day)
    # D2: Score >= 3.0 (Max 1/day)

    results = []

    # Baseline (Swing 3 Run 3 Base: Max 2/day, surge sorted)
    t_base = filter_daily(raw_signals, max_per_day=2, sort_key="deliv_surge_ratio")
    results.append(compute_metrics(t_base, "Baseline: Run 3 Base (Max 2/d)"))

    # 1. INDIVIDUAL LEVERS
    # Lever A isolated: Delivery >= 2.0x (Max 2/day)
    t_A = [t for t in raw_signals if t["cumul_ratio"] >= 2.0]
    results.append(compute_metrics(filter_daily(t_A, 2, "deliv_surge_ratio"), "Lever A: Delivery Accum >= 2.0x"))

    # Lever B isolated: OBV 20d High (Max 2/day)
    t_B = [t for t in raw_signals if t["is_obv_20max"]]
    results.append(compute_metrics(filter_daily(t_B, 2, "deliv_surge_ratio"), "Lever B: Demat OBV @ 20d High"))

    # Lever C isolated: Spread Squeeze <= 8% (Max 2/day)
    t_C = [t for t in raw_signals if t["spread"] <= 0.08]
    results.append(compute_metrics(filter_daily(t_C, 2, "deliv_surge_ratio"), "Lever C: Squeeze Spread <= 8%"))

    # Lever D1 isolated: Max 1/day with Score >= 2.5
    t_D1 = [t for t in raw_signals if t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_D1, 1, "conviction_score"), "Lever D1: Max 1/d + Score >= 2.5"))

    # Lever D2 isolated: Max 1/day with Score >= 3.0
    t_D2 = [t for t in raw_signals if t["conviction_score"] >= 3.0]
    results.append(compute_metrics(filter_daily(t_D2, 1, "conviction_score"), "Lever D2: Max 1/d + Score >= 3.0"))

    # 2. COMBINATIONS OF 2 LEVERS
    # Combo (A + B): Delivery 2.0x + OBV High (Max 2/day)
    t_AB = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["is_obv_20max"]]
    results.append(compute_metrics(filter_daily(t_AB, 2, "deliv_surge_ratio"), "Combo (A+B): Deliv 2.0x + OBV High"))

    # Combo (A + C): Delivery 2.0x + Spread <= 8% (Max 2/day)
    t_AC = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["spread"] <= 0.08]
    results.append(compute_metrics(filter_daily(t_AC, 2, "deliv_surge_ratio"), "Combo (A+C): Deliv 2.0x + Spread <= 8%"))

    # Combo (B + D1): OBV High + Max 1/d Score >= 2.5
    t_BD1 = [t for t in raw_signals if t["is_obv_20max"] and t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_BD1, 1, "conviction_score"), "Combo (B+D1): OBV High + Score >= 2.5 (1/d)"))

    # Combo (C + D1): Spread <= 8% + Max 1/d Score >= 2.5
    t_CD1 = [t for t in raw_signals if t["spread"] <= 0.08 and t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_CD1, 1, "conviction_score"), "Combo (C+D1): Spread <= 8% + Score >= 2.5 (1/d)"))

    # 3. COMBINATIONS OF 3 LEVERS
    # Combo (A + B + C): Deliv 2.0x + OBV High + Spread <= 8% (Max 2/day)
    t_ABC = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["is_obv_20max"] and t["spread"] <= 0.08]
    results.append(compute_metrics(filter_daily(t_ABC, 2, "deliv_surge_ratio"), "Combo (A+B+C): Deliv 2.0x + OBV + Spread 8%"))

    # Combo (A + B + D1): Deliv 2.0x + OBV High + Max 1/d Score >= 2.5
    t_ABD1 = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["is_obv_20max"] and t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_ABD1, 1, "conviction_score"), "Combo (A+B+D1): Deliv 2.0x + OBV + Score 2.5 (1/d)"))

    # Combo (B + C + D1): OBV High + Spread <= 8% + Max 1/d Score >= 2.5
    t_BCD1 = [t for t in raw_signals if t["is_obv_20max"] and t["spread"] <= 0.08 and t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_BCD1, 1, "conviction_score"), "Combo (B+C+D1): OBV + Spread 8% + Score 2.5 (1/d)"))

    # Combo (A + C + D1): Deliv 2.0x + Spread <= 8% + Max 1/d Score >= 2.5
    t_ACD1 = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["spread"] <= 0.08 and t["conviction_score"] >= 2.5]
    results.append(compute_metrics(filter_daily(t_ACD1, 1, "conviction_score"), "Combo (A+C+D1): Deliv 2.0x + Spread 8% + Score 2.5 (1/d)"))

    # 4. ALL 4 LEVERS COMBINED
    # All 4 with Score >= 2.5
    t_all4_d1 = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["is_obv_20max"] and t["spread"] <= 0.08 and t["conviction_score"] >= 2.5]
    final_all4_d1 = filter_daily(t_all4_d1, 1, "conviction_score")
    results.append(compute_metrics(final_all4_d1, "🎯 ALL 4 COMBINED (A + B + C + Score >= 2.5)"))

    # All 4 with Score >= 3.0
    t_all4_d2 = [t for t in raw_signals if t["cumul_ratio"] >= 2.0 and t["is_obv_20max"] and t["spread"] <= 0.08 and t["conviction_score"] >= 3.0]
    final_all4_d2 = filter_daily(t_all4_d2, 1, "conviction_score")
    results.append(compute_metrics(final_all4_d2, "🎯 ALL 4 COMBINED (A + B + C + Score >= 3.0)"))

    # RANK STRICTLY BY CLOSENESS TO 400 TRADES (Distance to 400 ascending)
    df_res = pd.DataFrame(results)
    df_sorted = df_res.sort_values(by=["Distance to 400", "Profit Factor"], ascending=[True, False]).reset_index(drop=True)
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    print("\n" + "=" * 124)
    print("🏆 SWING 3.0 RUN 3: ABLATION TARGETING ~400 TRADES (RANKED BY CLOSENESS TO 400)")
    print("=" * 124)
    print(df_sorted.drop(columns=["Distance to 400"]).to_string(index=False))
    print("=" * 124 + "\n")

    ranked_list = df_sorted.to_dict(orient="records")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": ranked_list,
        "Sample Trades": final_all4_d1[-100:]
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    run_target400_ablation()
