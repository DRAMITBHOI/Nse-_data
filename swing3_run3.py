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
# 1. DATA SANITIZATION & SPLIT ADJUSTMENT
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
# 3. EXTRACTION ENGINE (EXCLUDES NIFTY 50 SMA CHECK)
# -------------------------------------------------------------
def extract_signals_for_stock(symbol, clean_data, nifty_perf_map, nifty750_set):
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

    # Demat OBV & 20-Day Peak
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

            is_cumul_150 = False
            is_obv_20max = False
            is_spread_10 = False

            # SETUP 1: V-REVERSAL
            recent_20_high = highs[i - 20:i].max()
            is_steep_drop = (recent_20_high - l) / recent_20_high >= 0.18

            if is_steep_drop:
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
                    is_cumul_150 = True
                    is_obv_20max = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if i > 0 else True
                    is_spread_10 = True

            # SETUP 2: LAUNCHPAD BREAKOUT
            if not entry_triggered:
                valid_launchpad_13 = False
                launchpad_high = 0.0
                launchpad_low = 0.0
                actual_spread = 1.0

                for shelf_len in range(10, 19):
                    s_high = highs[i - shelf_len:i].max()
                    s_low = lows[i - shelf_len:i].min()
                    if s_low > 0:
                        spr = (s_high - s_low) / s_low
                        if spr <= 0.13:
                            valid_launchpad_13 = True
                            launchpad_high = round(float(s_high), 2)
                            launchpad_low = round(float(s_low), 2)
                            actual_spread = spr
                            break

                if valid_launchpad_13:
                    base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                    base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                    cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5

                    c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                    c_vol_bo = v >= (1.4 * v_avg)
                    c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
                    c_candle_bo = rc >= 0.65

                    if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (cumul_ratio >= 1.15):
                        entry_triggered = True
                        curr_setup = "LAUNCHPAD-BO"
                        calc_entry = launchpad_high
                        calc_sl = round(min(l, launchpad_low), 2)

                        is_cumul_150 = bool(cumul_ratio >= 1.50)
                        is_obv_20max = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if i > 0 else True
                        is_spread_10 = bool(actual_spread <= 0.10)

            if entry_triggered:
                r_dist = calc_entry - calc_sl
                if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                    # Stage 2 + 60d RS Filter (Nifty 50 SMA Regime removed)
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

                        # Delivery surge & conviction score
                        deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                        rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                        conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                        tag_info = {
                            "cumul_150": is_cumul_150,
                            "obv_20max": is_obv_20max,
                            "spread_10": is_spread_10,
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
                    "cumul_150": tag_info["cumul_150"],
                    "obv_20max": tag_info["obv_20max"],
                    "spread_10": tag_info["spread_10"],
                    "conviction_score": tag_info["conviction_score"],
                    "deliv_surge_ratio": tag_info["deliv_surge_ratio"],
                })

                in_position = False
                entry_price = 0.0
                current_stop = 0.0
                moved_to_be = False

    return signals


# -------------------------------------------------------------
# 4. METRICS & PORTFOLIO RANKING
# -------------------------------------------------------------
def compute_metrics(trades_list, label):
    if not trades_list:
        return {
            "Configuration": label,
            "Trades": 0, "Win Rate %": 0.0, "Profit Factor": 0.0,
            "Avg Gain %": 0.0, "Avg Loss %": 0.0, "Max Profit %": 0.0,
            "Avg Hold (Days)": 0.0
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
        "Avg Hold (Days)": avg_duration
    }


def filter_daily_concurrency(trades, max_entries_per_day, sort_key="conviction_score"):
    trades_by_date = {}
    for t in trades:
        trades_by_date.setdefault(t["Entry Date"], []).append(t)
    filtered = []
    for d, d_trades in sorted(trades_by_date.items()):
        d_trades.sort(key=lambda x: x[sort_key], reverse=True)
        filtered.extend(d_trades[:max_entries_per_day])
    return filtered


# -------------------------------------------------------------
# 5. EXECUTION MATRIX & RANKER
# -------------------------------------------------------------
def run_swing3_run3_ablation():
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

    print(f"🚀 Scanning {len(target_files)} symbols for Swing 3 Run 3 Ablation (All Market Regimes)...")
    raw_signals = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                t = extract_signals_for_stock(sym, clean, nifty_perf_map, nifty750_set)
                raw_signals.extend(t)
        except Exception:
            continue

    if not raw_signals:
        print("⚠️ No qualifying signals triggered.")
        return

    # 1. Swing 3 Run 3 Baseline (Run 2 Full without Nifty 50 SMA filter, Max 2/day by surge)
    base_trades = filter_daily_concurrency(raw_signals, max_entries_per_day=2, sort_key="deliv_surge_ratio")
    m_baseline = compute_metrics(base_trades, "Run 3 Baseline (No SMA50, Max 2/d)")

    # 2. Individual Improvements
    t_imp1 = [t for t in raw_signals if t["cumul_150"]]
    m_imp1 = compute_metrics(filter_daily_concurrency(t_imp1, 2, "deliv_surge_ratio"), "Imp 1: Delivery Accum >= 1.5x")

    t_imp2 = [t for t in raw_signals if t["obv_20max"]]
    m_imp2 = compute_metrics(filter_daily_concurrency(t_imp2, 2, "deliv_surge_ratio"), "Imp 2: Demat OBV @ 20d High")

    t_imp3 = [t for t in raw_signals if t["spread_10"]]
    m_imp3 = compute_metrics(filter_daily_concurrency(t_imp3, 2, "deliv_surge_ratio"), "Imp 3: Shelf Spread <= 10%")

    t_imp4 = filter_daily_concurrency(raw_signals, max_entries_per_day=1, sort_key="conviction_score")
    m_imp4 = compute_metrics(t_imp4, "Imp 4: Score-Based Max 1/Day")

    # 3. Combinations of 2 Improvements
    t_c_1_3 = [t for t in raw_signals if t["cumul_150"] and t["spread_10"]]
    m_c_1_3 = compute_metrics(filter_daily_concurrency(t_c_1_3, 2, "deliv_surge_ratio"), "Combo (1+3): Delivery 1.5x + Spread 10%")

    t_c_1_2 = [t for t in raw_signals if t["cumul_150"] and t["obv_20max"]]
    m_c_1_2 = compute_metrics(filter_daily_concurrency(t_c_1_2, 2, "deliv_surge_ratio"), "Combo (1+2): Delivery 1.5x + OBV 20d High")

    t_c_2_4 = [t for t in raw_signals if t["obv_20max"]]
    m_c_2_4 = compute_metrics(filter_daily_concurrency(t_c_2_4, 1, "conviction_score"), "Combo (2+4): OBV 20d High + Score Max 1/d")

    t_c_3_4 = [t for t in raw_signals if t["spread_10"]]
    m_c_3_4 = compute_metrics(filter_daily_concurrency(t_c_3_4, 1, "conviction_score"), "Combo (3+4): Spread 10% + Score Max 1/d")

    # 4. Combinations of 3 Improvements
    t_c_1_2_3 = [t for t in raw_signals if t["cumul_150"] and t["obv_20max"] and t["spread_10"]]
    m_c_1_2_3 = compute_metrics(filter_daily_concurrency(t_c_1_2_3, 2, "deliv_surge_ratio"), "Combo (1+2+3): Deliv 1.5x + OBV + Spr 10%")

    t_c_1_2_4 = [t for t in raw_signals if t["cumul_150"] and t["obv_20max"]]
    m_c_1_2_4 = compute_metrics(filter_daily_concurrency(t_c_1_2_4, 1, "conviction_score"), "Combo (1+2+4): Deliv 1.5x + OBV + Score 1/d")

    t_c_2_3_4 = [t for t in raw_signals if t["obv_20max"] and t["spread_10"]]
    m_c_2_3_4 = compute_metrics(filter_daily_concurrency(t_c_2_3_4, 1, "conviction_score"), "Combo (2+3+4): OBV + Spread 10% + Score 1/d")

    # 5. All 4 Improvements Combined
    t_all4 = [t for t in raw_signals if t["cumul_150"] and t["obv_20max"] and t["spread_10"]]
    t_all4_final = filter_daily_concurrency(t_all4, max_entries_per_day=1, sort_key="conviction_score")
    m_all4 = compute_metrics(t_all4_final, "🎯 SWING 3 RUN 3 (ALL 4 COMBINED)")

    all_results = [
        m_baseline,
        m_imp1, m_imp2, m_imp3, m_imp4,
        m_c_1_3, m_c_1_2, m_c_2_4, m_c_3_4,
        m_c_1_2_3, m_c_1_2_4, m_c_2_3_4,
        m_all4
    ]

    # Rank: First separate baseline, then rank candidates by highest Profit Factor with trades close to 400
    df_res = pd.DataFrame(all_results)
    
    # Sort by Profit Factor descending
    df_sorted = df_res.sort_values(by=["Profit Factor", "Win Rate %"], ascending=[False, False]).reset_index(drop=True)
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    print("\n" + "=" * 115)
    print("🏆 SWING 3.0 RUN 3: FULL IMPROVEMENT MATRIX RANKED BY PROFIT FACTOR (TARGET: ~400 TRADES)")
    print("=" * 115)
    print(df_sorted.to_string(index=False))
    print("=" * 115 + "\n")

    ranked_list = df_sorted.to_dict(orient="records")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": ranked_list,
        "Recent Trades": t_all4_final[-100:]
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    run_swing3_run3_ablation()
