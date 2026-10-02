import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
NIFTY_FILE = os.path.join(DATA_DIR, "nifty.json")
NIFTY_FALLBACK = os.path.join(DATA_DIR, "nifty50.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0  # Mandatory baseline turnover floor >= 10 Crore


# -------------------------------------------------------------
# DATA LOADERS & PREPARATION
# -------------------------------------------------------------
def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []

    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()[:10]
        if not raw_t:
            continue
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0:
                continue
            entry = {
                "time": raw_t,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0),
            }
            if raw_t not in date_map or entry["volume"] > date_map[raw_t]["volume"]:
                date_map[raw_t] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]

    # Split / Corporate Action Multipliers
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


def load_nifty_regime():
    """Loads Nifty daily history to calculate 50 SMA and 60-day relative strength."""
    file_to_use = NIFTY_FILE if os.path.exists(NIFTY_FILE) else NIFTY_FALLBACK
    if not os.path.exists(file_to_use):
        return {}

    with open(file_to_use, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    clean = clean_and_prepare(raw)
    if not clean:
        return {}

    df = pd.DataFrame(clean)
    df["sma50"] = df["close"].rolling(50, min_periods=20).mean()
    df["perf_60d"] = df["close"].pct_change(60)

    nifty_map = {}
    for _, row in df.iterrows():
        t = row["time"]
        c = row["close"]
        s50 = row["sma50"]
        p60 = row["perf_60d"]
        nifty_map[t] = {
            "above_sma50": bool(c >= s50) if pd.notnull(s50) else True,
            "perf_60d": float(p60) if pd.notnull(p60) else 0.0,
        }
    return nifty_map


def load_nifty750_symbols():
    if not os.path.exists(NIFTY750_FILE):
        return set()
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
# SIGNAL GENERATOR (TAGGED WITH COMPARISON METRICS)
# -------------------------------------------------------------
def generate_stock_signals(symbol, clean_data, nifty_map, nifty750_set):
    if len(clean_data) < 60:
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

    is_nifty750 = symbol in nifty750_set

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
            # Baseline turnover floor >= 10 Crore
            if turnover < MIN_TURNOVER_CR:
                continue

            entry_triggered = False
            curr_setup = ""
            calc_entry = 0.0
            calc_sl = 0.0
            delivery_surge_score = 0.0

            # ---------------------------------------------------------
            # ENGINE 1: V-REVERSAL
            # ---------------------------------------------------------
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
                    delivery_surge_score = dv / dv_avg if dv_avg > 0 else 1.0

            # ---------------------------------------------------------
            # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT
            # ---------------------------------------------------------
            if not entry_triggered:
                valid_launchpad = False
                launchpad_high = 0.0
                launchpad_low = 0.0

                for shelf_len in range(10, 19):
                    s_high = highs[i - shelf_len:i].max()
                    s_low = lows[i - shelf_len:i].min()
                    if s_low > 0 and ((s_high - s_low) / s_low) <= 0.13:
                        valid_launchpad = True
                        launchpad_high = round(float(s_high), 2)
                        launchpad_low = round(float(s_low), 2)
                        break

                if valid_launchpad:
                    base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                    base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                    cumulative_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5

                    c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                    c_vol_bo = v >= (1.4 * v_avg)
                    c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
                    c_candle_bo = rc >= 0.65

                    if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (cumulative_ratio >= 1.15):
                        entry_triggered = True
                        curr_setup = "LAUNCHPAD-BO"
                        calc_entry = launchpad_high
                        calc_sl = round(min(l, launchpad_low), 2)
                        delivery_surge_score = dv / dv_avg if dv_avg > 0 else 1.0

            if entry_triggered:
                r_dist = calc_entry - calc_sl
                if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                    in_position = True
                    entry_price = calc_entry
                    initial_stop = calc_sl
                    current_stop = initial_stop
                    r_unit = r_dist
                    moved_to_be = False
                    entry_idx = i
                    setup_name = curr_setup

                    # Benchmark conditions on entry day
                    nifty_state = nifty_map.get(t, {"above_sma50": True, "perf_60d": 0.0})
                    above_sma200 = bool(c >= sma200) if pd.notnull(sma200) else False
                    rs_outperforming = bool(stock_p60 > nifty_state["perf_60d"])

                    tag_info = {
                        "is_nifty750": is_nifty750,
                        "nifty_above_sma50": nifty_state["above_sma50"],
                        "stage2_and_rs": (above_sma200 and rs_outperforming),
                        "surge_score": round(delivery_surge_score, 2),
                    }

        else:
            # Trailing Stop Management
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
                    "is_nifty750": tag_info["is_nifty750"],
                    "nifty_above_sma50": tag_info["nifty_above_sma50"],
                    "stage2_and_rs": tag_info["stage2_and_rs"],
                    "surge_score": tag_info["surge_score"],
                })

                in_position = False
                entry_price = 0.0
                current_stop = 0.0
                moved_to_be = False

    return signals


# -------------------------------------------------------------
# METRICS COMPUTATION HELPER
# -------------------------------------------------------------
def compute_stats(trades_list):
    if not trades_list:
        return {
            "Trades": 0, "Win Rate %": 0.0, "Profit Factor": 0.0,
            "Avg Gain %": 0.0, "Avg Loss %": 0.0, "Avg Duration (Days)": 0.0
        }

    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["Outcome"] == "WIN"]
    losses = df[df["Outcome"] == "LOSS"]

    win_rate = round((len(wins) / total) * 100.0, 2)
    avg_gain = round(wins["PnL %"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["PnL %"].mean(), 2) if not losses.empty else 0.0
    avg_duration = round(df["Duration (Days)"].mean(), 1)

    gross_win = wins["PnL %"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["PnL %"].sum()) if not losses.empty else 1.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "Trades": total,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,
        "Avg Gain %": avg_gain,
        "Avg Loss %": avg_loss,
        "Avg Duration (Days)": avg_duration,
    }


# -------------------------------------------------------------
# RUNNER & COMPARISON PRINTER
# -------------------------------------------------------------
def run_comparative_backtest():
    nifty_map = load_nifty_regime()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json",
        "gap_margin_candidates.json", "swing3_results.json",
        "backtest_results.json", "swing3_compare.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"🚀 Scanning {len(target_files)} symbols with Turnover >= ₹{MIN_TURNOVER_CR} Cr/day...")
    all_raw_trades = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                trades = generate_stock_signals(sym, clean, nifty_map, nifty750_set)
                all_raw_trades.extend(trades)
        except Exception:
            continue

    if not all_raw_trades:
        print("⚠️ No qualifying trades found.")
        return

    # 1. Base Pool (Turnover >= 10 Cr, Unconstrained)
    base_stats = compute_stats(all_raw_trades)

    # 2. Comparison 1: Nifty 750 vs Non-Nifty 750
    t_nifty750 = [t for t in all_raw_trades if t["is_nifty750"]]
    t_non_nifty750 = [t for t in all_raw_trades if not t["is_nifty750"]]
    stats_nifty750 = compute_stats(t_nifty750)
    stats_non_nifty750 = compute_stats(t_non_nifty750)

    # 3. Comparison 2: Nifty 50 >= 50 SMA vs < 50 SMA
    t_above_sma50 = [t for t in all_raw_trades if t["nifty_above_sma50"]]
    t_below_sma50 = [t for t in all_raw_trades if not t["nifty_above_sma50"]]
    stats_above_sma50 = compute_stats(t_above_sma50)
    stats_below_sma50 = compute_stats(t_below_sma50)

    # 4. Comparison 3: Price >= 200 SMA & RS Outperforming vs Baseline
    t_stage2_rs = [t for t in all_raw_trades if t["stage2_and_rs"]]
    stats_stage2_rs = compute_stats(t_stage2_rs)

    # 5. Comparison 4: Max 2 Entries Per Day (Ranked by Delivery Surge) vs Unconstrained
    # Group by entry date and sort by surge_score descending
    trades_by_date = {}
    for t in all_raw_trades:
        trades_by_date.setdefault(t["Entry Date"], []).append(t)

    t_max2_daily = []
    for d, day_trades in sorted(trades_by_date.items()):
        day_trades.sort(key=lambda x: x["surge_score"], reverse=True)
        t_max2_daily.extend(day_trades[:2])
    stats_max2_daily = compute_stats(t_max2_daily)

    # 6. Combined "Ideal High-Conviction Portfolio" (Nifty750 + Nifty>=50SMA + RS + Max 2/Day)
    t_ideal = []
    ideal_by_date = {}
    for t in t_nifty750:
        if t["nifty_above_sma50"] and t["stage2_and_rs"]:
            ideal_by_date.setdefault(t["Entry Date"], []).append(t)
    for d, day_trades in sorted(ideal_by_date.items()):
        day_trades.sort(key=lambda x: x["surge_score"], reverse=True)
        t_ideal.extend(day_trades[:2])
    stats_ideal = compute_stats(t_ideal)

    # Compile Table
    report = [
        {"Configuration": "Baseline (Turnover >= 10 Cr)", **base_stats},
        {"Configuration": "1. Universe: Nifty 750", **stats_nifty750},
        {"Configuration": "1. Universe: Non-Nifty 750", **stats_non_nifty750},
        {"Configuration": "2. Regime: Nifty >= 50 SMA", **stats_above_sma50},
        {"Configuration": "2. Regime: Nifty < 50 SMA", **stats_below_sma50},
        {"Configuration": "3. Stage 2 + 60d RS vs Nifty", **stats_stage2_rs},
        {"Configuration": "4. Concurrency: Max 2/Day (Surge)", **stats_max2_daily},
        {"Configuration": "🔥 RUN 2 FULL (750 + Regime + RS + Max2)", **stats_ideal},
    ]

    df_report = pd.DataFrame(report)
    print("\n" + "=" * 94)
    print("🎯 SWING 3.0 MULTI-DIMENSIONAL ITERATION COMPARISON (TURNOVER >= ₹10 CR)")
    print("=" * 94)
    print(df_report.to_string(index=False))
    print("=" * 94 + "\n")

    # Export JSON
    with open(os.path.join(DATA_DIR, "swing3_compare.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)


if __name__ == "__main__":
    run_comparative_backtest()
