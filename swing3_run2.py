import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_run2_results.json")
MIN_TURNOVER_CR = 10.0  # Swing 3 Run 2 baseline floor: >= 10 Crore/day


# -------------------------------------------------------------
# ROBUST DATA CLEANING & SPLIT ADJUSTMENT
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
        # Standardize date to YYYY-MM-DD regardless of timestamp format
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


# -------------------------------------------------------------
# BULLETPROOF NIFTY 50 REGIME LOADER (WITH LIVE FALLBACK)
# -------------------------------------------------------------
def load_nifty_regime():
    """
    Loads Nifty 50, standardizes date indices, and computes rolling 50-day SMA.
    Falls back to fetching historical index quotes if local files are missing/malformed.
    """
    nifty_candidates = [
        os.path.join(DATA_DIR, "nifty.json"),
        os.path.join(DATA_DIR, "nifty50.json"),
        os.path.join(DATA_DIR, "NIFTY.json"),
        os.path.join(DATA_DIR, "NIFTY50.json"),
        os.path.join(DATA_DIR, "^NSEI.json")
    ]
    raw = None
    for path in nifty_candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list) and len(data) > 50:
                        raw = data
                        print(f"📖 Loaded Nifty index history from: {path}")
                        break
            except Exception:
                continue

    # Fallback to direct download if local file is missing
    if not raw:
        print("🌐 Local Nifty index file missing. Fetching historical benchmark data...")
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
        except Exception as e:
            print(f"⚠️ Could not fetch Nifty index externally: {e}")

    clean = clean_and_prepare(raw)
    if not clean:
        print("❌ CRITICAL: Failed to construct Nifty 50 regime data.")
        return {}

    df = pd.DataFrame(clean)
    df["sma50"] = df["close"].rolling(50, min_periods=20).mean()

    nifty_map = {}
    for _, row in df.iterrows():
        t = row["time"]
        c = row["close"]
        s50 = row["sma50"]
        if pd.notnull(s50):
            nifty_map[t] = bool(c >= s50)

    print(f"✅ Nifty 50 SMA regime map constructed for {len(nifty_map)} sessions.")
    return nifty_map


# -------------------------------------------------------------
# SWING 3.0 RUN 2 ENGINE
# -------------------------------------------------------------
def backtest_single_stock(symbol, clean_data, nifty_map):
    if len(clean_data) < 60:
        return []

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(0)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

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
    times = df["time"].values

    trades = []
    in_position = False
    entry_price = 0.0
    initial_stop = 0.0
    current_stop = 0.0
    r_unit = 0.0
    moved_to_be = False
    entry_idx = 0
    setup_name = ""
    is_nifty_above_sma = False

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
        t = times[i]

        if not in_position:
            # Baseline turnover floor >= 10 Crore
            if turnover < MIN_TURNOVER_CR:
                continue

            entry_triggered = False
            curr_setup = ""
            calc_entry = 0.0
            calc_sl = 0.0

            # -------------------------------------------------------------
            # ENGINE 1: DOWNTREND REVERSAL (V-SHAPE RECLAIM)
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # POSITION TRIGGER & NIFTY REGIME TAGGING
            # -------------------------------------------------------------
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
                    
                    # Exact date matching with fallback to closest prior trading date
                    if t in nifty_map:
                        is_nifty_above_sma = nifty_map[t]
                    else:
                        prior_dates = [d for d in nifty_map.keys() if d <= t]
                        is_nifty_above_sma = nifty_map[max(prior_dates)] if prior_dates else True

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

                trades.append({
                    "Symbol": symbol,
                    "Setup": setup_name,
                    "Entry Date": times[entry_idx],
                    "Exit Date": times[i],
                    "Duration (Days)": i - entry_idx,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Initial Stop": initial_stop,
                    "PnL %": pnl_pct,
                    "R Multiple": r_multiple,
                    "Outcome": "WIN" if pnl_pts > 0 else "LOSS",
                    "Nifty >= 50 SMA": is_nifty_above_sma
                })

                in_position = False
                entry_price = 0.0
                current_stop = 0.0
                moved_to_be = False

    return trades


# -------------------------------------------------------------
# STATISTICAL METRICS COMPILER
# -------------------------------------------------------------
def compute_metrics(trades_list, label):
    if not trades_list:
        return {
            "Configuration": label,
            "Trades": 0, "Win Rate %": 0.0, "Profit Factor": 0.0,
            "Avg Gain %": 0.0, "Avg Loss %": 0.0, "Max Gain %": 0.0,
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
        "Max Gain %": max_gain,
        "Avg Hold (Days)": avg_duration
    }


def run_swing3_run2():
    nifty_map = load_nifty_regime()
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "gap_margin_candidates.json", "swing3_results.json",
        "backtest_results.json", "swing3_compare.json", "swing3_run2_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"🚀 Scanning {len(target_files)} symbols for Swing 3.0 Run 2 (Turnover >= ₹{MIN_TURNOVER_CR} Cr/day)...")
    all_trades = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                t = backtest_single_stock(sym, clean, nifty_map)
                all_trades.extend(t)
        except Exception:
            continue

    if not all_trades:
        print("⚠️ No qualifying trades triggered.")
        return

    # Segregate Nifty Regime subsets
    trades_above = [t for t in all_trades if t["Nifty >= 50 SMA"]]
    trades_below = [t for t in all_trades if not t["Nifty >= 50 SMA"]]

    metrics_baseline = compute_metrics(all_trades, "Swing 3.0 Run 2 Baseline (Turnover >= 10 Cr)")
    metrics_above = compute_metrics(trades_above, "Nifty >= 50 SMA (Bull/Healthy Regime)")
    metrics_below = compute_metrics(trades_below, "Nifty < 50 SMA (Correction/Chop Regime)")

    report = [metrics_baseline, metrics_above, metrics_below]
    df_report = pd.DataFrame(report)

    print("\n" + "=" * 105)
    print("🎯 SWING 3.0 RUN 2: NIFTY 50 SMA REGIME COMPARISON (TURNOVER >= ₹10 CR)")
    print("=" * 105)
    print(df_report.to_string(index=False))
    print("=" * 105 + "\n")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": report,
        "Recent Trades": all_trades[-100:]
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    run_swing3_run2()
