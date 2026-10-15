import os
import glob
import json
import datetime
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
OUT_DIR = "backtest_results"
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

def simulate_stock_trades(clean_data, nifty_perf_map, mode="SWING3"):
    """
    mode:
      - 'SWING3': Spread <= 8%, Conviction Score >= 2.5
      - 'RUN2':   Spread <= 13%, No Score Cutoff
    """
    if len(clean_data) < 60:
        return []

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

    trades = []
    in_trade = False
    entry_p = 0.0
    initial_sl = 0.0
    current_sl = 0.0
    entry_idx = 0
    setup_name = ""

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

        # Trailing Exit Engine
        if in_trade:
            current_sl = max(initial_sl, round(float(ema), 2))
            if c < current_sl:
                exit_p = c
                pnl = (exit_p - entry_p) / entry_p
                trades.append({
                    "setup": setup_name,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                    "is_win": pnl > 0
                })
                in_trade = False
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
        cur_setup = ""

        # ENGINE 1: V-REVERSAL
        recent_20_high = highs[max(0, i - 20):i].max()
        if (recent_20_high - l) / recent_20_high >= 0.18:
            recent_trough = lows[max(0, i - 5):i].min()
            prior_3d_high = highs[max(0, i - 4):i].max()
            c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
            c_vol_rev = v >= (1.3 * v_avg)
            c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg)
            c_candle_rev = rc >= 0.55

            if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                entry_triggered = True
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)
                cur_setup = "V-REVERSAL"

        # ENGINE 2: LAUNCHPAD BREAKOUT
        if not entry_triggered:
            max_spread = 0.08 if mode == "SWING3" else 0.13
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[max(0, i - shelf_len):i].max()
                s_low = lows[max(0, i - shelf_len):i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= max_spread:
                    valid_launchpad = True
                    launchpad_high = round(float(s_high), 2)
                    launchpad_low = round(float(s_low), 2)
                    break

            if valid_launchpad:
                base_up_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] >= closes[k - 1])
                base_down_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] < closes[k - 1])
                req_ratio = 1.50 if mode == "SWING3" else 1.15
                base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else req_ratio

                c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg)
                c_candle_bo = rc >= 0.65
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if mode == "SWING3" else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= req_ratio) and obv_high:
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)
                    cur_setup = "LAUNCHPAD"

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                conv_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                score_pass = (conv_score >= 2.5) if mode == "SWING3" else True

                if score_pass:
                    in_trade = True
                    entry_p = calc_entry
                    initial_sl = calc_sl
                    current_sl = initial_sl
                    entry_idx = i
                    setup_name = cur_setup

    return trades

def main():
    print("🚀 Initializing Demat Delivery Backtesting Engine...")
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing3_results.json",
        "swing3_run2_results.json", "swing3_run2_global_results.json",
        "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
        "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
        "backtest_hp3_report.json", "live_intraday_breakouts.json"
    }

    stock_files = [f for f in json_files if os.path.basename(f) not in excluded]
    print(f"Loaded {len(stock_files)} stock files from {DATA_DIR}/")

    buckets = {
        "Swing 3: N750 V-Reversal": [],
        "Swing 3: N750 Launchpad": [],
        "Run 2: N750 V-Reversal": [],
        "Run 2: N750 Launchpad": [],
        "Run 2: Non-N750 V-Reversal": [],
        "Run 2: Non-N750 Launchpad": []
    }

    for path in stock_files:
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if not clean:
                continue

            is_n750 = sym in nifty750_set

            if is_n750:
                t_sw3 = simulate_stock_trades(clean, nifty_perf_map, mode="SWING3")
                for t in t_sw3:
                    buckets[f"Swing 3: N750 {t['setup'].title()}"].append(t)

                t_r2 = simulate_stock_trades(clean, nifty_perf_map, mode="RUN2")
                for t in t_r2:
                    buckets[f"Run 2: N750 {t['setup'].title()}"].append(t)
            else:
                t_r2 = simulate_stock_trades(clean, nifty_perf_map, mode="RUN2")
                for t in t_r2:
                    buckets[f"Run 2: Non-N750 {t['setup'].title()}"].append(t)
        except Exception:
            continue

    rows = []
    for k, t_list in buckets.items():
        if not t_list:
            rows.append({
                "Configuration": k, "Trades": 0, "Win Rate": "0.00%", "Profit Factor": "0.00",
                "Avg Gain": "+0.00%", "Avg Loss": "-0.00%", "Max Gain": "+0.00%", "Avg Days": "0.0d"
            })
            continue
        df_res = pd.DataFrame(t_list)
        wins = df_res[df_res["pnl"] > 0]
        losses = df_res[df_res["pnl"] <= 0]
        gross_p = wins["pnl"].sum()
        gross_l = abs(losses["pnl"].sum())
        pf = round(gross_p / gross_l, 2) if gross_l > 0 else 0.0

        rows.append({
            "Configuration": k,
            "Trades": len(df_res),
            "Win Rate": f"{(len(wins) / len(df_res)) * 100:.2f}%",
            "Profit Factor": f"{pf:.2f}",
            "Avg Gain": f"+{wins['pnl'].mean() * 100:.2f}%" if not wins.empty else "+0.00%",
            "Avg Loss": f"{losses['pnl'].mean() * 100:.2f}%" if not losses.empty else "-0.00%",
            "Max Gain": f"+{df_res['pnl'].max() * 100:.2f}%",
            "Avg Days": f"{df_res['holding_days'].mean():.1f}d"
        })

    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 115)
    print("🏆 SYSTEM COMPARISON DASHBOARD (TRUE NSE DEMAT DELIVERY)")
    print("=" * 115)
    print(out_df.to_string(index=False))
    print("=" * 115)

    os.makedirs(OUT_DIR, exist_ok=True)
    out
