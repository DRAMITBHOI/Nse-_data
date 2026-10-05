import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_vs_run2_nifty750_results.json")
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
            with open(NIFTY750_FILE, "r") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def extract_stock_setups(clean_data, nifty_perf_map, mode="swing3"):
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
    entry_price = 0.0
    initial_stop = 0.0
    current_stop = 0.0
    entry_date = ""
    entry_idx = 0

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

        # PLAN 2 EXIT (COMMON TO BOTH: TRAIL 20 EMA DIRECTLY)
        if in_trade:
            current_stop = max(initial_stop, round(float(ema), 2))
            if c < current_stop:
                exit_price = round(current_stop, 2)
                pnl_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": t,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_%": pnl_pct,
                    "hold_days": i - entry_idx,
                    "outcome": "WIN" if pnl_pct > 0 else "LOSS"
                })
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

        # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT
        if not entry_triggered:
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            max_spread = 0.08 if mode == "swing3" else 0.13

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= max_spread:
                    valid_launchpad = True
                    launchpad_high = round(float(s_high), 2)
                    launchpad_low = round(float(s_low), 2)
                    break

            if valid_launchpad:
                base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                req_ratio = 1.50 if mode == "swing3" else 1.15
                base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else req_ratio

                c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg)
                c_candle_bo = rc >= 0.65
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if (mode == "swing3" and i > 0) else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= req_ratio) and obv_high:
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                if mode == "swing3":
                    deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                    rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                    conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)
                    if conviction_score < 2.5:
                        continue

                in_trade = True
                entry_price = calc_entry
                initial_stop = calc_sl
                current_stop = initial_stop
                entry_date = t
                entry_idx = i

    return trades


def compute_metrics(trades_list, label):
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]

    win_rate = round((len(wins) / total) * 100.0, 2)
    avg_gain = round(wins["pnl_%"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["pnl_%"].mean(), 2) if not losses.empty else 0.0
    max_gain = round(df["pnl_%"].max(), 2) if not df.empty else 0.0
    avg_days = round(df["hold_days"].mean(), 1)

    gross_win = wins["pnl_%"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl_%"].sum()) if not losses.empty else 1.0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "configuration": label,
        "trades": total,
        "win_rate_%": win_rate,
        "profit_factor": pf,
        "avg_gain_%": avg_gain,
        "avg_loss_%": avg_loss,
        "max_gain_%": max_gain,
        "avg_days": avg_days,
    }


def run_comparative_backtest():
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f) not in excluded]

    print(f"🚀 Filtering to NIFTY 750 universe ({len(nifty750_set)} symbols in master set)...")

    swing3_trades = []
    run2_trades = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym not in nifty750_set:
            continue

        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if not clean:
                continue

            t_s3 = extract_stock_setups(clean, nifty_perf_map, mode="swing3")
            for t in t_s3:
                t["symbol"] = sym
            swing3_trades.extend(t_s3)

            t_r2 = extract_stock_setups(clean, nifty_perf_map, mode="run2")
            for t in t_r2:
                t["symbol"] = sym
            run2_trades.extend(t_r2)
        except Exception:
            continue

    m_s3 = compute_metrics(swing3_trades, "1. Swing 3.0 Locked (Nifty 750 • Spread <= 8% • Score >= 2.5)")
    m_r2 = compute_metrics(run2_trades, "2. Swing 3 Run 2 Full in Nifty 750 (Spread <= 13% • No Score Gate)")

    df_comp = pd.DataFrame([m_s3, m_r2])

    print("\n" + "=" * 115)
    print("🏆 HEAD-TO-HEAD COMPARISON: NIFTY 750 UNIVERSE (20 EMA TRAIL EXIT)")
    print("=" * 115)
    print(df_comp.to_string(index=False))
    print("=" * 115 + "\n")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": [m_s3, m_r2],
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"💾 Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    run_comparative_backtest()
