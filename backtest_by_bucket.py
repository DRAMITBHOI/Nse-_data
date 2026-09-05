import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "backtest_bucket_report.json")
START_DATE = "2021-01-01"

# BUCKET BASELINE RULES
BUCKET_CONFIGS = {
    "Bucket A": {
        "desc": "Large/Midcaps (>30 Cr/d)",
        "base_w": 60,
        "max_box": 35.0,
        "vol_m": 1.50,
        "pct_m": 1.30,
        "min_dots": 4,
        "max_risk": 8.0,
        "book_pct": 15.0,
        "be_trig": 10.0,
        "trail_window": 10,
        "min_vol": 40000
    },
    "Bucket B": {
        "desc": "Mid/Smallcaps (5-30 Cr/d)",
        "base_w": 90,
        "max_box": 35.0,
        "vol_m": 1.50,
        "pct_m": 1.30,
        "min_dots": 5,
        "max_risk": 10.0,
        "book_pct": 20.0,
        "be_trig": 12.0,
        "trail_window": 12,
        "min_vol": 25000
    },
    "Bucket C": {
        "desc": "Nifty 750 Low Turnover (<5 Cr/d)",
        "base_w": 120,
        "max_box": 40.0,
        "vol_m": 1.50,
        "pct_m": 1.30,
        "min_dots": 6,
        "max_risk": 12.0,
        "book_pct": 25.0,
        "be_trig": 15.0,
        "trail_window": 15,
        "min_vol": 15000
    },
    "Bucket D": {
        "desc": "Non-Nifty 750 Broader/Microcaps",
        "base_w": 160,
        "max_box": 35.0,
        "vol_m": 1.50,
        "pct_m": 1.40,
        "min_dots": 8,
        "max_risk": 12.0,
        "book_pct": 50.0,
        "be_trig": 20.0,
        "trail_window": 20,
        "min_vol": 15000
    }
}

def load_nifty_750_set():
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
                if isinstance(raw, list):
                    return set(str(x).strip().upper() for x in raw)
                elif isinstance(raw, dict):
                    return set(str(x).strip().upper() for x in raw.keys())
        except Exception:
            pass
    return set()

def clean_stock_records(raw_data, min_len=80):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        t_val = str(r.get("time", "")).strip()[:10]
        if len(t_val) < 10:
            continue
        try:
            c = float(r.get("close", 0) or 0)
        except Exception:
            continue
        if c <= 0 or np.isnan(c):
            continue
        
        try:
            entry = {
                "time": t_val,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            }
        except Exception:
            continue

        if t_val not in date_map or entry["volume"] > date_map[t_val]["volume"]:
            date_map[t_val] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < min_len:
        return []

    clean = [date_map[k] for k in sorted_dates]

    multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj = k
                        break
            if adj:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj
                    clean[j]["volume"] = clean[j]["volume"] * adj

    running_vol = 35000.0
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
            clean[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.40)
            clean[i]["deliv_pct"] = pct if pct > 0 else 40.0
        elif dv > v:
            clean[i]["delivery_vol"] = v
            clean[i]["deliv_pct"] = 100.0

    return clean

def fast_rolling(arr, window):
    if len(arr) < window:
        return np.full_like(arr, fill_value=1.0, dtype=float)
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res = np.empty_like(arr, dtype=float)
    res[:window-1] = np.nan
    res[window-1:] = ret[window-1:] / window
    res[:window-1] = res[window-1]
    return np.nan_to_num(res, nan=1.0)

def run_bucket_backtest():
    t0 = time.time()
    print("🚀 Running Bucket-Wise Historical Backtest (2021 to Present)...")

    n750_set = load_nifty_750_set()
    reserved = {
        "fundamentals.json", "screener_results.json", "nifty750.json",
        "NIFTY50.json", "NIFTY.json", "fno_history.json",
        "wyckoff_screener_results.json", "obv_backtest_report.json",
        "scana_vs_absorption_report.json", "scana_candidates.json",
        "optimal_strategies.json", "scana_sensitivity_report.json",
        "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
        "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
        "backtest_hp3_report.json", "scan_macro_results.json", "macro_combo_leaderboard.json",
        "bucket_optimization_leaderboard.json", "backtest_bucket_report.json"
    }

    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in reserved]
    print(f"📦 Pre-loading and classifying {len(stock_files)} stocks...")

    buckets_classified = {"Bucket A": [], "Bucket B": [], "Bucket C": [], "Bucket D": []}

    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_stock_records(raw, min_len=90)
            if not clean:
                continue

            closes = np.array([r["close"] for r in clean], dtype=float)
            highs = np.array([r["high"] for r in clean], dtype=float)
            lows = np.array([r["low"] for r in clean], dtype=float)
            pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
            d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
            volumes = np.array([r["volume"] for r in clean], dtype=float)
            times = [r["time"] for r in clean]
            N = len(closes)

            turnover_50d = fast_rolling((closes * volumes) / 1e7, 50)
            latest_to = turnover_50d[-1]
            is_n750 = sym in n750_set

            if not is_n750:
                b_name = "Bucket D"
            elif latest_to >= 30.0:
                b_name = "Bucket A"
            elif latest_to >= 5.0:
                b_name = "Bucket B"
            else:
                b_name = "Bucket C"

            deliv_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)
            vol_sma9 = fast_rolling(volumes, 9)

            obvs = np.zeros(N, dtype=float)
            cur_obv = 0.0
            for idx in range(N):
                dv = d_vols[idx]
                if idx > 0:
                    if closes[idx] > closes[idx - 1]: cur_obv += dv
                    elif closes[idx] < closes[idx - 1]: cur_obv -= dv
                else:
                    cur_obv = dv
                obvs[idx] = cur_obv

            buckets_classified[b_name].append({
                "sym": sym, "closes": closes, "highs": highs, "lows": lows,
                "times": times, "obvs": obvs, "vol_sma9": vol_sma9,
                "pcts": pcts, "pct_sma50": pct_sma50,
                "d_vols": d_vols, "deliv_sma20": deliv_sma20, "N": N
            })
        except Exception:
            continue

    for b, items in buckets_classified.items():
        print(f"  • {b}: {len(items)} stocks")

    bucket_reports = {}
    all_combined_trades = []

    for b_name, b_stocks in buckets_classified.items():
        cfg = BUCKET_CONFIGS[b_name]
        base_w = cfg["base_w"]
        max_box = cfg["max_box"]
        vol_m = cfg["vol_m"]
        pct_m = cfg["pct_m"]
        min_dots = cfg["min_dots"]
        max_risk = cfg["max_risk"]
        book_pct = cfg["book_pct"]
        be_trig = cfg["be_trig"]
        trail_w = cfg["trail_window"]
        min_vol = cfg["min_vol"]

        b_trades = []

        for s in b_stocks:
            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            times = s["times"]
            obvs = s["obvs"]
            vol_9 = s["vol_sma9"]
            N = s["N"]

            if N < (base_w + 15):
                continue

            dots = (s["pcts"] >= (pct_m * s["pct_sma50"])) & (s["d_vols"] >= (vol_m * s["deliv_sma20"]))
            dot_cumsum = np.cumsum(dots.astype(int))

            cooldown = 0
            for i in range(base_w + 5, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                if vol_9[i] < min_vol:
                    continue

                num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
                if num_dots < min_dots:
                    continue

                base_highs = highs[i - base_w : i]
                base_lows = lows[i - base_w : i]
                macro_high = float(np.nanmax(base_highs))
                macro_low = float(np.nanmin(base_lows))
                if macro_low <= 0:
                    continue

                if ((macro_high - macro_low) / macro_low) * 100.0 > max_box:
                    continue

                sw_idx = int(np.nanargmax(base_highs))
                obv_pos = i - base_w + sw_idx
                if obv_pos < 0 or obv_pos >= N:
                    continue

                if closes[i] > macro_high and closes[i - 1] <= macro_high and obvs[i] > obvs[obv_pos]:
                    entry_p = float(closes[i])
                    lookback_sl = min(15 if base_w > 90 else 10, base_w)
                    recent_low = float(np.nanmin(lows[i - lookback_sl : i]))
                    sl_p = round(recent_low * 0.992, 2)
                    risk_pct = ((entry_p - sl_p) / entry_p) * 100.0

                    if risk_pct <= 0 or risk_pct > max_risk:
                        continue

                    fwd_end = min(N, i + 1 + (160 if base_w > 90 else 80))
                    f_highs = highs[i + 1 : fwd_end]
                    f_lows = lows[i + 1 : fwd_end]
                    f_closes = closes[i + 1 : fwd_end]

                    if len(f_highs) < 2:
                        continue

                    max_run = 0.0
                    active_sl = sl_p
                    booked_partial = False
                    be_shifted = False
                    exit_p = float(f_closes[-1])
                    exit_date = times[fwd_end - 1]
                    hold_days = len(f_highs)

                    for bar_idx in range(len(f_highs)):
                        curr_bar = i + 1 + bar_idx
                        gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                        if gain > max_run:
                            max_run = gain

                        if max_run >= be_trig and not be_shifted and active_sl < entry_p:
                            active_sl = entry_p
                            be_shifted = True

                        if max_run >= book_pct and not booked_partial:
                            booked_partial = True
                            active_sl = entry_p

                        if booked_partial and bar_idx >= trail_w:
                            t_low = float(np.nanmin(lows[curr_bar - trail_w : curr_bar]))
                            if t_low > active_sl:
                                active_sl = t_low

                        if f_lows[bar_idx] <= active_sl:
                            exit_p = float(min(f_closes[bar_idx], active_sl))
                            exit_date = times[curr_bar]
                            hold_days = bar_idx + 1
                            break

                    raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                    realized_ret = (book_pct * 0.50) + (raw_ret * 0.50) if booked_partial else raw_ret

                    trade_record = {
                        "bucket": b_name,
                        "symbol": s["sym"],
                        "entry_date": times[i],
                        "exit_date": exit_date,
                        "entry_price": round(entry_p, 2),
                        "exit_price": round(exit_p, 2),
                        "initial_sl": sl_p,
                        "risk_pct": round(risk_pct, 2),
                        "realized_return": round(realized_ret, 2),
                        "max_run_gain": round(max_run, 2),
                        "hold_days": hold_days,
                        "win": bool(realized_ret > 0),
                        "r20": bool(max_run >= 20.0),
                        "r50": bool(max_run >= 50.0)
                    }
                    b_trades.append(trade_record)
                    all_combined_trades.append(trade_record)

                    cooldown = i + max(hold_days, 10)

        # Compute Bucket Statistics
        t_count = len(b_trades)
        wins = sum(1 for t in b_trades if t["win"])
        win_rate = round((wins / t_count) * 100.0, 1) if t_count > 0 else 0.0
        rets = [t["realized_return"] for t in b_trades]
        avg_ret = round(float(np.mean(rets)), 2) if t_count > 0 else 0.0
        pos_sum = sum(r for r in rets if r > 0)
        neg_sum = abs(sum(r for r in rets if r < 0))
        pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else (99.0 if pos_sum > 0 else 0.0)
        r20_pct = round((sum(1 for t in b_trades if t["r20"]) / t_count) * 100.0, 1) if t_count > 0 else 0.0
        r50_pct = round((sum(1 for t in b_trades if t["r50"]) / t_count) * 100.0, 1) if t_count > 0 else 0.0
        avg_hold = round(float(np.mean([t["hold_days"] for t in b_trades])), 1) if t_count > 0 else 0.0

        # Yearly breakdown
        y_map = {}
        for t in b_trades:
            yr = t["entry_date"][:4]
            if yr not in y_map: y_map[yr] = {"trades": 0, "wins": 0, "rets": []}
            y_map[yr]["trades"] += 1
            if t["win"]: y_map[yr]["wins"] += 1
            y_map[yr]["rets"].append(t["realized_return"])

        yearly_stats = []
        for yr in sorted(y_map.keys()):
            d = y_map[yr]
            tr = d["trades"]
            w = d["wins"]
            yearly_stats.append({
                "Year": yr,
                "Trades": tr,
                "Win Rate": f"{round((w/tr)*100.0, 1)}%",
                "Avg Return": f"{round(float(np.mean(d['rets'])), 2)}%"
            })

        bucket_reports[b_name] = {
            "Config": cfg,
            "Total Trades": t_count,
            "Trades Per Year": round(t_count / 5.67, 1),
            "Win Rate %": win_rate,
            "Profit Factor": pf,
            "Avg Return / Trade": avg_ret,
            "Avg Holding Days": avg_hold,
            "+20% Expansion %": r20_pct,
            "+50% Multibagger %": r50_pct,
            "Yearly Breakdown": yearly_stats,
            "Recent Trades": b_trades[-10:]
        }

    # Combined Summary
    tot_comb = len(all_combined_trades)
    wins_comb = sum(1 for t in all_combined_trades if t["win"])
    rets_comb = [t["realized_return"] for t in all_combined_trades]
    pos_comb = sum(r for r in rets_comb if r > 0)
    neg_comb = abs(sum(r for r in rets_comb if r < 0))

    final_payload = {
        "Generated Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Combined Metrics": {
            "Total Trades": tot_comb,
            "Win Rate %": round((wins_comb / tot_comb) * 100.0, 1) if tot_comb > 0 else 0.0,
            "Profit Factor": round(pos_comb / neg_comb, 2) if neg_comb > 0 else 0.0,
            "Avg Return": round(float(np.mean(rets_comb)), 2) if tot_comb > 0 else 0.0
        },
        "Bucket Breakdown": bucket_reports
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as fp:
        json.dump(final_payload, fp, indent=2)

    print(f"🎉 Backtest Complete in {time.time()-t0:.1f}s! Saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_bucket_backtest()
