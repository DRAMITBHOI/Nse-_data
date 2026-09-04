import os
import json
import time
import itertools
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_LEADERBOARD = os.path.join(DATA_DIR, "macro_combo_leaderboard.json")
START_DATE = "2021-01-01"

# STREAMLINED HIGH-CONVICTION MACRO GRID (36 Combinations)
BASE_WINDOWS = [120, 160]                     # ~6 months and ~8 months
BOX_DEPTHS = [35.0, 45.0]                     # Max consolidation depth
DOT_SPECS = [
    {"vol_m": 1.40, "pct_m": 1.30, "min_dots": 6},
    {"vol_m": 1.50, "pct_m": 1.30, "min_dots": 8},
    {"vol_m": 1.50, "pct_m": 1.40, "min_dots": 8}
]
MAX_RISKS = [12.0]                            # Anchored to 20d base low
EXIT_CONFIGS = [
    {"id": "Book25_BE15_Trail20d", "book_pct": 25.0, "be_trig": 15.0},
    {"id": "Book35_BE15_Trail20d", "book_pct": 35.0, "be_trig": 15.0},
    {"id": "Book50_BE20_Trail20d", "book_pct": 50.0, "be_trig": 20.0}
]

def clean_stock_records(raw_data, min_len=180):
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

def run_macro_optimizer():
    t0 = time.time()
    print("🚀 Initializing Ultra-Fast Vectorized Macro Optimizer...")

    reserved = {
        "fundamentals.json", "screener_results.json", "nifty750.json",
        "NIFTY50.json", "NIFTY.json", "fno_history.json",
        "wyckoff_screener_results.json", "obv_backtest_report.json",
        "scana_vs_absorption_report.json", "scana_candidates.json",
        "optimal_strategies.json", "scana_sensitivity_report.json",
        "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
        "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
        "backtest_hp3_report.json", "scan_macro_results.json", "macro_combo_leaderboard.json"
    }

    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in reserved]
    print(f"📦 Pre-processing {len(stock_files)} stock datasets...")

    parsed_stocks = []
    for f in stock_files:
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_stock_records(raw, min_len=180)
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

            # Precalculate dot masks
            dot_masks = {}
            for d_idx, d_spec in enumerate(DOT_SPECS):
                mask = (pcts >= (d_spec["pct_m"] * pct_sma50)) & (d_vols >= (d_spec["vol_m"] * deliv_sma20))
                dot_masks[d_idx] = np.cumsum(mask.astype(int))

            parsed_stocks.append({
                "sym": f.replace(".json", "").strip().upper(),
                "closes": closes, "highs": highs, "lows": lows,
                "times": times, "obvs": obvs, "vol_sma9": vol_sma9,
                "dot_cumsums": dot_masks, "N": N
            })
        except Exception:
            continue

    print(f"✅ Loaded {len(parsed_stocks)} viable stock series in {time.time()-t0:.1f}s. Evaluating configurations...")

    combos = list(itertools.product(BASE_WINDOWS, BOX_DEPTHS, range(len(DOT_SPECS)), MAX_RISKS, EXIT_CONFIGS))
    leaderboard = []

    for base_w, max_box, dot_idx, max_risk, exit_cfg in combos:
        dot_spec = DOT_SPECS[dot_idx]
        min_dots = dot_spec["min_dots"]
        book_pct = exit_cfg["book_pct"]
        be_trig = exit_cfg["be_trig"]

        trade_returns = []
        multibaggers_50 = 0

        for s in parsed_stocks:
            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            times = s["times"]
            obvs = s["obvs"]
            vol_9 = s["vol_sma9"]
            dot_cumsum = s["dot_cumsums"][dot_idx]
            N = s["N"]

            cooldown = 0
            for i in range(base_w + 10, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                if vol_9[i] < 20000:
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
                    recent_low = float(np.nanmin(lows[i - 20 : i]))
                    sl_p = round(recent_low * 0.99, 2)
                    risk_pct = ((entry_p - sl_p) / entry_p) * 100.0

                    if risk_pct <= 0 or risk_pct > max_risk:
                        continue

                    fwd_end = min(N, i + 1 + 180)
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

                        if booked_partial and bar_idx >= 15:
                            t_low = float(np.nanmin(lows[curr_bar - 20 : curr_bar]))
                            if t_low > active_sl:
                                active_sl = t_low

                        if f_lows[bar_idx] <= active_sl:
                            exit_p = float(min(f_closes[bar_idx], active_sl))
                            hold_days = bar_idx + 1
                            break

                    raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                    realized_ret = (book_pct * 0.50) + (raw_ret * 0.50) if booked_partial else raw_ret
                    trade_returns.append(realized_ret)
                    if max_run >= 50.0:
                        multibaggers_50 += 1

                    cooldown = i + max(hold_days, 15)

        total_trades = len(trade_returns)
        if total_trades >= 12:
            wins = sum(1 for r in trade_returns if r > 0)
            win_rate = round((wins / total_trades) * 100.0, 1)
            pos_sum = sum(r for r in trade_returns if r > 0)
            neg_sum = abs(sum(r for r in trade_returns if r < 0))
            pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0
            avg_ret = round(float(np.mean(trade_returns)), 2)
            multi_rate = round((multibaggers_50 / total_trades) * 100.0, 1)

            leaderboard.append({
                "base_w": base_w,
                "box_depth": max_box,
                "vol_m": dot_spec["vol_m"],
                "pct_m": dot_spec["pct_m"],
                "min_dots": min_dots,
                "max_risk": max_risk,
                "book_pct": book_pct,
                "be_trig": be_trig,
                "trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": pf,
                "avg_return": avg_ret,
                "multi50_rate": multi_rate
            })

    leaderboard.sort(key=lambda x: (x["win_rate"], x["profit_factor"], x["multi50_rate"]), reverse=True)
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank

    payload = {
        "Generated Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Total Permutations Tested": len(combos),
        "Viable Configurations Count": len(leaderboard),
        "Top 25 Leaderboard": leaderboard[:25]
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_LEADERBOARD, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Found {len(leaderboard)} ranked configurations.")

if __name__ == "__main__":
    run_macro_optimizer()
