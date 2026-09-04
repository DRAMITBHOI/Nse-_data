import os
import io
import json
import time
import urllib.request
import itertools
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "optimal_strategies.json")
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 50000
SPLIT_DATE = "2024-01-01"  # Train: < 2024-01-01 | Test: >= 2024-01-01

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def get_nifty_750_universe():
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fp:
                data = json.load(fp)
                if data:
                    return set(data)
        except Exception:
            pass

    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]
    symbols = set()
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception:
            pass

    sorted_universe = sorted(list(symbols))
    if sorted_universe:
        try:
            with open(local_path, "w") as fp:
                json.dump(sorted_universe, fp, indent=2)
        except Exception:
            pass
        return set(sorted_universe)
    return set()

def clean_data_fast(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = r.get("time", "")
        if not raw_t:
            continue
        d_str = str(raw_t)[:10]
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
            "deliv_pct": float(r.get("deliv_pct", 0) or 0)
        }
        if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
            date_map[d_str] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 60:
        return []

    clean = [date_map[k] for k in sorted_dates]

    deduped = []
    for r in clean:
        if deduped:
            prev = deduped[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    deduped.pop()
        deduped.append(r)

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(deduped) - 1, 0, -1):
        prev_c = deduped[i - 1]["close"]
        curr_o = deduped[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj = k
                        break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
                elif not adj and 8.50 <= ratio <= 11.50: adj = 10.0
            if adj:
                for j in range(0, i):
                    deduped[j]["open"] = round(deduped[j]["open"] / adj, 2)
                    deduped[j]["high"] = round(deduped[j]["high"] / adj, 2)
                    deduped[j]["low"] = round(deduped[j]["low"] / adj, 2)
                    deduped[j]["close"] = round(deduped[j]["close"] / adj, 2)
                    deduped[j]["delivery_vol"] = deduped[j]["delivery_vol"] * adj
                    deduped[j]["volume"] = deduped[j]["volume"] * adj

    running_vol = 50000.0
    for i in range(len(deduped)):
        v = deduped[i]["volume"]
        dv = deduped[i]["delivery_vol"]
        pct = deduped[i]["deliv_pct"]

        if v > 0: running_vol = 0.9 * running_vol + 0.1 * v
        else: deduped[i]["volume"] = running_vol; v = running_vol

        if dv <= 0:
            deduped[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            deduped[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            deduped[i]["delivery_vol"] = v
            deduped[i]["deliv_pct"] = 100.0

    return deduped

def fast_rolling_mean(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res = np.empty_like(arr, dtype=float)
    res[:window-1] = np.nan
    res[window-1:] = ret[window-1:] / window
    res[:window-1] = res[window-1] if len(res) >= window else 1.0
    return res

def precompute_stock(clean):
    N = len(clean)
    closes = np.array([r["close"] for r in clean], dtype=float)
    highs = np.array([r["high"] for r in clean], dtype=float)
    lows = np.array([r["low"] for r in clean], dtype=float)
    volumes = np.array([r["volume"] for r in clean], dtype=float)
    pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
    d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
    times = [r["time"] for r in clean]

    # Precalculate True Demat OBV
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

    deliv_sma20 = fast_rolling_mean(d_vols, 20)
    pct_sma50 = fast_rolling_mean(pcts, 50)
    vol_sma9 = fast_rolling_mean(volumes, 9)

    return {
        "N": N, "closes": closes, "highs": highs, "lows": lows,
        "volumes": volumes, "pcts": pcts, "d_vols": d_vols,
        "times": times, "obvs": obvs, "deliv_sma20": deliv_sma20,
        "pct_sma50": pct_sma50, "vol_sma9": vol_sma9
    }

def simulate_strategy(data_cache, base_w, vol_m, pct_m, min_dots, max_risk, target_pct, be_trigger):
    in_sample_trades = []
    out_sample_trades = []

    for sym, d in data_cache.items():
        N = d["N"]
        closes, highs, lows = d["closes"], d["highs"], d["lows"]
        pcts, d_vols = d["pcts"], d["d_vols"]
        pct_50, deliv_20 = d["pct_sma50"], d["deliv_sma20"]
        obvs, times, vol_9 = d["obvs"], d["times"], d["vol_sma9"]

        # Boolean vector of institutional dots
        dots = (pcts >= (pct_m * pct_50)) & (d_vols >= (vol_m * deliv_20))
        dot_cumsum = np.cumsum(dots.astype(int))

        cooldown = 0
        for i in range(base_w + 10, N - 1):
            if i < cooldown:
                continue

            if vol_9[i] < MIN_AVG_VOLUME_9D:
                continue

            # Dot count inside sliding base
            num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
            if num_dots < min_dots:
                continue

            base_highs = highs[i - base_w : i]
            sw_idx = int(np.argmax(base_highs))
            swing_high = base_highs[sw_idx]
            swing_obv = obvs[i - base_w + sw_idx]

            # Breakout condition
            if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                entry_p = closes[i]
                lookback_sl = min(12, base_w)
                recent_low = float(np.min(lows[i - lookback_sl : i]))
                sl_p = round(recent_low * 0.995, 2)
                risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                if risk_pct <= 0 or risk_pct > max_risk:
                    continue

                # Forward simulation up to 90 bars
                fwd_end = min(N, i + 1 + 90)
                f_highs = highs[i + 1 : fwd_end]
                f_lows = lows[i + 1 : fwd_end]
                f_closes = closes[i + 1 : fwd_end]

                if len(f_highs) < 2:
                    continue

                max_run = 0.0
                active_sl = sl_p
                booked_partial = False
                exit_p = f_closes[-1]
                days = len(f_highs)

                for bar_idx in range(len(f_highs)):
                    gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                    if gain > max_run:
                        max_run = gain

                    # Shift to BE
                    if max_run >= be_trigger and active_sl < entry_p:
                        active_sl = entry_p

                    # Book 50%
                    if max_run >= target_pct and not booked_partial:
                        booked_partial = True
                        active_sl = entry_p

                    # Trail swing lows on remainder
                    if booked_partial and bar_idx >= 10:
                        t_low = float(np.min(lows[i + 1 + bar_idx - 10 : i + 1 + bar_idx]))
                        if t_low > active_sl:
                            active_sl = t_low

                    if f_lows[bar_idx] <= active_sl:
                        exit_p = min(f_closes[bar_idx], active_sl)
                        days = bar_idx + 1
                        break

                raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                realized_ret = round((target_pct * 0.50) + (raw_ret * 0.50), 2) if booked_partial else round(raw_ret, 2)

                record = {
                    "return": realized_ret,
                    "win": realized_ret > 0,
                    "r20": max_run >= 20.0
                }

                if times[i] < SPLIT_DATE:
                    in_sample_trades.append(record)
                else:
                    out_sample_trades.append(record)

                cooldown = i + max(days, 8)

    def summarize(trade_list):
        if len(trade_list) < 15:
            return {"trades": len(trade_list), "win_rate": 0, "pf": 0, "avg_ret": 0, "r20": 0}
        total = len(trade_list)
        wins = sum(1 for t in trade_list if t["win"])
        win_rate = round((wins / total) * 100.0, 1)
        r20 = round((sum(1 for t in trade_list if t["r20"]) / total) * 100.0, 1)
        returns = [t["return"] for t in trade_list]
        avg_ret = round(float(np.mean(returns)), 2)
        pos = sum(r for r in returns if r > 0)
        neg = abs(sum(r for r in returns if r < 0))
        pf = round(pos / neg, 2) if neg > 0 else 99.0

        return {"trades": total, "win_rate": win_rate, "pf": pf, "avg_ret": avg_ret, "r20": r20}

    return summarize(in_sample_trades), summarize(out_sample_trades)

def run_grid_optimizer():
    print("🚀 Starting Genetic Grid Search across NIFTY Universe...")
    universe = get_nifty_750_universe()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as fp:
                fundamentals = json.load(fp)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "wyckoff_screener_results.json", "obv_backtest_report.json",
            "scana_vs_absorption_report.json", "scana_candidates.json",
            "optimal_strategies.json"
        ]
    ]

    if universe and len(universe) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in universe]

    # Pre-load and cache array computations once to allow microsecond simulations
    data_cache = {}
    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        if sym in fundamentals:
            pe_val = fundamentals[sym].get("pe", None)
            if pe_val:
                try:
                    if float(pe_val) > MAX_PE or float(pe_val) <= 0:
                        continue
                except (ValueError, TypeError):
                    pass

        try:
            with open(os.path.join(DATA_DIR, f), "r") as fp:
                raw = json.load(fp)
            clean = clean_data_fast(raw)
            if len(clean) >= 80:
                data_cache[sym] = precompute_stock(clean)
        except Exception:
            continue

    print(f"📦 Pre-cached {len(data_cache)} liquid stocks into memory.")

    # Define Parameter Grid
    grid_base_w = [15, 25, 35]
    grid_vol_m = [1.15, 1.30, 1.50]
    grid_pct_m = [1.15, 1.30]
    grid_min_dots = [2, 3]
    grid_max_risk = [5.0, 6.5, 8.0]
    grid_target = [10.0, 14.0, 18.0]
    grid_be = [6.0, 8.0]

    all_combos = list(itertools.product(
        grid_base_w, grid_vol_m, grid_pct_m, grid_min_dots,
        grid_max_risk, grid_target, grid_be
    ))

    total_runs = len(all_combos)
    print(f"⚙️ Running optimization across {total_runs} parameter combinations...")

    results = []
    start_t = time.time()

    for idx, (bw, vm, pm, dots, sl, tgt, be) in enumerate(all_combos):
        in_s, out_s = simulate_strategy(data_cache, bw, vm, pm, dots, sl, tgt, be)

        # Discard unstable parameter sets
        if in_s["trades"] >= 40 and out_s["trades"] >= 15:
            # Composite Scoring Formula: Weights Win Rate, Profit Factor, and Out-of-Sample Consistency
            consistency_penalty = abs(in_s["win_rate"] - out_s["win_rate"]) * 0.05
            score = (out_s["pf"] * 2.0) + (out_s["win_rate"] * 0.05) + (out_s["avg_ret"] * 0.5) - consistency_penalty

            results.append({
                "Score": round(score, 2),
                "Params": {
                    "Base Window": f"{bw}d",
                    "Vol Spike Multiplier": f"{vm}x",
                    "Deliv Pct Multiplier": f"{pm}x",
                    "Min Clustered Dots": dots,
                    "Max SL Risk %": f"{sl}%",
                    "50% Book Target %": f"+{tgt}%",
                    "BE Shift Trigger %": f"+{be}%"
                },
                "In_Sample_Pre2024": in_s,
                "Out_Sample_Live2024_26": out_s
            })

        if (idx + 1) % 150 == 0:
            print(f"⏳ Tested {idx + 1}/{total_runs} models (Elapsed: {round(time.time() - start_t, 1)}s)...")

    # Rank by robust out-of-sample score
    results.sort(key=lambda x: x["Score"], reverse=True)
    top_10 = results[:10]

    payload = {
        "Optimizer Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Total Models Tested": total_runs,
        "Valid Candidates": len(results),
        "Top 10 Optimal Strategies": top_10
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"\n🎉 Optimization Complete! Evaluated {total_runs} models.")
    if top_10:
        best = top_10[0]
        print(f"🏆 Rank #1 Strategy Score: {best['Score']}")
        print(f"   In-Sample Win Rate  : {best['In_Sample_Pre2024']['win_rate']}% | PF: {best['In_Sample_Pre2024']['pf']}")
        print(f"   Out-of-Sample Win Rate: {best['Out_Sample_Live2024_26']['win_rate']}% | PF: {best['Out_Sample_Live2024_26']['pf']}")
        print(f"📁 Full report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_grid_optimizer()
