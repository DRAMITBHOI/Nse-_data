import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scan_hp1_results.json")

# RANK #3 ENGINE SPECIFICATION
CONFIG = {
    "name": "Rank #3 HP1",
    "desc": "Base 40d | Box <= 35% | >=5 Dots (1.5x Vol, 1.4x Deliv) | Max SL 8% | Book +15% | BE +15%",
    "base_w": 40,
    "max_box": 35.0,
    "vol_m": 1.50,
    "pct_m": 1.40,
    "min_dots": 5,
    "max_risk": 8.0,
    "book_pct": 15.0,
    "be_trig": 15.0,
    "min_turnover_cr": 1.0  # Dynamic turnover floor: ₹1 Cr/d
}

def clean_stock_records(raw_data, min_len=60):
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
            if c <= 0 or np.isnan(c):
                continue
            
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
            if t_val not in date_map or entry["volume"] > date_map[t_val]["volume"]:
                date_map[t_val] = entry
        except Exception:
            continue

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

def load_fundamentals():
    f_path = os.path.join(DATA_DIR, "fundamentals.json")
    if os.path.exists(f_path):
        try:
            with open(f_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {}

def run_hp1_screener():
    t0 = time.time()
    print("🚀 Initializing SCAN_HP1 Engine...")

    fundamentals = load_fundamentals()

    reserved = {
        "fundamentals.json", "screener_results.json", "nifty750.json",
        "NIFTY50.json", "NIFTY.json", "fno_history.json",
        "wyckoff_screener_results.json", "obv_backtest_report.json",
        "scana_vs_absorption_report.json", "scana_candidates.json",
        "optimal_strategies.json", "scana_sensitivity_report.json",
        "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
        "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
        "backtest_hp3_report.json", "scan_macro_results.json", "macro_combo_leaderboard.json",
        "bucket_optimization_leaderboard.json", "backtest_bucket_report.json",
        "screener_macro_buckets_results.json", "scan_ultra_results.json"
    }

    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in reserved]
    print(f"📦 Evaluating {len(stock_files)} stocks...")

    candidates = []

    base_w = CONFIG["base_w"]
    max_box = CONFIG["max_box"]
    vol_m = CONFIG["vol_m"]
    pct_m = CONFIG["pct_m"]
    min_dots = CONFIG["min_dots"]
    max_risk = CONFIG["max_risk"]
    min_to = CONFIG["min_turnover_cr"]

    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_stock_records(raw, min_len=60)
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

            if N < (base_w + 15):
                continue

            turnover_cr = (closes * volumes) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)
            if float(to_50[-1]) < min_to:
                continue

            deliv_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)

            dots = (pcts >= (pct_m * pct_sma50)) & (d_vols >= (vol_m * deliv_sma20))
            dot_cumsum = np.cumsum(dots.astype(int))

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

            i = N - 1
            num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
            if num_dots < min_dots:
                continue

            base_highs = highs[i - base_w : i]
            base_lows = lows[i - base_w : i]
            macro_high = float(np.nanmax(base_highs))
            macro_low = float(np.nanmin(base_lows))
            if macro_low <= 0:
                continue

            box_depth = ((macro_high - macro_low) / macro_low) * 100.0
            if box_depth > max_box:
                continue

            sw_idx = int(np.nanargmax(base_highs))
            obv_pos = i - base_w + sw_idx
            if obv_pos < 0 or obv_pos >= N:
                continue

            cmp_val = float(closes[-1])
            dist_pct = ((macro_high - cmp_val) / macro_high) * 100.0

            status = None
            if closes[i] > macro_high and closes[i - 1] <= macro_high and obvs[i] > obvs[obv_pos]:
                status = "🟢 TRIGGERED TODAY"
            elif 0 <= dist_pct <= 3.0 and obvs[i] > obvs[obv_pos]:
                status = "🟡 READY AT CEILING"
            elif 3.0 < dist_pct <= 12.0:
                status = "🔵 DEEP ACCUMULATION"

            if not status:
                continue

            lookback_sl = min(12, base_w)
            recent_low = float(np.nanmin(lows[i - lookback_sl : i]))
            sl_val = round(recent_low * 0.992, 2)
            ref_price = cmp_val if "TRIGGERED" in status else macro_high
            risk_pct = round(((ref_price - sl_val) / ref_price) * 100.0, 1)

            if risk_pct > max_risk:
                continue

            pe_val = "N/A"
            if sym in fundamentals and isinstance(fundamentals[sym], dict):
                pe_val = str(fundamentals[sym].get("pe", "N/A"))

            candidates.append({
                "Symbol": sym,
                "Status": status,
                "LTP": round(cmp_val, 2),
                "Breakout Level": round(macro_high, 2),
                "Stop Loss": sl_val,
                "Risk %": f"{risk_pct}%",
                "50% Target (+15%)": round(ref_price * 1.15, 2),
                "Distance to Breakout": f"{round(dist_pct, 1)}%",
                "Demat Dots": f"{int(num_dots)} dots",
                "P/E": pe_val,
                "_dist": dist_pct,
                "_is_trig": 0 if "TRIGGERED" in status else (1 if "READY" in status else 2)
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: (x["_is_trig"], x["_dist"]))
    for c in candidates:
        del c["_dist"]
        del c["_is_trig"]

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Rank Config": CONFIG["desc"],
        "Total Found": len(candidates),
        "Candidates": candidates
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Found {len(candidates)} HP1 setups.")

if __name__ == "__main__":
    run_hp1_screener()
