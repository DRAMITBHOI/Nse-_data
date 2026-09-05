import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scan_ultra_results.json")

# SELECTED PRODUCTION PRESETS
PRESETS = {
    "Bucket A": {
        "label": "Large/Midcap (>=30 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 8,
        "max_risk": 10.0,
        "min_vol_bar": 15000,
        "target": "+50% Rally / Trailing 20d"
    },
    "Bucket B": {
        "label": "Mid/Smallcap (5-30 Cr/d)",
        "base_w": 120,
        "max_box": 40.0,
        "vol_m": 1.50,
        "pct_m": 1.40,
        "min_dots": 8,
        "max_risk": 12.0,
        "min_vol_bar": 10000,
        "target": "+50% Rally / Trailing 20d"
    },
    "Bucket C": {
        "label": "Smallcap (1-5 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 6,
        "max_risk": 12.0,
        "min_vol_bar": 4000,
        "target": "+50% Rally / Trailing 20d"
    },
    "Bucket D": {
        "label": "Microcap / Non-Nifty (<1 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 8,
        "max_risk": 10.0,
        "min_vol_bar": 4000,
        "target": "+50% Rally / Trailing 20d"
    }
}

def clean_stock_records(raw_data, min_len=70):
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

def run_scan_ultra():
    t0 = time.time()
    print("🚀 Initializing SCAN_ULTRA Engine...")

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
    print(f"📦 Scanning {len(stock_files)} stocks for SCAN_ULTRA signals...")

    results_by_bucket = {"Bucket A": [], "Bucket B": [], "Bucket C": [], "Bucket D": []}
    summary_stats = {"Total Scanned": len(stock_files), "Total Signals": 0}

    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_stock_records(raw, min_len=70)
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

            if N < 125:
                continue

            deliv_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)
            vol_sma9 = fast_rolling(volumes, 9)
            turnover_cr = (closes * volumes) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)
            latest_to = float(to_50[-1])

            # Classify into specific bucket
            if latest_to >= 30.0:
                b_name = "Bucket A"
            elif latest_to >= 5.0:
                b_name = "Bucket B"
            elif latest_to >= 1.0:
                b_name = "Bucket C"
            else:
                b_name = "Bucket D"

            cfg = PRESETS[b_name]
            base_w = cfg["base_w"]
            if N < (base_w + 15):
                continue

            # OBV Series
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

            # Delivery dots mask
            dots = (pcts >= (cfg["pct_m"] * pct_sma50)) & (d_vols >= (cfg["vol_m"] * deliv_sma20))
            dot_cumsum = np.cumsum(dots.astype(int))

            # Check the last 3 bars for an active trigger
            for bar_idx in range(N - 3, N):
                if vol_sma9[bar_idx] < cfg["min_vol_bar"]:
                    continue

                num_dots = dot_cumsum[bar_idx - 1] - dot_cumsum[max(0, bar_idx - 1 - base_w)]
                if num_dots < cfg["min_dots"]:
                    continue

                base_highs = highs[bar_idx - base_w : bar_idx]
                base_lows = lows[bar_idx - base_w : bar_idx]
                macro_high = float(np.nanmax(base_highs))
                macro_low = float(np.nanmin(base_lows))
                if macro_low <= 0:
                    continue

                box_depth = ((macro_high - macro_low) / macro_low) * 100.0
                if box_depth > cfg["max_box"]:
                    continue

                sw_idx = int(np.nanargmax(base_highs))
                obv_pos = bar_idx - base_w + sw_idx
                if obv_pos < 0 or obv_pos >= N:
                    continue

                if closes[bar_idx] > macro_high and closes[bar_idx - 1] <= macro_high and obvs[bar_idx] > obvs[obv_pos]:
                    entry_p = float(closes[bar_idx])
                    lookback_sl = min(15, base_w)
                    recent_low = float(np.nanmin(lows[bar_idx - lookback_sl : bar_idx]))
                    sl_p = round(recent_low * 0.992, 2)
                    risk_pct = ((entry_p - sl_p) / entry_p) * 100.0

                    if risk_pct <= 0 or risk_pct > cfg["max_risk"]:
                        continue

                    is_fresh = (bar_idx == N - 1)
                    alert_entry = {
                        "symbol": sym,
                        "bucket": b_name,
                        "trigger_date": times[bar_idx],
                        "is_fresh": is_fresh,
                        "current_price": round(float(closes[-1]), 2),
                        "breakout_price": round(entry_p, 2),
                        "stop_loss": sl_p,
                        "risk_pct": round(risk_pct, 2),
                        "macro_high": round(macro_high, 2),
                        "box_depth": round(box_depth, 1),
                        "dots_count": int(num_dots),
                        "avg_turnover_cr": round(latest_to, 2),
                        "target_rule": cfg["target"]
                    }
                    results_by_bucket[b_name].append(alert_entry)
                    summary_stats["Total Signals"] += 1
                    break
        except Exception:
            continue

    payload = {
        "Generated Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Summary": summary_stats,
        "Alerts": results_by_bucket
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! SCAN_ULTRA found {summary_stats['Total Signals']} candidates.")

if __name__ == "__main__":
    run_scan_ultra()
