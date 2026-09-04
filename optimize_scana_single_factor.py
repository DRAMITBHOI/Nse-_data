import os
import io
import json
import time
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "scana_sensitivity_report.json")
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 50000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

# BASELINE SCAN A ANCHOR
BASELINE = {
    "base_w": 25,
    "vol_m": 1.20,
    "pct_m": 1.20,
    "min_dots": 2,
    "max_risk": 8.0,
    "target_pct": 15.0,
    "be_trigger": 8.0
}

SWEEP_CONFIG = {
    "1. Base Window": [15, 20, 25, 30, 35, 40],
    "2. Delivery Vol Multiplier": [1.15, 1.20, 1.30, 1.40, 1.50],
    "3. Delivery Pct Multiplier": [1.20, 1.30, 1.40, 1.50],
    "4. Clustered Dots in Base": [2, 3, 4, 5],
    "5. Initial Stop Loss Cap": [5.0, 7.0, 8.0, 9.0, 10.0],
    "6. Partial Target (+15% Book)": [10.0, 12.0, 15.0, 18.0],
    "7. Breakeven Shift Trigger": [6.0, 8.0, 10.0, 13.0, 15.0]
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

def simulate_scan_a(data_cache, base_w, vol_m, pct_m, min_dots, max_risk, target_pct, be_trigger):
    trades = []

    for sym, d in data_cache.items():
        N = d["N"]
        closes, highs, lows = d["closes"], d["highs"], d["lows"]
        pcts, d_vols = d["pcts"], d["d_vols"]
        pct_50, deliv_20 = d["pct_sma50"], d["deliv_sma20"]
        obvs, vol_9 = d["obvs"], d["vol_sma9"]

        dots = (pcts >= (pct_m * pct_50)) & (d_vols >= (vol_m * deliv_20))
        dot_cumsum = np.cumsum(dots.astype(int))

        cooldown = 0
        for i in range(base_w + 10, N - 1):
            if i < cooldown:
                continue

            if vol_9[i] < MIN_AVG_VOLUME_9D:
                continue

            num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
            if num_dots < min_dots:
                continue

            base_highs = highs[i - base_w : i]
            sw_idx = int(np.argmax(base_highs))
            swing_high = base_highs[sw_idx]
            swing_obv = obvs[i - base_w + sw_idx]

            # Scan A Breakout Trigger
            if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                entry_p = closes[i]
                lookback_sl = min(12, base_w)
                recent_low = float(np.min(lows[i - lookback_sl : i]))
                sl_p = round(recent_low * 0.995, 2)
                risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                if risk_pct <= 0 or risk_pct > max_risk:
                    continue

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

                    # Breakeven stop shift
                    if max_run >= be_trigger and active_sl < entry_p:
                        active_sl = entry_p

                    # Book 50% at Partial Target
                    if max_run >= target_pct and not booked_partial:
                        booked_partial = True
                        active_sl = entry_p

                    # Trail 10-day swing lows on remaining half
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

                trades.append({
                    "return": realized_ret,
                    "win": realized_ret > 0,
                    "r20": max_run >= 20.0,
                    "days": days,
                    "risk": risk_pct
                })

                cooldown = i + max(days, 8)

    if not trades:
        return {"trades": 0, "win_rate": 0, "pf": 0, "avg_ret": 0, "r20": 0, "avg_risk": 0, "avg_hold": 0, "score": 0}

    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    win_rate = round((wins / total) * 100.0, 1)
    r20 = round((sum(1 for t in trades if t["r20"]) / total) * 100.0, 1)
    returns = [t["return"] for t in trades]
    avg_ret = round(float(np.mean(returns)), 2)
    pos = sum(r for r in returns if r > 0)
    neg = abs(sum(r for r in returns if r < 0))
    pf = round(pos / neg, 2) if neg > 0 else 99.0
    avg_risk = round(float(np.mean([t["risk"] for t in trades])), 2)
    avg_hold = round(float(np.mean([t["days"] for t in trades])), 1)

    # Composite Score = Profit Factor x 2.0 + Realized Return x 0.5 + Win Rate x 0.05
    score = round((pf * 2.0) + (avg_ret * 0.5) + (win_rate * 0.05), 2)

    return {
        "trades": total,
        "win_rate": f"{win_rate}%",
        "pf": pf,
        "avg_ret": f"{'+' if avg_ret >= 0 else ''}{avg_ret}%",
        "r20": f"{r20}%",
        "avg_risk": f"{avg_risk}%",
        "avg_hold": f"{avg_hold}d",
        "score": score
    }

def run_single_factor_optimization():
    print("🚀 Starting Scan A Single-Factor Parameter Optimization...")
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
            "optimal_strategies.json", "scana_sensitivity_report.json"
        ]
    ]

    if universe and len(universe) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in universe]

    # Pre-cache arrays
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

    print(f"📦 Pre-cached {len(data_cache)} stocks into RAM.")

    # Sweep each parameter one-at-a-time while locking others to BASELINE
    report_sections = {}

    for param_name, values in SWEEP_CONFIG.items():
        print(f"\n🔍 Testing Factor: {param_name} across {values}...")
        factor_results = []

        for val in values:
            cfg = BASELINE.copy()

            if "Base Window" in param_name: cfg["base_w"] = val
            elif "Delivery Vol Multiplier" in param_name: cfg["vol_m"] = val
            elif "Delivery Pct Multiplier" in param_name: cfg["pct_m"] = val
            elif "Clustered Dots in Base" in param_name: cfg["min_dots"] = val
            elif "Initial Stop Loss Cap" in param_name: cfg["max_risk"] = val
            elif "Partial Target" in param_name: cfg["target_pct"] = val
            elif "Breakeven Shift Trigger" in param_name: cfg["be_trigger"] = val

            stats = simulate_scan_a(
                data_cache,
                cfg["base_w"], cfg["vol_m"], cfg["pct_m"],
                cfg["min_dots"], cfg["max_risk"], cfg["target_pct"], cfg["be_trigger"]
            )

            is_base = (val == BASELINE.get(
                "base_w" if "Base Window" in param_name else
                "vol_m" if "Delivery Vol" in param_name else
                "pct_m" if "Delivery Pct" in param_name else
                "min_dots" if "Clustered Dots" in param_name else
                "max_risk" if "Initial Stop" in param_name else
                "target_pct" if "Partial Target" in param_name else
                "be_trigger"
            ))

            factor_results.append({
                "Value": f"{val}d" if "Window" in param_name else f"{val}x" if "Multiplier" in param_name else f"{val} dots" if "Dots" in param_name else f"{val}%",
                "Is_Baseline": is_base,
                "Metrics": stats
            })

        # Identify best preset for this parameter by Score
        factor_results.sort(key=lambda x: x["Metrics"]["score"], reverse=True)
        report_sections[param_name] = {
            "Best Preset": factor_results[0]["Value"],
            "Evaluations": factor_results
        }

    payload = {
        "Sensitivity Run Time": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Baseline Definition": BASELINE,
        "Factors Analyzed": report_sections
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"\n🎉 One-Factor-At-A-Time Sweep Complete! Results saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_single_factor_optimization()
