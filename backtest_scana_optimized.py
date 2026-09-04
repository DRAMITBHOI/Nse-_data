import os
import io
import json
import time
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "scana_optimized_report.json")
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 50000

# COMBINED OPTIMAL PRESET
OPT_CONFIG = {
    "base_w": 40,          # 1. Base Window: 40d
    "vol_m": 1.50,         # 2. Delivery Vol Multiplier: 1.5x
    "pct_m": 1.40,         # 3. Delivery Pct Multiplier: 1.4x
    "min_dots": 5,         # 4. Clustered Dots in Base: 5 dots
    "max_risk": 5.0,       # 5. Initial Stop Loss Cap: 5.0%
    "target_pct": 18.0,    # 6. Partial Target: +18.0%
    "be_trigger": 15.0     # 7. Breakeven Shift Trigger: +15.0%
}

# BASELINE FOR HEAD-TO-HEAD COMPARISON
BASE_CONFIG = {
    "base_w": 25,
    "vol_m": 1.20,
    "pct_m": 1.20,
    "min_dots": 2,
    "max_risk": 8.0,
    "target_pct": 15.0,
    "be_trigger": 8.0
}

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

def run_simulation(data_cache, cfg):
    trades = []
    base_w = cfg["base_w"]
    vol_m = cfg["vol_m"]
    pct_m = cfg["pct_m"]
    min_dots = cfg["min_dots"]
    max_risk = cfg["max_risk"]
    target_pct = cfg["target_pct"]
    be_trigger = cfg["be_trigger"]

    for sym, d in data_cache.items():
        N = d["N"]
        closes, highs, lows = d["closes"], d["highs"], d["lows"]
        pcts, d_vols = d["pcts"], d["d_vols"]
        pct_50, deliv_20 = d["pct_sma50"], d["deliv_sma20"]
        obvs, times, vol_9 = d["obvs"], d["times"], d["vol_sma9"]

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

                    # Breakeven shift
                    if max_run >= be_trigger and active_sl < entry_p:
                        active_sl = entry_p

                    # Partial profit booking
                    if max_run >= target_pct and not booked_partial:
                        booked_partial = True
                        active_sl = entry_p

                    # Trailing swing lows on remaining half
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
                    "Symbol": sym,
                    "Entry Date": times[i],
                    "Entry": entry_p,
                    "Exit": round(exit_p, 2),
                    "Risk %": risk_pct,
                    "Realized %": realized_ret,
                    "Max Run %": round(max_run, 2),
                    "Rally 20%": bool(max_run >= 20.0),
                    "Hold Days": days,
                    "Is Win": bool(realized_ret > 0)
                })

                cooldown = i + max(days, 8)

    return trades

def calc_stats(trades):
    if not trades:
        return {"Trades": 0, "Win Rate %": "0%", "+20% Expansion": "0%", "Avg Return %": "0%", "Profit Factor": 0.0, "Avg Risk %": "0%", "Avg Hold": "0 d", "Score": 0.0}
    df = pd.DataFrame(trades)
    total = len(df)
    wins = len(df[df["Is Win"] == True])
    win_rate = round((wins / total) * 100.0, 1)
    r20 = round((len(df[df["Rally 20%"] == True]) / total) * 100.0, 1)
    avg_ret = round(float(df["Realized %"].mean()), 2)
    pos = df[df["Is Win"] == True]["Realized %"].sum()
    neg = abs(df[df["Is Win"] == False]["Realized %"].sum())
    pf = round(pos / neg, 2) if neg > 0 else 99.0
    avg_risk = round(float(df["Risk %"].mean()), 2)
    avg_hold = round(float(df["Hold Days"].mean()), 1)
    score = round((pf * 2.0) + (avg_ret * 0.5) + (win_rate * 0.05), 2)

    return {
        "Trades": total,
        "Win Rate %": f"{win_rate}%",
        "+20% Expansion": f"{r20}%",
        "Avg Return %": f"{'+' if avg_ret >= 0 else ''}{avg_ret}%",
        "Profit Factor": pf,
        "Avg Risk %": f"{avg_risk}%",
        "Avg Hold": f"{avg_hold} d",
        "Score": score
    }

def main():
    print("🚀 Executing Combined Optimal Scan A Backtest...")
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
            "optimal_strategies.json", "scana_sensitivity_report.json",
            "scana_optimized_report.json"
        ]
    ]

    if universe and len(universe) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in universe]

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

    print(f"📦 Pre-cached {len(data_cache)} stocks into memory.")

    print("\n⏳ Running Baseline Model...")
    trades_base = run_simulation(data_cache, BASE_CONFIG)
    stats_base = calc_stats(trades_base)

    print("⏳ Running Combined Optimal Model...")
    trades_opt = run_simulation(data_cache, OPT_CONFIG)
    stats_opt = calc_stats(trades_opt)

    print("\n" + "="*75)
    print("🎯 COMBINED SCAN A OPTIMIZATION RESULTS")
    print("="*75)
    print(f"Original Baseline Model : {stats_base['Trades']} Trades | Win Rate {stats_base['Win Rate %']} | Return {stats_base['Avg Return %']} | PF {stats_base['Profit Factor']} | Score {stats_base['Score']}")
    print(f"Combined Optimal Model  : {stats_opt['Trades']} Trades | Win Rate {stats_opt['Win Rate %']} | Return {stats_opt['Avg Return %']} | PF {stats_opt['Profit Factor']} | Score {stats_opt['Score']}")
    print("="*75)

    payload = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Optimal Configuration": OPT_CONFIG,
        "Baseline Configuration": BASE_CONFIG,
        "Summary": {
            "Original Baseline": stats_base,
            "Combined Optimal Model": stats_opt
        },
        "Recent Optimal Trades": trades_opt[-30:] if len(trades_opt) >= 30 else trades_opt
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"📁 Full report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
