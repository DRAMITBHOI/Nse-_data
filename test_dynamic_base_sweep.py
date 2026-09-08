import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "dynamic_base_report.json")
START_DATE = "2021-01-01"

RESERVED = {
    "fundamentals.json", "screener_results.json", "nifty750.json",
    "NIFTY50.json", "NIFTY.json", "fno_history.json",
    "wyckoff_screener_results.json", "obv_backtest_report.json",
    "init_progress.json", "backfill_progress.json",
    "backtest_breakout_report.json", "backtest_universe_report.json",
    "duration_sweep_report.json"
}

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

def run_dynamic_base_sweep():
    t0 = time.time()
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED]
    print(f"📦 Evaluating dynamic organic bases across {len(stock_files)} stocks...")

    stocks = []
    for f in stock_files:
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            if not isinstance(raw, list) or len(raw) < 180:
                continue

            closes = np.array([float(r.get("close", 0) or 0) for r in raw], dtype=float)
            highs = np.array([float(r.get("high", 0) or 0) for r in raw], dtype=float)
            lows = np.array([float(r.get("low", 0) or 0) for r in raw], dtype=float)
            vols = np.array([float(r.get("volume", 0) or 0) for r in raw], dtype=float)
            d_vols = np.array([float(r.get("delivery_vol", 0) or 0) for r in raw], dtype=float)
            pcts = np.array([float(r.get("deliv_pct", 0) or 0) for r in raw], dtype=float)
            times = [str(r.get("time", ""))[:10] for r in raw]
            N = len(closes)

            # Institutional Demat OBV
            obvs = np.zeros(N, dtype=float)
            cur = 0.0
            for idx in range(N):
                dv = d_vols[idx]
                if idx > 0:
                    if closes[idx] > closes[idx - 1]: cur += dv
                    elif closes[idx] < closes[idx - 1]: cur -= dv
                else:
                    cur = dv
                obvs[idx] = cur

            d_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)
            vol_sma20 = fast_rolling(vols, 20)
            turnover_cr = (closes * vols) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)

            stocks.append({
                "closes": closes, "highs": highs, "lows": lows,
                "vols": vols, "times": times, "obvs": obvs,
                "to_50": to_50, "vol_sma20": vol_sma20,
                "d_vols": d_vols, "pcts": pcts,
                "d_sma20": d_sma20, "pct_sma50": pct_sma50, "N": N
            })
        except Exception:
            continue

    print(f"✅ Loaded {len(stocks)} valid tickers. Running simulation...")

    # TEST MATRIX: Minimum Base Length vs. Required Dot Density (dots per 20 trading sessions)
    MIN_BASE_LENGTHS = [20, 30, 45, 60]
    DOT_DENSITY_FLOORS = [0.8, 1.2, 1.6]  # Dots per month of base length

    results = []

    for min_base in MIN_BASE_LENGTHS:
        for density_req in DOT_DENSITY_FLOORS:
            total_triggers = 0
            hit_15_count = 0
            failed_stop_count = 0
            runs = []
            detected_lengths = []

            for s in stocks:
                N = s["N"]
                closes = s["closes"]
                highs = s["highs"]
                lows = s["lows"]
                vols = s["vols"]
                times = s["times"]
                obvs = s["obvs"]
                to_50 = s["to_50"]
                vol_sma20 = s["vol_sma20"]
                d_vols = s["d_vols"]
                pcts = s["pcts"]
                d_sma20 = s["d_sma20"]
                pct_sma50 = s["pct_sma50"]

                # Accumulation dot condition
                dots = (pcts >= (1.30 * pct_sma50)) & (d_vols >= (1.40 * d_sma20))
                dot_cumsum = np.cumsum(dots.astype(int))

                cooldown = 0
                for i in range(160, N - 1):
                    if i < cooldown or times[i] < START_DATE:
                        continue

                    # Liquidity Filter: >= ₹2 Cr/day
                    if to_50[i] < 2.0:
                        continue

                    # Step 1: Detect organic base dynamically
                    # Start at min_base and expand backwards as long as box <= 28%
                    best_w = 0
                    for w in range(min_base, 161, 5):
                        h_max = float(np.nanmax(highs[i - w : i]))
                        l_min = float(np.nanmin(lows[i - w : i]))
                        if l_min <= 0:
                            break
                        box_depth = ((h_max - l_min) / l_min) * 100.0
                        if box_depth <= 28.0:
                            best_w = w
                        else:
                            break

                    if best_w < min_base:
                        continue

                    # Base ceiling & floor
                    base_high = float(np.nanmax(highs[i - best_w : i]))

                    # Breakout check
                    if not (closes[i] > base_high and closes[i - 1] <= base_high):
                        continue

                    # Volume Expansion check
                    if vols[i] < (1.40 * vol_sma20[i]):
                        continue

                    # Step 2: Accumulation Density check inside this exact organic base
                    dot_count = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - best_w)]
                    density = (dot_count / best_w) * 20.0
                    if density < density_req:
                        continue

                    # Step 3: Demat OBV Confirmation
                    peak_idx = int(np.nanargmax(highs[i - best_w : i]))
                    obv_pos = i - best_w + peak_idx
                    if obv_pos < 0 or obv_pos >= N or obvs[i] <= obvs[obv_pos]:
                        continue

                    # Forward simulation
                    entry_p = float(closes[i])
                    fwd_end = min(N, i + 1 + 60)
                    f_highs = highs[i + 1 : fwd_end]
                    f_lows = lows[i + 1 : fwd_end]

                    if len(f_highs) < 5:
                        continue

                    total_triggers += 1
                    detected_lengths.append(best_w)
                    max_gain = ((np.max(f_highs) - entry_p) / entry_p) * 100.0
                    runs.append(max_gain)

                    hit_target = False
                    hit_stop = False
                    for b in range(len(f_highs)):
                        g = ((f_highs[b] - entry_p) / entry_p) * 100.0
                        l = ((f_lows[b] - entry_p) / entry_p) * 100.0

                        if g >= 15.0:
                            hit_target = True
                            break
                        if l <= -7.0:
                            hit_stop = True
                            break

                    if hit_target:
                        hit_15_count += 1
                    elif hit_stop:
                        failed_stop_count += 1

                    cooldown = i + 15

            if total_triggers >= 20:
                succ_rate = round((hit_15_count / total_triggers) * 100.0, 1)
                fail_rate = round((failed_stop_count / total_triggers) * 100.0, 1)
                avg_len = round(float(np.mean(detected_lengths)), 1)
                avg_run = round(float(np.mean(runs)), 1)

                results.append({
                    "Min Base Constraint": f"≥{min_base}d",
                    "Required Dot Density": f"≥{density_req} dots/mo",
                    "Total Triggers": total_triggers,
                    "Avg Detected Base": f"{avg_len}d (~{round(avg_len/20, 1)} mo)",
                    "Hit +15% First (Win %)": f"{succ_rate}%",
                    "Hit -7% Stop First": f"{fail_rate}%",
                    "Avg Post-Breakout Run": f"+{avg_run}%"
                })

    results.sort(key=lambda x: float(x["Hit +15% First (Win %)"].replace("%", "")), reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_dynamic_base_sweep()
