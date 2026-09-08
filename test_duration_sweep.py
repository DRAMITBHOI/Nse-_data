import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "duration_sweep_report.json")
START_DATE = "2021-01-01"

RESERVED_FILES = {
    "fundamentals.json", "screener_results.json", "nifty750.json",
    "NIFTY50.json", "NIFTY.json", "fno_history.json",
    "wyckoff_screener_results.json", "obv_backtest_report.json",
    "init_progress.json", "backfill_progress.json",
    "backtest_breakout_report.json", "backtest_universe_report.json"
}

DURATIONS_TO_TEST = [15, 30, 45, 60, 90, 120, 160]

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

def run_duration_analysis():
    t0 = time.time()
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED_FILES]
    print(f"📦 Analyzing consolidation durations across {len(stock_files)} stocks...")

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
            times = [str(r.get("time", ""))[:10] for r in raw]
            N = len(closes)

            turnover_cr = (closes * vols) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)
            vol_sma20 = fast_rolling(vols, 20)

            stocks.append({
                "closes": closes, "highs": highs, "lows": lows,
                "vols": vols, "times": times, "to_50": to_50,
                "vol_sma20": vol_sma20, "N": N
            })
        except Exception:
            continue

    print(f"✅ Loaded {len(stocks)} liquid tickers.")

    results = []

    for base_w in DURATIONS_TO_TEST:
        total_breakouts = 0
        hit_15_count = 0
        failed_first_count = 0
        max_runs = []

        for s in stocks:
            N = s["N"]
            if N < (base_w + 30):
                continue

            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            vols = s["vols"]
            times = s["times"]
            to_50 = s["to_50"]
            vol_sma20 = s["vol_sma20"]

            cooldown = 0
            for i in range(base_w + 10, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                # Liquidity Filter: >= ₹2.0 Cr/day average turnover
                if to_50[i] < 2.0:
                    continue

                base_highs = highs[i - base_w : i]
                base_lows = lows[i - base_w : i]
                macro_high = float(np.nanmax(base_highs))
                macro_low = float(np.nanmin(base_lows))
                if macro_low <= 0:
                    continue

                # Box Depth <= 30%
                box_pct = ((macro_high - macro_low) / macro_low) * 100.0
                if box_pct > 30.0:
                    continue

                # Breakout Condition: Close above base high with expanding volume
                if closes[i] > macro_high and closes[i - 1] <= macro_high and vols[i] >= (1.4 * vol_sma20[i]):
                    entry_p = float(closes[i])
                    stop_p = entry_p * 0.93  # 7% hard failure threshold

                    # Evaluate forward 60 trading days
                    fwd_end = min(N, i + 1 + 60)
                    f_highs = highs[i + 1 : fwd_end]
                    f_lows = lows[i + 1 : fwd_end]

                    if len(f_highs) < 5:
                        continue

                    total_breakouts += 1
                    highest_gain = ((np.max(f_highs) - entry_p) / entry_p) * 100.0
                    max_runs.append(highest_gain)

                    # Check what happened first: +15% gain or -7% stop hit
                    hit_target = False
                    hit_stop = False
                    for b in range(len(f_highs)):
                        gain = ((f_highs[b] - entry_p) / entry_p) * 100.0
                        loss = ((f_lows[b] - entry_p) / entry_p) * 100.0

                        if gain >= 15.0:
                            hit_target = True
                            break
                        if loss <= -7.0:
                            hit_stop = True
                            break

                    if hit_target:
                        hit_15_count += 1
                    elif hit_stop:
                        failed_first_count += 1

                    cooldown = i + 15

        if total_breakouts > 0:
            success_rate = round((hit_15_count / total_breakouts) * 100.0, 1)
            failure_rate = round((failed_first_count / total_breakouts) * 100.0, 1)
            median_run = round(float(np.median(max_runs)), 1)
            avg_run = round(float(np.mean(max_runs)), 1)

            results.append({
                "Duration (Days)": f"{base_w} sessions (~{round(base_w/20, 1)} mos)",
                "Total Setups": total_breakouts,
                "Hit +15% Before -7% (Success %)": f"{success_rate}%",
                "Hit -7% Stop First (Failure %)": f"{failure_rate}%",
                "Median Post-Breakout Run": f"+{median_run}%",
                "Average Post-Breakout Run": f"+{avg_run}%"
            })

    print("\n" + pd.DataFrame(results).to_string(index=False))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    run_duration_analysis()
