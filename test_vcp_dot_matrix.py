import os
import json
import time
import itertools
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "vcp_dot_matrix_report.json")
START_DATE = "2021-01-01"

RESERVED_FILES = {
    "fundamentals.json", "screener_results.json", "nifty750.json",
    "NIFTY50.json", "NIFTY.json", "fno_history.json",
    "wyckoff_screener_results.json", "obv_backtest_report.json",
    "init_progress.json", "backfill_progress.json",
    "backtest_breakout_report.json", "backtest_universe_report.json",
    "duration_sweep_report.json", "dynamic_base_report.json",
    "vcp_breakout_report.json", "vcp_comparison_report.json",
    "vcp_dot_matrix_report.json"
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

def compute_atr(highs, lows, closes, window=14):
    N = len(closes)
    tr = np.zeros(N, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, N):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    return fast_rolling(tr, window)

def run_matrix_sweep():
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED_FILES]
    print(f"📦 Loading {len(stock_files)} stocks into memory...")

    stocks = []
    for f in stock_files:
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            if not isinstance(raw, list) or len(raw) < 140:
                continue

            closes = np.array([float(r.get("close", 0) or 0) for r in raw], dtype=float)
            highs = np.array([float(r.get("high", 0) or 0) for r in raw], dtype=float)
            lows = np.array([float(r.get("low", 0) or 0) for r in raw], dtype=float)
            vols = np.array([float(r.get("volume", 0) or 0) for r in raw], dtype=float)
            d_vols = np.array([float(r.get("delivery_vol", 0) or 0) for r in raw], dtype=float)
            pcts = np.array([float(r.get("deliv_pct", 0) or 0) for r in raw], dtype=float)
            times = [str(r.get("time", ""))[:10] for r in raw]
            N = len(closes)

            if N < 140:
                continue

            # Demat OBV
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

            atr10 = compute_atr(highs, lows, closes, 10)
            atr40 = compute_atr(highs, lows, closes, 40)
            vol_sma20 = fast_rolling(vols, 20)
            vol_sma5 = fast_rolling(vols, 5)
            turnover_cr = (closes * vols) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)
            d_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)

            stocks.append({
                "closes": closes, "highs": highs, "lows": lows, "vols": vols,
                "times": times, "obvs": obvs, "to_50": to_50,
                "atr10": atr10, "atr40": atr40, "vol_sma20": vol_sma20,
                "vol_sma5": vol_sma5, "d_vols": d_vols, "pcts": pcts,
                "d_sma20": d_sma20, "pct_sma50": pct_sma50, "N": N
            })
        except Exception:
            continue

    print(f"✅ Loaded {len(stocks)} liquid tickers. Executing matrix sweep...")

    # Grid Dimensions
    GRID_DURATIONS = [30, 45, 60, 90]
    GRID_DOTS = [2, 4, 6]
    GRID_VCP = [True, False]

    combos = list(itertools.product(GRID_DURATIONS, GRID_DOTS, GRID_VCP))
    results = []

    for base_w, min_dots, use_vcp in combos:
        total_triggers = 0
        hit_15_count = 0
        hit_stop_count = 0
        runs = []

        for s in stocks:
            N = s["N"]
            if N < (base_w + 50):
                continue

            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            vols = s["vols"]
            times = s["times"]
            obvs = s["obvs"]
            to_50 = s["to_50"]
            atr10 = s["atr10"]
            atr40 = s["atr40"]
            vol_sma20 = s["vol_sma20"]
            vol_sma5 = s["vol_sma5"]
            d_vols = s["d_vols"]
            pcts = s["pcts"]
            d_sma20 = s["d_sma20"]
            pct_sma50 = s["pct_sma50"]

            dots = (pcts >= (1.30 * pct_sma50)) & (d_vols >= (1.40 * d_sma20))
            dot_cumsum = np.cumsum(dots.astype(int))

            cooldown = 0
            for i in range(base_w + 15, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                if to_50[i] < 2.0:
                    continue

                base_highs = highs[i - base_w : i]
                base_lows = lows[i - base_w : i]
                macro_high = float(np.nanmax(base_highs))
                macro_low = float(np.nanmin(base_lows))
                if macro_low <= 0:
                    continue

                if ((macro_high - macro_low) / macro_low) * 100.0 > 28.0:
                    continue

                # Accumulation dots filter
                dots_in_base = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
                if dots_in_base < min_dots:
                    continue

                # Demat OBV confirmation
                peak_idx = int(np.nanargmax(base_highs))
                obv_pos = i - base_w + peak_idx
                if obv_pos < 0 or obv_pos >= N or obvs[i] <= obvs[obv_pos]:
                    continue

                # Breakout price & volume check
                if not (closes[i] > macro_high and closes[i - 1] <= macro_high):
                    continue
                if vols[i] < (1.40 * vol_sma20[i]):
                    continue

                # VCP Filter evaluation
                last_10_h = float(np.nanmax(highs[i - 10 : i]))
                last_10_l = float(np.nanmin(lows[i - 10 : i]))
                last_10_box = ((last_10_h - last_10_l) / last_10_l) * 100.0 if last_10_l > 0 else 99.0

                is_vcp = (atr10[i - 1] <= (0.75 * atr40[i - 1])) and \
                         (vol_sma5[i - 1] <= (0.85 * vol_sma20[i - 1])) and \
                         (last_10_box <= 12.0)

                if use_vcp and not is_vcp:
                    continue
                if (not use_vcp) and is_vcp:
                    continue

                entry_p = float(closes[i])
                recent_l = float(np.nanmin(lows[i - 10 : i]))
                sl_p = max(recent_l * 0.995, entry_p * 0.92)

                fwd_end = min(N, i + 1 + 60)
                f_highs = highs[i + 1 : fwd_end]
                f_lows = lows[i + 1 : fwd_end]

                if len(f_highs) < 5:
                    continue

                total_triggers += 1
                max_gain = ((np.max(f_highs) - entry_p) / entry_p) * 100.0
                runs.append(max_gain)

                hit_15 = False
                hit_sl = False
                for b in range(len(f_highs)):
                    g = ((f_highs[b] - entry_p) / entry_p) * 100.0
                    if g >= 15.0:
                        hit_15 = True
                        break
                    if f_lows[b] <= sl_p:
                        hit_sl = True
                        break

                if hit_15:
                    hit_15_count += 1
                elif hit_sl:
                    hit_stop_count += 1

                cooldown = i + 15

        if total_triggers > 0:
            win_pct = round((hit_15_count / total_triggers) * 100.0, 1)
            fail_pct = round((hit_stop_count / total_triggers) * 100.0, 1)
            avg_run = round(float(np.mean(runs)), 1)
            median_run = round(float(np.median(runs)), 1)

            results.append({
                "Duration": f"{base_w}d",
                "Dots": f"≥{min_dots} dots",
                "VCP": "With VCP" if use_vcp else "Without VCP",
                "Triggers": total_triggers,
                "Hit +15% First (Win %)": win_pct,
                "Hit Stop First (Loss %)": fail_pct,
                "Median Run %": median_run,
                "Avg Run %": avg_run
            })

    results.sort(key=lambda x: (x["Hit +15% First (Win %)"], x["Avg Run %"]), reverse=True)

    payload = {
        "Audit Date": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Universe Size": len(stocks),
        "Results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_matrix_sweep()
