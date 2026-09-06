import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "backtest_universe_report.json")
START_DATE = "2021-01-01"

RESERVED_FILES = {
    "fundamentals.json", "screener_results.json", "nifty750.json",
    "NIFTY50.json", "NIFTY.json", "fno_history.json",
    "wyckoff_screener_results.json", "obv_backtest_report.json",
    "scana_vs_absorption_report.json", "scana_candidates.json",
    "optimal_strategies.json", "scana_sensitivity_report.json",
    "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
    "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
    "backtest_hp3_report.json", "scan_macro_results.json", "macro_combo_leaderboard.json",
    "bucket_optimization_leaderboard.json", "backtest_bucket_report.json",
    "screener_macro_buckets_results.json", "scan_ultra_results.json",
    "init_progress.json", "backfill_progress.json", "backtest_universe_matrix_report.json",
    "backtest_universe_report.json"
}

STRATEGIES = [
    {
        "id": "HP1",
        "name": "HP1 (Rank #3 Baseline)",
        "base_w": 40,
        "max_box": 35.0,
        "vol_m": 1.50,
        "pct_m": 1.40,
        "min_dots": 5,
        "max_risk": 8.0,
        "book_pct": 15.0,
        "be_trig": 15.0,
        "trail_w": 10,
        "min_turnover_cr": 1.0,
        "max_turnover_cr": 99999.0
    },
    {
        "id": "HP2",
        "name": "HP2 (Rank #16 Baseline)",
        "base_w": 40,
        "max_box": 35.0,
        "vol_m": 1.50,
        "pct_m": 1.30,
        "min_dots": 4,
        "max_risk": 8.0,
        "book_pct": 15.0,
        "be_trig": 10.0,
        "trail_w": 10,
        "min_turnover_cr": 1.0,
        "max_turnover_cr": 99999.0
    },
    {
        "id": "ULTRA_A",
        "name": "SCAN_ULTRA: Bucket A (>=30 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 8,
        "max_risk": 10.0,
        "book_pct": 50.0,
        "be_trig": 20.0,
        "trail_w": 20,
        "min_turnover_cr": 30.0,
        "max_turnover_cr": 99999.0
    },
    {
        "id": "ULTRA_B",
        "name": "SCAN_ULTRA: Bucket B (5-30 Cr/d)",
        "base_w": 120,
        "max_box": 40.0,
        "vol_m": 1.50,
        "pct_m": 1.40,
        "min_dots": 8,
        "max_risk": 12.0,
        "book_pct": 50.0,
        "be_trig": 20.0,
        "trail_w": 20,
        "min_turnover_cr": 5.0,
        "max_turnover_cr": 30.0
    },
    {
        "id": "ULTRA_C",
        "name": "SCAN_ULTRA: Bucket C (1-5 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 6,
        "max_risk": 12.0,
        "book_pct": 50.0,
        "be_trig": 20.0,
        "trail_w": 20,
        "min_turnover_cr": 1.0,
        "max_turnover_cr": 5.0
    },
    {
        "id": "ULTRA_D",
        "name": "SCAN_ULTRA: Bucket D (<1 Cr/d)",
        "base_w": 80,
        "max_box": 40.0,
        "vol_m": 1.40,
        "pct_m": 1.30,
        "min_dots": 8,
        "max_risk": 10.0,
        "book_pct": 50.0,
        "be_trig": 20.0,
        "trail_w": 20,
        "min_turnover_cr": 0.0,
        "max_turnover_cr": 1.0
    }
]

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

def run_universe_backtest():
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED_FILES]
    print(f"📦 Found {len(stock_files)} stock files in {DATA_DIR}. Loading into memory...")

    stocks = []
    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            if not isinstance(raw, list) or len(raw) < 80:
                continue

            clean_rows = []
            for r in raw:
                if not isinstance(r, dict):
                    continue
                c = float(r.get("close", 0) or 0)
                if c <= 0:
                    continue
                clean_rows.append(r)

            if len(clean_rows) < 80:
                continue

            closes = np.array([float(r.get("close", 0) or 0) for r in clean_rows], dtype=float)
            highs = np.array([float(r.get("high", 0) or 0) for r in clean_rows], dtype=float)
            lows = np.array([float(r.get("low", 0) or 0) for r in clean_rows], dtype=float)
            vols = np.array([float(r.get("volume", 0) or 0) for r in clean_rows], dtype=float)
            d_vols = np.array([float(r.get("delivery_vol", 0) or 0) for r in clean_rows], dtype=float)
            pcts = np.array([float(r.get("deliv_pct", 0) or 0) for r in clean_rows], dtype=float)
            times = [str(r.get("time", ""))[:10] for r in clean_rows]
            N = len(closes)

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

            d_sma20 = fast_rolling(d_vols, 20)
            pct_sma50 = fast_rolling(pcts, 50)
            turnover_cr = (closes * vols) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)

            stocks.append({
                "sym": sym, "closes": closes, "highs": highs, "lows": lows,
                "times": times, "obvs": obvs, "to_50": to_50,
                "d_vols": d_vols, "pcts": pcts, "d_sma20": d_sma20,
                "pct_sma50": pct_sma50, "N": N
            })
        except Exception:
            continue

    print(f"✅ Loaded {len(stocks)} valid tickers. Simulating strategies...")

    results_table = []

    for strat in STRATEGIES:
        strat_id = strat["id"]
        strat_name = strat["name"]
        base_w = strat["base_w"]
        max_box = strat["max_box"]
        vol_m = strat["vol_m"]
        pct_m = strat["pct_m"]
        min_dots = strat["min_dots"]
        max_risk = strat["max_risk"]
        book_pct = strat["book_pct"]
        be_trig = strat["be_trig"]
        trail_w = strat["trail_w"]
        min_to = strat["min_turnover_cr"]
        max_to = strat["max_turnover_cr"]

        trade_returns = []

        for s in stocks:
            N = s["N"]
            if N < (base_w + 20):
                continue

            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            times = s["times"]
            obvs = s["obvs"]
            d_vols = s["d_vols"]
            pcts = s["pcts"]
            d_sma = s["d_sma20"]
            p_sma = s["pct_sma50"]
            to_50 = s["to_50"]

            dots = (pcts >= (pct_m * p_sma)) & (d_vols >= (vol_m * d_sma))
            dot_cumsum = np.cumsum(dots.astype(int))

            cooldown = 0
            for i in range(base_w + 10, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                if not (min_to <= to_50[i] < max_to):
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

                box_depth = ((macro_high - macro_low) / macro_low) * 100.0
                if box_depth > max_box:
                    continue

                sw_idx = int(np.nanargmax(base_highs))
                obv_pos = i - base_w + sw_idx
                if obv_pos < 0 or obv_pos >= N:
                    continue

                if closes[i] > macro_high and closes[i - 1] <= macro_high and obvs[i] > obvs[obv_pos]:
                    entry_p = float(closes[i])
                    lookback_sl = min(15 if base_w >= 80 else 12, base_w)
                    recent_low = float(np.nanmin(lows[i - lookback_sl : i]))
                    sl_p = round(recent_low * 0.992, 2)
                    risk_pct = ((entry_p - sl_p) / entry_p) * 100.0

                    if risk_pct <= 0 or risk_pct > max_risk:
                        continue

                    fwd_end = min(N, i + 1 + 120)
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
                            curr_bar = i + 1 + bar_idx
                            t_low = float(np.nanmin(lows[curr_bar - trail_w : curr_bar]))
                            if t_low > active_sl:
                                active_sl = t_low

                        if f_lows[bar_idx] <= active_sl:
                            exit_p = float(min(f_closes[bar_idx], active_sl))
                            hold_days = bar_idx + 1
                            break

                    raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                    realized_ret = (book_pct * 0.50) + (raw_ret * 0.50) if booked_partial else raw_ret
                    trade_returns.append(realized_ret)
                    cooldown = i + max(hold_days, 8)

        total_trades = len(trade_returns)
        if total_trades > 0:
            wins = sum(1 for r in trade_returns if r > 0)
            win_rate = round((wins / total_trades) * 100.0, 1)
            pos_sum = sum(r for r in trade_returns if r > 0)
            neg_sum = abs(sum(r for r in trade_returns if r < 0))
            pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0
            avg_ret = round(float(np.mean(trade_returns)), 2)

            results_table.append({
                "Strategy ID": strat_id,
                "Strategy Name": strat_name,
                "Base Window": f"{base_w}d",
                "Max Box": f"<={int(max_box)}%",
                "Total Trades": total_trades,
                "Win Rate %": win_rate,
                "Profit Factor": pf,
                "Avg Return %": avg_ret
            })

    results_table.sort(key=lambda x: (x["Win Rate %"], x["Profit Factor"]), reverse=True)

    payload = {
        "Backtest Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Total Universe Tested": len(stocks),
        "Results": results_table
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_universe_backtest()
