import os
import json
import time
import itertools
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "backtest_breakout_report.json")
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
    "backtest_universe_report.json", "backtest_breakout_report.json"
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

def load_universe_in_memory():
    os.makedirs(DATA_DIR, exist_ok=True)
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED_FILES]
    print(f"📦 Loading {len(stock_files)} stocks into fast memory...")

    stocks = []
    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            if not isinstance(raw, list) or len(raw) < 160:
                continue

            clean = []
            for r in raw:
                if isinstance(r, dict) and float(r.get("close", 0) or 0) > 0:
                    clean.append(r)

            if len(clean) < 160:
                continue

            closes = np.array([float(r.get("close", 0)) for r in clean], dtype=float)
            highs = np.array([float(r.get("high", 0)) for r in clean], dtype=float)
            lows = np.array([float(r.get("low", 0)) for r in clean], dtype=float)
            vols = np.array([float(r.get("volume", 0)) for r in clean], dtype=float)
            d_vols = np.array([float(r.get("delivery_vol", 0)) for r in clean], dtype=float)
            pcts = np.array([float(r.get("deliv_pct", 0)) for r in clean], dtype=float)
            times = [str(r.get("time", ""))[:10] for r in clean]
            N = len(closes)

            # Institutional Demat OBV
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
            vol_sma20 = fast_rolling(vols, 20)
            turnover_cr = (closes * vols) / 1e7
            to_50 = fast_rolling(turnover_cr, 50)

            stocks.append({
                "sym": sym, "closes": closes, "highs": highs, "lows": lows, "vols": vols,
                "times": times, "obvs": obvs, "to_50": to_50,
                "d_vols": d_vols, "pcts": pcts, "d_sma20": d_sma20,
                "pct_sma50": pct_sma50, "vol_sma20": vol_sma20, "N": N
            })
        except Exception:
            continue

    print(f"✅ Universe ready: {len(stocks)} liquid equities configured.")
    return stocks

def run_consolidation_sweep():
    t0 = time.time()
    stocks = load_universe_in_memory()
    if not stocks:
        print("No stock data available.")
        return

    # -------------------------------------------------------------------------
    # PARAMETER COMBINATION GRID FOR CONSOLIDATION BREAKOUT
    # -------------------------------------------------------------------------
    GRID_BASE_WINDOWS = [60, 80, 120, 160]         # 3 months, 4 months, 6 months, 8 months
    GRID_MAX_BOX_DEPTH = [25.0, 35.0, 45.0]        # Tight (<25%), Standard (<35%), Loose (<45%)
    GRID_MIN_DOTS = [4, 6, 8]                      # Accumulation density in base
    GRID_BOOK_TARGETS = [
        {"book_pct": 25.0, "be_trig": 12.0, "trail_w": 10}, # Swing runner
        {"book_pct": 40.0, "be_trig": 18.0, "trail_w": 20}, # Trend runner
    ]

    combos = list(itertools.product(
        GRID_BASE_WINDOWS,
        GRID_MAX_BOX_DEPTH,
        GRID_MIN_DOTS,
        GRID_BOOK_TARGETS
    ))

    print(f"⚙️ Sweeping across {len(combos)} distinct consolidation presets...")

    leaderboard = []

    for idx, (base_w, max_box, min_dots, exit_plan) in enumerate(combos):
        trade_returns = []
        book_pct = exit_plan["book_pct"]
        be_trig = exit_plan["be_trig"]
        trail_w = exit_plan["trail_w"]

        for s in stocks:
            N = s["N"]
            if N < (base_w + 30):
                continue

            closes = s["closes"]
            highs = s["highs"]
            lows = s["lows"]
            vols = s["vols"]
            times = s["times"]
            obvs = s["obvs"]
            d_vols = s["d_vols"]
            pcts = s["pcts"]
            d_sma = s["d_sma20"]
            p_sma = s["pct_sma50"]
            v_sma = s["vol_sma20"]
            to_50 = s["to_50"]

            # Institutional Accumulation Spike (Dot)
            dots = (pcts >= (1.30 * p_sma)) & (d_vols >= (1.40 * d_sma))
            dot_cumsum = np.cumsum(dots.astype(int))

            cooldown = 0
            for i in range(base_w + 15, N - 1):
                if i < cooldown or times[i] < START_DATE:
                    continue

                # Liquidity filter: >= ₹2.0 Cr/day average turnover
                if to_50[i] < 2.0:
                    continue

                # Must have sufficient accumulation dots inside the base lookback
                num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - base_w)]
                if num_dots < min_dots:
                    continue

                base_highs = highs[i - base_w : i]
                base_lows = lows[i - base_w : i]
                macro_high = float(np.nanmax(base_highs))
                macro_low = float(np.nanmin(base_lows))
                if macro_low <= 0:
                    continue

                # Box tightness check
                box_pct = ((macro_high - macro_low) / macro_low) * 100.0
                if box_pct > max_box:
                    continue

                # Demat OBV Confirmation
                sw_idx = int(np.nanargmax(base_highs))
                obv_pos = i - base_w + sw_idx
                if obv_pos < 0 or obv_pos >= N:
                    continue

                # Breakout Trigger: Closes above base high with volume expansion and OBV confirmation
                if closes[i] > macro_high and closes[i - 1] <= macro_high:
                    if obvs[i] <= obvs[obv_pos]:
                        continue
                    if vols[i] < (1.50 * v_sma[i]):
                        continue

                    entry_p = float(closes[i])
                    sl_lookback = min(15, base_w)
                    recent_low = float(np.nanmin(lows[i - sl_lookback : i]))
                    sl_p = round(recent_low * 0.992, 2)
                    risk_pct = ((entry_p - sl_p) / entry_p) * 100.0

                    if risk_pct <= 0 or risk_pct > 10.0:
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

                        # Breakeven stop move
                        if max_run >= be_trig and not be_shifted and active_sl < entry_p:
                            active_sl = entry_p
                            be_shifted = True

                        # 50% Profit Booking
                        if max_run >= book_pct and not booked_partial:
                            booked_partial = True
                            active_sl = entry_p

                        # Trailing low on remainder
                        if booked_partial and bar_idx >= trail_w:
                            curr_bar = i + 1 + bar_idx
                            t_low = float(np.nanmin(lows[curr_bar - trail_w : curr_bar]))
                            if t_low > active_sl:
                                active_sl = t_low

                        # Hit stop or trail exit
                        if f_lows[bar_idx] <= active_sl:
                            exit_p = float(min(f_closes[bar_idx], active_sl))
                            hold_days = bar_idx + 1
                            break

                    raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                    # Deduct 0.8% round-trip friction (taxes + slippage)
                    realized_ret = ((book_pct * 0.50) + (raw_ret * 0.50)) - 0.80 if booked_partial else (raw_ret - 0.80)
                    trade_returns.append(realized_ret)
                    cooldown = i + max(hold_days, 10)

        total_trades = len(trade_returns)
        if total_trades >= 30:
            wins = sum(1 for r in trade_returns if r > 0)
            win_rate = round((wins / total_trades) * 100.0, 1)
            pos_sum = sum(r for r in trade_returns if r > 0)
            neg_sum = abs(sum(r for r in trade_returns if r < 0))
            pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0
            avg_ret = round(float(np.mean(trade_returns)), 2)

            leaderboard.append({
                "Preset ID": f"PRESET_{idx+1:02d}",
                "Base Window": f"{base_w}d",
                "Max Box": f"≤{int(max_box)}%",
                "Min Accum Dots": min_dots,
                "Target / BE": f"+{int(book_pct)}% / BE +{int(be_trig)}%",
                "Total Trades": total_trades,
                "Win Rate %": win_rate,
                "Profit Factor": pf,
                "Avg Net Return %": avg_ret
            })

    # Rank by Profit Factor descending, then Win Rate
    leaderboard.sort(key=lambda x: (x["Profit Factor"], x["Win Rate %"]), reverse=True)

    payload = {
        "Audit Date": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Universe Size": len(stocks),
        "Total Presets Tested": len(combos),
        "Top Ranked Presets": leaderboard[:25]
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"🎉 Complete in {time.time()-t0:.1f}s! Leaderboard saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_consolidation_sweep()
