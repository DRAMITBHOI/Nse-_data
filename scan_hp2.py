import os
import json
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scan_hp2_results.json")

# RANK #16 EXACT PARAMETERS
CONFIG = {
    "name": "Rank #16 HP2",
    "desc": "Base 40d | Box <= 35% | >=4 Dots (1.5x Vol, 1.3x Deliv) | Max SL 8% | Book +15% | BE +10%",
    "base_w": 40,
    "max_box": 35.0,
    "vol_m": 1.50,
    "pct_m": 1.30,
    "min_dots": 4,
    "max_risk": 8.0,
    "book_pct": 15.0,
    "be_trig": 10.0,
    "min_turnover_cr": 1.0,  # Turnover floor: ₹1 Cr/d replaces fragile nifty750.json dependency
    "max_pe": 35.0
}

def clean_data_fast(raw_data, min_len=60):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()[:10]
        if len(raw_t) < 10:
            continue
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0 or np.isnan(c):
                continue
            
            entry = {
                "time": raw_t,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            }
            if raw_t not in date_map or entry["volume"] > date_map[raw_t]["volume"]:
                date_map[raw_t] = entry
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

def fast_rolling_mean(arr, window):
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
    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {}

def scan_hp2():
    t0 = time.time()
    print("🚀 Running Scan HP2 Production Scanner (Rank #16 Configuration)...")
    print(f"⚙️ Settings: {CONFIG['desc']}")

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

    live_candidates = []
    trade_visualizer_cache = {}

    base_w = CONFIG["base_w"]
    max_box = CONFIG["max_box"]
    vol_m = CONFIG["vol_m"]
    pct_m = CONFIG["pct_m"]
    min_dots = CONFIG["min_dots"]
    max_risk = CONFIG["max_risk"]
    target_pct = CONFIG["book_pct"]
    be_trig = CONFIG["be_trig"]
    min_to = CONFIG["min_turnover_cr"]
    max_pe = CONFIG["max_pe"]

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        pe_float = None
        if pe_val is not None:
            try:
                pe_float = float(pe_val)
                if pe_float <= 0 or pe_float >= max_pe:
                    continue
            except (ValueError, TypeError):
                pass

        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except Exception:
            continue

        clean = clean_data_fast(raw, min_len=60)
        if len(clean) < (base_w + 15):
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
        d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # Dynamic turnover check (>= ₹1.0 Cr/d)
        turnover_cr = (closes * volumes) / 1e7
        to_50 = fast_rolling_mean(turnover_cr, 50)
        if float(to_50[-1]) < min_to:
            continue

        deliv_sma20 = fast_rolling_mean(d_vols, 20)
        pct_sma50 = fast_rolling_mean(pcts, 50)

        # Demat OBV
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

        # Rank #16 dots: 1.5x Vol, 1.3x Deliv%
        dots = (pcts >= (pct_m * pct_sma50)) & (d_vols >= (vol_m * deliv_sma20))
        dot_indices = [int(x) for x in np.where(dots)[0]]
        dot_cumsum = np.cumsum(dots.astype(int))

        # -----------------------------------------------------------------
        # 1. HISTORICAL SIMULATION (CACHE FOR VISUALIZER)
        # -----------------------------------------------------------------
        symbol_trades = []
        cooldown = 0

        for i in range(base_w + 10, N):
            if i < cooldown:
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
            swing_high = base_highs[sw_idx]
            swing_obv = obvs[i - base_w + sw_idx]

            if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                entry_p = float(closes[i])
                lookback_sl = min(12, base_w)
                recent_low = float(np.nanmin(lows[i - lookback_sl : i]))
                sl_p = round(recent_low * 0.992, 2)
                risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                if risk_pct <= 0 or risk_pct > max_risk:
                    continue

                fwd_end = min(N, i + 1 + 90)
                f_highs = highs[i + 1 : fwd_end]
                f_lows = lows[i + 1 : fwd_end]
                f_closes = closes[i + 1 : fwd_end]

                max_run = 0.0
                active_sl = sl_p
                booked_partial = False
                be_shifted = False
                partial_booked_idx = None
                exit_p = float(f_closes[-1]) if len(f_closes) > 0 else entry_p
                exit_idx = fwd_end - 1 if len(f_closes) > 0 else i
                is_closed = False

                for bar_idx in range(len(f_highs)):
                    curr_bar = i + 1 + bar_idx
                    gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                    if gain > max_run:
                        max_run = gain

                    if max_run >= be_trig and not be_shifted and active_sl < entry_p:
                        active_sl = entry_p
                        be_shifted = True

                    if max_run >= target_pct and not booked_partial:
                        booked_partial = True
                        partial_booked_idx = curr_bar
                        active_sl = entry_p

                    if booked_partial and bar_idx >= 10:
                        t_low = float(np.nanmin(lows[curr_bar - 10 : curr_bar]))
                        if t_low > active_sl:
                            active_sl = t_low

                    if f_lows[bar_idx] <= active_sl:
                        exit_p = float(min(f_closes[bar_idx], active_sl))
                        exit_idx = curr_bar
                        is_closed = True
                        break

                raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                realized_ret = round((target_pct * 0.50) + (raw_ret * 0.50), 2) if booked_partial else round(raw_ret, 2)

                symbol_trades.append({
                    "entry_idx": i,
                    "entry_date": times[i],
                    "entry_price": entry_p,
                    "initial_sl": sl_p,
                    "target_price": round(entry_p * (1.0 + target_pct / 100.0), 2),
                    "partial_booked_idx": partial_booked_idx,
                    "exit_idx": exit_idx if is_closed else None,
                    "exit_date": times[exit_idx] if is_closed else None,
                    "exit_price": round(exit_p, 2) if is_closed else None,
                    "risk_pct": risk_pct,
                    "realized_ret": realized_ret if is_closed else round(raw_ret, 2),
                    "is_closed": is_closed
                })

                cooldown = exit_idx + 8 if is_closed else N

        if symbol_trades:
            trade_visualizer_cache[sym] = {
                "dots": dot_indices,
                "trades": symbol_trades
            }

        # -----------------------------------------------------------------
        # 2. LIVE SCREENER EVALUATION (3-STAGE RADAR)
        # -----------------------------------------------------------------
        curr_idx = N - 1
        num_dots_curr = dot_cumsum[curr_idx] - dot_cumsum[max(0, curr_idx - base_w)]

        if num_dots_curr >= min_dots:
            base_highs = highs[curr_idx - base_w : curr_idx]
            base_lows = lows[curr_idx - base_w : curr_idx]
            macro_high = float(np.nanmax(base_highs))
            macro_low = float(np.nanmin(base_lows))
            if macro_low <= 0:
                continue

            box_depth = ((macro_high - macro_low) / macro_low) * 100.0
            if box_depth > max_box:
                continue

            sw_idx = int(np.nanargmax(base_highs))
            curr_swing_high = base_highs[sw_idx]
            curr_swing_obv = obvs[curr_idx - base_w + sw_idx]

            lookback_sl = min(12, base_w)
            recent_low = float(np.nanmin(lows[curr_idx - lookback_sl : curr_idx]))
            sl_price = round(recent_low * 0.992, 2)
            curr_close = float(closes[curr_idx])
            risk_pct = round(((curr_close - sl_price) / curr_close) * 100.0, 2)

            is_triggered = (curr_close >= curr_swing_high) and (obvs[curr_idx] > curr_swing_obv)
            dist_pct = round(((curr_swing_high - curr_close) / curr_swing_high) * 100.0, 2)

            if is_triggered and (0 < risk_pct <= max_risk):
                status = "🟢 BREAKOUT TRIGGERED"
            elif 0.0 <= dist_pct <= 3.0 and obvs[curr_idx] > curr_swing_obv:
                status = "🟡 READY AT CEILING"
            elif 3.0 < dist_pct <= 12.0:
                status = "🔵 DEEP ACCUMULATION"
            else:
                status = None

            if status:
                live_candidates.append({
                    "Symbol": sym,
                    "Status": status,
                    "LTP": round(curr_close, 2),
                    "Breakout Level": round(curr_swing_high, 2),
                    "Stop Loss": sl_price,
                    "Risk %": f"{risk_pct}%" if risk_pct > 0 else "N/A",
                    "50% Target (+15%)": round(curr_close * 1.15, 2),
                    "Demat Dots": f"{int(num_dots_curr)} dots (≥1.5x Vol, 1.3x Del)",
                    "Distance to Breakout": "0.0%" if is_triggered else f"-{dist_pct}%",
                    "P/E": f"{pe_float:.1f}" if pe_float else "N/A"
                })

    status_weights = {
        "🟢 BREAKOUT TRIGGERED": 0,
        "🟡 READY AT CEILING": 1,
        "🔵 DEEP ACCUMULATION": 2
    }
    live_candidates.sort(key=lambda x: (
        status_weights.get(x["Status"], 99),
        float(x["Distance to Breakout"].replace("%", "").replace("-", ""))
    ))

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Execution Duration Sec": round(time.time() - t0, 1),
        "Rank Config": CONFIG["desc"],
        "Total Live Candidates": len(live_candidates),
        "Candidates": live_candidates,
        "Visualizer Data": trade_visualizer_cache
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ HP2 Scan Complete in {time.time()-t0:.1f}s! Found {len(live_candidates)} candidates. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    scan_hp2()
