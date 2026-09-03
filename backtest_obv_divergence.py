import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "obv_backtest_report.json")
MAX_PE = 35.0
MAX_RISK_PCT = 8.0          # Stop loss strictly <= 8%
PARTIAL_TARGET_PCT = 15.0   # Book 50% at +15%
BREAKEVEN_TRIGGER_PCT = 9.0 # Move SL to Cost once gain touches +9%
MIN_AVG_VOLUME_9D = 50000
MIN_PRICE_DROP_PCT = -7.5
MIN_OBV_GAIN_PCT = 8.0
MIN_LOOKBACK_BARS = 5
MAX_LOOKBACK_BARS = 40
BREAKOUT_WINDOW_BARS = 30   # 6 weeks (30 sessions) after Point B
MAX_HOLD_DAYS = 90

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
            "volume": float(r.get("volume", r.get("delivery_vol", 0)) or 0)
        }
        if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
            date_map[d_str] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 50:
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

    return deduped

def find_swing_lows_fast(arr, left=3, right=3):
    lows = []
    n = len(arr)
    for i in range(left, n - right):
        val = arr[i]
        is_min = True
        for j in range(i - left, i + right + 1):
            if j != i and arr[j] <= val:
                is_min = False
                break
        if is_min:
            lows.append(i)
    return lows

def fast_rolling_mean(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res = np.empty_like(arr, dtype=float)
    res[:window-1] = np.nan
    res[window-1:] = ret[window-1:] / window
    res[:window-1] = res[window-1] if len(res) >= window else 1.0
    return res

def run_backtest():
    print("🚀 Running Upgraded OBV Strategy Backtest (50 SMA + Delivery Confirmation)...")
    nifty_750_set = get_nifty_750_universe()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "wyckoff_screener_results.json", "obv_backtest_report.json"
        ]
    ]

    if nifty_750_set and len(nifty_750_set) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in nifty_750_set]

    print(f"📦 Evaluating across {len(stock_files)} stocks...")

    trades_swing_mode = []
    trades_obv_mode = []
    processed = 0

    for f_name in stock_files:
        processed += 1
        if processed % 300 == 0:
            print(f"⏳ Evaluated {processed}/{len(stock_files)} stocks...")

        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        if pe_val is not None:
            try:
                pe_float = float(pe_val)
                if pe_float <= 0 or pe_float >= MAX_PE:
                    continue
            except (ValueError, TypeError):
                pass

        try:
            with open(json_path, "r") as fp:
                raw = json.load(fp)
        except Exception:
            continue

        clean = clean_data_fast(raw)
        if len(clean) < 60:
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        traded_vols = np.array([r["volume"] for r in clean], dtype=float)
        deliv_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # Indicator Calculations
        vol_sma9 = fast_rolling_mean(traded_vols, 9)
        deliv_sma20 = fast_rolling_mean(deliv_vols, 20)
        sma_50 = fast_rolling_mean(closes, 50)

        # True Demat Delivery OBV
        obvs = np.zeros(N, dtype=float)
        cur_obv = 0.0
        for idx in range(N):
            dv = min(deliv_vols[idx], traded_vols[idx]) if traded_vols[idx] > 0 else deliv_vols[idx]
            if idx > 0:
                if closes[idx] > closes[idx - 1]: cur_obv += dv
                elif closes[idx] < closes[idx - 1]: cur_obv -= dv
            else:
                cur_obv = dv
            obvs[idx] = cur_obv

        obv_sma20 = fast_rolling_mean(obvs, 20)
        obv_lows = find_swing_lows_fast(obvs, 3, 3)

        cooldown_idx = 0

        for i in range(50, N - 1):
            if i < cooldown_idx:
                continue

            # 1. Broad Trend Filter: Price must trade above 50 SMA
            if closes[i] < sma_50[i]:
                continue

            if vol_sma9[i] < MIN_AVG_VOLUME_9D:
                continue

            valid_lows = [idx for idx in obv_lows if (i - idx) <= (MAX_LOOKBACK_BARS + BREAKOUT_WINDOW_BARS) and (i - idx) >= 2]
            if len(valid_lows) < 2:
                continue

            found_setup = False
            for idx_b in reversed(valid_lows):
                bars_since_b = i - idx_b
                if bars_since_b > BREAKOUT_WINDOW_BARS or bars_since_b < 1:
                    continue

                # 2. Base Quality Filter: Reject divergence happening at fresh 52-week lows
                low_200 = np.min(lows[max(0, idx_b - 150) : idx_b + 1])
                if lows[idx_b] <= (low_200 * 1.01):
                    continue

                for idx_a in reversed(valid_lows):
                    span = idx_b - idx_a
                    if MIN_LOOKBACK_BARS <= span <= MAX_LOOKBACK_BARS:
                        if closes[idx_a] <= 0 or abs(obvs[idx_a]) == 0:
                            continue

                        p_drop = ((closes[idx_b] - closes[idx_a]) / closes[idx_a]) * 100.0
                        o_gain = ((obvs[idx_b] - obvs[idx_a]) / abs(obvs[idx_a])) * 100.0

                        if p_drop <= MIN_PRICE_DROP_PCT and o_gain >= MIN_OBV_GAIN_PCT:
                            target_swing_high = None
                            target_swing_obv = None
                            sl_price = None
                            pattern_name = ""

                            # Pattern 1: Breakout above intermediate peak between A & B
                            mid_high_idx = idx_a + int(np.argmax(highs[idx_a : idx_b + 1]))
                            sh1_price = highs[mid_high_idx]
                            sh1_obv = obvs[mid_high_idx]

                            if (closes[i] > sh1_price and closes[i - 1] <= sh1_price) and (obvs[i] > sh1_obv):
                                target_swing_high = sh1_price
                                target_swing_obv = sh1_obv
                                sl_price = round(lows[idx_b] * 0.995, 2)
                                pattern_name = "Pattern 1 (Base Breakout)"

                            # Pattern 2: Right-Side Pivot Breakout after Point B
                            elif bars_since_b >= 4:
                                post_b_high_idx = idx_b + int(np.argmax(highs[idx_b : i]))
                                sh2_price = highs[post_b_high_idx]
                                sh2_obv = obvs[post_b_high_idx]

                                if post_b_high_idx > idx_b and (closes[i] > sh2_price and closes[i - 1] <= sh2_price) and (obvs[i] > sh2_obv):
                                    recent_pullback_low = float(np.min(lows[post_b_high_idx : i]))
                                    target_swing_high = sh2_price
                                    target_swing_obv = sh2_obv
                                    sl_price = round(recent_pullback_low * 0.995, 2)
                                    pattern_name = "Pattern 2 (Right-Side Pivot Breakout)"

                            if not target_swing_high or not sl_price:
                                continue

                            # 3. Institutional Delivery Expansion Gate on Breakout Day
                            if deliv_vols[i] < (1.25 * deliv_sma20[i]):
                                continue

                            entry_price = closes[i]
                            entry_date = times[i]
                            risk_pct = round(((entry_price - sl_price) / entry_price) * 100.0, 2)

                            # Risk filter: Must be <= 8%
                            if risk_pct > MAX_RISK_PCT or risk_pct <= 0:
                                continue

                            # Forward simulation
                            fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                            fwd_highs = highs[i + 1 : fwd_end]
                            fwd_lows = lows[i + 1 : fwd_end]
                            fwd_closes = closes[i + 1 : fwd_end]
                            fwd_obvs = obvs[i + 1 : fwd_end]
                            fwd_obv_sma = obv_sma20[i + 1 : fwd_end]

                            if len(fwd_highs) < 2:
                                continue

                            # -----------------------------------------------------------------
                            # SIMULATION 1: Swing Low Trailing
                            # -----------------------------------------------------------------
                            max_run_1 = 0.0
                            active_sl_1 = sl_price
                            booked_15_1 = False
                            exit_p_1 = fwd_closes[-1]
                            days_1 = len(fwd_highs)

                            for d_idx in range(len(fwd_highs)):
                                h_bar = fwd_highs[d_idx]
                                l_bar = fwd_lows[d_idx]
                                gain = ((h_bar - entry_price) / entry_price) * 100.0
                                if gain > max_run_1:
                                    max_run_1 = gain

                                # Shift SL to Breakeven at +9%
                                if max_run_1 >= BREAKEVEN_TRIGGER_PCT and active_sl_1 < entry_price:
                                    active_sl_1 = entry_price

                                # Book 50% at +15%
                                if max_run_1 >= PARTIAL_TARGET_PCT and not booked_15_1:
                                    booked_15_1 = True
                                    active_sl_1 = entry_price

                                if booked_15_1 and d_idx >= 10:
                                    trail_low = float(np.min(lows[i + 1 + d_idx - 10 : i + 1 + d_idx]))
                                    if trail_low > active_sl_1:
                                        active_sl_1 = trail_low

                                if l_bar <= active_sl_1:
                                    exit_p_1 = min(fwd_closes[d_idx], active_sl_1)
                                    days_1 = d_idx + 1
                                    break

                            raw_ret_1 = ((exit_p_1 - entry_price) / entry_price) * 100.0
                            ret_1 = round((PARTIAL_TARGET_PCT * 0.50) + (raw_ret_1 * 0.50), 2) if booked_15_1 else round(raw_ret_1, 2)

                            trades_swing_mode.append({
                                "Symbol": sym,
                                "Pattern": pattern_name,
                                "Entry Date": entry_date,
                                "Entry Price": entry_price,
                                "Exit Price": round(exit_p_1, 2),
                                "Risk %": risk_pct,
                                "Return %": ret_1,
                                "Max Run %": round(max_run_1, 2),
                                "Rally 20%": bool(max_run_1 >= 20.0),
                                "Days Held": days_1,
                                "Is Win": bool(ret_1 > 0)
                            })

                            # -----------------------------------------------------------------
                            # SIMULATION 2: OBV Breakdown Exit
                            # -----------------------------------------------------------------
                            max_run_2 = 0.0
                            active_sl_2 = sl_price
                            booked_15_2 = False
                            exit_p_2 = fwd_closes[-1]
                            days_2 = len(fwd_highs)

                            for d_idx in range(len(fwd_highs)):
                                h_bar = fwd_highs[d_idx]
                                l_bar = fwd_lows[d_idx]
                                gain = ((h_bar - entry_price) / entry_price) * 100.0
                                if gain > max_run_2:
                                    max_run_2 = gain

                                if max_run_2 >= BREAKEVEN_TRIGGER_PCT and active_sl_2 < entry_price:
                                    active_sl_2 = entry_price

                                if max_run_2 >= PARTIAL_TARGET_PCT and not booked_15_2:
                                    booked_15_2 = True
                                    active_sl_2 = entry_price

                                if l_bar <= active_sl_2:
                                    exit_p_2 = min(fwd_closes[d_idx], active_sl_2)
                                    days_2 = d_idx + 1
                                    break

                                # OBV Breakdown
                                if d_idx >= 5 and fwd_obvs[d_idx] < fwd_obv_sma[d_idx]:
                                    exit_p_2 = fwd_closes[d_idx]
                                    days_2 = d_idx + 1
                                    break

                            raw_ret_2 = ((exit_p_2 - entry_price) / entry_price) * 100.0
                            ret_2 = round((PARTIAL_TARGET_PCT * 0.50) + (raw_ret_2 * 0.50), 2) if booked_15_2 else round(raw_ret_2, 2)

                            trades_obv_mode.append({
                                "Symbol": sym,
                                "Pattern": pattern_name,
                                "Entry Date": entry_date,
                                "Entry Price": entry_price,
                                "Exit Price": round(exit_p_2, 2),
                                "Risk %": risk_pct,
                                "Return %": ret_2,
                                "Max Run %": round(max_run_2, 2),
                                "Rally 20%": bool(max_run_2 >= 20.0),
                                "Days Held": days_2,
                                "Is Win": bool(ret_2 > 0)
                            })

                            cooldown_idx = i + max(days_1, 10)
                            found_setup = True
                            break
                    if found_setup:
                        break

    def calc_stats(trade_list):
        if not trade_list:
            return {"Trades": 0, "Win Rate %": "0%", "+20% Expansion": "0%", "Avg Return %": "0%", "Profit Factor": 0.0, "Avg Hold": "0 d", "Avg Risk %": "0%"}
        df = pd.DataFrame(trade_list)
        total = len(df)
        wins = len(df[df["Is Win"] == True])
        win_rate = round((wins / total) * 100, 1)
        r20 = round((len(df[df["Rally 20%"] == True]) / total) * 100, 1)
        avg_ret = round(float(df["Return %"].mean()), 2)
        gross_win = df[df["Is Win"] == True]["Return %"].sum()
        gross_loss = abs(df[df["Is Win"] == False]["Return %"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_hold = round(float(df["Days Held"].mean()), 1)
        avg_risk = round(float(df["Risk %"].mean()), 2)

        return {
            "Trades": total,
            "Win Rate %": f"{win_rate}%",
            "+20% Expansion": f"{r20}%",
            "Avg Return %": f"{'+' if avg_ret >= 0 else ''}{avg_ret}%",
            "Profit Factor": pf,
            "Avg Hold": f"{avg_hold} d",
            "Avg Risk %": f"{avg_risk}%"
        }

    stats_swing = calc_stats(trades_swing_mode)
    stats_obv = calc_stats(trades_obv_mode)

    print("\n" + "="*70)
    print("🎯 RESULTS (UPGRADED OBV DIVERGENCE ENGINE)")
    print("="*70)
    print(f"Mode 1 (Swing Low Trailing SL) : {stats_swing['Trades']} Trades | Win Rate {stats_swing['Win Rate %']} | Return {stats_swing['Avg Return %']} | PF {stats_swing['Profit Factor']}")
    print(f"Mode 2 (OBV Breakdown Exit)    : {stats_obv['Trades']} Trades | Win Rate {stats_obv['Win Rate %']} | Return {stats_obv['Avg Return %']} | PF {stats_obv['Profit Factor']}")
    print("="*70)

    payload = {
        "Comparison Summary": {
            "Swing Low Trailing Exit (Mode 1)": stats_swing,
            "OBV Breakdown Exit (Mode 2)": stats_obv
        },
        "Recent Trades Mode 1": trades_swing_mode[-25:] if len(trades_swing_mode) >= 25 else trades_swing_mode,
        "Recent Trades Mode 2": trades_obv_mode[-25:] if len(trades_obv_mode) >= 25 else trades_obv_mode
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"📁 Full report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_backtest()
