import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "scana_vs_absorption_report.json")
MAX_PE = 35.0
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
            "volume": float(r.get("volume", 0) or 0),
            "deliv_pct": float(r.get("deliv_pct", 0) or 0)
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

def run_comparative_backtest():
    print("🚀 Running Scan A vs. Institutional Absorption Engine Backtest...")
    nifty_750_set = get_nifty_750_universe()

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
            "scana_vs_absorption_report.json"
        ]
    ]

    if nifty_750_set and len(nifty_750_set) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in nifty_750_set]

    print(f"📦 Evaluating across {len(stock_files)} stocks...")

    trades_scan_a = []
    trades_absorption = []
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
        opens = np.array([r["open"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
        d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # Baseline Indicators
        to_cr = (closes * volumes) / 1e7
        to_50 = fast_rolling_mean(to_cr, 50)
        pct_50 = fast_rolling_mean(pcts, 50)
        deliv_sma20 = fast_rolling_mean(d_vols, 20)
        vol_sma10 = fast_rolling_mean(volumes, 10)
        vol_sma50 = fast_rolling_mean(volumes, 50)

        # Volatility Contraction
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        natr10 = (fast_rolling_mean(tr, 10) / np.maximum(closes, 1e-4)) * 100.0
        natr50 = (fast_rolling_mean(tr, 50) / np.maximum(closes, 1e-4)) * 100.0

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

        cooldown_a = 0
        cooldown_abs = 0

        for i in range(50, N - 1):
            curr_to = to_50[i]
            if curr_to >= 30.0:
                pct_m, vol_m, min_c, base_w = 1.35, 1.0, 3, 40
            elif curr_to >= 5.0:
                pct_m, vol_m, min_c, base_w = 1.20, 1.0, 2, 20
            else:
                pct_m, vol_m, min_c, base_w = 1.35, 1.0, 3, 20

            base_start = max(0, i - base_w)
            qualifying_dots = (pcts[base_start:i] >= (pct_m * pct_50[base_start:i])) & \
                              (d_vols[base_start:i] >= (vol_m * deliv_sma20[base_start:i]))
            delivery_cluster_present = bool(np.sum(qualifying_dots) >= min_c)

            if not delivery_cluster_present:
                continue

            base_highs = highs[base_start:i]
            sw_idx = int(np.argmax(base_highs))
            major_swing_high = base_highs[sw_idx]
            major_swing_obv = obvs[base_start + sw_idx]

            # -------------------------------------------------------------
            # MODEL 1: Scan A (Breakout of Major Swing High)
            # -------------------------------------------------------------
            if i > cooldown_a:
                if (closes[i] > major_swing_high and closes[i - 1] <= major_swing_high) and (obvs[i] > major_swing_obv):
                    entry_p = closes[i]
                    lookback_sl = min(12, i - base_start)
                    recent_low = float(np.min(lows[i - lookback_sl : i]))
                    sl_p = round(recent_low * 0.995, 2)
                    risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                    if 0 < risk_pct <= 8.0:
                        # Forward simulation Model 1
                        fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                        f_highs = highs[i + 1 : fwd_end]
                        f_lows = lows[i + 1 : fwd_end]
                        f_closes = closes[i + 1 : fwd_end]

                        if len(f_highs) >= 2:
                            max_run = 0.0
                            active_sl = sl_p
                            booked_15 = False
                            exit_p = f_closes[-1]
                            days = len(f_highs)

                            for d in range(len(f_highs)):
                                gain = ((f_highs[d] - entry_p) / entry_p) * 100.0
                                if gain > max_run: max_run = gain

                                if max_run >= 15.0 and not booked_15:
                                    booked_15 = True
                                    active_sl = entry_p

                                if booked_15 and d >= 10:
                                    trail = float(np.min(lows[i + 1 + d - 10 : i + 1 + d]))
                                    if trail > active_sl: active_sl = trail

                                if f_lows[d] <= active_sl:
                                    exit_p = min(f_closes[d], active_sl)
                                    days = d + 1
                                    break

                            raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                            fin_ret = round((15.0 * 0.50) + (raw_ret * 0.50), 2) if booked_15 else round(raw_ret, 2)

                            trades_scan_a.append({
                                "Symbol": sym,
                                "Entry Date": times[i],
                                "Entry": entry_p,
                                "Exit": round(exit_p, 2),
                                "Risk %": risk_pct,
                                "Realized %": fin_ret,
                                "Max Run %": round(max_run, 2),
                                "Rally 20%": bool(max_run >= 20.0),
                                "Hold Days": days,
                                "Is Win": bool(fin_ret > 0)
                            })
                            cooldown_a = i + max(days, 10)

            # -------------------------------------------------------------
            # MODEL 2: Institutional Absorption (Supply Dry-Up + Shelf Entry)
            # -------------------------------------------------------------
            if i > cooldown_abs:
                # 1. Supply Drought: volume dried up vs 50-day average
                supply_dry = (vol_sma50[i] > 0) and (vol_sma10[i] <= (0.80 * vol_sma50[i]))
                
                # 2. Float Lock-up: Volatility contraction
                vcp_tight = (natr50[i] > 0) and ((natr10[i] / natr50[i]) <= 0.75)

                # 3. Tight 5-Day Shelf right before breakout (range <= 5.5%)
                shelf_high = float(np.max(highs[i - 5 : i]))
                shelf_low = float(np.min(lows[i - 5 : i]))
                shelf_depth_pct = ((shelf_high - shelf_low) / shelf_high) * 100.0
                is_shelf_tight = shelf_depth_pct <= 5.5

                # 4. Entry: price clears the 5-day shelf, positioned safely below or at resistance
                if supply_dry and vcp_tight and is_shelf_tight and (closes[i] > shelf_high):
                    entry_p = closes[i]
                    sl_p = round(shelf_low * 0.995, 2)
                    risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                    # Shelf stops are structurally compact: 2.0% to 5.0%
                    if 0 < risk_pct <= 5.0:
                        fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                        f_highs = highs[i + 1 : fwd_end]
                        f_lows = lows[i + 1 : fwd_end]
                        f_closes = closes[i + 1 : fwd_end]

                        if len(f_highs) >= 2:
                            max_run = 0.0
                            active_sl = sl_p
                            booked_12 = False
                            exit_p = f_closes[-1]
                            days = len(f_highs)

                            for d in range(len(f_highs)):
                                gain = ((f_highs[d] - entry_p) / entry_p) * 100.0
                                if gain > max_run: max_run = gain

                                # Shift SL to Breakeven early at +6.0% (protects tight shelf entries)
                                if max_run >= 6.0 and active_sl < entry_p:
                                    active_sl = entry_p

                                # Book 50% at +12.0%
                                if max_run >= 12.0 and not booked_12:
                                    booked_12 = True
                                    active_sl = entry_p

                                # Trail 10-day swing lows on remainder
                                if booked_12 and d >= 10:
                                    trail = float(np.min(lows[i + 1 + d - 10 : i + 1 + d]))
                                    if trail > active_sl: active_sl = trail

                                if f_lows[d] <= active_sl:
                                    exit_p = min(f_closes[d], active_sl)
                                    days = d + 1
                                    break

                            raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                            fin_ret = round((12.0 * 0.50) + (raw_ret * 0.50), 2) if booked_12 else round(raw_ret, 2)

                            trades_absorption.append({
                                "Symbol": sym,
                                "Entry Date": times[i],
                                "Entry": entry_p,
                                "Exit": round(exit_p, 2),
                                "Risk %": risk_pct,
                                "Realized %": fin_ret,
                                "Max Run %": round(max_run, 2),
                                "Rally 20%": bool(max_run >= 20.0),
                                "Hold Days": days,
                                "Is Win": bool(fin_ret > 0)
                            })
                            cooldown_abs = i + max(days, 8)

    def calc_stats(trades):
        if not trades:
            return {"Trades": 0, "Win Rate %": "0%", "+20% Expansion": "0%", "Avg Return %": "0%", "Profit Factor": 0.0, "Avg Hold": "0 d", "Avg Risk %": "0%"}
        df = pd.DataFrame(trades)
        total = len(df)
        wins = len(df[df["Is Win"] == True])
        win_rate = round((wins / total) * 100, 1)
        r20 = round((len(df[df["Rally 20%"] == True]) / total) * 100, 1)
        avg_ret = round(float(df["Realized %"].mean()), 2)
        gross_win = df[df["Is Win"] == True]["Realized %"].sum()
        gross_loss = abs(df[df["Is Win"] == False]["Realized %"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_hold = round(float(df["Hold Days"].mean()), 1)
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

    stats_scan_a = calc_stats(trades_scan_a)
    stats_abs = calc_stats(trades_absorption)

    print("\n" + "="*75)
    print("🎯 COMPARISON: SCAN A vs INSTITUTIONAL ABSORPTION (SHELF ENTRY)")
    print("="*75)
    print(f"Model 1: Scan A Breakout       : {stats_scan_a['Trades']} Trades | Win Rate {stats_scan_a['Win Rate %']} | Return {stats_scan_a['Avg Return %']} | PF {stats_scan_a['Profit Factor']} | Risk {stats_scan_a['Avg Risk %']}")
    print(f"Model 2: Absorption Shelf      : {stats_abs['Trades']} Trades | Win Rate {stats_abs['Win Rate %']} | Return {stats_abs['Avg Return %']} | PF {stats_abs['Profit Factor']} | Risk {stats_abs['Avg Risk %']}")
    print("="*75)

    payload = {
        "Summary": {
            "Scan A Breakout": stats_scan_a,
            "Institutional Absorption Shelf": stats_abs
        },
        "Recent Trades Absorption": trades_absorption[-25:] if len(trades_absorption) >= 25 else trades_absorption,
        "Recent Trades Scan A": trades_scan_a[-25:] if len(trades_scan_a) >= 25 else trades_scan_a
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"📁 Comparative report written to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_comparative_backtest()
