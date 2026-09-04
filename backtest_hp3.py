import os
import json
import time
import traceback
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "backtest_hp3_report.json")

# HP3 SPECIFICATIONS
BASE_W = 60
VOL_M = 1.50
PCT_M = 1.30
MIN_DOTS = 4
MAX_RISK = 15.0
TARGET_PCT = 15.0
BE_TRIGGER = 10.0
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 10000
START_DATE = "2021-01-01"

def load_nifty_750_set():
    """Load exclusively from local disk without hitting external network."""
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
                if isinstance(raw, list):
                    return set(str(x).strip().upper() for x in raw)
                elif isinstance(raw, dict):
                    return set(str(x).strip().upper() for x in raw.keys())
        except Exception as e:
            print(f"Warning reading nifty750.json: {e}")
    return set()

def clean_stock_records(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        t_val = str(r.get("time", "")).strip()[:10]
        if len(t_val) < 10:
            continue
        try:
            c = float(r.get("close", 0) or 0)
        except Exception:
            continue
        if c <= 0 or np.isnan(c):
            continue
        
        try:
            entry = {
                "time": t_val,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            }
        except Exception:
            continue

        if t_val not in date_map or entry["volume"] > date_map[t_val]["volume"]:
            date_map[t_val] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < (BASE_W + 15):
        return []

    clean = [date_map[k] for k in sorted_dates]

    # Split / Corporate Action Adjustments
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

    # Synthetic volume smoothing for non-liquid candles
    running_vol = 25000.0
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

def run_backtest():
    all_trades = []
    log_messages = []
    
    try:
        print("Starting HP3 Historical Backtest...")
        n750_set = load_nifty_750_set()
        log_messages.append(f"Loaded {len(n750_set)} Nifty 750 tickers to exclude.")

        # Exclude system files
        reserved = {
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "wyckoff_screener_results.json", "obv_backtest_report.json",
            "scana_vs_absorption_report.json", "scana_candidates.json",
            "optimal_strategies.json", "scana_sensitivity_report.json",
            "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
            "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
            "backtest_hp3_report.json", "scan_macro_results.json"
        }

        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in reserved]
        # Restrict strictly to non-Nifty 750
        target_files = [f for f in all_files if f.replace(".json", "").strip().upper() not in n750_set]
        log_messages.append(f"Target files outside Nifty 750: {len(target_files)}")

        # Optional P/E check
        fund_map = {}
        fund_path = os.path.join(DATA_DIR, "fundamentals.json")
        if os.path.exists(fund_path):
            try:
                with open(fund_path, "r", encoding="utf-8") as fp:
                    fund_map = json.load(fp)
            except Exception:
                pass

        processed_stocks = 0

        for f_name in target_files:
            try:
                sym = f_name.replace(".json", "").strip().upper()
                json_path = os.path.join(DATA_DIR, f_name)

                if sym in fund_map:
                    pe_val = fund_map[sym].get("pe")
                    if pe_val is not None:
                        try:
                            pe_flt = float(pe_val)
                            if pe_flt <= 0 or pe_flt > MAX_PE:
                                continue
                        except Exception:
                            pass

                with open(json_path, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)

                clean = clean_stock_records(raw)
                if len(clean) < (BASE_W + 20):
                    continue

                processed_stocks += 1

                closes = np.array([r["close"] for r in clean], dtype=float)
                highs = np.array([r["high"] for r in clean], dtype=float)
                lows = np.array([r["low"] for r in clean], dtype=float)
                volumes = np.array([r["volume"] for r in clean], dtype=float)
                pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
                d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
                times = [r["time"] for r in clean]
                N = len(closes)

                deliv_sma20 = fast_rolling(d_vols, 20)
                pct_sma50 = fast_rolling(pcts, 50)
                vol_sma9 = fast_rolling(volumes, 9)

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

                dots = (pcts >= (PCT_M * pct_sma50)) & (d_vols >= (VOL_M * deliv_sma20))
                dot_cumsum = np.cumsum(dots.astype(int))

                cooldown = 0

                for i in range(BASE_W + 5, N - 1):
                    if i < cooldown:
                        continue

                    entry_date = times[i]
                    if entry_date < START_DATE:
                        continue

                    if vol_sma9[i] < MIN_AVG_VOLUME_9D:
                        continue

                    num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - BASE_W)]
                    if num_dots < MIN_DOTS:
                        continue

                    base_highs = highs[i - BASE_W : i]
                    if len(base_highs) == 0 or np.isnan(base_highs).all():
                        continue

                    sw_idx = int(np.nanargmax(base_highs))
                    swing_high = float(base_highs[sw_idx])

                    obv_pos = i - BASE_W + sw_idx
                    if obv_pos < 0 or obv_pos >= N:
                        continue
                    swing_obv = float(obvs[obv_pos])

                    if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                        entry_p = float(closes[i])
                        lookback = min(15, BASE_W)
                        recent_low = float(np.nanmin(lows[i - lookback : i]))
                        sl_p = round(recent_low * 0.995, 2)
                        risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                        if risk_pct <= 0 or risk_pct > MAX_RISK or np.isnan(risk_pct):
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
                        be_shifted = False
                        exit_p = float(f_closes[-1])
                        exit_date = times[fwd_end - 1]
                        hold_days = len(f_highs)

                        for bar_idx in range(len(f_highs)):
                            curr_bar = i + 1 + bar_idx
                            gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                            if gain > max_run:
                                max_run = gain

                            if max_run >= BE_TRIGGER and not be_shifted and active_sl < entry_p:
                                active_sl = entry_p
                                be_shifted = True

                            if max_run >= TARGET_PCT and not booked_partial:
                                booked_partial = True
                                active_sl = entry_p

                            if booked_partial and bar_idx >= 10:
                                t_low = float(np.nanmin(lows[curr_bar - 10 : curr_bar]))
                                if t_low > active_sl:
                                    active_sl = t_low

                            if f_lows[bar_idx] <= active_sl:
                                exit_p = float(min(f_closes[bar_idx], active_sl))
                                exit_date = times[curr_bar]
                                hold_days = bar_idx + 1
                                break

                        raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                        realized_ret = round((TARGET_PCT * 0.50) + (raw_ret * 0.50), 2) if booked_partial else round(raw_ret, 2)

                        all_trades.append({
                            "symbol": sym,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "entry_price": round(entry_p, 2),
                            "exit_price": round(exit_p, 2),
                            "initial_sl": sl_p,
                            "risk_pct": risk_pct,
                            "realized_return": realized_ret,
                            "max_run_gain": round(max_run, 2),
                            "hold_days": hold_days,
                            "partial_booked": booked_partial,
                            "win": bool(realized_ret > 0),
                            "r20": bool(max_run >= 20.0)
                        })

                        cooldown = i + max(hold_days, 8)
            except Exception:
                continue

        log_messages.append(f"Successfully processed {processed_stocks} stocks. Total trades: {len(all_trades)}")

    except Exception as e:
        log_messages.append(f"Fatal error encountered: {str(e)}")
        traceback.print_exc()

    # Calculation and payload compilation
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t["win"])
    losses = total_trades - wins
    win_rate = round((wins / total_trades) * 100.0, 2) if total_trades > 0 else 0.0
    returns = [t["realized_return"] for t in all_trades]
    avg_return = round(float(np.mean(returns)), 2) if total_trades > 0 else 0.0

    pos_sum = sum(r for r in returns if r > 0)
    neg_sum = abs(sum(r for r in returns if r < 0))
    pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else (99.0 if pos_sum > 0 else 0.0)
    r20_pct = round((sum(1 for t in all_trades if t["r20"]) / total_trades) * 100.0, 2) if total_trades > 0 else 0.0

    all_trades.sort(key=lambda x: x["entry_date"], reverse=True)

    yearly_breakdown = {}
    for t in all_trades:
        yr = t["entry_date"][:4]
        if yr not in yearly_breakdown:
            yearly_breakdown[yr] = {"trades": 0, "wins": 0, "returns": []}
        yearly_breakdown[yr]["trades"] += 1
        if t["win"]: yearly_breakdown[yr]["wins"] += 1
        yearly_breakdown[yr]["returns"].append(t["realized_return"])

    yearly_stats = []
    for yr in sorted(yearly_breakdown.keys()):
        d = yearly_breakdown[yr]
        tr = d["trades"]
        w = d["wins"]
        wr = round((w / tr) * 100.0, 1)
        ar = round(float(np.mean(d["returns"])), 2)
        yearly_stats.append({
            "Year": yr,
            "Trades": tr,
            "Win Rate": f"{wr}%",
            "Avg Return": f"{'+' if ar >= 0 else ''}{ar}%"
        })

    report = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Backtest Period": f"{START_DATE} to Present",
        "Target Universe": "NSE Stocks Outside Nifty 750",
        "Strategy Config": {
            "Base Window": f"{BASE_W} days",
            "Accumulation Dots": f">={MIN_DOTS} dots (1.5x Vol, 1.3x Deliv)",
            "Max Risk (SL Cap)": f"{MAX_RISK}%",
            "Target (50% Booking)": f"+{TARGET_PCT}%",
            "BE Shift Trigger": f"+{BE_TRIGGER}%",
            "Min 9D Volume": MIN_AVG_VOLUME_9D
        },
        "Summary Metrics": {
            "Total Trades": total_trades,
            "Win Rate": f"{win_rate}%",
            "Profit Factor": pf,
            "Average Return Per Trade": f"{'+' if avg_return >= 0 else ''}{avg_return}%",
            "+20% Expansion Rate": f"{r20_pct}%",
            "Winning Trades": wins,
            "Losing Trades": losses
        },
        "Yearly Performance": yearly_stats,
        "Execution Logs": log_messages,
        "Recent Trades Sample": all_trades[:50]
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"Report written to {OUTPUT_REPORT} with {total_trades} trades.")

if __name__ == "__main__":
    run_backtest()
