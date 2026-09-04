import os
import io
import json
import time
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "backtest_hp3_report.json")

# SCAN_HP3 BACKTEST PARAMETERS
BASE_W = 60
VOL_M = 1.50
PCT_M = 1.30
MIN_DOTS = 4
MAX_RISK = 15.0
TARGET_PCT = 15.0
BE_TRIGGER = 10.0
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 25000
START_DATE = "2021-01-01"

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
    if len(sorted_dates) < (BASE_W + 20):
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

def run_backtest():
    print(f"🚀 Initializing HP3 Backtest (From 2021 to Present)...")
    print(f"⚙️ Target Universe: Strictly Outside Nifty 750 | Base: {BASE_W}d | Max Risk: {MAX_RISK}%")

    nifty_750 = get_nifty_750_universe()
    print(f"📦 Loaded {len(nifty_750)} Nifty 750 symbols to exclude.")

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as fp:
                fundamentals = json.load(fp)
        except Exception:
            pass

    all_stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "wyckoff_screener_results.json", "obv_backtest_report.json",
            "scana_vs_absorption_report.json", "scana_candidates.json",
            "optimal_strategies.json", "scana_sensitivity_report.json",
            "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
            "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
            "backtest_hp3_report.json"
        ]
    ]

    stock_files = [f for f in all_stock_files if f.replace(".json", "").strip().upper() not in nifty_750]
    print(f"🎯 Evaluating {len(stock_files)} stocks outside Nifty 750...")

    all_trades = []

    for f_name in stock_files:
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
        if len(clean) < (BASE_W + 20):
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
        d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        deliv_sma20 = fast_rolling_mean(d_vols, 20)
        pct_sma50 = fast_rolling_mean(pcts, 50)
        vol_sma9 = fast_rolling_mean(volumes, 9)

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

        for i in range(BASE_W + 10, N - 1):
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
            sw_idx = int(np.argmax(base_highs))
            swing_high = base_highs[sw_idx]
            swing_obv = obvs[i - BASE_W + sw_idx]

            if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                entry_p = closes[i]
                lookback_sl = min(15, BASE_W)
                recent_low = float(np.min(lows[i - lookback_sl : i]))
                sl_p = round(recent_low * 0.995, 2)
                risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                if risk_pct <= 0 or risk_pct > MAX_RISK:
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
                exit_p = f_closes[-1]
                exit_date = times[fwd_end - 1]
                hold_days = len(f_highs)

                for bar_idx in range(len(f_highs)):
                    curr_bar = i + 1 + bar_idx
                    gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                    if gain > max_run:
                        max_run = gain

                    # BE Shift at +10%
                    if max_run >= BE_TRIGGER and not be_shifted and active_sl < entry_p:
                        active_sl = entry_p
                        be_shifted = True

                    # Book 50% at +15%
                    if max_run >= TARGET_PCT and not booked_partial:
                        booked_partial = True
                        active_sl = entry_p

                    # Trail 10-day lows
                    if booked_partial and bar_idx >= 10:
                        t_low = float(np.min(lows[curr_bar - 10 : curr_bar]))
                        if t_low > active_sl:
                            active_sl = t_low

                    if f_lows[bar_idx] <= active_sl:
                        exit_p = min(f_closes[bar_idx], active_sl)
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
                    "win": realized_ret > 0,
                    "r20": max_run >= 20.0
                })

                cooldown = i + max(hold_days, 8)

    total_trades = len(all_trades)
    print(f"\n📊 Backtest Execution Complete! Simulated {total_trades} trades since 2021.")

    if total_trades == 0:
        print("❌ No trades triggered. Verify dataset depth and pathing.")
        return

    wins = sum(1 for t in all_trades if t["win"])
    losses = total_trades - wins
    win_rate = round((wins / total_trades) * 100.0, 2)
    returns = [t["realized_return"] for t in all_trades]
    avg_return = round(float(np.mean(returns)), 2)
    
    pos_sum = sum(r for r in returns if r > 0)
    neg_sum = abs(sum(r for r in returns if r < 0))
    pf = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0
    r20_pct = round((sum(1 for t in all_trades if t["r20"]) / total_trades) * 100.0, 2)

    # Sort trades chronologically
    all_trades.sort(key=lambda x: x["entry_date"], reverse=True)

    # Annual breakdown
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
        "Recent Trades Sample": all_trades[:50]
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(report, fp, indent=2)

    print(f"📁 Detailed report written to {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_backtest()
